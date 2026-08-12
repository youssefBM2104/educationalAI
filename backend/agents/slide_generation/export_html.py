import logging
import re
import tempfile
from pathlib import Path

from backend.agents.state import LectureState
from backend.agents.slide_generation._common import build_image_index, images_of_chunk
from backend.agents.slide_generation.template_store import reference_from_pptx, default_reference
from backend.agents.slide_generation.html_slide_agent import generate_slide_html
from backend.agents.slide_generation.html_render import html_to_png, pngs_to_pptx

logger = logging.getLogger(__name__)

_IMG_TOKEN = re.compile(r"\{\{IMG:([^}]+)\}\}")
_FORBIDDEN = r'<>:"/\|?*'

def _safe_stem(name: str, max_len: int = 120) -> str:
    cleaned = re.sub(f"[{re.escape(_FORBIDDEN)}]", "-", name or "")
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    cleaned = re.sub(r"[-_]{2,}", "_", cleaned).strip("._-")
    return (cleaned or "lecture")[:max_len]


def _slide_lines(slide: dict) -> list[str]:
    """Flatten a slide's content (bullets + slots) into text lines — used to detect empty slides."""
    lines = list(slide.get("bullets") or [])
    slots = slide.get("slots") or {}
    if slots.get("subtitle"):
        lines.append(str(slots["subtitle"]))
    if slots.get("stat"):
        lines.append(f"{slots['stat']} — {slots.get('explanation', '')}".strip(" —"))
    for card in slots.get("cards") or []:
        lines.append(f"{card.get('label', '')}: {card.get('text', '')}".strip(": ")
                     if isinstance(card, dict) else str(card))
    if slots.get("caption"):
        lines.append(str(slots["caption"]))
    return lines


def _backfill_empty_slides(slides: list[dict], headings: list[dict]) -> int:
    """Safety net for planner drop-outs: a slide with a title but no bullets/slots is backfilled from
    the Composer's `content_points` so it is never blank. Deterministic."""
    by_heading = {h.get("heading_id"): (h.get("content_points") or []) for h in headings or []}
    fixed = 0
    for s in slides:
        if _slide_lines(s) or (s.get("image_refs") or []):
            continue
        points = by_heading.get(s.get("heading_id"))
        if points:
            s["bullets"] = list(points)
            fixed += 1
    return fixed


def _full_image_map(chunks: list | None) -> dict:
    """image_id -> the FULL image object (incl. vlm_description) — for prompt reasoning fields."""
    out = {}
    for c in chunks or []:
        for img in images_of_chunk(c):
            iid = img.get("image_id")
            if iid and iid not in out:
                out[iid] = img
    return out


def _data_uri(entry: dict) -> str:
    raw = entry.get("image_base64") or ""
    if raw.startswith("data:"):
        return raw
    return f"data:{entry.get('mime_type', 'image/jpeg')};base64,{raw}"


def _inject_images(html: str, image_index: dict) -> str:
    """Replace {{IMG:<id>}} tokens with real data-URIs; drop the token if the id is unknown."""
    def sub(m):
        iid = m.group(1).strip()
        entry = image_index.get(iid)
        if not entry or not entry.get("image_base64"):
            logger.warning("HTML export: image_id %s not found — dropping token", iid)
            return ""
        return _data_uri(entry)
    return _IMG_TOKEN.sub(sub, html)


def export_html_agent(state: LectureState) -> dict:
    passed = state.get("verification_passed", False)
    if passed:
        deck = state.get("lecture_slides") or {}
        composer = state.get("composer_output") or {}
        status = "verified"
    else:
        best = state.get("best_attempt") or {}
        deck = best.get("lecture_slides") or state.get("lecture_slides") or {}
        composer = best.get("composer_output") or state.get("composer_output") or {}
        status = "best_attempt_unverified"

    slides = deck.get("slides", [])
    _backfill_empty_slides(slides, composer.get("headings", []))

    # Build the template style reference INLINE from the request's .pptx (no disk, stateless-safe);
    # no template -> a neutral default style.
    src = state.get("slide_template_pptx")
    reference = reference_from_pptx(src) if src else default_reference()
    tpl_name = "custom" if src else "default"
    image_index = build_image_index(state.get("rag_chunks"))     # id -> base64/mime
    full_images = _full_image_map(state.get("rag_chunks"))       # id -> full obj (vlm_description)

    stem = _safe_stem(state.get("query", "lecture"))
    out_dir = Path(tempfile.gettempdir()) / "slide_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / f"{stem}_html"
    png_dir.mkdir(parents=True, exist_ok=True)

    png_paths = []
    for i, s in enumerate(slides):
        imgs = [full_images[iid] for iid in (s.get("image_refs") or []) if iid in full_images]
        html = generate_slide_html(reference, s, imgs)
        html = _inject_images(html, image_index)
        png = html_to_png(html, png_dir / f"slide_{i:02d}.png")
        png_paths.append(png)

    fmt = state.get("output_format", "pptx")
    out_path = out_dir / f"{stem}.{fmt}"
    if fmt == "pdf":
        from PIL import Image
        imgs = [Image.open(p).convert("RGB") for p in png_paths]
        imgs[0].save(str(out_path), save_all=True, append_images=imgs[1:]) if imgs else None
    else:
        pngs_to_pptx(png_paths, out_path)

    data = out_path.read_bytes()
    logger.info("HTML Export: %d slides -> %s (%s, template=%s, %d bytes)",
                len(png_paths), out_path, status, tpl_name, len(data))
    return {
        "lecture_output_path": str(out_path),
        "lecture_output_bytes": data,
        "export_status": status,
        "final_output": f"Slide deck via HTML template ({tpl_name}, {status}): {out_path}",
    }
