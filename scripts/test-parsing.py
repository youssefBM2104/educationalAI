"""
test_docling_enriched.py
------------------------
Standalone test for the Docling + Ollama VLM enrichment pipeline.
Run from the project root:

    python scripts/test_docling_enriched.py path/to/test.pdf

No imports from the main codebase — runs independently.
Requires: docling, httpx, Ollama running at localhost:11434 with moondream pulled.
"""

import sys
import re
import base64
import time
from io import BytesIO
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_HOST          = "http://localhost:11434"
VLM_MODEL            = "moondream"
TIMEOUT_SECS         = 120
OUTPUT_PREVIEW_CHARS = 3000
# ─────────────────────────────────────────────────────────────────────────────

VLM_PROMPT = (
    "Describe exactly what is shown in this chart or figure. "
    "Read all axis labels, tick values, legend entries, and titles literally as written — "
    "do not infer units or domain context. "
    "Describe the shape and direction of each curve or data series. "
    "Be concise and factual. Start directly with the description."
)


def sep(label: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")


# ── Sanitizer ─────────────────────────────────────────────────────────────────
def sanitize_formula_output(markdown: str) -> str:
    """
    Remove hallucinated repetition loops from formula enrichment model output.
    Detects any LaTeX command repeated 5+ times consecutively.
    """
    cleaned = re.sub(
        r'(\\[a-zA-Z]+\s*\{[^}]*\}\s*){5,}',
        '<!-- formula-enrichment-failed -->',
        markdown
    )
    removed = markdown.count('\\intertext') - cleaned.count('\\intertext')
    if removed > 0:
        print(f"   ⚠️  Sanitizer removed hallucinated loop ({removed} occurrences)")
    return cleaned


# ── VLM call with retry ───────────────────────────────────────────────────────
def describe_image(pil_image) -> str:
    import httpx

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    image_b64 = base64.standard_b64encode(buffer.getvalue()).decode()

    payload = {
        "model": VLM_MODEL,
        "prompt": VLM_PROMPT,
        "images": [image_b64],
        "stream": False,
    }

    for attempt in range(2):
        try:
            response = httpx.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
                timeout=TIMEOUT_SECS,
            )
            response.raise_for_status()
            description = response.json()["response"].strip()

            if description:
                return f"[Figure: {description}]"

            print(f"   ⚠️  Empty response from VLM (attempt {attempt + 1}/2) — retrying...")
            time.sleep(3)

        except Exception as e:
            print(f"   ⚠️  VLM call failed (attempt {attempt + 1}/2): {e}")
            time.sleep(3)

    return "<!-- image -->"


# ── Step 1: Ollama health check ───────────────────────────────────────────────
def check_ollama() -> bool:
    sep("STEP 1 — Ollama connectivity")
    try:
        import httpx
        r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"✅ Ollama is reachable at {OLLAMA_HOST}")
        print(f"   Pulled models: {models if models else '(none)'}")
        if not any(VLM_MODEL in m for m in models):
            print(f"⚠️  '{VLM_MODEL}' not found. Run: ollama pull {VLM_MODEL}")
            print("   Image enrichment will be skipped.")
            return False
        print(f"✅ Model '{VLM_MODEL}' is available")
        return True
    except Exception as e:
        print(f"❌ Ollama not reachable: {e}")
        print("   Image enrichment will be skipped.")
        return False


# ── Step 2: Docling converter init ────────────────────────────────────────────
def build_converter():
    sep("STEP 2 — Docling converter initialisation")
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    t0 = time.time()
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.do_formula_enrichment   = True
    pipeline_options.images_scale            = 2.0

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    print(f"✅ Converter ready ({time.time() - t0:.1f}s)")
    return converter


# ── Step 3: Parse PDF + sanitize ─────────────────────────────────────────────
def parse_pdf(converter, pdf_path: str):
    sep("STEP 3 — Docling PDF parse + sanitize")
    t0 = time.time()
    result = converter.convert(pdf_path)
    elapsed = time.time() - t0

    markdown = result.document.export_to_markdown()

    # Count before sanitizing
    n_images   = markdown.count("<!-- image -->")
    n_formulas = markdown.count("<!-- formula-not-decoded -->")

    print(f"✅ Parsed in {elapsed:.1f}s")
    print(f"   Total chars : {len(markdown):,}")
    print(f"   <!-- image --> placeholders              : {n_images}")
    print(f"   <!-- formula-not-decoded --> placeholders: {n_formulas}")

    # Sanitize hallucination loops
    markdown = sanitize_formula_output(markdown)
    n_failed = markdown.count("<!-- formula-enrichment-failed -->")
    if n_failed:
        print(f"   <!-- formula-enrichment-failed --> replaced: {n_failed}")

    print(f"\n── Markdown preview (first {OUTPUT_PREVIEW_CHARS} chars) ──")
    print(markdown[:OUTPUT_PREVIEW_CHARS])

    return result, markdown


# ── Step 4: Count picture items ──────────────────────────────────────────────
def inspect_picture_items(result):
    sep("STEP 4 — Picture item inspection")
    from docling_core.types.doc import PictureItem

    items = [
        item
        for item, _level in result.document.iterate_items()
        if isinstance(item, PictureItem)
    ]
    print(f"   PictureItem objects found: {len(items)}")
    for i, item in enumerate(items):
        has_pil = item.image and item.image.pil_image is not None
        size = f"{item.image.pil_image.size}" if has_pil else "N/A"
        print(f"   [{i}] has_image={has_pil}  size={size}")
    return items


# ── Step 5: VLM description test (first image only) ──────────────────────────
def test_vlm_single(picture_items) -> str | None:
    sep("STEP 5 — VLM description (first image only)")

    if not picture_items:
        print("   No picture items to test.")
        return None

    item = next(
        (p for p in picture_items if p.image and p.image.pil_image), None
    )
    if item is None:
        print("   No picture items with extractable pixels.")
        return None

    pil_image = item.image.pil_image
    print(f"   Image size: {pil_image.size}  mode: {pil_image.mode}")
    print(f"   Sending to Ollama ({VLM_MODEL})...")

    t0 = time.time()
    result = describe_image(pil_image)
    elapsed = time.time() - t0

    if result.startswith("[Figure:"):
        print(f"✅ VLM response received ({elapsed:.1f}s)")
        print(f"\n── VLM description ──")
        print(result)
    else:
        print(f"❌ VLM failed after retries ({elapsed:.1f}s) — got: {result}")

    return result


# ── Step 6: Full enrichment pass ─────────────────────────────────────────────
def full_enrichment_pass(result, markdown: str, picture_items, ollama_ok: bool) -> str:
    sep("STEP 6 — Full enrichment pass")
    from docling_core.types.doc import PictureItem

    if not ollama_ok:
        print("   Skipped — Ollama not available.")
        return markdown

    enriched = markdown
    replaced = 0
    failed   = 0

    for i, item in enumerate(picture_items):
        if "<!-- image -->" not in enriched:
            break

        if item.image and item.image.pil_image:
            t0 = time.time()
            desc = describe_image(item.image.pil_image)
            elapsed = time.time() - t0

            if desc.startswith("[Figure:"):
                enriched = enriched.replace("<!-- image -->", desc, 1)
                replaced += 1
                print(f"   ✅ [{i}] replaced ({elapsed:.1f}s)")
            else:
                enriched = enriched.replace("<!-- image -->", "<!-- image -->", 1)
                failed += 1
                print(f"   ❌ [{i}] failed after retries ({elapsed:.1f}s)")
        else:
            enriched = enriched.replace("<!-- image -->", "<!-- image-no-data -->", 1)
            failed += 1
            print(f"   ⚠️  [{i}] no image data")

    print(f"\n   Replaced : {replaced}")
    print(f"   Failed   : {failed}")
    print(f"   Remaining <!-- image --> placeholders: {enriched.count('<!-- image -->')}")

    print(f"\n── Enriched markdown preview (first {OUTPUT_PREVIEW_CHARS} chars) ──")
    print(enriched[:OUTPUT_PREVIEW_CHARS])

    return enriched


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python test_docling_enriched.py path/to/test.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"\n🔍 Testing Docling + Ollama enrichment pipeline")
    print(f"   PDF      : {pdf_path}")
    print(f"   VLM      : {VLM_MODEL} @ {OLLAMA_HOST}")

    ollama_ok     = check_ollama()
    converter     = build_converter()
    result, md    = parse_pdf(converter, pdf_path)
    picture_items = inspect_picture_items(result)

    if ollama_ok and picture_items:
        test_vlm_single(picture_items)

    enriched_md = full_enrichment_pass(result, md, picture_items, ollama_ok)

    output_path = Path(pdf_path).with_suffix(".enriched.md")
    output_path.write_text(enriched_md, encoding="utf-8")
    print(f"\n✅ Enriched markdown written to {output_path}")

    sep("DONE")
    print("All steps complete.\n")


if __name__ == "__main__":
    main()