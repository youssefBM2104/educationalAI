from __future__ import annotations


# --- Image handling ----------------------------------------------------------------

# Fields an LLM is allowed to see about an image. `image_base64` is deliberately absent.
_IMAGE_PROMPT_FIELDS = ("image_id", "vlm_description", "page_number")


def image_reasoning_view(image: dict) -> dict:
    """One image object reduced to the fields safe to put in a prompt (no base64)."""
    return {k: image.get(k) for k in _IMAGE_PROMPT_FIELDS if k in image}


def images_of_chunk(chunk: dict) -> list[dict]:
    return chunk.get("images") or []


def build_image_index(chunks: list | None) -> dict[str, dict]:
    """image_id -> {image_base64, mime_type} across all chunks. Used ONLY by Export."""
    index: dict[str, dict] = {}
    for c in chunks or []:
        for img in images_of_chunk(c):
            iid = img.get("image_id")
            if iid and iid not in index:
                index[iid] = {
                    "image_base64": img.get("image_base64"),
                    "mime_type": img.get("mime_type", "image/jpeg"),
                }
    return index


def available_images(chunks: list | None) -> list[dict]:
    """All images across chunks, in prompt-safe form (no base64)."""
    out = []
    for c in chunks or []:
        for img in images_of_chunk(c):
            out.append(image_reasoning_view(img))
    return out


# --- Concept / chunk helpers -------------------------------------------------------

def concept_ids(chunk: dict) -> list[str]:
    out = []
    for c in chunk.get("covers_concepts", []) or []:
        cid = c.get("id") if isinstance(c, dict) else c
        if cid:
            out.append(str(cid))
    return out


def all_concepts(chunks: list | None) -> list[str]:
    seen: dict[str, None] = {}
    for c in chunks or []:
        for cid in concept_ids(c):
            seen.setdefault(cid, None)
    return list(seen)


# --- Prompt formatting (base64-free) -----------------------------------------------

def format_chunks_for_prompt(chunks: list | None) -> str:
    """Render chunks for an LLM: text + concepts + image reasoning fields, NEVER base64."""
    if not chunks:
        return "(no chunks)"
    blocks = []
    for c in chunks:
        head = f"[{c.get('source', '?')}#{c.get('chunk_index')}] concepts: {', '.join(concept_ids(c)) or '-'}"
        imgs = images_of_chunk(c)
        img_lines = "\n".join(
            f"  - image {i.get('image_id')}: {i.get('vlm_description', '')}" for i in imgs
        )
        block = f"{head}\n{c.get('text', '')}"
        if img_lines:
            block += f"\nimages:\n{img_lines}"
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


def format_relations(kg: dict | None) -> str:
    rels = (kg or {}).get("relations", [])
    if not rels:
        return "(no relations)"
    return "\n".join(f"{r.get('from')} --[{r.get('type')}]--> {r.get('to')}" for r in rels)


DEFAULT_TEMPLATE = {
    "name": "default",
    "max_words_per_bullet": 16,
    "layouts": [
        {"id": "title", "purpose": "deck title or a section opener — no body content",
         "slots": {"title": "<= 8 words", "subtitle": "<= 15 words"}},
        {"id": "title_bullets", "purpose": "one idea explained as a few bullet points",
         "slots": {"title": "...", "bullets": "3-5 bullets, each <= max_words_per_bullet"}},
        {"id": "two_column_image", "purpose": "a concept explained alongside a supporting figure",
         "slots": {"title": "...", "bullets": "2-4 bullets", "image": "one image_ref"}},
        {"id": "three_card", "purpose": "exactly three parallel items / examples / properties",
         "slots": {"title": "...", "cards": "3 x { label, one-line text }"}},
        {"id": "stat", "purpose": "one headline number and what it means",
         "slots": {"title": "...", "stat": "a % or number", "explanation": "one short sentence"}},
        {"id": "image_caption", "purpose": "a figure that IS the point, with a short caption",
         "slots": {"title": "...", "image": "one image_ref", "caption": "<= 20 words"}},
    ],
}
