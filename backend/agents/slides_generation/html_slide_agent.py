import logging
import re

from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.slides_generation._common import image_reasoning_view

logger = logging.getLogger(__name__)

llm = MODELS["gpt-5"]

_FENCE = re.compile(r"^\s*```(?:html)?\s*|\s*```\s*$", re.IGNORECASE)


def _prompt(reference_html: str, slide: dict, images: list[dict]) -> str:
    img_lines = "\n".join(
        f"- image_id={i.get('image_id')}: {i.get('vlm_description', '')[:200]}" for i in images
    ) or "(none)"
    bullets = slide.get("bullets") or []
    slots = slide.get("slots") or {}
    return f"""# Role
You are a slide designer. Output ONE complete, standalone HTML document for a SINGLE slide.

# Hard requirements
- The document renders at EXACTLY 1280x720 px, no scrollbars, nothing clipped. Root box is 1280x720.
- ALL CSS inline in a <style> tag. No external fonts/URLs/scripts.
- Reproduce the LOOK of the style reference below: same background, colour palette, fonts, accent
  treatment and overall feel. Do not copy its placeholder text — use the real content.
- If a figure is provided, place it with an <img> whose src is the literal token `{{{{IMG:<image_id>}}}}`
  (e.g. src="{{{{IMG:abc-123}}}}"). NEVER invent an image URL or a data URI. Size it to fit tastefully.
  Only use an image_id from the list; if the list is empty, use no <img>.
- Output ONLY the HTML document — no markdown fences, no commentary.

# Style reference (reproduce this look, not this text)
{reference_html}

# Slide content to lay out
title: {slide.get('title', '')}
layout hint: {slide.get('layout', 'title_bullets')}
bullets: {bullets}
slots: {slots}

# Available figures for THIS slide (reference by token; bytes are injected later)
{img_lines}
"""


def generate_slide_html(reference_html: str, slide: dict, images: list[dict]) -> str:
    """Return a standalone HTML document for one slide, styled to the template reference."""
    safe_images = [image_reasoning_view(i) for i in images]   # strips base64 defensively
    msg = llm.invoke([
        SystemMessage(content=_prompt(reference_html, slide, safe_images)),
        HumanMessage(content="Write the slide's HTML now."),
    ])
    html = msg.content if isinstance(msg.content, str) else str(msg.content)
    html = _FENCE.sub("", html).strip()
    logger.info("HTML slide [%s]: %d chars", slide.get("slide_id"), len(html))
    return html
