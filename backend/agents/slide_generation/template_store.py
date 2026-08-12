from __future__ import annotations

import io

_A = "http://schemas.openxmlformats.org/drawingml/2006/main"   # DrawingML namespace


# --- Theme extraction -------------------------------------------------------------------

def _theme_xml(prs):
    """Return the template's theme1.xml element (colour + font scheme), or None."""
    from lxml import etree
    for part in prs.part.package.iter_parts():
        if str(part.partname).endswith("theme1.xml"):
            return etree.fromstring(part.blob)
    return None


def _color(el) -> str | None:
    """Resolve a theme colour child (<a:dk1>/<a:accent1>/...) to a #RRGGBB hex."""
    srgb = el.find(f"{{{_A}}}srgbClr")
    if srgb is not None:
        return "#" + srgb.get("val")
    sysc = el.find(f"{{{_A}}}sysClr")
    if sysc is not None and sysc.get("lastClr"):
        return "#" + sysc.get("lastClr")
    return None


def _extract_theme(prs) -> dict:
    """Pull the palette + fonts out of the template. Everything has a sane fallback so a template
    with an unusual theme still yields a usable style card."""
    tokens = {
        "bg": "#ffffff", "text": "#111111", "accent": "#2f5597",
        "accent2": "#c00000", "light": "#f2f2f2",
        "major_font": "Calibri", "minor_font": "Calibri",
    }
    root = _theme_xml(prs)
    if root is not None:
        clr = root.find(f".//{{{_A}}}clrScheme")
        if clr is not None:
            m = {child.tag.split("}")[1]: _color(child) for child in clr}
            tokens["text"] = m.get("dk1") or tokens["text"]
            tokens["bg"] = m.get("lt1") or tokens["bg"]
            tokens["accent"] = m.get("accent1") or tokens["accent"]
            tokens["accent2"] = m.get("accent2") or tokens["accent2"]
            tokens["light"] = m.get("lt2") or tokens["light"]
        fonts = root.find(f".//{{{_A}}}fontScheme")
        if fonts is not None:
            major = fonts.find(f"{{{_A}}}majorFont/{{{_A}}}latin")
            minor = fonts.find(f"{{{_A}}}minorFont/{{{_A}}}latin")
            if major is not None and major.get("typeface"):
                tokens["major_font"] = major.get("typeface")
            if minor is not None and minor.get("typeface"):
                tokens["minor_font"] = minor.get("typeface")

    # Background: prefer an explicit solid fill on the slide master, else keep the theme's lt1.
    bg = _master_bg(prs)
    if bg:
        tokens["bg"] = bg
    return tokens


def _master_bg(prs) -> str | None:
    """A solid background colour set directly on the slide master, if any (#RRGGBB)."""
    try:
        el = prs.slide_masters[0].element
        srgb = el.find(f".//{{{_A}}}bg//{{{_A}}}solidFill/{{{_A}}}srgbClr")
        if srgb is not None:
            return "#" + srgb.get("val")
    except Exception:
        pass
    return None


# --- Contrast helpers -------------------------------------------------------------------

def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return 0.5
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _pick_title_color(tokens: dict) -> str:
    """A title colour that actually reads against the background — some themes set accent1 == bg
    (title would be invisible). Prefer accent, else accent2, else the body text colour."""
    bg = _luminance(tokens["bg"])
    for key in ("accent", "accent2", "text"):
        if abs(_luminance(tokens[key]) - bg) >= 0.25:
            return tokens[key]
    return tokens["text"]


# --- Reference HTML ---------------------------------------------------------------------

def _reference_html(tokens: dict) -> str:
    """A single 1280x720 styled slide that SHOWS the template's look — the style card the LLM mimics."""
    title_color = _pick_title_color(tokens)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
  :root {{
    --bg: {tokens['bg']}; --text: {tokens['text']}; --accent: {tokens['accent']};
    --accent2: {tokens['accent2']}; --light: {tokens['light']}; --title: {title_color};
    --major: '{tokens['major_font']}', system-ui, sans-serif;
    --minor: '{tokens['minor_font']}', system-ui, sans-serif;
  }}
  html,body {{ margin:0; padding:0; }}
  .slide {{
    width:1280px; height:720px; box-sizing:border-box; background:var(--bg); color:var(--text);
    font-family:var(--minor); padding:64px 72px; position:relative; overflow:hidden;
  }}
  .slide h1 {{ font-family:var(--major); color:var(--title); font-size:44px; margin:0 0 28px; }}
  .slide ul {{ font-size:26px; line-height:1.5; margin:0; padding-left:1.1em; }}
  .slide li {{ margin:0 0 14px; }}
  .accent-bar {{ position:absolute; left:0; top:0; width:14px; height:100%; background:var(--accent); }}
</style></head>
<body>
  <div class="slide">
    <div class="accent-bar"></div>
    <h1>Section title</h1>
    <ul>
      <li>First supporting point in the template's body style.</li>
      <li>Second point — same font, colour and spacing.</li>
      <li>Third point showing the overall look.</li>
    </ul>
  </div>
</body></html>"""


# --- Public API -------------------------------------------------------------------------

def reference_from_pptx(source) -> str:
    """Build the style-reference HTML from a .pptx IN MEMORY — no disk, nothing persisted, so it is
    safe on a stateless server. `source` is a path string or raw bytes (e.g. an uploaded file); the
    theme (colours/fonts/background) is extracted and rendered into the style card the LLM mimics."""
    from pptx import Presentation
    src = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    prs = Presentation(src)
    return _reference_html(_extract_theme(prs))


def default_reference() -> str:
    """A neutral built-in style card, used by the HTML renderer when no template is supplied."""
    return _reference_html({
        "bg": "#ffffff", "text": "#1a1a1a", "accent": "#2f5597",
        "accent2": "#c00000", "light": "#f2f2f2",
        "major_font": "Calibri", "minor_font": "Calibri",
    })
