import json
import logging
from pathlib import Path

from backend.agents.state import LearningMaterialsState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import — only fail at call time
# ---------------------------------------------------------------------------
try:
    import markmap  # markmap-python or equivalent wrapper if available
    _MARKMAP_AVAILABLE = True
except ImportError:
    _MARKMAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Mermaid renderer
# ---------------------------------------------------------------------------

def _node_to_mermaid(node: dict, depth: int = 0) -> list[str]:
    """
    Recursively convert a MindmapNode dict into Mermaid mindmap lines.
    Mermaid mindmap uses indentation to express hierarchy.
    """
    indent = "  " * (depth + 1)
    label = node["label"]

    # Root node uses double parentheses, others use plain text
    if depth == 0:
        lines = [f"{indent}root(({label}))"]
    else:
        lines = [f"{indent}{label}"]

    for child in node.get("children", []):
        lines.extend(_node_to_mermaid(child, depth + 1))

    return lines


def _build_mermaid(tree: dict, topic: str) -> str:
    root = tree["root"]
    lines = ["---", f'title: "{topic}"', "---", "mindmap"]
    lines.extend(_node_to_mermaid(root, depth=0))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markmap HTML renderer
# ---------------------------------------------------------------------------

def _node_to_markdown(node: dict, depth: int = 0) -> list[str]:
    """
    Convert the tree to a nested Markdown list — the format markmap expects.
    Descriptions are added as sub-items in italics.
    """
    prefix = "#" * (depth + 1) if depth < 3 else "-" * (depth - 2)
    if depth == 0:
        lines = [f"# {node['label']}"]
    else:
        indent = "  " * (depth - 1)
        lines = [f"{indent}- **{node['label']}**"]
        if node.get("description"):
            lines.append(f"{indent}  *{node['description']}*")

    for child in node.get("children", []):
        lines.extend(_node_to_markdown(child, depth + 1))

    return lines


def _build_markmap_html(tree: dict, topic: str) -> str:
    md_lines = _node_to_markdown(tree["root"], depth=0)
    md_content = "\n".join(md_lines)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{topic} — Mindmap</title>
  <script src="https://cdn.jsdelivr.net/npm/markmap-autoloader@0.16"></script>
</head>
<body>
  <div class="markmap">
    <script type="text/template">
{md_content}
    </script>
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def mindmap_export(state: LearningMaterialsState) -> dict:
    tree  = state["mindmap_tree"]
    topic = state["query"]
    fmt   = state.get("mindmap_format", "mermaid")  # "mermaid" | "markmap"

    output_dir = Path("outputs/mindmaps")
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = topic.replace(" ", "_")[:60]

    if fmt == "markmap":
        content = _build_markmap_html(tree, topic)
        fname   = f"{safe_name}_mindmap.html"
        output_path = output_dir / fname
        output_path.write_text(content, encoding="utf-8")
        raw = content.encode("utf-8")
    else:
        # Default: Mermaid .md file
        content = _build_mermaid(tree, topic)
        fname   = f"{safe_name}_mindmap.md"
        output_path = output_dir / fname
        output_path.write_text(content, encoding="utf-8")
        raw = content.encode("utf-8")

    # Always also save the raw JSON tree for downstream use
    json_path = output_dir / f"{safe_name}_mindmap.json"
    json_path.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "MindmapExport: format=%s path=%s size=%d bytes nodes_total=%d",
        fmt,
        output_path,
        len(raw),
        len(json.dumps(tree)),
    )

    return {
        "mindmap_output_path":  str(output_path),
        "mindmap_output_bytes": raw,
    }