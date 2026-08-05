from pathlib import Path
import uuid
import time
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import re
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling_core.types.doc import PictureItem
import httpx
import base64
from io import BytesIO
from backend.core.config import settings
import logging


logger = logging.getLogger(__name__)

_markitdown = MarkItDown()

def parse(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf_docling(file_path)
    else:
        result = _markitdown.convert(file_path)
        return result.text_content

_docling_converter: DocumentConverter | None = None

def _get_docling_converter() -> DocumentConverter:
    global _docling_converter
    if _docling_converter is None:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True
        pipeline_options.do_formula_enrichment = True
        pipeline_options.images_scale = 2.0
        device = AcceleratorDevice.CUDA if settings.docling_device == 'cuda' else AcceleratorDevice.CPU
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=device,
        )
        _docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    return _docling_converter


def _describe_image(pil_image) -> str:
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    image_b64 = base64.standard_b64encode(buffer.getvalue()).decode()

    payload = {
        "model": settings.vlm_model,
        "prompt": (
            "Describe exactly what is shown in this chart or figure. "
            "Read all axis labels, tick values, legend entries, and titles literally as written — "
            "do not infer units or domain context. "
            "Describe the shape and direction of each curve or data series. "
            "Be concise and factual. Start directly with the description."
        ),
        "images": [image_b64],
        "stream": False,
    }

    for attempt in range(2):  # 1 retry on empty response
        try:
            response = httpx.post(
                f"{settings.ollama_host}/api/generate",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            description = response.json()["response"].strip()

            if description:
                return f"[Figure: {description}]"

            logger.warning(f"VLM returned empty response (attempt {attempt + 1}/2)")
            time.sleep(3)  # brief pause before retry

        except Exception as e:
            logger.warning(f"VLM call failed (attempt {attempt + 1}/2): {e}")
            time.sleep(3)

    return "<!-- image-failed -->"  # both attempts failed or returned empty


def _sanitize_formula_output(markdown: str) -> str:
    """
    Remove hallucinated repetition loops produced by the formula enrichment model
    on complex multi-line equations. Detects any LaTeX command repeated 5+ times
    consecutively and replaces the entire block with a failure placeholder.
    """
    return re.sub(
        r'(\\[a-zA-Z]+\s*\{[^}]*\}\s*){5,}',
        '<!-- formula-enrichment-failed -->',
        markdown
    )

def _ollama_reachable() -> bool:
    try:
        httpx.get(f"{settings.ollama_host}/api/tags", timeout=5.0).raise_for_status()
        return True
    except Exception:
        return False


def _parse_pdf_docling(file_path: str, document_id: str) -> tuple[str, list[dict]]:
    converter = _get_docling_converter()
    result = converter.convert(file_path)
    markdown = result.document.export_to_markdown()
    markdown = _sanitize_formula_output(markdown)

    image_records: list[dict] = []

    if not settings.vlm_enrichment_enabled:
        return markdown, image_records

    picture_items = [
        item for item, _level in result.document.iterate_items()
        if isinstance(item, PictureItem)
    ]
    if not picture_items:
        return markdown, image_records

    if not _ollama_reachable():
        logger.warning("Ollama unreachable — skipping VLM image enrichment for %s", file_path)
        return markdown, image_records

    for idx, item in enumerate(picture_items):
        if "<!-- image -->" not in markdown:
            break

        if item.image and item.image.pil_image:
            image_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_img_{idx:04d}"))
            description = _describe_image(item.image.pil_image)

            bbox = item.prov[0].bbox if item.prov else None
            page_number = item.prov[0].page_no if item.prov else None

            image_records.append({
                "image_id": image_id,
                "document_id": document_id,
                "pil_image": item.image.pil_image,
                "vlm_description": description,
                "page_number": page_number,
                "bbox": bbox,
                "chunk_id": None,  # resolved after chunking
            })
            replacement = f"{description}<!-- IMG:{image_id} -->"
        else:
            replacement = "<!-- image-no-data -->"

        markdown = markdown.replace("<!-- image -->", replacement, 1)

    return markdown, image_records

def clean(text: str) -> str:
    # 1. Normalize Windows line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # 2. Collapse 3+ newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 3. Collapse multiple spaces/tabs into one space
    text = re.sub(r'[ \t]+', ' ', text)
    # 4. Strip trailing whitespace from each line
    text = '\n'.join(line.rstrip() for line in text.split('\n'))

    text = text.replace('\x0c', '')
    return text.strip()



_IMG_MARKER = re.compile(r'<!-- IMG:([0-9a-f\-]+) -->')

def chunk(text: str, document_id: str, course_id: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )
    splits = splitter.split_text(text)
    chunks = []
    for i, split in enumerate(splits):
        image_ids = _IMG_MARKER.findall(split)
        clean_text = _IMG_MARKER.sub('', split).strip()
        chunks.append({
            "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_chunk_{i:04d}")),
            "document_id": document_id,
            "course_id": course_id,
            "chunk_index": i,
            "text": clean_text,
            "covers_concepts": [],
            "image_ids": image_ids,
        })
    return chunks

# Semantic + Hierarchical chunking

_embeddings = None


def _get_embeddings():
    """Load the local sentence-transformers embedding model once and reuse it."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings

def semantic_hierarchical_chunk(text: str, document_id: str, course_id: str) -> list[dict]:
    """
    Combined strategy:
      1. HIERARCHICAL — split the markdown into sections by headers (#, ##, ###),
         using the structure pymupdf4llm produced.
      2. SEMANTIC — within each section, split further at points where the topic
         shifts (measured by embedding similarity between adjacent sentences).
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    sections = md_splitter.split_text(text)
    semantic_splitter = SemanticChunker(_get_embeddings())
    chunks = []
    idx = 0
    for section in sections:
        content = section.page_content.strip()
        if not content:
            continue
        for piece in semantic_splitter.split_text(content):
            image_ids = _IMG_MARKER.findall(piece)
            clean_piece = _IMG_MARKER.sub('', piece).strip()
            chunks.append({
                "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_chunk_{idx:04d}")),
                "document_id": document_id,
                "course_id": course_id,
                "chunk_index": idx,
                "text": clean_piece,
                "covers_concepts": [],
                "image_ids": image_ids,
            })
            idx += 1
    return chunks

def parse_and_chunk(file_path: str, document_id: str, course_id: str) -> list[dict]:
    """Parse -> clean -> save markdown -> RECURSIVE chunk."""
    raw_text = parse(file_path)
    clean_text = clean(raw_text)
    Path(file_path + ".md").write_text(clean_text, encoding="utf-8")
    chunks = chunk(clean_text, document_id, course_id)
    if not chunks:
        raise ValueError(
            f"Document produced no chunks after parsing — "
            f"file may be empty or unreadable: {file_path}"
        )
    return chunks


def parse_and_semantic_hierarchical_chunk(file_path: str, document_id: str, course_id: str) -> tuple[list[dict], list[dict]]:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        raw_text, image_records = _parse_pdf_docling(file_path, document_id)
    else:
        raw_text = _markitdown.convert(file_path).text_content
        image_records = []

    clean_text = clean(raw_text)
    Path(file_path + ".md").write_text(clean_text, encoding="utf-8")
    chunks = semantic_hierarchical_chunk(clean_text, document_id, course_id)
    if not chunks:
        raise ValueError(f"Document produced no chunks — {file_path}")

    id_to_chunk = {}
    for c in chunks:
        for img_id in c["image_ids"]:
            id_to_chunk[img_id] = c["chunk_id"]
    for rec in image_records:
        rec["chunk_id"] = id_to_chunk.get(rec["image_id"])

    return chunks, image_records