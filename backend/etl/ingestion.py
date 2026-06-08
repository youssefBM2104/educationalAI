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
from docling_core.types.doc import PictureItem
import httpx
import base64
from io import BytesIO
from backend.core.config import settings
import logging


logger = logging.getLogger(__name__)

def parse(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf_docling(file_path)
    else:
        md = MarkItDown()
        result = md.convert(file_path)
        return result.text_content

_docling_converter: DocumentConverter | None = None

def _get_docling_converter() -> DocumentConverter:
    global _docling_converter
    if _docling_converter is None:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True
        pipeline_options.do_formula_enrichment = True
        pipeline_options.images_scale = 2.0
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
        "model": settings.VLM_MODEL,
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
                f"{settings.OLLAMA_HOST}/api/generate",
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

    return "<!-- image -->"  # both attempts failed or returned empty


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

def _parse_pdf_docling(file_path: str) -> str:
    converter = _get_docling_converter()
    result = converter.convert(file_path)
    markdown = result.document.export_to_markdown()

    # Remove hallucinated repetition loops from formula enrichment
    markdown = _sanitize_formula_output(markdown)

    if not settings.VLM_ENRICHMENT_ENABLED:
        return markdown

    picture_items = [
        item
        for item, _level in result.document.iterate_items()
        if isinstance(item, PictureItem)
    ]

    for item in picture_items:
        if "<!-- image -->" not in markdown:
            break
        if item.image and item.image.pil_image:
            description = _describe_image(item.image.pil_image)
        else:
            description = "<!-- image-no-data -->"
        markdown = markdown.replace("<!-- image -->", description, 1)

    return markdown


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



def chunk(text: str, document_id: str, course_id: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )
    splits = splitter.split_text(text)
    return [
        {
            "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_chunk_{i:04d}")),
            "document_id": document_id,
            "course_id": course_id,
            "chunk_index": i,
            "text": split,
            "covers_concepts": [],
        }
        for i, split in enumerate(splits)
    ]
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
            chunks.append({
                "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_chunk_{idx:04d}")),
                "document_id": document_id,
                "course_id": course_id,
                "chunk_index": idx,
                "text": piece,
                "covers_concepts": [],
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


def parse_and_semantic_hierarchical_chunk(file_path: str, document_id: str, course_id: str) -> list[dict]:
    """Parse -> clean -> save markdown -> SEMANTIC + HIERARCHICAL chunk.

    Sibling of parse_and_chunk(): same prepare steps, different chunking strategy.
    Use this when you want richer chunks with header metadata + semantic boundaries
    (slower; embeds every sentence via sentence-transformers).
    """
    raw_text = parse(file_path)
    clean_text = clean(raw_text)
    Path(file_path + ".md").write_text(clean_text, encoding="utf-8")
    chunks = semantic_hierarchical_chunk(clean_text, document_id, course_id)
    if not chunks:
        raise ValueError(
            f"Document produced no chunks after parsing — "
            f"file may be empty or unreadable: {file_path}"
        )
    return chunks
