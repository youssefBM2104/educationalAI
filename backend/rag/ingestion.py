from pathlib import Path
import uuid

import pymupdf4llm
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import re


def parse(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf_pymupdf(file_path)
    else:
        md = MarkItDown()
        result = md.convert(file_path)
        return result.text_content

def _parse_pdf_pymupdf(file_path: str) -> str:
    page_chunks = pymupdf4llm.to_markdown(file_path, page_chunks=True)
    pages = []
    for chunk in page_chunks:
        text = chunk["text"]
        if text.strip():
            page_num = chunk["metadata"]["page_number"]
            pages.append(f"<!-- page {page_num} -->\n{text}")
    return "\n\n".join(pages)

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

# check kỹ lại 
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
        # header path gives the hierarchical context, e.g. "Preface > Overview".
        # Strip markdown bold/italic markers so "**Preface**" -> "Preface" in metadata.
        header_path = " > ".join(section.metadata.values()) if section.metadata else ""
        header_path = re.sub(r"\*+|_+", "", header_path).strip()
        for piece in semantic_splitter.split_text(content):
            chunks.append({
                "chunk_id": f"{document_id}_chunk_{idx:04d}",
                "document_id": document_id,
                "course_id": course_id,
                "chunk_index": idx,
                "text": piece,
                "section": header_path,
                "covers_concepts": [],
            })
            idx += 1
    return chunks




def parse_and_chunk(file_path: str, document_id: str, course_id: str) -> list[dict]:
    raw_text = parse(file_path)
    clean_text = clean(raw_text)
    Path(file_path + ".md").write_text(clean_text, encoding="utf-8")
    return chunk(clean_text, document_id, course_id)
