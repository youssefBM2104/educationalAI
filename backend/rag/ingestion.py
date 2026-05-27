from pathlib import Path

import pymupdf4llm
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    # pymupdf4llm extracts structured Markdown (headings, tables) with reliable
    # word spacing, unlike a raw page.get_text() dump.
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
            "chunk_id": f"{document_id}_chunk_{i:04d}",
            "document_id": document_id,
            "course_id": course_id,
            "chunk_index": i,
            "text": split,
            "covers_concepts": [],
        }
        for i, split in enumerate(splits)
    ]

def parse_and_chunk(file_path: str, document_id: str, course_id: str) -> list[dict]:
    raw_text = parse(file_path)
    clean_text = clean(raw_text)
    Path(file_path + ".md").write_text(clean_text, encoding="utf-8")
    return chunk(clean_text, document_id, course_id)
