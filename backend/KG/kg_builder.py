# kg_builder.py — Throttled + retry-on-429 version
import asyncio
import logging
from typing import List, Dict, Any

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class KGBuilder:
    def __init__(
        self,
        llm,
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        max_concurrent: int = 3,
        min_chunk_chars: int = 200,
        max_retries: int = 5,
    ):
        self.transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=["Concept", "Formula", "Theorem", "Example", "Method", "Definition"],
            allowed_relationships=[
                "PREREQUISITE", "EXTENDS", "DEFINES",
                "APPLIES_TO", "ILLUSTRATES", "PART_OF",
            ],
            ignore_tool_usage=True,  # ChatNVIDIA doesn't support include_raw=True in structured output
        )
        self.graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_password)
        self.max_concurrent = max_concurrent
        self.min_chunk_chars = min_chunk_chars
        self.max_retries = max_retries

    async def _extract_one(self, doc: Document, sem: asyncio.Semaphore):
        """Extract graph from 1 document with concurrency + 429 retry."""
        async with sem:
            for attempt in range(self.max_retries + 1):
                try:
                    result = await self.transformer.aconvert_to_graph_documents([doc])
                    return result[0]
                except Exception as e:
                    msg = str(e)
                    if "429" in msg and attempt < self.max_retries:
                        wait = 2 ** attempt  # 1, 2, 4, 8, 16 s
                        logger.warning(
                            f"[KG] 429 rate limit — backoff {wait}s "
                            f"(attempt {attempt+1}/{self.max_retries})"
                        )
                        await asyncio.sleep(wait)
                        continue
                    logger.error(f"[KG] extraction failed for chunk: {msg}")
                    raise

    async def _extract_all(self, documents: List[Document]):
        sem = asyncio.Semaphore(self.max_concurrent)
        tasks = [self._extract_one(doc, sem) for doc in documents]
        return await asyncio.gather(*tasks)

    def build_from_dicts(self, chunks_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Skip very short chunks (TOC lines, headers, empty bullets) — wasteful LLM calls.
        filtered = [c for c in chunks_dicts if len(c["text"]) >= self.min_chunk_chars]
        skipped = len(chunks_dicts) - len(filtered)
        logger.info(
            f"[KG] {len(filtered)}/{len(chunks_dicts)} chunks will be sent to LLM "
            f"(skipped {skipped} chunks shorter than {self.min_chunk_chars} chars)"
        )

        documents = [
            Document(
                page_content=c["text"],
                metadata={
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "course_id": c["course_id"],
                },
            )
            for c in filtered
        ]

        logger.info(
            f"[KG] starting extraction with max_concurrent={self.max_concurrent}, "
            f"max_retries={self.max_retries}"
        )
        graph_docs = asyncio.run(self._extract_all(documents))
        self.graph.add_graph_documents(graph_docs, include_source=True)

        # Write covers_concepts back onto the chunk dicts (mutates caller's list).
        chunk_id_to_dict = {c["chunk_id"]: c for c in chunks_dicts}
        for graph_doc in graph_docs:
            chunk_id = graph_doc.source.metadata.get("chunk_id")
            if chunk_id and chunk_id in chunk_id_to_dict:
                chunk_id_to_dict[chunk_id]["covers_concepts"] = [
                    {"id": node.id, "name": node.id}
                    for node in graph_doc.nodes
                ]

        total_nodes = sum(len(gd.nodes) for gd in graph_docs)
        total_rels = sum(len(gd.relationships) for gd in graph_docs)
        logger.info(f"[KG] done — {total_nodes} nodes, {total_rels} relationships extracted")

        return {
            "total_chunks": len(chunks_dicts),
            "filtered_chunks": len(filtered),
            "total_nodes": total_nodes,
            "total_relations": total_rels,
            "covers_concepts": chunks_dicts,
        }
