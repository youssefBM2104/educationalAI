# kg_builder.py — Spec-compliant version
import logging
import time
from typing import List, Dict, Any

from langchain_experimental.graph_transformers import LLMGraphTransformer
#from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document

from tqdm import tqdm

logger = logging.getLogger(__name__)


# --- Ontology ----------------------------------------------------------------------
# Constrains what the LLM is allowed to extract. A richer relation set gives the
# downstream consumers more edges to reason over: the retriever's KG expansion finds
# more neighbours, and the exam generator can build longer, more varied paths.
ALLOWED_NODES = [
    "Concept",
    "Formula",
    "Theorem",
    "Method",
    "Example",
    "Quantity",   # lets an edge express monotonicity between two quantities
]

ALLOWED_RELATIONSHIPS = [
    # --- structural / curricular
    "PREREQUISITE",
    "PART_OF",
    "EXTENDS",
    "ILLUSTRATES",
    "APPLIES_TO",

    # --- causal
    "CAUSES",
    "INCREASES",
    "REDUCES",
    "PREVENTS",

    # --- comparative & evaluative
    "CONTRASTS_WITH",
    "TRADES_OFF_AGAINST",
    "IS_INSTANCE_OF",

    # --- pedagogical
    "COMMON_MISCONCEPTION_OF",
]

NODE_PROPERTIES = ["definition", "aliases", "status"]


class KGBuilder:
    def __init__(self, llm, neo4j_uri, neo4j_user, neo4j_password):
        self.transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=ALLOWED_NODES,
            allowed_relationships=ALLOWED_RELATIONSHIPS,
            node_properties=NODE_PROPERTIES,
            # node_properties require native function calling; keep this False (an OpenAI model is
            # used — the NVIDIA prompt-based path raises ValueError when properties are requested).
            ignore_tool_usage=False,
        )
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password
        )

    # --- shared helpers ------------------------------------------------------------

    @staticmethod
    def _documents(chunks_dicts: List[Dict[str, Any]]) -> List[Document]:
        return [
            Document(
                page_content=chunk["text"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "course_id": chunk["course_id"],
                },
            )
            for chunk in chunks_dicts
        ]

    @staticmethod
    def _write_covers_concepts(graph_docs: list, chunks_dicts: List[Dict[str, Any]]) -> None:
        """Tag each chunk dict in-place with the concepts extracted from it. This is the join key
        the retriever's KG expansion and the exam generator's soft filter depend on."""
        chunk_id_to_dict = {c["chunk_id"]: c for c in chunks_dicts}
        for graph_doc in graph_docs:
            chunk_id = graph_doc.source.metadata.get("chunk_id")
            if chunk_id and chunk_id in chunk_id_to_dict:
                chunk_id_to_dict[chunk_id]["covers_concepts"] = [
                    {"id": node.id}  # LangChain uses the name as the id
                    for node in graph_doc.nodes
                ]

    @staticmethod
    def _summary(chunks_dicts: List[Dict[str, Any]], graph_docs: list) -> Dict[str, Any]:
        return {
            "total_chunks": len(chunks_dicts),
            "total_nodes": sum(len(gd.nodes) for gd in graph_docs),
            "total_relations": sum(len(gd.relationships) for gd in graph_docs),
            "covers_concepts": chunks_dicts,
        }

    def _convert_one_with_retry(self, doc: Document, max_retries: int = 5):
        # NVIDIA NIM free tier ~40 req/min — retry on 429 with exponential backoff
        delay = 2
        for attempt in range(max_retries):
            try:
                return self.transformer.convert_to_graph_documents([doc])[0]
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

    # --- sync path -----------------------------------------------------------------

    def build_from_dicts(self, chunks_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sequential + throttled extraction. Safe under a tight per-minute quota; see
        abuild_from_dicts for the faster concurrent path."""
        documents = self._documents(chunks_dicts)

        # Sequential extraction with throttle — avoids 429 on a per-minute limit
        graph_docs = []
        for doc in tqdm(documents):
            graph_docs.append(self._convert_one_with_retry(doc))
            time.sleep(1.5)
        self.graph.add_graph_documents(graph_docs, include_source=True)

        self._write_covers_concepts(graph_docs, chunks_dicts)
        return self._summary(chunks_dicts, graph_docs)

    # --- async path ----------------------------------------------------------------

    async def abuild_from_dicts(self, chunks_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Concurrent extraction — a large speedup on a full document. LLMGraphTransformer fans the
        per-chunk LLM calls out concurrently instead of one-at-a-time-with-a-sleep, so wall-clock is
        bounded by the slowest chunk, not the sum of all of them. Requires a model/quota that
        tolerates parallel requests (the OpenAI path does; the NVIDIA free tier may 429)."""
        documents = self._documents(chunks_dicts)

        graph_docs = await self.transformer.aconvert_to_graph_documents(documents)
        self.graph.add_graph_documents(graph_docs, include_source=True)

        self._write_covers_concepts(graph_docs, chunks_dicts)
        return self._summary(chunks_dicts, graph_docs)
