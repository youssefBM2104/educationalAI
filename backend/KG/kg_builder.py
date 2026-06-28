# kg_builder.py — Spec-compliant version
import asyncio

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts import ChatPromptTemplate
#from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document
from typing import List, Dict, Any

import time

from tqdm import tqdm

KG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert knowledge graph extraction system.

Extract entities and relationships from the text.

IMPORTANT ENTITY NAMING RULES:

- Use lowercase names only.
- Use singular nouns whenever possible.
- Avoid abbreviations.
- Prefer canonical terminology.
- Do not create duplicate concepts with different casing.
- Normalize synonymous forms when obvious.

Examples:

operating systems -> operating system
OS -> operating system
Maximum -> maximum

Entity types allowed:
Concept
Formula
Theorem
Example
Method
Definition

Relationship types allowed:
PREREQUISITE
EXTENDS
DEFINES
APPLIES_TO
ILLUSTRATES
PART_OF
"""
    ),
    ("human", "{input}")
])

class KGBuilder:
    def __init__(self, llm, neo4j_uri, neo4j_user, neo4j_password):
        self.transformer = LLMGraphTransformer(
            llm=llm,
            prompt=KG_PROMPT,  
            allowed_nodes=["Concept", "Formula", "Theorem", "Example", "Method", "Definition"],
            allowed_relationships=["PREREQUISITE", "EXTENDS", "DEFINES", "APPLIES_TO", "ILLUSTRATES", "PART_OF"],
            ignore_tool_usage=True,  # ChatNVIDIA doesn't support include_raw=True in structured output
                                     # so we fall back to prompt-based extraction instead
        )
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password
        )
        for label in ["Concept","Formula","Theorem","Example","Method","Definition"]:
            self.graph.query(f"""
            CREATE CONSTRAINT {label.lower()}_unique
            IF NOT EXISTS
            FOR (n:{label})
            REQUIRE n.id IS UNIQUE
            """)


    def normalize_entity_name(name: str) -> str:
        if not name:
            return name

        name = name.lower().strip()
        # remove space
        name = re.sub(r"\s+", " ", name)
        # remove punctuations
        name = re.sub(r"[^\w\s\-]", "", name)

        return name


    def _normalize_graph_doc(self, graph_doc):

        for node in graph_doc.nodes:
            node.id = normalize_entity_name(node.id)

            if hasattr(node, "properties"):
                for key, value in node.properties.items():
                    if isinstance(value, str):
                        node.properties[key] = value.strip()

        # normalize relations
        for rel in graph_doc.relationships:
            rel.source.id = normalize_entity_name(rel.source.id)
            rel.target.id = normalize_entity_name(rel.target.id)

        return graph_doc

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

    def build_from_dicts(self, chunks_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Convert chunk dicts to LangChain Documents
        documents = [
            Document(
                page_content=chunk["text"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "course_id": chunk["course_id"],
                }
            )
            for chunk in chunks_dicts
        ]

        # LangChain handles extraction + schema enforcement
        # TODO  ( uncomment when access to GPU is available)
        #graph_docs = asyncio.run(self.transformer.aconvert_to_graph_documents(documents))

        # Sequential extraction with throttle — avoids 429 on NVIDIA's per-minute limit
        graph_docs = []
        for doc in tqdm(documents):
            graph_doc = self._convert_one_with_retry(doc)

            graph_doc = self._normalize_graph_doc(graph_doc) #normalize nodes and relations

            graph_docs.append(graph_doc)

            time.sleep(1.5)
        
        
        self.graph.add_graph_documents(graph_docs, include_source=True)
        
        # Write covers_concepts
        chunk_id_to_dict = {c["chunk_id"]: c for c in chunks_dicts}
        for graph_doc in graph_docs:
            chunk_id = graph_doc.source.metadata.get("chunk_id")
            if chunk_id and chunk_id in chunk_id_to_dict:
                chunk_id_to_dict[chunk_id]["covers_concepts"] = [
                    {"id": node.id}  # LangChain uses name as ID
                    for node in graph_doc.nodes
                ]

        total_nodes = sum(len(gd.nodes) for gd in graph_docs)
        total_rels = sum(len(gd.relationships) for gd in graph_docs)

        return {
            "total_chunks": len(chunks_dicts),
            "total_nodes": total_nodes,
            "total_relations": total_rels,
            "covers_concepts": chunks_dicts,
        }