# kg_builder.py
# Knowledge Graph Construction

import os
import json
import hashlib
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field, validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class NodeType(str, Enum):
    """Allowed node types for the knowledge graph"""
    CONCEPT = "Concept"
    FORMULA = "Formula"
    THEOREM = "Theorem"
    EXAMPLE = "Example"
    METHOD = "Method"
    DEFINITION = "Definition"


class RelationType(str, Enum):
    """Allowed relation types between nodes"""
    PREREQUISITE = "prerequisite"
    EXTENDS = "extends"
    DEFINES = "defines"
    APPLIES_TO = "applies_to"
    ILLUSTRATES = "illustrates"
    PART_OF = "part_of"


class KGNode(BaseModel):
    """Knowledge Graph Node - represents a concept, formula, theorem, etc."""
    id: str = Field(default="", description="Unique ID generated from name and type")
    name: str = Field(description="Name of the entity")
    node_type: NodeType = Field(description="Type following controlled schema")
    description: str = Field(description="Description extracted from chunk")
    chunk_ids: List[str] = Field(default_factory=list, description="Source chunk IDs")
    
    def generate_id(self) -> str:
        """Generate deterministic ID from name and type"""
        raw = f"{self.name.lower().strip()}_{self.node_type.value}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]
    
    def model_post_init(self, __context):
        if not self.id:
            self.id = self.generate_id()


class KGRelation(BaseModel):
    """Relation between two KG nodes"""
    source_node_id: str = Field(description="Source node ID")
    target_node_id: str = Field(description="Target node ID")
    relation_type: RelationType = Field(description="Type following controlled schema")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    chunk_id: str = Field(description="Source chunk ID that generated this relation")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {v}")
        return v


class KGChunkExtraction(BaseModel):
    """Complete extraction result for one chunk"""
    nodes: List[KGNode] = Field(default_factory=list)
    relations: List[KGRelation] = Field(default_factory=list)
    
    def is_empty(self) -> bool:
        return len(self.nodes) == 0 and len(self.relations) == 0


# =============================================================================
# LLM GRAPH TRANSFORMER
# =============================================================================

class LLMGraphTransformer:
    """
    Transforms text chunks into nodes and relations using LLM
    Now accepts dictionaries from ingestion.py directly
    """
    
    def __init__(self, llm, temperature: float = 0.1, max_concurrent: int = 5):
        self.llm = llm
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        
        self.parser = PydanticOutputParser(pydantic_object=KGChunkExtraction)
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with controlled schema definition"""
        node_types_list = [nt.value for nt in NodeType]
        relation_types_list = [rt.value for rt in RelationType]
        
        return f"""You are an expert at extracting structured knowledge from educational text.

CONTROLLED SCHEMA - ONLY use these types:
- Node types: {', '.join(node_types_list)}
- Relation types: {', '.join(relation_types_list)}

EXTRACTION RULES:
1. Extract ONLY nodes that match the allowed node types
2. Extract ONLY relations between nodes that match allowed relation types
3. Each node must have a clear name and brief description from the text
4. Relations must be directional (source → target)
5. Set confidence based on textual evidence:
   - 0.9-1.0: explicitly stated
   - 0.7-0.8: strongly implied  
   - 0.5-0.6: weakly implied
6. Return empty lists if the chunk has no meaningful educational content

{self.parser.get_format_instructions()}

IMPORTANT: Return ONLY valid JSON. No markdown, no extra text."""
    
    def _chunk_dict_to_input(self, chunk_dict: Dict[str, Any]) -> tuple:
        
        chunk_id = chunk_dict.get("chunk_id", f"chunk_{hash(chunk_dict.get('text', ''))}")
        text = chunk_dict.get("text", "")
        
        # Build metadata from available fields
        metadata = {
            "document_id": chunk_dict.get("document_id", "unknown"),
            "course_id": chunk_dict.get("course_id", "unknown"),
            "chunk_index": chunk_dict.get("chunk_index", 0),
            "source_file": chunk_dict.get("document_id", "unknown"),  # For compatibility
            "subject": chunk_dict.get("course_id", "unknown")  # Use course_id as subject
        }
        
        return chunk_id, text, metadata
    
    async def transform_chunk_dict(self, chunk_dict: Dict[str, Any]) -> KGChunkExtraction:
        """
        Transform ONE chunk dictionary into KG extraction
        
        Args:
            chunk_dict: Dictionary parse_and_chunk()
            
        Returns:
            Validated KGChunkExtraction
        """
        chunk_id, text, metadata = self._chunk_dict_to_input(chunk_dict)
        
        # Build user prompt
        user_prompt = f"""Chunk ID: {chunk_id}
Document ID: {metadata.get('document_id', 'unknown')}
Course ID: {metadata.get('course_id', 'unknown')}

Text:
{text}

Extract nodes and relations according to the controlled schema."""
        
        try:
            # LLM call
            response = await self.llm.ainvoke([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            # Clean response
            content = response.content if hasattr(response, 'content') else str(response)
            content = self._clean_json_response(content)
            
            # Parse with Pydantic validation
            extraction = self.parser.parse(content)
            
            # Add chunk_id to all nodes
            for node in extraction.nodes:
                if chunk_id not in node.chunk_ids:
                    node.chunk_ids.append(chunk_id)
            
            # Add chunk_id to all relations
            for rel in extraction.relations:
                if not rel.chunk_id:
                    rel.chunk_id = chunk_id
            
            return extraction

        except Exception as e:
            # Distinguish failure types for ChatNVIDIA
            err_str = str(e).lower()
            err_type = type(e).__name__

            if any(k in err_str for k in ("429", "rate limit", "too many requests")):
                print(
                    f"[RATE LIMIT] Chunk {chunk_id} dropped — ChatNVIDIA quota exceeded. "
                    f"Lower max_concurrent or add a retry delay. ({err_type}: {e})"
                )
            elif any(k in err_str for k in ("401", "403", "unauthorized", "forbidden", "api key", "authentication")):
                print(
                    f"[AUTH ERROR] Chunk {chunk_id} dropped — ChatNVIDIA credentials rejected. "
                    f"Check your NVIDIA_API_KEY. ({err_type}: {e})"
                )
            elif any(k in err_str for k in ("500", "502", "503", "504", "server error", "service unavailable", "timeout", "timed out", "connection")):
                print(
                    f"[NETWORK/SERVER ERROR] Chunk {chunk_id} dropped — ChatNVIDIA endpoint unreachable or timed out. "
                    f"Consider retrying. ({err_type}: {e})"
                )
            elif "json" in err_str or isinstance(e, (ValueError, KeyError)) and "parse" in err_str:
                print(
                    f"[JSON PARSE ERROR] Chunk {chunk_id} dropped — LLM returned malformed JSON. "
                    f"Raw content may contain markdown or extra text. ({err_type}: {e})"
                )
            elif err_type in ("ValidationError", "PydanticValidationError") or "validation" in err_str:
                print(
                    f"[SCHEMA VALIDATION ERROR] Chunk {chunk_id} dropped — extracted JSON doesn't match "
                    f"KGChunkExtraction schema (wrong node_type or relation_type?). ({err_type}: {e})"
                )
            else:
                print(
                    f"[UNEXPECTED ERROR] Chunk {chunk_id} dropped — unhandled exception during LLM extraction. "
                    f"({err_type}: {e})"
                )

            return KGChunkExtraction()
    
    def _clean_json_response(self, raw: str) -> str:
        """Clean LLM response to get valid JSON"""
        # Remove markdown code fences
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        return raw.strip()
    
    async def transform_batch_dict(self, chunks_dicts: List[Dict[str, Any]]) -> List[KGChunkExtraction]:

        #Transform multiple chunk dictionaries in parallel with controlled concurrency

        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_limit(chunk_dict):
            async with semaphore:
                return await self.transform_chunk_dict(chunk_dict)
        
        tasks = [process_with_limit(chunk) for chunk in chunks_dicts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        extractions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Failed chunk {i}: {result}")
                extractions.append(KGChunkExtraction())
            else:
                extractions.append(result)
        
        return extractions


# =============================================================================
# KNOWLEDGE GRAPH STORAGE
# =============================================================================

class KnowledgeGraphStore:
    """KG Storage with Neo4j + mapping chunk → nodes"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Args:
            neo4j_uri: ex: "bolt://localhost:7687"
            neo4j_user: ex: "neo4j"
            neo4j_password: Neo4j password
        """
        from neo4j import GraphDatabase
        self.uri = neo4j_uri  # store URI
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self._init_constraints()
    
    def _init_constraints(self):
        """Initialize uniqueness constraints and indexes"""
        with self.driver.session() as session:
            # Constraints per node type
            for node_type in NodeType:
                session.run(f"""
                    CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node_type.value}) 
                    REQUIRE n.id IS UNIQUE
                """)
            
            
            #Chunk nodes need their own uniqueness constraint
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk)
                REQUIRE c.chunk_id IS UNIQUE
            """)
            
            # Index for relations — one index per relationship type (Neo4j 5 syntax)
            for rel_type in RelationType:
                session.run(f"""
                    CREATE INDEX rel_chunk_id_{rel_type.name} IF NOT EXISTS
                    FOR ()-[r:{rel_type.value.upper()}]-() ON (r.chunk_id)
                """)
    
    def add_extractions(self, extractions: List[KGChunkExtraction]):
        """Add extractions to KG (synchronous version)"""
        with self.driver.session() as session:
            for extraction in extractions:
                if extraction.is_empty():
                    continue
                
                # Add nodes
                for node in extraction.nodes:
                    # Fetch existing chunk_ids first, merge in Python
                    result = session.run(
                        "MATCH (n {id: $id}) RETURN n.chunk_ids AS existing, n.description AS existing_desc",
                        id=node.id
                    ).single()
                    
                    existing_chunk_ids = result["existing"] if result else []
                    existing_description = result["existing_desc"] if result else ""

                    # Keep the longest description
                    best_description = (
                        node.description
                        if len(node.description) >= len(existing_description or "")
                        else existing_description
                    )


                    merged_chunk_ids = list(set((existing_chunk_ids  or []) + node.chunk_ids))
                    
                    session.run(
                        f"""
                        MERGE (n:{node.node_type.value} {{id: $id}})
                        SET n.name = $name,
                            n.description = $description,
                            n.chunk_ids = $chunk_ids
                        """,
                        id=node.id,
                        name=node.name,
                        description=best_description,
                        chunk_ids=merged_chunk_ids
                    )
                    #For each source chunk, create a Chunk node and link it
                    for chunk_id in node.chunk_ids:
                        session.run(
                            """
                            MERGE (c:Chunk {chunk_id: $chunk_id})
                            WITH c
                            MATCH (n {id: $node_id})
                            MERGE (n)-[:SOURCED_FROM]->(c)
                            """,
                            chunk_id=chunk_id,
                            node_id=node.id
                        )
            for extraction in extractions:
                if extraction.is_empty():
                    continue
  
                # Add relations
                for rel in extraction.relations:
                    session.run(
                        f"""
                        MATCH (source {{id: $source_id}})
                        MATCH (target {{id: $target_id}})
                        MERGE (source)-[r:{rel.relation_type.value.upper()}]->(target)
                        SET r.confidence = $confidence,
                            r.chunk_id = $chunk_id
                        """,
                        source_id=rel.source_node_id,
                        target_id=rel.target_node_id,
                        confidence=rel.confidence,
                        chunk_id=rel.chunk_id
                    )
                    
    
    async def add_extractions_batch(self, extractions: List[KGChunkExtraction], batch_size: int = 50):
        """Async batch version for better performance"""
        valid = [e for e in extractions if not e.is_empty()]
        
        if not valid:
            return
        
        for i in range(0, len(valid), batch_size):
            batch = valid[i:i+batch_size]
            await asyncio.to_thread(self.add_extractions, batch)
            print(f" Stored batch {i//batch_size + 1}/{(len(valid)-1)//batch_size + 1}")
    

    def get_nodes_by_chunk(self, chunk_id: str) -> List[Dict]:
        """Return all nodes associated with a specific chunk"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n)-[:SOURCED_FROM]->(c:Chunk {chunk_id: $chunk_id_in_neo4j})
                RETURN n.id AS id,
                    n.name AS name,
                    labels(n)[0] AS type,
                    n.description AS description
                """,
                chunk_id_in_neo4j=chunk_id
            )
            return [dict(record) for record in result]

    
    def get_related_chunks(self, node_id: str, max_hops: int = 2) -> List[str]:
        """Find chunks related to a node (for augmented retrieval)"""
        with self.driver.session() as session:
            rel_pattern = "|".join([rt.value.upper() for rt in RelationType])
            
            result = session.run(
                f"""
                MATCH (n {{id: $node_id_in_neo4j}})-[:{rel_pattern}*1..{max_hops}]-(connected)
                UNWIND connected.chunk_ids AS chunk_id
                RETURN DISTINCT chunk_id
                LIMIT 50
                """,
                node_id_in_neo4j=node_id
            )
            return [record["chunk_id"] for record in result]
    
    def get_chunk_neighborhood(self, chunk_id: str, max_nodes: int = 20) -> Dict:
        """Get the KG neighborhood of a chunk (nodes + local relations)"""
        with self.driver.session() as session:
            nodes = self.get_nodes_by_chunk(chunk_id)
            
            if not nodes:
                return {"nodes": [], "relations": []}
            
            node_ids = [n["id"] for n in nodes]
            
            result = session.run(
                f"""
                MATCH (n)-[r]->(m)
                WHERE n.id IN $node_ids_in_neo4j AND m.id IN $node_ids_in_neo4j
                RETURN n.id AS source, type(r) AS type, m.id AS target, r.confidence AS confidence
                """,
                node_ids_in_neo4j=node_ids
            )
            relations = [dict(record) for record in result]
            
            return {"nodes": nodes, "relations": relations}
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()


# =============================================================================
# MAIN PIPELINE - KGBuilder
# =============================================================================

class KGBuilder:
    """
    Complete KG construction pipeline
    Assembles all 3 blocks: Schema → Transformer → Store with Linking
    Now accepts dictionaries from ingestion.py directly
    """
    
    def __init__(
        self,
        llm,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        max_concurrent: int = 5
    ):
        """
        Args:
            llm: LLM instance (ChatNVIDIA, ChatOpenAI, etc.)
            neo4j_uri: Neo4j URI (ex: "bolt://localhost:7687")
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            max_concurrent: Maximum parallel LLM calls
        """
        self.transformer = LLMGraphTransformer(llm, max_concurrent=max_concurrent)
        self.store = KnowledgeGraphStore(neo4j_uri, neo4j_user, neo4j_password)
    
    async def build_from_dicts(self, chunks_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build KG from dictionaries produced
        
        Args:
            chunks_dicts: List of dictionaries from parse_and_chunk()
                         Each dict has: chunk_id, document_id, course_id, 
                                       chunk_index, text, covers_concepts
            
        Returns:
            Statistics about the construction
        """
        print(f"\n{'='*60}")
        print(f"KNOWLEDGE GRAPH CONSTRUCTION")
        print(f"{'='*60}")
        print(f"📦 Input: {len(chunks_dicts)} chunks (from ingestion.py format)")
        
        # LLM Transformation
        print(f"\nLLM Graph Transformation...")
        extractions = await self.transformer.transform_batch_dict(chunks_dicts)
        
        # Statistics
        total_nodes = sum(len(e.nodes) for e in extractions)
        total_relations = sum(len(e.relations) for e in extractions)
        non_empty = sum(1 for e in extractions if not e.is_empty())
        
        print(f"  ✓ Extracted: {total_nodes} nodes, {total_relations} relations")
        print(f"  ✓ Non-empty chunks: {non_empty}/{len(chunks_dicts)}")
        
        #Storage with chunk linking
        print(f"\n Storage + Chunk ID Linking...")
        await self.store.add_extractions_batch(extractions)
        
        print(f"\n✅ KG Construction complete!")
        print(f"  • Nodes stored: {total_nodes}")
        print(f"  • Relations stored: {total_relations}")
        print(f"  • Neo4j: {self.store.uri}")
        
        # Write covers_concepts back into each chunk dict.
        # Uses chunk_id_to_dict for explicit, order-independent mapping.
        # Each entry is {"id": ..., "name": ...} so callers have both the
        # graph-traversal key and the human-readable label in one field,
        # consistent with the covers_concepts: [] schema from ingestion.py.
        chunk_id_to_extraction = {
            chunk_dict["chunk_id"]: extraction
            for chunk_dict, extraction in zip(chunks_dicts, extractions)
        }
        chunk_id_to_dict = {c["chunk_id"]: c for c in chunks_dicts}

        for chunk_id, chunk_dict in chunk_id_to_dict.items():
            extraction = chunk_id_to_extraction[chunk_id]
            chunk_dict["covers_concepts"] = [
                {"id": node.id, "name": node.name}
                for node in extraction.nodes
            ]

        return {
            "total_chunks": len(chunks_dicts),
            "non_empty_chunks": non_empty,
            "total_nodes": total_nodes,
            "total_relations": total_relations,
            "extractions": extractions,
            "chunks_with_concepts": chunks_dicts
        }
        
    
    def get_chunk_nodes(self, chunk_id: str) -> List[Dict]:
        """Get nodes associated with a chunk (BLOCK 3: linking query)"""
        return self.store.get_nodes_by_chunk(chunk_id)
    
    def get_related_chunks(self, node_id: str, max_hops: int = 2) -> List[str]:
        """Get chunks related to a node"""
        return self.store.get_related_chunks(node_id, max_hops)
    
    def close(self):
        """Close Neo4j connection"""
        self.store.close()