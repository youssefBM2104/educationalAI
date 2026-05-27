
import logging


from langchain_nvidia_ai_endpoints import ChatNVIDIA
from backend.core.config import settings
from backend.KG.kg_builder import KGBuilder


_kg = None

logger = logging.getLogger(__name__)


def get_kg():
    global _kg
    if _kg:
        logger.info(f"Using existing kg")

        return _kg
    try:
        llm = ChatNVIDIA(
            model="meta/llama-3.1-70b-instruct",
            api_key=settings.nim_api_key,
            temperature=0.1,
        )
        _kg = KGBuilder(
            llm=llm,
            neo4j_uri=settings.neo4j_uri,
            neo4j_user=settings.neo4j_user,
            neo4j_password=settings.neo4j_password,
        )
        logger.info("Created kg")
    except Exception as e:
        logger.error("Failed to initialize KGBuilder: %s", e)
        raise
    return _kg

