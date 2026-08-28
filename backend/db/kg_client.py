
import logging


from backend.core.models import MODELS
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
        # OpenAI (native function calling) — enables node_properties in the KG builder and gives
        # much stronger extraction than the old llama-3.1-70b/NVIDIA prompt-based path.
        llm = MODELS["gpt-5-mini"]
        _kg = KGBuilder(
            llm=llm,
            neo4j_uri=settings.neo4j_uri,
            neo4j_user=settings.neo4j_user,
            neo4j_password=settings.neo4j_password,
        )
        logger.info("Creating kg")
    except Exception as e:
        logger.error("Failed to initialize KGBuilder: %s", e)
        raise
    return _kg

