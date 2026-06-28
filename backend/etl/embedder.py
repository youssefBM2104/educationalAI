import logging
from FlagEmbedding import BGEM3FlagModel
from backend.core.config import settings

logger = logging.getLogger(__name__)
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device=settings.embedding_device)
logger.info("BGE-M3 model loaded")

def embed_chunks(chunks: list[dict]) -> list[dict]:
    texts = [chunk["text"] for chunk in chunks]
    output = model.encode(
        texts,
        batch_size=16,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    for i, chunk in enumerate(chunks):
        chunk["dense_vector"] = output["dense_vecs"][i].tolist()
        token_weights = output["lexical_weights"][i]
        id_to_weight: dict[int, float] = {}
        for token_string, weight in token_weights.items():
            token_id = (
                token_string
                if isinstance(token_string, int)
                else model.tokenizer.convert_tokens_to_ids(token_string)
            )
            if isinstance(token_id, int) and token_id >= 0:
                id_to_weight[token_id] = id_to_weight.get(token_id, 0.0) + float(weight)
        chunk["sparse_vector"] = {
            "indices": list(id_to_weight.keys()),
            "values": list(id_to_weight.values()),
        }
    return chunks