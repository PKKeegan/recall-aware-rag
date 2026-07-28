from llama_index.embeddings.ollama import OllamaEmbedding

from src.common.config import EMBED_MODEL


def get_embedding_model():
    return OllamaEmbedding(
        model_name=EMBED_MODEL
    )