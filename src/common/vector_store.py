from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage

from src.common.config import VECTOR_STORE_DIR
from src.common.embeddings import get_embedding_model


def load_vector_index():

    storage_context = StorageContext.from_defaults(
        persist_dir=str(VECTOR_STORE_DIR)
    )

    index = load_index_from_storage(
        storage_context=storage_context,
        embed_model=get_embedding_model()
    )

    return index