from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from src.common.config import (
    DATA_DIR,
    VECTOR_STORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from src.common.embeddings import get_embedding_model


def main():

    print("Loading documents...")

    documents = SimpleDirectoryReader(
        input_dir=str(DATA_DIR)
    ).load_data()

    print(f"Loaded {len(documents)} documents")

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    print(f"Created {len(nodes)} chunks")

    print("Loading embedding model...")

    embed_model = get_embedding_model()

    print("Building index...")

    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model,
    )

    VECTOR_STORE_DIR.mkdir(
        exist_ok=True,
        parents=True,
    )

    index.storage_context.persist(
        persist_dir=str(VECTOR_STORE_DIR)
    )

    print("Index saved!")


if __name__ == "__main__":
    main()