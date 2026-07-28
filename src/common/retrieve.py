from src.common.vector_store import load_vector_index
from src.common.config import TOP_K


class Retriever:

    def __init__(self):

        self.index = load_vector_index()

        self.retriever = self.index.as_retriever(
            similarity_top_k=TOP_K
        )

    def retrieve(self, query):

        return self.retriever.retrieve(query)


retriever = Retriever()


def retrieve(query):

    return retriever.retrieve(query)