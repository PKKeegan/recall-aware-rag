import numpy as np

from src.common.embeddings import get_embedding_model


class SemanticOverlap:

    def __init__(self):
        self.embed_model = get_embedding_model()

    def compute(self, answer: str, context: str) -> float:
        """
        Compute semantic similarity between the generated
        answer and the retrieved context.

        Returns a score between 0 and 1.
        """

        if not answer.strip() or not context.strip():
            return 0.0

        answer_embedding = np.array(
            self.embed_model.get_text_embedding(answer)
        )

        context_embedding = np.array(
            self.embed_model.get_text_embedding(context)
        )

        similarity = np.dot(
            answer_embedding,
            context_embedding
        ) / (
            np.linalg.norm(answer_embedding)
            * np.linalg.norm(context_embedding)
            + 1e-8
        )

        # Convert cosine similarity [-1, 1] to [0, 1]
        normalized_similarity = (similarity + 1) / 2

        return float(normalized_similarity)