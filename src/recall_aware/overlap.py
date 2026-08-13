"""
overlap.py

Semantic similarity utilities for Recall-Aware Abstention RAG.

Responsibilities
----------------
✓ Generate embeddings
✓ Compute cosine similarity
✓ Compute semantic overlap between texts

This module DOES NOT:
- Retrieve documents
- Generate answers
- Estimate recall
- Estimate entropy

Those responsibilities belong to other modules.
"""

from __future__ import annotations

import numpy as np

from src.common.embeddings import get_embedding_model


class SemanticOverlap:
    """
    Compute semantic similarity between texts.
    """

    def __init__(self):

        self.embed_model = get_embedding_model()


    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def compute(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        Compute semantic similarity between two texts.

        Returns
        -------
        float
            Similarity score normalized to [0,1]
        """

        if (
            not text1
            or not text2
            or not text1.strip()
            or not text2.strip()
        ):

            return 0.0


        embedding1 = self.embed(
            text1
        )

        embedding2 = self.embed(
            text2
        )


        return self.similarity(
            embedding1,
            embedding2,
        )


    # ---------------------------------------------------------
    # Embedding
    # ---------------------------------------------------------

    def embed(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Generate embedding vector.
        """

        embedding = (
            self.embed_model
            .get_text_embedding(text)
        )


        return np.asarray(
            embedding,
            dtype=np.float32,
        )


    # ---------------------------------------------------------
    # Cosine Similarity
    # ---------------------------------------------------------

    @staticmethod
    def similarity(
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity.

        Returns
        -------
        float
            Value normalized to [0,1]
        """

        norm1 = np.linalg.norm(
            embedding1
        )

        norm2 = np.linalg.norm(
            embedding2
        )


        if norm1 == 0 or norm2 == 0:

            return 0.0


        cosine = (
            np.dot(
                embedding1,
                embedding2,
            )
            /
            (
                norm1
                *
                norm2
            )
        )


        cosine = float(
            np.clip(
                cosine,
                -1.0,
                1.0,
            )
        )


        return (
            cosine + 1.0
        ) / 2.0



    # ---------------------------------------------------------
    # Batch Similarity
    # ---------------------------------------------------------

    def similarities(
        self,
        query_embedding: np.ndarray,
        texts: list[str],
    ) -> list[float]:
        """
        Compute similarity between one embedding
        and multiple texts.
        """

        scores = []


        for text in texts:

            if (
                not text
                or not text.strip()
            ):

                scores.append(
                    0.0
                )

                continue


            embedding = self.embed(
                text
            )


            score = self.similarity(
                query_embedding,
                embedding,
            )


            scores.append(
                score
            )


        return scores