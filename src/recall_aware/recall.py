"""
recall.py

Estimate retrieval recall for Recall-Aware Abstention RAG.

Responsibilities
----------------
✓ Estimate retrieval quality
✓ Measure evidence coverage
✓ Measure ranking quality
✓ Produce a normalized recall estimate

This module DOES NOT:
- Retrieve documents
- Generate answers
- Compute entropy
- Compute final confidence

Those responsibilities belong to other modules.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.common.config import (
    RECALL_SIMILARITY_WEIGHT,
    RECALL_COVERAGE_WEIGHT,
    RECALL_RANK_WEIGHT,
    RELEVANCE_THRESHOLD,
)


class RecallEstimator:
    """
    Estimate retrieval recall using retrieval quality
    rather than ground-truth relevance labels.
    """

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def estimate(
        self,
        retrieved_nodes: List,
    ) -> float:
        """
        Estimate retrieval recall.

        Parameters
        ----------
        retrieved_nodes : List

        Returns
        -------
        float
            Estimated recall in [0,1]
        """

        scores = self._extract_scores(
            retrieved_nodes
        )

        if len(scores) == 0:

            return 0.0

        average_similarity = self._average_similarity(
            scores
        )

        coverage = self._coverage(
            scores
        )

        rank_quality = self._rank_quality(
            scores
        )

        recall = (

            RECALL_SIMILARITY_WEIGHT
            * average_similarity

            +

            RECALL_COVERAGE_WEIGHT
            * coverage

            +

            RECALL_RANK_WEIGHT
            * rank_quality

        )

        return float(
            np.clip(
                recall,
                0.0,
                1.0,
            )
        )

    # ---------------------------------------------------------
    # Score extraction
    # ---------------------------------------------------------

    @staticmethod
    def _extract_scores(
        retrieved_nodes: List,
    ) -> np.ndarray:
        """
        Extract similarity scores from retrieved nodes.
        """

        scores = []

        for node in retrieved_nodes:

            score = getattr(
                node,
                "score",
                None,
            )

            if score is not None:

                scores.append(
                    float(score)
                )

        return np.asarray(
            scores,
            dtype=float,
        )

    # ---------------------------------------------------------
    # Average similarity
    # ---------------------------------------------------------

    @staticmethod
    def _average_similarity(
        scores: np.ndarray,
    ) -> float:
        """
        Mean retrieval similarity.
        """

        return float(
            np.mean(scores)
        )

    # ---------------------------------------------------------
    # Coverage
    # ---------------------------------------------------------

    @staticmethod
    def _coverage(
        scores: np.ndarray,
    ) -> float:
        """
        Fraction of retrieved chunks considered relevant.
        """

        relevant = np.sum(
            scores >= RELEVANCE_THRESHOLD
        )

        return float(
            relevant / len(scores)
        )

    # ---------------------------------------------------------
    # Rank quality
    # ---------------------------------------------------------

    @staticmethod
    def _rank_quality(
        scores: np.ndarray,
    ) -> float:
        """
        Measure how evenly retrieval quality is distributed.

        Good retrieval should have several strong chunks,
        not only one excellent chunk.
        """

        if len(scores) == 1:

            return 1.0

        top = scores[0]

        remaining = np.mean(
            scores[1:]
        )

        gap = max(
            0.0,
            top - remaining,
        )

        quality = 1.0 - gap

        return float(
            np.clip(
                quality,
                0.0,
                1.0,
            )
        )