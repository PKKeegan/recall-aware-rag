"""
confidence.py

Compute the final confidence score for
Recall-Aware Abstention RAG.

Responsibilities
----------------
✓ Combine recall
✓ Combine semantic overlap
✓ Combine entropy
✓ Produce final confidence score
✓ Decide whether to answer or abstain

This module DOES NOT:
- Retrieve documents
- Generate answers
- Compute embeddings
- Compute recall
- Compute entropy

Those responsibilities belong to other modules.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.common.config import (
    RECALL_WEIGHT,
    OVERLAP_WEIGHT,
    ENTROPY_WEIGHT,
    CONFIDENCE_THRESHOLD,
)


class ConfidenceEstimator:
    """
    Compute the final confidence score
    from Recall, Semantic Overlap,
    and Entropy.
    """

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def estimate(
        self,
        recall: float,
        overlap: float,
        entropy: float,
    ) -> Dict:
        """
        Estimate confidence.

        Parameters
        ----------
        recall : float

        overlap : float

        entropy : float

        Returns
        -------
        Dict
        """

        confidence = self._combine(
            recall=recall,
            overlap=overlap,
            entropy=entropy,
        )

        decision = self._decision(
            confidence
        )

        return {

            "confidence": confidence,

            "decision": decision,

            "recall": recall,

            "semantic_overlap": overlap,

            "entropy": entropy,

        }

    # ---------------------------------------------------------
    # Confidence Formula
    # ---------------------------------------------------------

    @staticmethod
    def _combine(
        recall: float,
        overlap: float,
        entropy: float,
    ) -> float:
        """
        Compute

        Confidence =
            w1 * Recall
          + w2 * Overlap
          + w3 * (1 - Entropy)
        """

        confidence = (

            RECALL_WEIGHT * recall

            +

            OVERLAP_WEIGHT * overlap

            +

            ENTROPY_WEIGHT * (1.0 - entropy)

        )

        return float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )

    # ---------------------------------------------------------
    # Decision
    # ---------------------------------------------------------

    @staticmethod
    def _decision(
        confidence: float,
    ) -> str:
        """
        Determine whether the model
        should answer or abstain.
        """

        if confidence >= CONFIDENCE_THRESHOLD:

            return "answer"

        return "abstain"