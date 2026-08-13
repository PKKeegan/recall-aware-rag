"""
abstention.py

Decision policy for Recall-Aware Abstention RAG.

Responsibilities
----------------
- Decide whether to answer or abstain
- Apply configurable confidence threshold
- Return a simple decision

This module DOES NOT:
- Retrieve documents
- Generate answers
- Compute confidence
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.config import CONFIDENCE_THRESHOLD


@dataclass
class AbstentionResult:
    """
    Result returned by the abstention policy.
    """

    decision: str
    confidence: float
    threshold: float


class AbstentionPolicy:
    """
    Decide whether the system should answer
    or abstain based on confidence.
    """

    def __init__(
        self,
        threshold: float = CONFIDENCE_THRESHOLD,
    ):

        self.threshold = threshold

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def decide(
        self,
        confidence: float,
    ) -> AbstentionResult:
        """
        Parameters
        ----------
        confidence : float

        Returns
        -------
        AbstentionResult
        """

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        if confidence >= self.threshold:
            decision = "answer"
        else:
            decision = "abstain"

        return AbstentionResult(
            decision=decision,
            confidence=confidence,
            threshold=self.threshold,
        )

    # ---------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------

    def should_answer(
        self,
        confidence: float,
    ) -> bool:

        return confidence >= self.threshold

    def should_abstain(
        self,
        confidence: float,
    ) -> bool:

        return confidence < self.threshold
    