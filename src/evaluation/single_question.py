"""
single_question.py

Single-question evaluation wrapper for the
Recall-Aware RAG pipeline.

Responsibilities
----------------
- Run one evaluation question through RecallAwareRAG
- Record the generated answer
- Record the system decision
- Record confidence-related metrics
- Record retrieval information
- Record timing information
- Preserve the raw pipeline result for debugging

This module DOES NOT:
- Load the Excel evaluation dataset
- Run multiple questions
- Calculate aggregate evaluation metrics
- Compare baseline RAG with Recall-Aware RAG
"""

from __future__ import annotations

import time
from typing import Any, Dict

from src.recall_aware.recall_rag import RecallAwareRAG


class SingleQuestionEvaluator:
    """
    Evaluate a single question using the existing
    RecallAwareRAG pipeline.
    """

    def __init__(
        self,
        rag: RecallAwareRAG | None = None,
    ) -> None:
        """
        Initialize the evaluator.

        Parameters
        ----------
        rag:
            Existing RecallAwareRAG instance.

            Passing an existing instance is recommended when
            evaluating many questions because the RAG pipeline
            does not need to be initialized repeatedly.
        """

        if rag is None:
            self.rag = RecallAwareRAG()
        else:
            self.rag = rag

    # =========================================================
    # PUBLIC API
    # =========================================================

    def evaluate(
        self,
        question_id: str,
        question: str,
    ) -> Dict[str, Any]:
        """
        Run one question through RecallAwareRAG.

        Parameters
        ----------
        question_id:
            Unique ID from the evaluation dataset.

        question:
            Evaluation question.

        Returns
        -------
        Dict[str, Any]
            Structured evaluation result.
        """

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        # -----------------------------------------------------
        # Start timing
        # -----------------------------------------------------

        start_time = time.perf_counter()

        # -----------------------------------------------------
        # Execute the existing RAG pipeline
        # -----------------------------------------------------

        raw_result = self.rag.query(
            question
        )

        # -----------------------------------------------------
        # End timing
        # -----------------------------------------------------

        total_time = (
            time.perf_counter()
            - start_time
        )

        # -----------------------------------------------------
        # Extract values
        # -----------------------------------------------------

        answer = self._get_value(
            raw_result,
            "answer",
            "response",
            "generated_answer",
            "output",
        )

        decision = self._get_value(
            raw_result,
            "decision",
        )

        reason = self._get_value(
            raw_result,
            "reason",
            "decision_reason",
        )

        recall = self._get_value(
            raw_result,
            "recall",
            "recall_score",
        )

        semantic_overlap = self._get_value(
            raw_result,
            "semantic_overlap",
            "overlap",
            "overlap_score",
        )

        entropy = self._get_value(
            raw_result,
            "entropy",
            "combined_entropy",
        )

        confidence = self._get_value(
            raw_result,
            "confidence",
            "confidence_score",
        )

        # -----------------------------------------------------
        # Retrieval timing
        #
        # IMPORTANT:
        # retrieval_time is stored inside the nested
        # RetrievalResult object:
        #
        # raw_result.retrieval.retrieval_time
        # -----------------------------------------------------

        retrieval_time = self._get_nested_value(
            raw_result,
            "retrieval",
            "retrieval_time",
        )

        # -----------------------------------------------------
        # Generation timing
        # -----------------------------------------------------

        generation_time = self._get_value(
            raw_result,
            "generation_time",
        )

        # -----------------------------------------------------
        # Retrieval information
        # -----------------------------------------------------

        retrieved_chunks = self._get_value(
            raw_result,
            "retrieved_chunks",
            "num_retrieved",
            "retrieval_count",
        )

        node_ids = self._get_value(
            raw_result,
            "node_ids",
        )

        similarity_scores = self._get_value(
            raw_result,
            "similarity_scores",
        )

        # -----------------------------------------------------
        # Return structured evaluation record
        # -----------------------------------------------------

        return {
            # -------------------------------------------------
            # Question information
            # -------------------------------------------------

            "question_id": question_id,

            "question": question,

            # -------------------------------------------------
            # Generated response
            # -------------------------------------------------

            "answer": answer,

            # -------------------------------------------------
            # RAG decision
            # -------------------------------------------------

            "decision": decision,

            "reason": reason,

            # -------------------------------------------------
            # Recall-Aware metrics
            # -------------------------------------------------

            "recall": recall,

            "semantic_overlap": semantic_overlap,

            "entropy": entropy,

            "confidence": confidence,

            # -------------------------------------------------
            # Retrieval information
            # -------------------------------------------------

            "retrieved_chunks": retrieved_chunks,

            "node_ids": node_ids,

            "similarity_scores": similarity_scores,

            # -------------------------------------------------
            # Timing
            # -------------------------------------------------

            "retrieval_time": retrieval_time,

            "generation_time": generation_time,

            "total_time": total_time,

            # -------------------------------------------------
            # Preserve original result
            #
            # This is extremely useful while developing
            # the evaluation framework.
            # -------------------------------------------------

            "raw_result": raw_result,
        }

    # =========================================================
    # SAFE VALUE EXTRACTION
    # =========================================================

    @staticmethod
    def _get_value(
        result: Any,
        *keys: str,
    ) -> Any:
        """
        Safely retrieve a value from the result returned
        by RecallAwareRAG.query().

        Supports:
        - dictionaries
        - objects with attributes

        If none of the requested keys exist, returns None.
        """

        if result is None:
            return None

        # -----------------------------------------------------
        # Dictionary result
        # -----------------------------------------------------

        if isinstance(
            result,
            dict,
        ):

            for key in keys:

                if key in result:

                    return result[key]

        # -----------------------------------------------------
        # Object result
        # -----------------------------------------------------

        for key in keys:

            if hasattr(
                result,
                key,
            ):

                return getattr(
                    result,
                    key,
                )

        return None

    # =========================================================
    # NESTED VALUE EXTRACTION
    # =========================================================

    @staticmethod
    def _get_nested_value(
        result: Any,
        parent_key: str,
        child_key: str,
    ) -> Any:
        """
        Safely retrieve a nested value.

        Example
        -------
        For:

            result.retrieval.retrieval_time

        use:

            _get_nested_value(
                result,
                "retrieval",
                "retrieval_time",
            )

        Supports:
        - dictionaries
        - objects with attributes
        """

        if result is None:
            return None

        # -----------------------------------------------------
        # Get parent object
        # -----------------------------------------------------

        parent = None

        if isinstance(
            result,
            dict,
        ):
            parent = result.get(
                parent_key
            )

        else:
            parent = getattr(
                result,
                parent_key,
                None,
            )

        if parent is None:
            return None

        # -----------------------------------------------------
        # Get child value
        # -----------------------------------------------------

        if isinstance(
            parent,
            dict,
        ):
            return parent.get(
                child_key
            )

        return getattr(
            parent,
            child_key,
            None,
        )