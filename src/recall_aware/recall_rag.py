"""
recall_rag.py

End-to-end Recall-Aware Abstention RAG pipeline.

Responsibilities
----------------
- Retrieve relevant chunks
- Build retrieval context
- Generate answer
- Estimate recall
- Estimate semantic overlap
- Estimate predictive entropy
- Compute confidence
- Apply abstention policy
- Return RecallAwareResult

This module only orchestrates the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from src.common.config import (
    TOP_K,
    CONFIDENCE_THRESHOLD,
    RECALL_WEIGHT,
    OVERLAP_WEIGHT,
    ENTROPY_WEIGHT,
)

from src.common.llm import get_llm
from src.common.retrieve import retrieve

from src.recall_aware.recall import RecallEstimator
from src.recall_aware.overlap import SemanticOverlap
from src.recall_aware.entropy import EntropyEstimator
from src.recall_aware.confidence import ConfidenceEstimator
from src.recall_aware.abstention import AbstentionPolicy


# ============================================================
# Retrieval Result
# ============================================================

@dataclass
class RetrievalResult:
    """
    Stores retrieval stage information.
    """

    nodes: list[Any]
    node_ids: list[str]
    scores: list[float]
    context: str
    retrieval_time: float


# ============================================================
# Generation Result
# ============================================================

@dataclass
class GenerationResult:
    """
    Stores generation stage information.
    """

    prompt: str
    answer: str
    generation_time: float


# ============================================================
# Final Pipeline Result
# ============================================================

@dataclass
class RecallAwareResult:
    """
    Complete Recall-Aware RAG output.
    """

    question: str
    answer: str
    final_response: str

    retrieval: RetrievalResult

    recall: float
    semantic_overlap: float
    entropy: float
    confidence: float

    decision: str
    decision_reason: str

    generation_time: float
    confidence_time: float
    total_time: float

    config: dict


# ============================================================
# Recall-Aware RAG
# ============================================================

class RecallAwareRAG:
    """
    End-to-end Recall-Aware Abstention RAG pipeline.

    Components are dependency-injected so they can be
    replaced or mocked during testing.
    """

    def __init__(
        self,
        retriever: Callable = retrieve,
        llm=None,
        recall_estimator=None,
        overlap_estimator=None,
        entropy_estimator=None,
        confidence_estimator=None,
        abstention_policy=None,
    ):

        # -----------------------------------------------------
        # External dependencies
        # -----------------------------------------------------

        self.retriever = retriever

        self.llm = (
            llm
            or get_llm()
        )

        # -----------------------------------------------------
        # Metric components
        # -----------------------------------------------------

        self.recall = (
            recall_estimator
            or RecallEstimator()
        )

        self.overlap = (
            overlap_estimator
            or SemanticOverlap()
        )

        # Predictive entropy requires the same LLM
        # used by the main pipeline.

        self.entropy = (
            entropy_estimator
            or EntropyEstimator(
                llm=self.llm
            )
        )

        self.confidence = (
            confidence_estimator
            or ConfidenceEstimator()
        )

        self.abstention = (
            abstention_policy
            or AbstentionPolicy()
        )

        # -----------------------------------------------------
        # Configuration snapshot
        # -----------------------------------------------------

        self.config = {

            "top_k": TOP_K,

            "confidence_threshold":
                CONFIDENCE_THRESHOLD,

            "recall_weight":
                RECALL_WEIGHT,

            "overlap_weight":
                OVERLAP_WEIGHT,

            "entropy_weight":
                ENTROPY_WEIGHT,
        }


    # ========================================================
    # Retrieval
    # ========================================================

    def _retrieve(
        self,
        question: str,
    ) -> RetrievalResult:
        """
        Retrieve relevant nodes.

        The common retrieve.py exposes:

            retrieve(query)

        Therefore we intentionally pass only
        the question.

        retrieve.py remains unchanged.
        """

        start = perf_counter()

        nodes = self.retriever(
            question
        )

        retrieval_time = (
            perf_counter()
            - start
        )

        node_ids: list[str] = []
        scores: list[float] = []

        for node in nodes:

            # -------------------------------------------------
            # Similarity score
            # -------------------------------------------------

            score = getattr(
                node,
                "score",
                None,
            )

            if score is None:

                scores.append(0.0)

            else:

                scores.append(
                    float(score)
                )

            # -------------------------------------------------
            # Node ID
            # -------------------------------------------------

            node_id = None

            # NodeWithScore
            if hasattr(
                node,
                "node",
            ):

                node_id = getattr(
                    node.node,
                    "node_id",
                    None,
                )

            # Direct node
            if node_id is None:

                node_id = getattr(
                    node,
                    "node_id",
                    None,
                )

            node_ids.append(
                str(
                    node_id
                    or "unknown"
                )
            )

        return RetrievalResult(

            nodes=nodes,

            node_ids=node_ids,

            scores=scores,

            context="",

            retrieval_time=retrieval_time,
        )


    # ========================================================
    # Context Construction
    # ========================================================

    def _build_context(
        self,
        retrieval: RetrievalResult,
    ) -> RetrievalResult:
        """
        Build retrieval context.

        Only:
        - extracts text
        - preserves retrieval order
        - concatenates chunks

        No reranking, summarization,
        compression, or deduplication.
        """

        context_parts: list[str] = []

        for node in retrieval.nodes:

            text = self._extract_node_text(
                node
            )

            if text:

                context_parts.append(
                    text.strip()
                )

        retrieval.context = (
            "\n\n".join(
                context_parts
            )
        )

        return retrieval


    # ========================================================
    # Node Text Extraction
    # ========================================================

    @staticmethod
    def _extract_node_text(
        node: Any,
    ) -> str:
        """
        Robustly extract text from LlamaIndex
        NodeWithScore or TextNode objects.
        """

        # NodeWithScore
        if hasattr(
            node,
            "node",
        ):

            underlying_node = node.node

            if hasattr(
                underlying_node,
                "get_content",
            ):

                return underlying_node.get_content()

            if hasattr(
                underlying_node,
                "text",
            ):

                return underlying_node.text

        # Direct node
        if hasattr(
            node,
            "get_content",
        ):

            return node.get_content()

        if hasattr(
            node,
            "text",
        ):

            return node.text

        return ""


    # ========================================================
    # Prompt Construction
    # ========================================================

    def _build_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Construct the answer-generation prompt.
        """

        return f"""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer cannot be found in the context,
say that the information is unavailable.

Context:

{context}

Question:

{question}

Answer:
""".strip()


    # ========================================================
    # Answer Generation
    # ========================================================

    def _generate_answer(
        self,
        question: str,
        context: str,
    ) -> GenerationResult:
        """
        Generate the primary answer using the LLM.
        """

        prompt = self._build_prompt(
            question,
            context,
        )

        start = perf_counter()

        response = self.llm.complete(
            prompt
        )

        generation_time = (
            perf_counter()
            - start
        )

        answer = response.text.strip()

        return GenerationResult(

            prompt=prompt,

            answer=answer,

            generation_time=generation_time,
        )


    # ========================================================
    # Recall Estimation
    # ========================================================

    def _estimate_recall(
        self,
        nodes: list[Any],
    ) -> float:
        """
        Delegate recall estimation to RecallEstimator.
        """

        return float(
            self.recall.estimate(
                nodes
            )
        )


    # ========================================================
    # Semantic Overlap Estimation
    # ========================================================

    def _estimate_overlap(
        self,
        answer: str,
        context: str,
    ) -> float:
        """
        Delegate semantic overlap calculation
        to SemanticOverlap.
        """

        return float(
            self.overlap.compute(
                answer,
                context,
            )
        )


    # ========================================================
    # Predictive Entropy Estimation
    # ========================================================

    def _estimate_entropy(
        self,
        question: str,
        context: str,
    ) -> float:
        """
        Estimate predictive entropy.

        The current EntropyEstimator is responsible
        for generating multiple candidate answers
        and measuring semantic disagreement.

        Therefore recall_rag.py only passes:

            question
            context
        """

        entropy_result = (
            self.entropy.estimate(
                question=question,
                context=context,
            )
        )

        # Current EntropyEstimator should return
        # the final entropy using the "entropy" key.

        if "entropy" in entropy_result:

            return float(
                entropy_result["entropy"]
            )

        # Compatibility fallback in case the estimator
        # returns the older key name.

        if "combined_entropy" in entropy_result:

            return float(
                entropy_result[
                    "combined_entropy"
                ]
            )

        raise KeyError(
            "EntropyEstimator result must contain "
            "'entropy' or 'combined_entropy'."
        )


    # ========================================================
    # Confidence Estimation
    # ========================================================

    def _estimate_confidence(
        self,
        recall: float,
        overlap: float,
        entropy: float,
    ) -> dict:
        """
        Delegate composite confidence calculation
        to ConfidenceEstimator.
        """

        return self.confidence.estimate(
            recall=recall,
            overlap=overlap,
            entropy=entropy,
        )


    # ========================================================
    # Abstention
    # ========================================================

    def _apply_abstention(
        self,
        confidence: float,
    ) -> tuple[str, str, str | None]:
        """
        Apply the abstention policy.

        Returns
        -------
        decision
        reason
        refusal_message
        """

        result = self.abstention.decide(
            confidence
        )

        if result.decision == "answer":

            return (
                "answer",
                "Confidence above threshold",
                None,
            )

        return (
            "abstain",
            "Confidence below threshold",
            (
                "I’m not finding enough relevant "
                "information to give you a reliable "
                "answer right now."
            ),
        )


    # ========================================================
    # Public API
    # ========================================================

    def query(
        self,
        question: str,
    ) -> RecallAwareResult:
        """
        Execute the complete Recall-Aware RAG pipeline.

        Pipeline:

        1. Retrieve
        2. Build context
        3. Generate answer
        4. Estimate recall
        5. Estimate semantic overlap
        6. Estimate predictive entropy
        7. Compute confidence
        8. Apply abstention
        9. Return result
        """

        total_start = perf_counter()

        # -----------------------------------------------------
        # Validate question
        # -----------------------------------------------------

        if not question or not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )

        question = question.strip()

        # -----------------------------------------------------
        # 1. Retrieve
        # -----------------------------------------------------

        retrieval = self._retrieve(
            question
        )

        # -----------------------------------------------------
        # 2. Build context
        # -----------------------------------------------------

        retrieval = self._build_context(
            retrieval
        )

        # -----------------------------------------------------
        # 3. Generate primary answer
        # -----------------------------------------------------

        generation = self._generate_answer(
            question,
            retrieval.context,
        )

        # -----------------------------------------------------
        # 4. Recall
        # -----------------------------------------------------

        recall = self._estimate_recall(
            retrieval.nodes
        )

        # -----------------------------------------------------
        # 5. Semantic overlap
        # -----------------------------------------------------

        overlap = self._estimate_overlap(
            generation.answer,
            retrieval.context,
        )

        # -----------------------------------------------------
        # 6. Predictive entropy
        # -----------------------------------------------------

        entropy = self._estimate_entropy(
            question=question,
            context=retrieval.context,
        )

        # -----------------------------------------------------
        # 7. Composite confidence
        # -----------------------------------------------------

        confidence_start = perf_counter()

        confidence_result = (
            self._estimate_confidence(
                recall=recall,
                overlap=overlap,
                entropy=entropy,
            )
        )

        confidence_time = (
            perf_counter()
            - confidence_start
        )

        confidence = float(
            confidence_result[
                "confidence"
            ]
        )

        # -----------------------------------------------------
        # 8. Abstention
        # -----------------------------------------------------

        (
            decision,
            reason,
            refusal,
        ) = self._apply_abstention(
            confidence
        )

        if decision == "answer":

            final_response = (
                generation.answer
            )

        else:

            final_response = refusal

        # -----------------------------------------------------
        # Total time
        # -----------------------------------------------------

        total_time = (
            perf_counter()
            - total_start
        )

        # -----------------------------------------------------
        # 9. Final result
        # -----------------------------------------------------

        return RecallAwareResult(

            question=question,

            answer=generation.answer,

            final_response=final_response,

            retrieval=retrieval,

            recall=recall,

            semantic_overlap=overlap,

            entropy=entropy,

            confidence=confidence,

            decision=decision,

            decision_reason=reason,

            generation_time=(
                generation.generation_time
            ),

            confidence_time=confidence_time,

            total_time=total_time,

            config=self.config.copy(),
        )

