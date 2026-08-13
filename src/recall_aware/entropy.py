"""
entropy.py

Predictive entropy estimator for Recall-Aware Abstention RAG.

Responsibilities
----------------
✓ Generate multiple candidate answers
✓ Embed candidate answers
✓ Measure semantic similarity between answers
✓ Group semantically similar answers
✓ Construct a probability distribution
✓ Calculate normalized Shannon entropy

This module DOES NOT:
- Retrieve documents
- Compute retrieval recall
- Compute final semantic overlap
- Compute final confidence
- Apply abstention

Those responsibilities belong to other modules.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.recall_aware.overlap import SemanticOverlap


class EntropyEstimator:
    """
    Estimate predictive uncertainty using
    multiple candidate LLM responses.

    Higher entropy:
        Greater disagreement between candidate answers.

    Lower entropy:
        Greater agreement between candidate answers.
    """

    def __init__(
        self,
        llm,
        num_samples: int = 5,
        similarity_threshold: float = 0.80,
    ):
        """
        Parameters
        ----------
        llm:
            LlamaIndex-compatible LLM.

        num_samples:
            Number of candidate answers generated
            for entropy estimation.

        similarity_threshold:
            Minimum semantic similarity required
            for two answers to be considered part
            of the same semantic cluster.
        """

        if llm is None:
            raise ValueError(
                "EntropyEstimator requires an LLM."
            )

        self.llm = llm

        self.num_samples = max(
            2,
            int(num_samples),
        )

        self.similarity_threshold = float(
            np.clip(
                similarity_threshold,
                0.0,
                1.0,
            )
        )

        self.overlap = SemanticOverlap()

    # =========================================================
    # Public API
    # =========================================================

    def estimate(
        self,
        question: str,
        context: str,
    ) -> Dict[str, float]:
        """
        Estimate predictive entropy.

        Pipeline:

        Question + Context
                |
                v
        Generate N candidate answers
                |
                v
        Embed candidate answers
                |
                v
        Compare semantic similarity
                |
                v
        Group similar answers
                |
                v
        Create probability distribution
                |
                v
        Calculate Shannon entropy

        Parameters
        ----------
        question : str
            Original user question.

        context : str
            Retrieved evidence.

        Returns
        -------
        Dict[str, float]
            Entropy result and supporting statistics.
        """

        if not question or not question.strip():

            return {
                "entropy": 1.0,
                "num_samples": 0,
                "num_clusters": 0,
            }

        if not context or not context.strip():

            return {
                "entropy": 1.0,
                "num_samples": 0,
                "num_clusters": 0,
            }

        # -----------------------------------------------------
        # Generate candidate answers
        # -----------------------------------------------------

        answers = self._generate_answers(
            question=question,
            context=context,
        )

        # -----------------------------------------------------
        # Insufficient samples
        # -----------------------------------------------------

        if len(answers) < 2:

            return {
                "entropy": 1.0,
                "num_samples": len(answers),
                "num_clusters": 0,
            }

        # -----------------------------------------------------
        # Generate answer embeddings
        # -----------------------------------------------------

        embeddings = self._embed_answers(
            answers
        )

        if len(embeddings) < 2:

            return {
                "entropy": 1.0,
                "num_samples": len(answers),
                "num_clusters": 0,
            }

        # -----------------------------------------------------
        # Cluster semantically similar answers
        # -----------------------------------------------------

        clusters = self._cluster_answers(
            embeddings
        )

        # -----------------------------------------------------
        # Convert clusters to probabilities
        # -----------------------------------------------------

        probabilities = self._cluster_probabilities(
            clusters=clusters,
            num_samples=len(answers),
        )

        # -----------------------------------------------------
        # Shannon entropy
        # -----------------------------------------------------

        entropy = self._shannon_entropy(
            probabilities
        )

        return {
            "entropy": entropy,
            "num_samples": len(answers),
            "num_clusters": len(clusters),
        }

    # =========================================================
    # Candidate Answer Generation
    # =========================================================

    def _generate_answers(
        self,
        question: str,
        context: str,
    ) -> List[str]:
        """
        Generate multiple candidate answers from
        the same question and retrieval context.

        Each generation is performed independently.
        """

        prompt = self._build_prompt(
            question=question,
            context=context,
        )

        answers: List[str] = []

        for _ in range(
            self.num_samples
        ):

            try:

                response = self.llm.complete(
                    prompt
                )

                answer = response.text.strip()

                if answer:

                    answers.append(
                        answer
                    )

            except Exception as exc:

                print(
                    "Entropy sampling warning:",
                    exc,
                )
        # -----------------------------------------------------
        # DEBUG: Show generated entropy samples
        # -----------------------------------------------------

        for i, answer in enumerate(answers, start=1):
            print(f"\nEntropy Sample {i}:")
            print(answer)

        return answers

    # =========================================================
    # Prompt Construction
    # =========================================================

    @staticmethod
    def _build_prompt(
        question: str,
        context: str,
    ) -> str:
        """
        Construct the prompt used for entropy sampling.

        The prompt deliberately matches the main
        Recall-Aware RAG answering instructions.
        """

        return f"""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer cannot be determined from the context,
say that the information is not available.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    # =========================================================
    # Answer Embeddings
    # =========================================================

    def _embed_answers(
        self,
        answers: List[str],
    ) -> List[np.ndarray]:
        """
        Generate embeddings for all candidate answers.
        """

        embeddings: List[np.ndarray] = []

        for answer in answers:

            if not answer.strip():
                continue

            try:

                embedding = self.overlap.embed(
                    answer
                )

                embeddings.append(
                    embedding
                )

            except Exception as exc:

                print(
                    "Entropy embedding warning:",
                    exc,
                )

        return embeddings

    # =========================================================
    # Semantic Clustering
    # =========================================================

    def _cluster_answers(
        self,
        embeddings: List[np.ndarray],
    ) -> List[List[int]]:
        """
        Group semantically similar answers.

        A greedy clustering approach is used.

        Each answer is compared with the first
        answer in an existing cluster.

        If similarity is above the configured
        threshold, the answer joins that cluster.

        Otherwise, a new cluster is created.
        """

        clusters: List[List[int]] = []

        for index, embedding in enumerate(
            embeddings
        ):

            assigned = False

            for cluster in clusters:

                representative_index = cluster[0]

                representative = embeddings[
                    representative_index
                ]

                similarity = (
                    self.overlap.similarity(
                        embedding,
                        representative,
                    )
                )

                if (
                    similarity
                    >= self.similarity_threshold
                ):

                    cluster.append(
                        index
                    )

                    assigned = True

                    break

            if not assigned:

                clusters.append(
                    [index]
                )

        return clusters

    # =========================================================
    # Probability Distribution
    # =========================================================

    @staticmethod
    def _cluster_probabilities(
        clusters: List[List[int]],
        num_samples: int,
    ) -> np.ndarray:
        """
        Convert cluster sizes into probabilities.

        Example:

        5 answers

        Cluster sizes:

        [4, 1]

        Probability distribution:

        [0.8, 0.2]
        """

        if num_samples <= 0:

            return np.asarray(
                [],
                dtype=float,
            )

        probabilities = np.asarray(
            [
                len(cluster) / num_samples
                for cluster in clusters
            ],
            dtype=float,
        )

        return probabilities

    # =========================================================
    # Shannon Entropy
    # =========================================================

    @staticmethod
    def _shannon_entropy(
        probabilities: np.ndarray,
    ) -> float:
        """
        Calculate normalized Shannon entropy.

        Entropy is normalized to [0,1].

        0:
            All candidate answers belong
            to one semantic cluster.

        1:
            Candidate answers are maximally
            distributed across clusters.
        """

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        probabilities = probabilities[
            probabilities > 0
        ]

        if len(probabilities) <= 1:

            return 0.0

        total = probabilities.sum()

        if total <= 0:

            return 1.0

        probabilities = (
            probabilities / total
        )

        entropy = -np.sum(
            probabilities
            * np.log(probabilities)
        )

        maximum_entropy = np.log(
            len(probabilities)
        )

        if maximum_entropy <= 0:

            return 0.0

        normalized_entropy = (
            entropy
            / maximum_entropy
        )

        return float(
            np.clip(
                normalized_entropy,
                0.0,
                1.0,
            )
        )