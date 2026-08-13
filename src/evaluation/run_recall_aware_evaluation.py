"""
run_recall_aware_evaluation.py

Evaluation runner for Recall-Aware Abstention RAG.

Loads an XLSX evaluation dataset, runs every question through
RecallAwareRAG, records the generated response and pipeline
metrics, and saves the results to an XLSX file.

This script does NOT modify the RAG pipeline.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.recall_aware.recall_rag import RecallAwareRAG


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = Path(
    "data/evaluation/evaluation_dataset.xlsx"
)

OUTPUT_PATH = Path(
    "data/evaluation/recall_aware_results.xlsx"
)


# ============================================================
# Helper functions
# ============================================================

def safe_get(
    result: Any,
    key: str,
    default: Any = None,
):
    """
    Safely retrieve a value from the pipeline result.

    Supports dictionaries and normal Python objects.
    """

    if isinstance(result, dict):
        return result.get(key, default)

    return getattr(
        result,
        key,
        default,
    )


def extract_metric(
    result: Any,
    *keys: str,
    default: Any = None,
):
    """
    Try multiple possible metric names.

    This makes the evaluator tolerant of small naming
    differences in the current RecallAwareRAG result.
    """

    for key in keys:

        value = safe_get(
            result,
            key,
            None,
        )

        if value is not None:
            return value

    return default


# ============================================================
# Main evaluation
# ============================================================

def main():

    print("=" * 70)
    print("RECALL-AWARE RAG EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Evaluation dataset not found:\n"
            f"{DATASET_PATH}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_excel(
        DATASET_PATH
    )

    print(
        f"\nLoaded {len(df)} evaluation questions."
    )

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "question_id",
        "question",
        "category",
        "difficulty",
        "gold_answer",
        "expected_behavior",
        "evidence_source",
        "evidence_location",
        "evidence_summary",
        "support_level",
        "reason_for_abstention",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Dataset is missing required columns:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_columns
            )
        )

    # --------------------------------------------------------
    # Initialize RAG
    # --------------------------------------------------------

    print("\nInitializing Recall-Aware RAG...")

    rag = RecallAwareRAG()

    print("RAG initialized successfully.")

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Evaluate each question
    # --------------------------------------------------------

    total_questions = len(df)

    for index, row in df.iterrows():

        question_id = row["question_id"]
        question = row["question"]

        print(
            f"\n[{index + 1}/{total_questions}] "
            f"{question_id}"
        )

        print(
            f"Question: {question}"
        )

        # ----------------------------------------------------
        # Execute pipeline
        # ----------------------------------------------------

        start_time = time.perf_counter()

        try:

            result = rag.query(
                question
            )

            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            # ------------------------------------------------
            # Extract result fields
            # ------------------------------------------------

            answer = extract_metric(
                result,
                "answer",
                "response",
                "generated_answer",
                default="",
            )

            decision = extract_metric(
                result,
                "decision",
                "action",
                default="",
            )

            confidence = extract_metric(
                result,
                "confidence",
                "confidence_score",
                default=None,
            )

            recall = extract_metric(
                result,
                "recall",
                "recall_score",
                default=None,
            )

            semantic_overlap = extract_metric(
                result,
                "semantic_overlap",
                "overlap",
                "overlap_score",
                default=None,
            )

            entropy = extract_metric(
                result,
                "entropy",
                "entropy_score",
                default=None,
            )

            retrieval_time = extract_metric(
                result,
                "retrieval_time",
                "retrieval_seconds",
                default=None,
            )

            generation_time = extract_metric(
                result,
                "generation_time",
                "generation_seconds",
                default=None,
            )

            retrieved_chunks = extract_metric(
                result,
                "retrieved_chunks",
                "num_retrieved",
                "retrieval_count",
                default=None,
            )

            # ------------------------------------------------
            # Determine whether system behavior matched
            # expected behavior
            # ------------------------------------------------

            expected_behavior = str(
                row["expected_behavior"]
            ).strip().lower()

            actual_decision = str(
                decision
            ).strip().lower()

            if expected_behavior == "answer":

                behavior_correct = (
                    actual_decision == "answer"
                )

            elif expected_behavior == "abstain":

                behavior_correct = (
                    actual_decision == "abstain"
                )

            else:

                behavior_correct = None

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            evaluation_row = {

                # Dataset information
                "question_id": question_id,

                "question": question,

                "category": row[
                    "category"
                ],

                "difficulty": row[
                    "difficulty"
                ],

                "gold_answer": row[
                    "gold_answer"
                ],

                "expected_behavior": row[
                    "expected_behavior"
                ],

                "support_level": row[
                    "support_level"
                ],

                "evidence_source": row[
                    "evidence_source"
                ],

                "evidence_location": row[
                    "evidence_location"
                ],

                "evidence_summary": row[
                    "evidence_summary"
                ],

                "reason_for_abstention": row[
                    "reason_for_abstention"
                ],

                # RAG output
                "generated_answer": answer,

                "actual_decision": decision,

                # Core evaluation metrics
                "confidence": confidence,

                "recall": recall,

                "semantic_overlap": semantic_overlap,

                "entropy": entropy,

                # Retrieval information
                "retrieved_chunks": retrieved_chunks,

                # Timing
                "retrieval_time_seconds": retrieval_time,

                "generation_time_seconds": generation_time,

                "total_time_seconds": elapsed_time,

                # Behavioral evaluation
                "behavior_correct": behavior_correct,

                # Error information
                "error": None,
            }

        except Exception as exc:

            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            print(
                f"ERROR: {exc}"
            )

            evaluation_row = {

                "question_id": question_id,

                "question": question,

                "category": row[
                    "category"
                ],

                "difficulty": row[
                    "difficulty"
                ],

                "gold_answer": row[
                    "gold_answer"
                ],

                "expected_behavior": row[
                    "expected_behavior"
                ],

                "support_level": row[
                    "support_level"
                ],

                "evidence_source": row[
                    "evidence_source"
                ],

                "evidence_location": row[
                    "evidence_location"
                ],

                "evidence_summary": row[
                    "evidence_summary"
                ],

                "reason_for_abstention": row[
                    "reason_for_abstention"
                ],

                "generated_answer": None,

                "actual_decision": None,

                "confidence": None,

                "recall": None,

                "semantic_overlap": None,

                "entropy": None,

                "retrieved_chunks": None,

                "retrieval_time_seconds": None,

                "generation_time_seconds": None,

                "total_time_seconds": elapsed_time,

                "behavior_correct": False,

                "error": str(exc),
            }

        results.append(
            evaluation_row
        )

        print(
            f"Decision: "
            f"{evaluation_row['actual_decision']}"
        )

        print(
            f"Confidence: "
            f"{evaluation_row['confidence']}"
        )

        print(
            f"Total time: "
            f"{elapsed_time:.3f}s"
        )

    # --------------------------------------------------------
    # Create results DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_excel(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    successful = (
        results_df["error"]
        .isna()
        .sum()
    )

    failed = (
        results_df["error"]
        .notna()
        .sum()
    )

    behavior_correct = (
        results_df[
            "behavior_correct"
        ]
        .fillna(False)
        .sum()
    )

    print("\n")
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Questions evaluated: "
        f"{len(results_df)}"
    )

    print(
        f"Successful: "
        f"{successful}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    print(
        f"Correct expected behavior: "
        f"{behavior_correct}"
    )

    if len(results_df) > 0:

        behavior_accuracy = (
            behavior_correct
            / len(results_df)
        )

        print(
            f"Behavior accuracy: "
            f"{behavior_accuracy:.4f}"
        )

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_PATH}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()