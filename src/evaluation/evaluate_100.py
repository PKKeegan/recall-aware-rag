"""
evaluate_100.py

Run the full 100-question evaluation dataset through
the Recall-Aware RAG pipeline.

Responsibilities
----------------
- Load evaluation questions from:
    data/evaluation/evaluation_dataset.xlsx
- Run every question through RecallAwareRAG
- Reuse one RAG instance for the entire evaluation
- Save each successful result immediately
- Resume safely if the script is interrupted
- Preserve failed iterations separately
- Preserve raw RAG output for debugging

This module DOES NOT:
- Run baseline RAG
- Calculate aggregate evaluation metrics
- Compare Recall-Aware RAG with baseline RAG
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.evaluation.single_question import SingleQuestionEvaluator


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_dataset.xlsx"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "results"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "recall_aware_results.xlsx"
)

FAILURES_PATH = (
    RESULTS_DIR
    / "recall_aware_failures.xlsx"
)


# =========================================================
# DATASET COLUMNS
# =========================================================

REQUIRED_COLUMNS = [
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


# =========================================================
# HELPERS
# =========================================================

def make_json_safe(value: Any) -> Any:
    """
    Convert pipeline values into objects that can safely
    be stored inside an Excel cell.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    if isinstance(value, (list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    return str(value)


def load_dataset() -> pd.DataFrame:
    """
    Load and validate the evaluation dataset.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_excel(
        DATASET_PATH
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The evaluation dataset is missing "
            f"required columns: {missing_columns}"
        )

    return df


def load_existing_results() -> pd.DataFrame:
    """
    Load previously completed results.

    If the file does not exist, return an empty DataFrame.

    This allows the evaluator to resume after interruption.
    """

    if not RESULTS_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_excel(
            RESULTS_PATH
        )

    except Exception as exc:
        print(
            "\nWARNING: Could not read existing results file."
        )

        print(
            f"Reason: {type(exc).__name__}: {exc}"
        )

        print(
            "A new results file will be created."
        )

        return pd.DataFrame()


def save_results(results: list[Dict[str, Any]]) -> None:
    """
    Persist all successful results to Excel.

    This function is intentionally called after every
    successful question.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        results
    )

    dataframe.to_excel(
        RESULTS_PATH,
        index=False,
    )


def save_failures(
    failures: list[Dict[str, Any]],
) -> None:
    """
    Persist failed questions separately.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not failures:
        return

    dataframe = pd.DataFrame(
        failures
    )

    dataframe.to_excel(
        FAILURES_PATH,
        index=False,
    )


def build_result_record(
    question_row: pd.Series,
    evaluation_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Combine the original evaluation dataset information
    with the Recall-Aware RAG evaluation result.
    """

    record: Dict[str, Any] = {}

    # -----------------------------------------------------
    # Original evaluation dataset fields
    # -----------------------------------------------------

    for column in REQUIRED_COLUMNS:

        record[column] = question_row.get(
            column
        )

    # -----------------------------------------------------
    # Evaluation result fields
    # -----------------------------------------------------

    result_fields = [
        "answer",
        "decision",
        "reason",
        "recall",
        "semantic_overlap",
        "entropy",
        "confidence",
        "retrieved_chunks",
        "node_ids",
        "similarity_scores",
        "retrieval_time",
        "generation_time",
        "total_time",
    ]

    for field in result_fields:

        record[field] = make_json_safe(
            evaluation_result.get(field)
        )

    # -----------------------------------------------------
    # Preserve raw pipeline result
    # -----------------------------------------------------

    record["raw_result"] = make_json_safe(
        evaluation_result.get(
            "raw_result"
        )
    )

    # -----------------------------------------------------
    # Evaluation status
    # -----------------------------------------------------

    record["evaluation_status"] = "success"

    return record


# =========================================================
# MAIN EVALUATION
# =========================================================

def main() -> None:

    print("\n" + "=" * 80)
    print("RECALL-AWARE RAG — 100 QUESTION EVALUATION")
    print("=" * 80)

    print(
        f"\nDataset:\n{DATASET_PATH}"
    )

    print(
        f"\nResults:\n{RESULTS_PATH}"
    )

    print(
        f"\nFailures:\n{FAILURES_PATH}"
    )

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    print(
        "\nLoading evaluation dataset..."
    )

    dataset = load_dataset()

    print(
        f"Loaded {len(dataset)} questions."
    )

    # -----------------------------------------------------
    # Load previous results
    # -----------------------------------------------------

    existing_results = (
        load_existing_results()
    )

    results: list[Dict[str, Any]] = []

    failures: list[Dict[str, Any]] = []

    completed_ids: set[str] = set()

    # -----------------------------------------------------
    # Recover completed questions
    # -----------------------------------------------------

    if not existing_results.empty:

        if "question_id" in existing_results.columns:

            completed_ids = set(
                existing_results[
                    "question_id"
                ]
                .dropna()
                .astype(str)
                .tolist()
            )

            results = (
                existing_results
                .to_dict("records")
            )

            print(
                f"\nFound {len(completed_ids)} "
                "previously completed questions."
            )

            print(
                "The evaluation will resume "
                "from the remaining questions."
            )

    # -----------------------------------------------------
    # Initialize RAG ONCE
    # -----------------------------------------------------

    print(
        "\nInitializing Recall-Aware RAG..."
    )

    evaluator = SingleQuestionEvaluator()

    print(
        "Recall-Aware RAG initialized successfully."
    )

    # -----------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------

    total_questions = len(dataset)

    start_all = time.perf_counter()

    for index, question_row in dataset.iterrows():

        question_id = str(
            question_row["question_id"]
        )

        question = str(
            question_row["question"]
        )

        question_number = index + 1

        # -------------------------------------------------
        # Skip already completed questions
        # -------------------------------------------------

        if question_id in completed_ids:

            print(
                f"\n[{question_number}/{total_questions}] "
                f"{question_id} — SKIPPED "
                "(already completed)"
            )

            continue

        print("\n" + "-" * 80)

        print(
            f"[{question_number}/{total_questions}] "
            f"Evaluating {question_id}"
        )

        print(
            f"Question: {question}"
        )

        question_start = time.perf_counter()

        try:

            # ---------------------------------------------
            # Run Recall-Aware RAG
            # ---------------------------------------------

            evaluation_result = evaluator.evaluate(
                question_id=question_id,
                question=question,
            )

            # ---------------------------------------------
            # Build persistent result
            # ---------------------------------------------

            result_record = build_result_record(
                question_row,
                evaluation_result,
            )

            # ---------------------------------------------
            # Add result to memory
            # ---------------------------------------------

            results.append(
                result_record
            )

            completed_ids.add(
                question_id
            )

            # ---------------------------------------------
            # SAVE IMMEDIATELY
            #
            # This is critical.
            #
            # If question 67 crashes, questions 1–66
            # are already written to disk.
            # ---------------------------------------------

            save_results(
                results
            )

            elapsed = (
                time.perf_counter()
                - question_start
            )

            print(
                f"\n✓ {question_id} completed"
            )

            print(
                f"Decision: "
                f"{evaluation_result.get('decision')}"
            )

            print(
                f"Confidence: "
                f"{evaluation_result.get('confidence')}"
            )

            print(
                f"Retrieval Time: "
                f"{evaluation_result.get('retrieval_time')}"
            )

            print(
                f"Generation Time: "
                f"{evaluation_result.get('generation_time')}"
            )

            print(
                f"Total Time: "
                f"{evaluation_result.get('total_time')}"
            )

            print(
                f"Iteration Time: "
                f"{elapsed:.2f}s"
            )

            print(
                f"Saved to: {RESULTS_PATH}"
            )

        except Exception as exc:

            # ---------------------------------------------
            # IMPORTANT:
            #
            # A failed question must NOT terminate the
            # entire evaluation.
            # ---------------------------------------------

            elapsed = (
                time.perf_counter()
                - question_start
            )

            error_record = {

                "question_id": question_id,

                "question": question,

                "category": question_row.get(
                    "category"
                ),

                "difficulty": question_row.get(
                    "difficulty"
                ),

                "error_type": type(
                    exc
                ).__name__,

                "error_message": str(
                    exc
                ),

                "traceback": traceback.format_exc(),

                "elapsed_time": elapsed,

                "evaluation_status": "failed",
            }

            failures.append(
                error_record
            )

            # ---------------------------------------------
            # Save failure immediately too
            # ---------------------------------------------

            save_failures(
                failures
            )

            print(
                f"\n✗ {question_id} FAILED"
            )

            print(
                f"Error: "
                f"{type(exc).__name__}: {exc}"
            )

            print(
                "Failure saved. "
                "Continuing to next question..."
            )

            continue

    # -----------------------------------------------------
    # Final save
    # -----------------------------------------------------

    save_results(
        results
    )

    save_failures(
        failures
    )

    total_elapsed = (
        time.perf_counter()
        - start_all
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    successful_count = len(
        results
    )

    failed_count = len(
        failures
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"\nTotal dataset questions: "
        f"{total_questions}"
    )

    print(
        f"Successful evaluations: "
        f"{successful_count}"
    )

    print(
        f"Failed evaluations: "
        f"{failed_count}"
    )

    print(
        f"Total execution time: "
        f"{total_elapsed:.2f} seconds"
    )

    print(
        f"\nResults saved to:"
        f"\n{RESULTS_PATH}"
    )

    if failures:

        print(
            f"\nFailures saved to:"
            f"\n{FAILURES_PATH}"
        )

    else:

        print(
            "\nNo failed questions."
        )

    print(
        "\nThe evaluation data has been persisted "
        "incrementally after each successful question."
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()