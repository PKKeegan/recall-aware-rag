"""
test_single_question.py

Test the evaluation wrapper using one question.

Run with:

    python -m src.evaluation.test_single_question
"""

from __future__ import annotations

import traceback

from src.evaluation.single_question import SingleQuestionEvaluator


def main() -> None:

    print("\n" + "=" * 70)
    print("SINGLE QUESTION EVALUATION TEST")
    print("=" * 70)

    question_id = "Q001"
    question = "What is Kangaroo Mother Care?"

    print(f"\nQuestion ID: {question_id}")
    print(f"Question: {question}")

    try:

        # -----------------------------------------------------
        # Initialize evaluator
        # -----------------------------------------------------

        print("\nInitializing Recall-Aware RAG...")

        evaluator = SingleQuestionEvaluator()

        print("RAG initialized successfully.")

        # -----------------------------------------------------
        # Execute one question
        # -----------------------------------------------------

        print("\nRunning question through Recall-Aware RAG...")
        print("Please wait...\n")

        result = evaluator.evaluate(
            question_id=question_id,
            question=question,
        )

        # -----------------------------------------------------
        # Display result
        # -----------------------------------------------------

        print("\n" + "=" * 70)
        print("EVALUATION RESULT")
        print("=" * 70)

        for key, value in result.items():

            print(f"\n{key}:")
            print(value)

        # -----------------------------------------------------
        # Summary
        # -----------------------------------------------------

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        print(
            f"\nQuestion ID: "
            f"{result.get('question_id')}"
        )

        print(
            f"Decision: "
            f"{result.get('decision')}"
        )

        print(
            f"Recall: "
            f"{result.get('recall')}"
        )

        print(
            f"Semantic Overlap: "
            f"{result.get('semantic_overlap')}"
        )

        print(
            f"Entropy: "
            f"{result.get('entropy')}"
        )

        print(
            f"Confidence: "
            f"{result.get('confidence')}"
        )

        print(
            f"Retrieval Time: "
            f"{result.get('retrieval_time')}"
        )

        print(
            f"Generation Time: "
            f"{result.get('generation_time')}"
        )

        print(
            f"Total Time: "
            f"{result.get('total_time')}"
        )

        print("\nTest completed successfully.")

    except Exception as exc:

        print("\n" + "=" * 70)
        print("ERROR")
        print("=" * 70)

        print(
            f"\n{type(exc).__name__}: {exc}"
        )

        print("\nFull traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()