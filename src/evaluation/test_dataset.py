"""
test_dataset.py

Verify that the evaluation Excel dataset
can be loaded correctly.
"""

from src.evaluation.load_dataset import (
    load_evaluation_dataset,
)


def main():

    dataset = load_evaluation_dataset()

    print(
        f"Loaded {len(dataset)} evaluation questions."
    )

    print(
        "\nFirst question:"
    )

    first = dataset[0]

    print(
        f"ID: {first['question_id']}"
    )

    print(
        f"Question: {first['question']}"
    )

    print(
        f"Category: {first['category']}"
    )

    print(
        f"Difficulty: {first['difficulty']}"
    )

    print(
        f"Expected behavior: "
        f"{first['expected_behavior']}"
    )

    print(
        f"Support level: "
        f"{first['support_level']}"
    )

    print(
        "\nDataset columns:"
    )

    for column in first.keys():

        print(
            f"- {column}"
        )


if __name__ == "__main__":
    main()