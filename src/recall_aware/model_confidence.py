import re

from src.common.llm import get_llm


class ModelConfidence:

    def __init__(self):
        self.llm = get_llm()

    def estimate(
        self,
        question: str,
        answer: str,
        context: str
    ) -> float:

        prompt = f"""
You are evaluating whether an answer is adequately supported
by the provided context.

Question:
{question}

Context:
{context}

Answer:
{answer}

Evaluate how strongly the context supports the answer.

Consider:

1. Is the answer directly supported by the context?
2. Does the answer contain information not present in the context?
3. Does the answer contradict the context?
4. How confident are you that the answer is reliable based ONLY
   on the provided context?

Return ONLY one decimal number between 0 and 1.

Examples:
0.0
0.2
0.5
0.7
0.9
1.0

Do not provide an explanation.
"""

        response = self.llm.complete(prompt)

        raw_output = response.text.strip()

        print("\nDEBUG MODEL OUTPUT:")
        print(repr(raw_output))

        # Find the first number between 0 and 1
        match = re.search(
            r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b",
            raw_output
        )

        if match is None:
            print("WARNING: Could not extract confidence score.")
            return 0.0

        score = float(match.group())

        return max(0.0, min(1.0, score))