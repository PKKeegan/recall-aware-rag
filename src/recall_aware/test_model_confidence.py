from src.baseline.baseline_rag import BaselineRAG
from src.recall_aware.model_confidence import ModelConfidence


question = "What is family planning?"


# Run baseline RAG
rag = BaselineRAG()

result = rag.query(question)

# answer = result["answer"]
answer = """
it is the getting children
"""

context = result["context"]


# Estimate model confidence
confidence_estimator = ModelConfidence()

confidence = confidence_estimator.estimate(
    question=question,
    answer=answer,
    context=context
)


print("=" * 70)
print("QUESTION")
print("=" * 70)

print(question)

print("\n" + "=" * 70)
print("ANSWER")
print("=" * 70)

print(answer)

print("\n" + "=" * 70)
print("MODEL CONFIDENCE")
print("=" * 70)

print(f"{confidence:.4f}")