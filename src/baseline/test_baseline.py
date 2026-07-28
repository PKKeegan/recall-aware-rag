from src.baseline.baseline_rag import BaselineRAG

rag = BaselineRAG()

question = input("Question: ")

result = rag.query(question)

print("\n")
print("="*80)

print("ANSWER\n")

print(result["answer"])

print("="*80)