from src.common.retrieve import retrieve
from src.recall_aware.recall import RecallEstimator

question = "What is adolescence"

nodes = retrieve(question)

recall = RecallEstimator().estimate(nodes)

print("Estimated Recall")

print(recall)