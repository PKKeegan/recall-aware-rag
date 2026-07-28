from src.common.retrieve import retrieve
from src.recall_aware.overlap import SemanticOverlap


question = "What is family planning?"

nodes = retrieve(question)

context = "\n\n".join(
    node.text for node in nodes
)

answer = """
Family planning allows individuals and couples to determine
the number and spacing of their children by using appropriate
contraceptive methods and reproductive health services.
"""


overlap_calculator = SemanticOverlap()

overlap = overlap_calculator.compute(
    answer=answer,
    context=context
)

print("Semantic Overlap:")
print(overlap)