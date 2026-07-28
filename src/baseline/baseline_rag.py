from src.common.retrieve import retrieve
from src.common.llm import get_llm

from src.baseline.prompt import BASELINE_PROMPT


class BaselineRAG:

    def __init__(self):

        self.llm = get_llm()

    def query(self, question):

        nodes = retrieve(question)

        context = "\n\n".join(
            node.text for node in nodes
        )

        prompt = BASELINE_PROMPT.format(
            context=context,
            query=question
        )

        response = self.llm.complete(prompt)

        return {
            "question": question,
            "answer": response.text,
            "context": context,
            "nodes": nodes
        }