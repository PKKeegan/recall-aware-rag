from llama_index.core import PromptTemplate

BASELINE_PROMPT = PromptTemplate(
"""
You are an expert assistant.

Answer ONLY using the retrieved context.

If the answer is not contained in the context, reply exactly:

I don't know.

-----------------------
Context
-----------------------

{context}

-----------------------
Question
-----------------------

{query}

-----------------------
Answer
-----------------------
"""
)