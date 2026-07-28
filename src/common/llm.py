from llama_index.llms.ollama import Ollama

from src.common.config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
)


def get_llm():

    return Ollama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )