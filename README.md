# Improving Reliability in Retrieval-Augmented Generation through Recall-Aware Abstention

## Project Description

Large Language Models (LLMs) often generate incorrect or hallucinated responses when retrieved information is incomplete or irrelevant. This project proposes a Recall-Aware Abstention framework that combines retrieval recall and model confidence to determine when a RAG system should answer a question or abstain. The goal is to improve the reliability and trustworthiness of RAG systems by reducing hallucinations while maintaining useful answer coverage.

## Author

Patrick Keegan Gichovi Kariuki

BSc Data Science and Analytics

United States International University-Africa (USIU-Africa)

## Dataset

This project uses a custom document corpus consisting of publicly available health policies, clinical guidelines, and health information resources. The documents are indexed using FAISS to support Retrieval-Augmented Generation (RAG). The dataset serves as the knowledge base for evaluating the proposed Recall-Aware Abstention framework by assessing retrieval quality, answer reliability, and abstention decisions.

## Technologies

- Python
- Ollama
- Llama 3.1 8B
- FAISS
- LlamaIndex
- Nomic Embed Text
- Pandas
- NumPy
- Matplotlib

## Objectives

- Build a baseline RAG system
- Evaluate retrieval quality
- Measure retrieval recall
- Estimate model confidence
- Develop a Recall-Aware Abstention framework
- Compare baseline RAG against the proposed method
- Evaluate accuracy, faithfulness, hallucination rate and coverage

## Current Progress

 Literature review completed

 Local RAG environment configured

 Ollama and Llama 3.1 set up

 FAISS indexing implemented

 Embedding model configured

 Initial retrieval pipeline built


## Pending Activities
 Recall estimation

 Composite confidence scoring

 Abstention decision module

 Experimental evaluation

 Dissertation writing