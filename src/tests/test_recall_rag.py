"""
test_recall_rag.py

Integration test for Recall-Aware RAG pipeline.

Tests:
- Retrieval
- Context construction
- Generation
- Recall estimation
- Semantic overlap
- Entropy estimation
- Confidence calculation
- Abstention decision
"""

from src.recall_aware.recall_rag import RecallAwareRAG



def main():

    print("=" * 70)
    print("Recall-Aware RAG Pipeline Test")
    print("=" * 70)


    # --------------------------------------------------------
    # Initialize pipeline
    # --------------------------------------------------------

    rag = RecallAwareRAG()


    # --------------------------------------------------------
    # Test query
    # --------------------------------------------------------

    question = (
        "What are the benefits of family planning?"
    )


    print("\nQuestion:")
    print(question)



    # --------------------------------------------------------
    # Execute pipeline
    # --------------------------------------------------------

    result = rag.query(
        question
    )

    print("\n" + "=" * 60)
    print("RESULT TYPE")
    print("=" * 60)

    print(type(result))

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print(result)

    if isinstance(result, dict):

        print("\n" + "=" * 60)
        print("RESULT KEYS")
        print("=" * 60)

        for key in result.keys():
            print(f"- {key}")


    # --------------------------------------------------------
    # Retrieval Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)


    print(
        "Retrieved chunks:",
        len(result.retrieval.nodes)
    )


    print("\nNode IDs:")

    for node_id in result.retrieval.node_ids:

        print(
            "-",
            node_id
        )


    print("\nSimilarity Scores:")

    for score in result.retrieval.scores:

        print(
            round(score, 4)
        )


    print("\nContext Preview:")

    print(
        result.retrieval.context[:500]
    )



    # --------------------------------------------------------
    # Confidence Metrics
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RECALL-AWARE METRICS")
    print("=" * 70)


    print(
        "Recall:",
        round(
            result.recall,
            4
        )
    )


    print(
        "Semantic Overlap:",
        round(
            result.semantic_overlap,
            4
        )
    )


    print(
        "Entropy:",
        round(
            result.entropy,
            4
        )
    )


    print(
        "Confidence:",
        round(
            result.confidence,
            4
        )
    )



    # --------------------------------------------------------
    # Abstention Decision
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ABSTENTION DECISION")
    print("=" * 70)


    print(
        "Decision:",
        result.decision
    )


    print(
        "Reason:",
        result.decision_reason
    )



    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL RESPONSE")
    print("=" * 70)


    print(
        result.final_response
    )



    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TIMING")
    print("=" * 70)


    print(
        "Retrieval:",
        round(
            result.retrieval.retrieval_time,
            4
        ),
        "seconds"
    )


    print(
        "Generation:",
        round(
            result.generation_time,
            4
        ),
        "seconds"
    )


    print(
        "Confidence:",
        round(
            result.confidence_time,
            4
        ),
        "seconds"
    )


    print(
        "Total:",
        round(
            result.total_time,
            4
        ),
        "seconds"
    )



    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)


    for key, value in result.config.items():

        print(
            f"{key}: {value}"
        )



if __name__ == "__main__":

    main()