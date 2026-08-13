"""
test_splitter.py

Validation test for AdaptiveSplitter.

Run:

python -m src.preprocessing.test_splitter
"""

from src.preprocessing.Chunking.splitter import AdaptiveSplitter



def create_test_merged_unit():
    """
    Simulates output from merger.py
    """

    text = """
## Understanding Adolescence

Adolescence is a stage of human development where young people experience significant physical, emotional and social changes.

During this period, adolescents develop new skills, build relationships, and begin making important decisions about their health and wellbeing.

## Physical Changes

Young people experience changes in their bodies during puberty. These changes are normal parts of development and require proper health education and support.

## Emotional Development

Adolescents may experience changes in emotions, identity, confidence, and relationships with family members and peers.

Supportive environments help adolescents navigate these transitions safely.

## Health Decisions

Access to accurate information allows adolescents to make informed decisions about reproductive health, mental wellbeing, and personal safety.

""" * 5



    return {

        "merged_section_id":
            "merged-test-001",


        "document_name":
            "ASRH_test_document.pdf",


        "section_titles":
            [
                "Understanding Adolescence",
                "Physical Changes",
                "Emotional Development",
                "Health Decisions"
            ],


        "section_ids":
            [
                "s1",
                "s2",
                "s3",
                "s4"
            ],


        "merged_section_count":
            4,


        "text":
            text
    }



def main():

    print(
        "\n========== SPLITTER TEST ==========\n"
    )


    splitter = AdaptiveSplitter(
        target_words=100,
        max_words=150,
        overlap_sentences=2
    )


    merged_units = [
        create_test_merged_unit()
    ]


    chunks = splitter.split(
        merged_units
    )



    print(
        "Generated chunks:",
        len(chunks)
    )



    for index, chunk in enumerate(chunks):

        print(
            "\n--------------------------------"
        )


        print(
            f"Chunk {index + 1}"
        )


        print(
            "Chunk ID:",
            chunk["chunk_id"]
        )


        print(
            "Words:",
            chunk["word_count"]
        )


        print(
            "Characters:",
            chunk["character_count"]
        )


        print(
            "Document:",
            chunk["document_name"]
        )


        print(
            "Merged Section ID:",
            chunk["merged_section_id"]
        )


        print(
            "Sections:",
            chunk["section_titles"]
        )


        print(
            "\nPreview:"
        )


        print(
            chunk["text"][:300]
        )



    # =====================================================
    # Validation checks
    # =====================================================


    print(
        "\n========== VALIDATION ==========\n"
    )


    # Check chunks were created

    assert len(chunks) > 1, (
        "Splitter failed: "
        "Expected multiple chunks"
    )


    # Check max size

    for chunk in chunks:

        assert chunk["word_count"] <= 150, (
            "Chunk exceeds max_words"
        )



    # Check metadata

    for chunk in chunks:

        assert chunk["document_name"] == (
            "ASRH_test_document.pdf"
        )


        assert chunk["merged_section_id"] == (
            "merged-test-001"
        )



    print(
        "✓ Multiple chunks generated"
    )

    print(
        "✓ Chunk size validation passed"
    )

    print(
        "✓ Metadata preservation passed"
    )

    print(
        "\nSplitter test completed successfully."
    )



if __name__ == "__main__":

    main()