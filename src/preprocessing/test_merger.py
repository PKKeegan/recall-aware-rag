"""
test_merger.py

Validation tests for AdaptiveSectionMerger.

Run:

python -m src.preprocessing.test_merger
"""

from src.preprocessing.Chunking.merger import AdaptiveSectionMerger


def create_test_sections():

    return [

        # ------------------------------------------
        # Same document, same hierarchy
        # Should merge
        # ------------------------------------------

        {
            "section_id": "s1",
            "document_name": "tb_guidelines.pdf",
            "title": "TB Symptoms",
            "heading_level": 2,
            "parent_heading": "Clinical Management",
            "text": "Symptoms of tuberculosis include cough and fever.",
            "word_count": 40,
        },


        {
            "section_id": "s2",
            "document_name": "tb_guidelines.pdf",
            "title": "TB Diagnosis",
            "heading_level": 2,
            "parent_heading": "Clinical Management",
            "text": "Diagnosis involves testing and clinical assessment.",
            "word_count": 60,
        },


        # ------------------------------------------
        # Different hierarchy
        # Should NOT merge
        # ------------------------------------------

        {
            "section_id": "s3",
            "document_name": "tb_guidelines.pdf",
            "title": "TB Treatment",
            "heading_level": 2,
            "parent_heading": "Treatment Protocol",
            "text": "Patients receive medication according to guidelines.",
            "word_count": 120,
        },


        # ------------------------------------------
        # Different document
        # Should NOT merge
        # ------------------------------------------

        {
            "section_id": "s4",
            "document_name": "asrh_booklet.pdf",
            "title": "Adolescent Health",
            "heading_level": 2,
            "parent_heading": "Adolescent Development",
            "text": "Adolescence is a period of growth.",
            "word_count": 90,
        },

    ]



def main():

    merger = AdaptiveSectionMerger(
        target_words=200,
        min_words=100,
        max_words=400
    )


    sections = create_test_sections()


    results = merger.merge(
        sections
    )


    print("\n========== MERGER TEST ==========\n")


    print(
        f"Input sections: {len(sections)}"
    )


    print(
        f"Merged units: {len(results)}\n"
    )


    for i, unit in enumerate(results, start=1):

        print(
            f"--- Unit {i} ---"
        )

        print(
            "Document:",
            unit["document_name"]
        )

        print(
            "Sections:",
            unit["section_titles"]
        )

        print(
            "Section IDs:",
            unit["section_ids"]
        )

        print(
            "Word count:",
            unit["word_count"]
        )

        print(
            "Merged ID:",
            unit["merged_section_id"]
        )

        print()



if __name__ == "__main__":

    main()