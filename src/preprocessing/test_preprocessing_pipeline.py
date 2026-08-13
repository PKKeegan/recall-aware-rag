from src.preprocessing.Chunking.merger import AdaptiveSectionMerger
from src.preprocessing.Chunking.splitter import AdaptiveSplitter


sections = [
    {
        "document_name": "tb_guidelines.pdf",
        "section_id": "s1",
        "section_title": "TB Symptoms",
        "text": "..."
    },
    {
        "document_name": "tb_guidelines.pdf",
        "section_id": "s2",
        "section_title": "TB Diagnosis",
        "text": "..."
    },
]


merger = AdaptiveSectionMerger()

merged = merger.merge(sections)


print("MERGED")
for m in merged:
    print(
        m["document_name"],
        m["section_titles"],
        m["merged_section_count"]
    )


splitter = AdaptiveSplitter()

chunks = splitter.split(merged)


print("\nCHUNKS")

for c in chunks:
    print(
        c["document_name"],
        c["section_titles"],
        c["word_count"]
    )