from pathlib import Path
from statistics import mean

from src.preprocessing.docling_parser import DoclingParser
from src.preprocessing.normalizer import TextNormalizer
from src.preprocessing.metadata import MetadataExtractor
from src.preprocessing.chunker import HierarchicalChunker


# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

PDF_PATH = Path(
    "data/raw/ASRH-Booklet_Launch_Print_Signed_Final.pdf"
)

# ----------------------------------------------------
# Parse document
# ----------------------------------------------------

print("=" * 70)
print("PARSING DOCUMENT")
print("=" * 70)

parser = DoclingParser()

document = parser.parse(PDF_PATH)

print("✓ Document parsed")

# ----------------------------------------------------
# Normalize markdown
# ----------------------------------------------------

print("\n" + "=" * 70)
print("NORMALIZING")
print("=" * 70)

markdown = document.export_to_markdown()

normalizer = TextNormalizer()

markdown = normalizer.normalize(markdown)

print("✓ Markdown normalized")

# ----------------------------------------------------
# Metadata
# ----------------------------------------------------

print("\n" + "=" * 70)
print("EXTRACTING METADATA")
print("=" * 70)

metadata = MetadataExtractor().extract(
    document=document,
    source_path=PDF_PATH,
)

print("✓ Metadata extracted")

# ----------------------------------------------------
# Chunking
# ----------------------------------------------------

print("\n" + "=" * 70)
print("CHUNKING DOCUMENT")
print("=" * 70)

chunker = HierarchicalChunker()

chunks = chunker.chunk_document(
    markdown,
    metadata,
)

print(f"✓ Generated {len(chunks)} chunks")

# ----------------------------------------------------
# Chunk Statistics
# ----------------------------------------------------

word_counts = [
    chunk["word_count"]
    for chunk in chunks
]

print("\n" + "=" * 70)
print("CHUNK STATISTICS")
print("=" * 70)

print(f"Total Chunks : {len(chunks)}")
print(f"Average Words: {mean(word_counts):.1f}")
print(f"Minimum Words: {min(word_counts)}")
print(f"Maximum Words: {max(word_counts)}")

# ----------------------------------------------------
# Sample Chunk
# ----------------------------------------------------

print("\n" + "=" * 70)
print("FIRST CHUNK")
print("=" * 70)

first = chunks[0]

for key in [
    "chunk_id",
    "chunk_index",
    "document_name",
    "section_title",
    "word_count",
    "character_count",
]:

    print(f"{key}: {first[key]}")

print("\nTEXT PREVIEW\n")

print(first["text"][:1000])

# ----------------------------------------------------
# Metadata Validation
# ----------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATING CHUNKS")
print("=" * 70)

required_fields = [

    "chunk_id",
    "chunk_index",
    "total_chunks",

    "document_id",
    "document_name",
    "document_stem",

    "section_id",
    "section_number",
    "section_title",

    "word_count",
    "character_count",

    "text",
]

errors = 0

for i, chunk in enumerate(chunks):

    for field in required_fields:

        if field not in chunk:

            print(
                f"Chunk {i+1} missing '{field}'"
            )

            errors += 1

if errors == 0:

    print("✓ All chunks contain required metadata.")

else:

    print(f"{errors} metadata errors found.")

# ----------------------------------------------------
# Section Summary
# ----------------------------------------------------

print("\n" + "=" * 70)
print("SECTION SUMMARY")
print("=" * 70)

seen = set()

for chunk in chunks:

    title = chunk["section_title"]

    if title not in seen:

        seen.add(title)

        print(title)

print("\n✓ Chunking pipeline completed successfully.")