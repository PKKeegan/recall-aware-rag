from pathlib import Path
import json

from src.preprocessing.docling_parser import DoclingParser
from src.preprocessing.normalizer import TextNormalizer
from src.preprocessing.metadata import MetadataExtractor


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

print("✓ Document parsed successfully")

# ----------------------------------------------------
# Normalize markdown
# ----------------------------------------------------

print("\n" + "=" * 70)
print("NORMALIZING MARKDOWN")
print("=" * 70)

markdown = document.export_to_markdown()

normalizer = TextNormalizer()

clean_markdown = normalizer.normalize(markdown)

print("✓ Markdown normalized")

# ----------------------------------------------------
# Extract metadata
# ----------------------------------------------------

print("\n" + "=" * 70)
print("EXTRACTING METADATA")
print("=" * 70)

extractor = MetadataExtractor()

metadata = extractor.extract(
    document=document,
    source_path=PDF_PATH,
)

print("✓ Metadata extracted")

# ----------------------------------------------------
# Document metadata
# ----------------------------------------------------

print("\n" + "=" * 70)
print("DOCUMENT METADATA")
print("=" * 70)

print(
    json.dumps(
        metadata["document"],
        indent=4,
    )
)

# ----------------------------------------------------
# Statistics
# ----------------------------------------------------

print("\n" + "=" * 70)
print("DOCUMENT STATISTICS")
print("=" * 70)

print(
    json.dumps(
        metadata["statistics"],
        indent=4,
    )
)

# ----------------------------------------------------
# Sections
# ----------------------------------------------------

print("\n" + "=" * 70)
print("FIRST FIVE SECTIONS")
print("=" * 70)

for section in metadata["sections"][:5]:

    print()

    print(
        json.dumps(
            section,
            indent=4,
        )
    )

# ----------------------------------------------------
# Summary
# ----------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Document : {metadata['document']['document_name']}")
print(f"Title    : {metadata['document']['title']}")
print(f"Sections : {metadata['document']['num_sections']}")
print(f"Words    : {metadata['statistics']['words']}")
print(f"Images   : {metadata['statistics']['images']}")
print(f"Tables   : {metadata['statistics']['tables']}")

print("\n✓ Metadata pipeline completed successfully.")