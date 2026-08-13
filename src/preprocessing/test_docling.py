"""
Test Docling parser.

Run:

python -m src.preprocessing.test_docling
"""

from pathlib import Path

from src.preprocessing.docling_parser import DoclingParser


RAW_DATA = Path("data/raw")

EXPORT_DIR = Path("data/exports")

pdf_files = sorted(RAW_DATA.glob("*.pdf"))

if len(pdf_files) == 0:
    raise FileNotFoundError(
        "No PDF files found inside data/raw/"
    )

pdf = pdf_files[0]

print()

print("=" * 70)

print(f"Parsing document: {pdf.name}")

print("=" * 70)

parser = DoclingParser()

document = parser.parse(pdf)

print()

print("SUCCESS")

print(type(document))

print()

parser.print_summary(document)

markdown_file = EXPORT_DIR / f"{pdf.stem}.md"

json_file = EXPORT_DIR / f"{pdf.stem}.json"

parser.export_markdown(
    document,
    markdown_file,
)

parser.export_json(
    document,
    json_file,
)

print()

print("=" * 70)

print("FILES GENERATED")

print("=" * 70)

print(markdown_file)

print(json_file)

print()

print("=" * 70)

print("FIRST 1500 CHARACTERS OF MARKDOWN")

print("=" * 70)

print(document.export_to_markdown()[:1500])

print()

print("=" * 70)

print("DONE")

print("=" * 70)