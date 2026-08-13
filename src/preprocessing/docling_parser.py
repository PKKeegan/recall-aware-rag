"""
docling_parser.py

Production wrapper around Docling.

Responsibilities
----------------
- Parse supported documents (PDF, DOCX, HTML, PPTX, etc.)
- Return DoclingDocument objects
- Export Markdown
- Export JSON
- Provide simple document statistics

This module deliberately DOES NOT:
- Chunk
- Embed
- Index
- Perform retrieval
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from docling.document_converter import DocumentConverter


class DoclingParser:

    def __init__(self):

        self.converter = DocumentConverter()

    # ----------------------------------------------------------
    # Parsing
    # ----------------------------------------------------------

    def parse(self, file_path: Union[str, Path]):

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        result = self.converter.convert(file_path)

        return result.document

    # ----------------------------------------------------------
    # Export
    # ----------------------------------------------------------

    def export_markdown(
        self,
        document,
        output_path: Union[str, Path],
    ):

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            document.export_to_markdown(),
            encoding="utf-8",
        )

    def export_json(
        self,
        document,
        output_path: Union[str, Path],
    ):

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                document.export_to_dict(),
                f,
                indent=2,
                ensure_ascii=False,
            )

    # ----------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------

    def print_summary(self, document):

        print("=" * 70)

        print("DOCUMENT SUMMARY")

        print("=" * 70)

        try:
            print(f"Name      : {document.name}")
        except Exception:
            pass

        try:
            print(f"Pages     : {len(document.pages)}")
        except Exception:
            pass

        print()

        print("Available attributes:")

        print("-" * 70)

        attrs = [
            a
            for a in dir(document)
            if not a.startswith("_")
        ]

        for attr in attrs:
            print(attr)

        print("=" * 70)