"""
metadata.py

Extract reusable metadata from a DoclingDocument.

Responsibilities
----------------
- Document metadata
- Section metadata
- Corpus statistics

This module DOES NOT:
- Chunk documents
- Generate embeddings
- Create LlamaIndex Documents

Those responsibilities belong to later pipeline stages.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List
import uuid
import re


class MetadataExtractor:
    """
    Extract metadata from a DoclingDocument.
    """

    def __init__(self):
        pass

    # -------------------------------------------------------
    # Public API
    # -------------------------------------------------------

    def extract(
        self,
        document,
        source_path: str | Path,
    ) -> Dict:

        source_path = Path(source_path)

        markdown = document.export_to_markdown()

        sections = self.extract_sections(markdown)

        document_metadata = self.extract_document_metadata(
            document=document,
            markdown=markdown,
            source_path=source_path,
        )

        document_metadata["num_sections"] = len(sections)

        return {
            "document": document_metadata,
            "sections": sections,
            "statistics": self.extract_statistics(markdown),
        }

    # -------------------------------------------------------
    # Document Metadata
    # -------------------------------------------------------

    def extract_document_metadata(
        self,
        document,
        markdown: str,
        source_path: Path,
    ) -> Dict:

        return {

            "document_id": str(uuid.uuid4()),

            "document_name": source_path.name,

            "document_stem": source_path.stem,

            "source_path": str(source_path),

            "file_extension": source_path.suffix,

            "title": self.extract_title(markdown),

            "source": self.detect_source(markdown),

            "language": "English",

            "processed_at": datetime.now(
                timezone.utc
            ).isoformat(),

        }

    # -------------------------------------------------------
    # Title
    # -------------------------------------------------------

    @staticmethod
    def extract_title(markdown: str) -> str:

        for line in markdown.splitlines():

            line = line.strip()

            if line.startswith("#"):

                return line.lstrip("#").strip()

        return "Unknown Title"

    # -------------------------------------------------------
    # Source Detection
    # -------------------------------------------------------

    @staticmethod
    def detect_source(markdown: str) -> str:

        text = markdown.lower()

        sources = {

            "ministry of health": "Ministry of Health Kenya",

            "who": "World Health Organization",

            "unicef": "UNICEF",

            "usaid": "USAID",

            "family health": "Division of Family Health",

        }

        for keyword, source in sources.items():

            if re.search(
                rf"\b{re.escape(keyword)}\b",
                text,
                flags=re.IGNORECASE,
            ):
                return source

        return "Unknown"

    # -------------------------------------------------------
    # Sections
    # -------------------------------------------------------

    @staticmethod
    def extract_sections(markdown: str) -> List[Dict]:

        lines = markdown.splitlines()

        sections = []

        section_number = 0

        current = None

        heading_stack = []

        for idx, line in enumerate(lines):

            line = line.rstrip()

            if not line.startswith("#"):
                continue

            # -------------------------------------------------
            # Finish previous section
            # -------------------------------------------------

            if current:

                current["end_line"] = idx - 1

                current["line_count"] = (
                    current["end_line"]
                    - current["start_line"]
                    + 1
                )

                section_lines = lines[
                    current["start_line"]:
                    current["end_line"] + 1
                ]

                section_text = "\n".join(
                    section_lines
                ).strip()

                current["text"] = section_text

                current["word_count"] = len(
                    section_text.split()
                )

                current["character_count"] = len(
                    section_text
                )

                sections.append(current)

            # -------------------------------------------------
            # New section
            # -------------------------------------------------

            section_number += 1

            heading_marker = line.split()[0]

            heading_level = len(
                heading_marker
            )

            title = line.lstrip("#").strip()

            while (
                heading_stack
                and heading_stack[-1]["heading_level"] >= heading_level
            ):
                heading_stack.pop()

            parent_section = (
                heading_stack[-1]["section_id"]
                if heading_stack
                else None
            )

            section_path = [
                item["title"]
                for item in heading_stack
            ]

            current = {

                "section_id": str(uuid.uuid4()),

                "section_number": section_number,

                "title": title,

                "heading_level": heading_level,

                "heading_marker": heading_marker,

                "parent_section": parent_section,

                "section_path": section_path,

                "start_line": idx,

                "end_line": None,

                "line_count": 0,

                "text": "",

                "word_count": 0,

                "character_count": 0,

            }

            heading_stack.append(
                {
                    "section_id": current["section_id"],
                    "title": current["title"],
                    "heading_level": current["heading_level"],
                }
            )

        # -------------------------------------------------
        # Finish final section
        # -------------------------------------------------

        if current:

            current["end_line"] = len(lines) - 1

            current["line_count"] = (
                current["end_line"]
                - current["start_line"]
                + 1
            )

            section_lines = lines[
                current["start_line"]:
                current["end_line"] + 1
            ]

            section_text = "\n".join(
                section_lines
            ).strip()

            current["text"] = section_text

            current["word_count"] = len(
                section_text.split()
            )

            current["character_count"] = len(
                section_text
            )

            sections.append(current)

        return sections

    # -------------------------------------------------------
    # Statistics
    # -------------------------------------------------------

    @staticmethod
    def extract_statistics(
        markdown: str
    ) -> Dict:

        return {

            "characters": len(markdown),

            "words": len(markdown.split()),

            "lines": len(markdown.splitlines()),

            "headings": len(
                re.findall(
                    r"^#+ ",
                    markdown,
                    flags=re.MULTILINE,
                )
            ),

            "images": markdown.count(
                "<!-- image -->"
            ),

            "tables": len(
                re.findall(
                    r"^\|.*\|$",
                    markdown,
                    flags=re.MULTILINE,
                )
            ),

            "lists": len(
                re.findall(
                    r"^\s*[-*]",
                    markdown,
                    flags=re.MULTILINE,
                )
            ),

        }
    