"""
chunker.py

Hierarchical semantic chunking for Recall-Aware Abstention RAG.

Pipeline
--------
Docling
    ↓
Markdown
    ↓
Metadata
    ↓
Sections
    ↓
Paragraphs
    ↓
Semantic Chunks
"""

from __future__ import annotations

import re
import uuid
from typing import Dict, List


class HierarchicalChunker:
    """
    Creates semantic chunks from normalized markdown.

    Chunks never cross section boundaries.
    """

    TARGET_WORDS = 400
    MIN_WORDS = 200
    MAX_WORDS = 500

    def __init__(
        self,
        target_words: int = TARGET_WORDS,
        min_words: int = MIN_WORDS,
        max_words: int = MAX_WORDS,
    ):

        self.target_words = target_words
        self.min_words = min_words
        self.max_words = max_words

    # -------------------------------------------------------
    # Public API
    # -------------------------------------------------------

    def chunk_document(
        self,
        markdown: str,
        metadata: Dict,
    ) -> List[Dict]:
        """
        Chunk an entire document.

        Parameters
        ----------
        markdown
            Normalized markdown.

        metadata
            Metadata extracted by MetadataExtractor.

        Returns
        -------
        List[Dict]
        """

        chunks = []

        document = metadata["document"]
        sections = metadata["sections"]

        for section in sections:

            text = self._extract_section_text(
                markdown,
                section,
            )

            section_chunks = self._chunk_section(
                text=text,
                document=document,
                section=section,
            )

            chunks.extend(section_chunks)

        return chunks

    # -------------------------------------------------------
    # Extract Section Text
    # -------------------------------------------------------

    @staticmethod
    def _extract_section_text(
        markdown: str,
        section: Dict,
    ) -> str:

        lines = markdown.splitlines()

        return "\n".join(
            lines[
                section["start_line"]:
                section["end_line"] + 1
            ]
        )

    # -------------------------------------------------------
    # Chunk One Section
    # -------------------------------------------------------

    def _chunk_section(
        self,
        text: str,
        document: Dict,
        section: Dict,
    ) -> List[Dict]:

        paragraphs = self._paragraph_split(text)

        chunks = []

        current = []
        current_words = 0

        for paragraph in paragraphs:

            words = len(paragraph.split())

            # Huge paragraph
            if words > self.max_words:

                if current:

                    chunks.append("\n\n".join(current))

                    current = []
                    current_words = 0

                chunks.extend(
                    self._split_large_paragraph(
                        paragraph
                    )
                )

                continue

            # Fits current chunk
            if (
                current_words + words
                <= self.target_words
            ):

                current.append(paragraph)
                current_words += words

            else:

                chunks.append("\n\n".join(current))

                current = [paragraph]
                current_words = words

        if current:

            chunks.append("\n\n".join(current))

        return self._attach_metadata(
            chunks,
            document,
            section,
        )

    # -------------------------------------------------------
    # Paragraph Split
    # -------------------------------------------------------

    @staticmethod
    def _paragraph_split(
        text: str,
    ) -> List[str]:

        paragraphs = re.split(
            r"\n\s*\n",
            text,
        )

        return [
            p.strip()
            for p in paragraphs
            if p.strip()
        ]

    # -------------------------------------------------------
    # Split Large Paragraph
    # -------------------------------------------------------

    def _split_large_paragraph(
        self,
        paragraph: str,
    ) -> List[str]:

        words = paragraph.split()

        chunks = []

        for i in range(
            0,
            len(words),
            self.target_words,
        ):

            piece = words[
                i:i + self.target_words
            ]

            chunks.append(
                " ".join(piece)
            )

        return chunks

    # -------------------------------------------------------
    # Attach Metadata
    # -------------------------------------------------------

    def _attach_metadata(
        self,
        chunks: List[str],
        document: Dict,
        section: Dict,
    ) -> List[Dict]:

        total = len(chunks)

        output = []

        for index, text in enumerate(chunks, start=1):

            output.append({

                "chunk_id": str(uuid.uuid4()),

                "chunk_index": index,

                "total_chunks": total,

                # -------------------------
                # Document
                # -------------------------

                "document_id":
                    document["document_id"],

                "document_name":
                    document["document_name"],

                "document_stem":
                    document["document_stem"],

                "source":
                    document["source"],

                "language":
                    document["language"],

                # -------------------------
                # Section
                # -------------------------

                "section_id":
                    section["section_id"],

                "section_number":
                    section["section_number"],

                "section_title":
                    section["title"],

                "heading_level":
                    section["heading_level"],

                # -------------------------
                # Chunk stats
                # -------------------------

                "word_count":
                    len(text.split()),

                "character_count":
                    len(text),

                # -------------------------
                # Content
                # -------------------------

                "text":
                    text,

            })

        return output