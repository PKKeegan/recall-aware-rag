"""
splitter.py

Adaptive semantic chunk splitter for Docling-based RAG pipeline.

Responsibilities
----------------
✓ Split oversized merged sections
✓ Preserve section boundaries
✓ Preserve paragraph meaning
✓ Preserve sentence boundaries
✓ Add controlled sentence overlap
✓ Maintain metadata lineage
✓ Produce retrieval-ready chunks

This module DOES NOT:
- Generate embeddings
- Create LlamaIndex Documents
- Store vectors
"""


from __future__ import annotations


import re
import uuid

from typing import Dict, List


from src.common.config import (
    TARGET_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    CHUNK_OVERLAP_SENTENCES,
)



class AdaptiveSplitter:
    """
    Adaptive semantic splitter.

    Input:
        Output from merger.py

    Output:
        Retrieval-ready chunk dictionaries
    """



    def __init__(
        self,
        target_words: int = TARGET_CHUNK_SIZE,
        max_words: int = MAX_CHUNK_SIZE,
        overlap_sentences: int = CHUNK_OVERLAP_SENTENCES,
    ):

        self.target_words = target_words
        self.max_words = max_words
        self.overlap_sentences = overlap_sentences



    # =====================================================
    # Public API
    # =====================================================

    def split(
        self,
        merged_units: List[Dict],
    ) -> List[Dict]:


        chunks = []

        chunk_index = 0



        for unit in merged_units:


            text = unit.get(
                "text",
                ""
            ).strip()



            if not text:

                continue



            if self.word_count(text) <= self.max_words:


                chunks.append(
                    self.build_chunk(
                        text,
                        unit,
                        chunk_index
                    )
                )

                chunk_index += 1



            else:


                parts = self.recursive_split(
                    text
                )


                for part in parts:


                    chunks.append(
                        self.build_chunk(
                            part,
                            unit,
                            chunk_index
                        )
                    )


                    chunk_index += 1



        return chunks



    # =====================================================
    # Recursive semantic splitting
    # =====================================================

    def recursive_split(
        self,
        text: str,
    ) -> List[str]:


        sections = self.split_markdown_sections(
            text
        )


        paragraphs = []


        for section in sections:


            paragraphs.extend(
                self.split_paragraphs(
                    section
                )
            )



        chunks = []

        current = []

        current_words = 0



        for paragraph in paragraphs:


            paragraph_words = self.word_count(
                paragraph
            )



            if (
                current_words + paragraph_words
                <= self.target_words
            ):


                current.append(
                    paragraph
                )


                current_words += paragraph_words



            else:


                if current:


                    chunks.append(
                        "\n\n".join(current)
                    )



                if paragraph_words > self.max_words:


                    chunks.extend(
                        self.split_sentences(
                            paragraph
                        )
                    )


                    current = []

                    current_words = 0



                else:


                    current = [
                        paragraph
                    ]

                    current_words = paragraph_words



        if current:


            chunks.append(
                "\n\n".join(current)
            )



        return self.add_overlap(
            chunks
        )



    # =====================================================
    # Markdown section handling
    # =====================================================

    def split_markdown_sections(
        self,
        text: str,
    ) -> List[str]:

        """
        Split markdown headings while keeping
        headings attached to their content.
        """


        sections = re.split(
            r"(?=\n#{1,6}\s)",
            text
        )


        return [

            section.strip()

            for section in sections

            if section.strip()

        ]



    # =====================================================
    # Paragraph handling
    # =====================================================

    @staticmethod
    def split_paragraphs(
        text: str,
    ) -> List[str]:


        return [

            paragraph.strip()

            for paragraph in re.split(
                r"\n\s*\n",
                text
            )

            if paragraph.strip()

        ]



    # =====================================================
    # Sentence fallback splitting
    # =====================================================

    def split_sentences(
        self,
        text: str,
    ) -> List[str]:


        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )


        chunks = []

        current = []

        count = 0



        for sentence in sentences:


            words = self.word_count(
                sentence
            )



            if (
                count + words
                <= self.target_words
            ):


                current.append(
                    sentence
                )


                count += words



            else:


                if current:

                    chunks.append(
                        " ".join(current)
                    )



                current = [
                    sentence
                ]

                count = words



        if current:


            chunks.append(
                " ".join(current)
            )



        return chunks



    # =====================================================
    # Controlled sentence overlap
    # =====================================================

    def add_overlap(
        self,
        chunks: List[str],
    ) -> List[str]:


        if len(chunks) <= 1:

            return chunks



        output = []

        previous_sentences = []



        for chunk in chunks:


            if previous_sentences:


                overlap = " ".join(
                    previous_sentences
                )


                chunk = (
                    overlap
                    +
                    "\n\n"
                    +
                    chunk
                )



            output.append(
                chunk
            )


            previous_sentences = (
                self.extract_sentences(
                    chunk
                )
                [
                    -self.overlap_sentences:
                ]
            )



        return output



    # =====================================================
    # Sentence extraction
    # =====================================================

    def extract_sentences(
        self,
        text: str,
    ) -> List[str]:


        lines = text.splitlines()


        cleaned = []



        for line in lines:


            line = line.strip()



            if not line:

                continue



            if line.startswith("#"):

                continue



            cleaned.append(
                line
            )



        plain_text = " ".join(
            cleaned
        )



        sentences = re.split(
            r"(?<=[.!?])\s+",
            plain_text
        )



        return [

            sentence.strip()

            for sentence in sentences

            if sentence.strip()

        ]



    # =====================================================
    # Chunk builder
    # =====================================================

    def build_chunk(
        self,
        text: str,
        metadata: Dict,
        index: int,
    ) -> Dict:


        return {


            "chunk_id":
                str(uuid.uuid4()),



            "chunk_index":
                index,



            "text":
                text,



            "word_count":
                self.word_count(text),



            "character_count":
                len(text),



            "document_name":
                metadata.get(
                    "document_name"
                ),



            "merged_section_id":
                metadata.get(
                    "merged_section_id"
                ),



            "section_titles":
                metadata.get(
                    "section_titles",
                    []
                ),



            "section_ids":
                metadata.get(
                    "section_ids",
                    []
                ),



            "merged_section_count":
                metadata.get(
                    "merged_section_count",
                    1
                ),

        }



    # =====================================================
    # Utility
    # =====================================================

    @staticmethod
    def word_count(
        text: str,
    ) -> int:


        return len(
            text.split()
        )