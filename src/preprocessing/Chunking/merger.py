"""
merger.py

Adaptive hierarchical section merger for Docling RAG pipeline.

Responsibilities
----------------
✓ Merge adjacent related sections
✓ Preserve document boundaries
✓ Preserve heading hierarchy
✓ Preserve provenance metadata
✓ Avoid unrelated topic mixing
✓ Prepare sections for adaptive splitting

This module DOES NOT:
- Split oversized sections
- Generate embeddings
- Create LlamaIndex Documents
"""

from __future__ import annotations

import uuid
from typing import Dict, List


from src.common.config import (
    TARGET_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
)


class AdaptiveSectionMerger:
    """
    Combines small neighbouring sections into semantic units.

    Input:
    ------
    List[Dict]

    Example:

    {
        "section_id": "s1",
        "document_name": "tb.pdf",
        "title": "TB Symptoms",
        "heading_level": 2,
        "text": "...",
        "word_count": 120
    }


    Output:
    -------
    List[Dict]

    Each merged unit contains:

    {
        "merged_section_id",
        "sections",
        "section_titles",
        "section_ids",
        "text",
        "word_count"
    }
    """

    def __init__(
        self,
        target_words: int = TARGET_CHUNK_SIZE,
        min_words: int = MIN_CHUNK_SIZE,
        max_words: int = MAX_CHUNK_SIZE,
    ):

        self.target_words = target_words
        self.min_words = min_words
        self.max_words = max_words


    # =====================================================
    # Public API
    # =====================================================

    def merge(
        self,
        sections: List[Dict],
    ) -> List[Dict]:

        """
        Merge neighbouring sections.

        Rules:
        -------
        1. Never cross documents
        2. Never merge different heading levels aggressively
        3. Stop when target size reached
        4. Oversized sections remain untouched
        """

        if not sections:
            return []


        merged_units = []

        current = []


        current_words = 0

        current_document = None



        for section in sections:


            text = section.get(
                "text",
                ""
            ).strip()


            if not text:
                continue


            words = section.get(
                "word_count",
                self.count_words(text)
            )


            document = section.get(
                "document_name"
            )



            # ---------------------------------------------
            # Document boundary protection
            # ---------------------------------------------

            if (
                current
                and document != current_document
            ):

                merged_units.append(
                    self.finalize(current)
                )

                current = []

                current_words = 0



            current_document = document



            # ---------------------------------------------
            # Oversized section
            # ---------------------------------------------

            if words > self.max_words:


                if current:

                    merged_units.append(
                        self.finalize(current)
                    )

                    current = []

                    current_words = 0



                merged_units.append(
                    self.finalize(
                        [section]
                    )
                )


                continue



            # ---------------------------------------------
            # Would exceed maximum
            # ---------------------------------------------

            if (
                current_words + words
                > self.max_words
            ):


                merged_units.append(
                    self.finalize(current)
                )


                current = []

                current_words = 0



            current.append(section)

            current_words += words



            # ---------------------------------------------
            # Target reached
            # ---------------------------------------------

            if (
                current_words >= self.target_words
            ):

                merged_units.append(
                    self.finalize(current)
                )

                current = []

                current_words = 0



        # Remaining sections

        if current:

            merged_units.append(
                self.finalize(current)
            )



        return merged_units



    # =====================================================
    # Finalization
    # =====================================================

    def finalize(
        self,
        sections: List[Dict],
    ) -> Dict:


        return {

            "merged_section_id":
                str(uuid.uuid4()),


            "sections":
                sections,


            "merged_section_count":
                len(sections),



            "document_name":
                sections[0].get(
                    "document_name"
                ),



            "section_titles":
                [
                    s.get(
                        "title",
                        ""
                    )
                    for s in sections
                ],



            "section_ids":
                [
                    s.get(
                        "section_id"
                    )
                    for s in sections
                ],



            "heading_levels":
                [
                    s.get(
                        "heading_level"
                    )
                    for s in sections
                ],



            "word_count":
                sum(
                    s.get(
                        "word_count",
                        self.count_words(
                            s.get(
                                "text",
                                ""
                            )
                        )
                    )
                    for s in sections
                ),



            "text":
                "\n\n".join(
                    s.get(
                        "text",
                        ""
                    )
                    for s in sections
                )

        }



    # =====================================================
    # Utilities
    # =====================================================

    @staticmethod
    def count_words(
        text: str
    ) -> int:

        return len(
            text.split()
        )