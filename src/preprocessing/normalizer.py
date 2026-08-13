"""
normalizer.py

Document text normalization for Recall-Aware Abstention RAG.

Responsibilities
----------------
- Normalize Unicode characters
- Normalize whitespace
- Remove excessive blank lines
- Preserve Markdown structure

This module operates on Markdown exported by Docling and prepares
clean text for metadata extraction and chunking.
"""

from __future__ import annotations

import re
import unicodedata


class TextNormalizer:
    """
    Main normalization pipeline.

    Example
    -------
    >>> normalizer = TextNormalizer()
    >>> clean_text = normalizer.normalize(markdown)
    """

    def __init__(self):
        """
        OCR correction is intentionally disabled.

        Docling already performs OCR and layout analysis.
        OCR post-processing can be reintroduced later if
        evaluation shows it improves retrieval performance.
        """
        pass

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def normalize(self, text: str) -> str:
        """
        Run the complete normalization pipeline.

        Parameters
        ----------
        text : str
            Markdown exported by Docling.

        Returns
        -------
        str
            Cleaned markdown.
        """

        text = self.normalize_unicode(text)

        text = self.normalize_whitespace(text)

        text = self.remove_blank_lines(text)

        text = self.normalize_markdown(text)

        return text.strip()

    # ---------------------------------------------------------
    # Unicode normalization
    # ---------------------------------------------------------

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize Unicode characters.

        Converts fancy quotation marks, ligatures,
        compatibility characters, etc.
        """

        return unicodedata.normalize("NFKC", text)

    # ---------------------------------------------------------
    # Whitespace normalization
    # ---------------------------------------------------------

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Collapse unnecessary spaces while preserving
        Markdown line structure.
        """

        # Windows → Unix
        text = text.replace("\r\n", "\n")

        # Tabs → spaces
        text = text.replace("\t", " ")

        # Multiple spaces
        text = re.sub(r"[ ]{2,}", " ", text)

        # Remove trailing spaces
        text = re.sub(r"[ \t]+\n", "\n", text)

        return text

    # ---------------------------------------------------------
    # Blank lines
    # ---------------------------------------------------------

    @staticmethod
    def remove_blank_lines(text: str) -> str:
        """
        Reduce long sequences of blank lines.
        """

        return re.sub(r"\n{3,}", "\n\n", text)

    # ---------------------------------------------------------
    # Markdown cleanup
    # ---------------------------------------------------------

    @staticmethod
    def normalize_markdown(text: str) -> str:
        """
        Preserve Markdown while removing obvious noise.
        """

        # Standardize image placeholders
        text = re.sub(
            r"<!--\s*image\s*-->",
            "<!-- image -->",
            text,
        )

        # Remove trailing whitespace
        text = re.sub(
            r"[ \t]+$",
            "",
            text,
            flags=re.MULTILINE,
        )

        return text

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    @staticmethod
    def statistics(text: str) -> dict:
        """
        Return simple statistics for inspection.
        """

        return {

            "characters": len(text),

            "words": len(text.split()),

            "lines": len(text.splitlines()),

            "headings": len(
                re.findall(
                    r"^#+ ",
                    text,
                    flags=re.MULTILINE,
                )
            ),

            "images": len(
                re.findall(
                    r"<!-- image -->",
                    text,
                )
            ),

        }

    # ---------------------------------------------------------
    # File helper
    # ---------------------------------------------------------

    def normalize_file(self, input_path, output_path):
        """
        Normalize a Markdown file.

        Parameters
        ----------
        input_path : str | Path

        output_path : str | Path
        """

        from pathlib import Path

        input_path = Path(input_path)
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        text = input_path.read_text(
            encoding="utf-8"
        )

        cleaned = self.normalize(text)

        output_path.write_text(
            cleaned,
            encoding="utf-8",
        )

        return self.statistics(cleaned)