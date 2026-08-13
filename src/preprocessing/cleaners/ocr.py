"""
ocr.py

OCR-aware text cleaner using SymSpell.

Responsibilities
----------------
- Repair OCR split words
- Fix line-break hyphenation
- Preserve genuine word boundaries
- Validate merges using an English frequency dictionary

This implementation is intentionally generic and suitable
for research purposes.
"""

from __future__ import annotations

import re
from pathlib import Path

from symspellpy import SymSpell


class OCRCleaner:

    def __init__(self):

        dictionary = (
            Path("data")
            / "dictionaries"
            / "frequency_dictionary_en_82_765.txt"
        )

        if not dictionary.exists():
            raise FileNotFoundError(
                f"Dictionary not found: {dictionary}"
            )

        self.symspell = SymSpell(
            max_dictionary_edit_distance=2,
            prefix_length=7,
        )

        loaded = self.symspell.load_dictionary(
            str(dictionary),
            term_index=0,
            count_index=1,
        )

        if not loaded:
            raise RuntimeError(
                "Failed to load SymSpell dictionary."
            )

    # ----------------------------------------------------
    # Public API
    # ----------------------------------------------------

    def clean(self, text: str) -> str:

        text = self.fix_hyphenation(text)

        text = self.fix_newline_breaks(text)

        text = self.fix_split_words(text)

        return text

    # ----------------------------------------------------
    # adoles-
    # cence
    # ->
    # adolescence
    # ----------------------------------------------------

    @staticmethod
    def fix_hyphenation(text: str):

        return re.sub(r"-\n", "", text)

    # ----------------------------------------------------
    # W
    # ebsite
    # ->
    # Website
    # ----------------------------------------------------

    @staticmethod
    def fix_newline_breaks(text):

        return re.sub(
            r"([A-Za-z])\n([a-z])",
            r"\1\2",
            text,
        )

    # ----------------------------------------------------
    # publica tion
    # ->
    # publication
    # ----------------------------------------------------

    def fix_split_words(self, text: str):

        pattern = re.compile(
            r"\b([A-Za-z]{4,})\s([A-Za-z]{1,5})\b"
        )

        def repair(match):

            left = match.group(1)

            right = match.group(2)

            merged = left + right

            suggestions = self.symspell.lookup(
                merged,
                verbosity=0,
            )

            if suggestions:

                best = suggestions[0]

                if (
                    best.term.lower()
                    == merged.lower()
                ):
                    return best.term

            return match.group(0)

        return pattern.sub(repair, text)