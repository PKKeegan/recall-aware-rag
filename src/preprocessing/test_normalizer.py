from pathlib import Path

from src.preprocessing.normalizer import TextNormalizer

markdown = Path(
    "data/exports/ASRH-Booklet_Launch_Print_Signed_Final.md"
).read_text(encoding="utf-8")

normalizer = TextNormalizer()

clean = normalizer.normalize(markdown)

print(clean[:2000])

Path(
    "data/processed/ASRH-Booklet_Launch_Print_Signed_Final.cleaned.md"
).write_text(
    clean,
    encoding="utf-8"
)

print("\nNormalization complete.")