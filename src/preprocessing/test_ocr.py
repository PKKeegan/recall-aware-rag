from src.preprocessing.cleaners.ocr import OCRCleaner

cleaner = OCRCleaner()

sample = """
Healt h

publica tion

Repr oduct ive

W
ebsite

adoles-
cence
"""

print("Before\n")
print(sample)

print("\n" + "=" * 60 + "\n")

clean = cleaner.clean(sample)

print("After\n")
print(clean)