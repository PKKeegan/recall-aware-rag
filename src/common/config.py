from pathlib import Path


# ==========================================================
# Project Directories
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

EXPORT_DIR = DATA_DIR / "exports"

VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"



# ==========================================================
# Models
# ==========================================================

LLM_MODEL = "llama3.1:8b"

EMBED_MODEL = "nomic-embed-text"


# LLM generation settings

LLM_TEMPERATURE = 0.3



# ==========================================================
# Document Processing
# ==========================================================

PRESERVE_MARKDOWN = True

ENABLE_OCR = True



# ==========================================================
# Section Merger Configuration
# ==========================================================

MERGER_TARGET_WORDS = 400

MERGER_MIN_WORDS = 200

MERGER_MAX_WORDS = 500



# ==========================================================
# Adaptive Chunking Configuration
# ==========================================================

TARGET_CHUNK_SIZE = 400

MAX_CHUNK_SIZE = 500

MIN_CHUNK_SIZE = 200


# Sentence overlap
# Number of previous sentences carried forward

CHUNK_OVERLAP_SENTENCES = 2



# ==========================================================
# Retrieval
# ==========================================================

TOP_K = 5



# ==========================================================
# Recall-Aware Abstention
# ==========================================================

MIN_RECALL = 0.45


CONFIDENCE_THRESHOLD = 0.65



# ==========================================================
# Confidence Formula Weights
# ==========================================================

RECALL_WEIGHT = 0.40

OVERLAP_WEIGHT = 0.40

ENTROPY_WEIGHT = 0.20




# ==========================================================
# Evaluation
# ==========================================================

RANDOM_SEED = 42

# ============================================================
# Recall Estimation Weights
# ============================================================

RECALL_SIMILARITY_WEIGHT = 0.40
RECALL_COVERAGE_WEIGHT = 0.30
RECALL_RANK_WEIGHT = 0.30

RELEVANCE_THRESHOLD = 0.60

# ==========================================================
# Entropy Estimation Weights
# ==========================================================

ENTROPY_RETRIEVAL_WEIGHT = 0.40

ENTROPY_EVIDENCE_WEIGHT = 0.40

ENTROPY_COVERAGE_WEIGHT = 0.20