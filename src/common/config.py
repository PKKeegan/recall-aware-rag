from pathlib import Path

# ==========================================================
# Project Directories
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# ==========================================================
# Models
# ==========================================================

LLM_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"

# LLM generation settings
LLM_TEMPERATURE = 0.0

# ==========================================================
# Chunking
# ==========================================================

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128

# ==========================================================
# Retrieval
# ==========================================================

TOP_K = 5

# Early abstention
MIN_RECALL = 0.45

# Composite confidence
CONFIDENCE_THRESHOLD = 0.75

# Weights
RECALL_WEIGHT = 0.40
OVERLAP_WEIGHT = 0.40
CONFIDENCE_WEIGHT = 0.20