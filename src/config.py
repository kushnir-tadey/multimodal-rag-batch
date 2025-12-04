from pathlib import Path

# ----------------------
# Project Paths
# ----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = DATA_DIR / "images"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------
# External Sources
# ----------------------
BASE_URL = "https://www.deeplearning.ai"
BATCH_ARCHIVE_URL = f"{BASE_URL}/the-batch/"


# ----------------------
# Model Settings
# ----------------------
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"

# ----------------------
# Retrieval Settings
# ----------------------
TOP_K_TEXT = 5
TOP_K_IMAGES = 3
MULTIMODAL_ALPHA = 0.7  # Weight for text vs image
