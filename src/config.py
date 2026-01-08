from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

# ----------------------
# Project Paths
# ----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = DATA_DIR / "images"
INDEX_DIR = DATA_DIR / "faiss_index"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------
# External Sources
# ----------------------
BASE_URL = "https://www.deeplearning.ai"
BATCH_ARCHIVE_URL = f"{BASE_URL}/the-batch/"

# ----------------------
# Index Paths
# ----------------------
TEXT_INDEX_PATH = INDEX_DIR / "text.index"
TEXT_METADATA_PATH = INDEX_DIR / "text_metadata.json"

IMAGE_INDEX_PATH = INDEX_DIR / "image.index"
IMAGE_METADATA_PATH = INDEX_DIR / "image_metadata.json"

# ----------------------
# API Keys / Models (from .env)
# ----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "text-embedding-3-small")
IMAGE_EMBEDDING_MODEL = os.getenv("IMAGE_EMBEDDING_MODEL", "ViT-B-16-SigLIP")
IMAGE_EMBEDDING_PRETRAINED = os.getenv("IMAGE_EMBEDDING_PRETRAINED", "webli")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

