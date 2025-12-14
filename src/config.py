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
# API Keys / Models (from .env)
# ----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "clip-ViT-B-32")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

