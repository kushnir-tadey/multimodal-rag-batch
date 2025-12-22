import json
import logging
import re
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# Custom Modules
from src.indexing.chunker import Chunker
from src.config import PROCESSED_DIR, DATA_DIR, EMBEDDING_MODEL, RAW_DIR

# ----------------------
# Configuration
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config Paths
MODEL_NAME = EMBEDDING_MODEL 
INDEX_DIR = DATA_DIR / "faiss_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Input/Output Files
RAW_DATA_PATH = RAW_DIR / "articles.json" 
CLEAN_DATA_PATH = PROCESSED_DIR / "articles_clean.json"

# ----------------------
# Helper Functions (Ported from prepare_embeddings.py)
# ----------------------
def load_data(file_path: Path) -> List[Dict]:
    if not file_path.exists():
        logger.error(f"❌ File not found: {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(text: str) -> str:
    """Simple cleaning: remove extra whitespace and newlines."""
    if not text: return ""
    # Collapse multiple spaces/newlines into single spaces (optional, or keep \n for structure)
    # For recursive chunking, we actually want to KEEP \n\n structure!
    # So we only strip mostly.
    text = text.strip()
    return text

def save_clean_data(articles: List[Dict], out_path: Path):
    """Saves the intermediate JSON for the Analytics Dashboard."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Saved cleaned data to {out_path}")

# ----------------------
# Main Pipeline
# ----------------------
def build_multimodal_index():
    # 1. Load Raw Data
    logger.info(f"Loading raw data from {RAW_DATA_PATH}...")
    raw_articles = load_data(RAW_DATA_PATH)
    if not raw_articles:
        return

    # 2. Initialize Logic
    model = SentenceTransformer(MODEL_NAME)
    chunker = Chunker(chunk_size=800, chunk_overlap=100)
    
    # buffers
    processed_articles_for_json = []
    text_chunks = []
    image_paths = []
    metadata = []
    doc_id_counter = 0

    logger.info("Processing articles (Cleaning + Chunking)...")

    # 3. Process Loop
    for art in raw_articles:
        # A. Clean
        raw_text = art.get("text", "")
        cleaned_text = clean_text(raw_text)
        
        # B. Chunk (Recursive)
        chunks = chunker.chunk_text(cleaned_text)
        
        # C. Store for JSON (Analytics needs this!)
        processed_articles_for_json.append({
            "url": art.get("url"),
            "title": art.get("title"),
            "text": cleaned_text,       # Full text
            "chunks": chunks,           # The new recursive chunks
            "local_image_path": art.get("local_image_path")
        })

        # D. Prepare for Embedding (Index needs this)
        for chunk in chunks:
            combined_text = f"{art.get('title', 'Unknown')}: {chunk}"
            text_chunks.append(combined_text)
            
            metadata.append({
                "id": doc_id_counter,
                "type": "text",
                "url": art.get("url", ""),
                "title": art.get("title", ""),
                "content": chunk,
                "full_text_preview": cleaned_text[:200]
            })
            doc_id_counter += 1
            
        # E. Process Image
        if art.get("local_image_path"):
            img_p = Path(art["local_image_path"])
            if img_p.exists():
                image_paths.append(str(img_p))
                metadata.append({
                    "id": doc_id_counter,
                    "type": "image",
                    "url": art.get("url", ""),
                    "title": art.get("title", ""),
                    "image_path": str(img_p),
                    "content": "[Image Associated with Article]"
                })
                doc_id_counter += 1

    # 4. Save Intermediate JSON (Crucial for Analytics)
    save_clean_data(processed_articles_for_json, CLEAN_DATA_PATH)

    # 5. Embed & Index
    logger.info(f"Embedding {len(text_chunks)} text chunks...")
    if not text_chunks:
        logger.warning("No text chunks generated.")
        return

    text_embeddings = model.encode(text_chunks, convert_to_numpy=True, show_progress_bar=True)
    
    if image_paths:
        logger.info(f"Embedding {len(image_paths)} images...")
        images = [Image.open(p) for p in image_paths]
        image_embeddings = model.encode(images, convert_to_numpy=True, show_progress_bar=True)
        all_embeddings = np.vstack([text_embeddings, image_embeddings])
    else:
        all_embeddings = text_embeddings

    faiss.normalize_L2(all_embeddings)
    
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(all_embeddings)
    
    # 6. Save Index
    faiss.write_index(index, str(INDEX_DIR / "multimodal.index"))
    with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"✅ Indexing complete! Saved to {INDEX_DIR}")

if __name__ == "__main__":
    build_multimodal_index()