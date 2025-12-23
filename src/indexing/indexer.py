import json
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict
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
# Helper Functions
# ----------------------
def load_data(file_path: Path) -> List[Dict]:
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(text: str) -> str:
    """Simple cleaning: remove extra whitespace."""
    if not text: return ""
    return text.strip()

def save_clean_data(articles: List[Dict], out_path: Path):
    """Saves the intermediate JSON for the Analytics Dashboard."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved cleaned data to {out_path}")

# ----------------------
# Main Pipeline
# ----------------------
def build_multimodal_index():
    # 1. Load Raw Data
    logger.info(f"Loading raw data from {RAW_DATA_PATH}...")
    raw_articles = load_data(RAW_DATA_PATH)
    if not raw_articles:
        logger.warning("No articles found in raw data.")
        return

    # 2. Initialize Logic
    logger.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    chunker = Chunker(chunk_size=800, chunk_overlap=100)
    
    # Buffers
    processed_articles_for_json = []
    text_chunks = []
    image_paths_to_embed = [] # Paths for the model
    metadata = []
    doc_id_counter = 0

    logger.info("Processing articles (Cleaning + Chunking + Image Path Fix)...")

    # 3. Process Loop
    for art in raw_articles:
        # A. Clean
        raw_text = art.get("text", "")
        cleaned_text = clean_text(raw_text)
        
        # B. Chunk (Recursive)
        chunks = chunker.chunk_text(cleaned_text)
        
        # C. Store for JSON (Analytics)
        processed_articles_for_json.append({
            "url": art.get("url"),
            "title": art.get("title"),
            "text": cleaned_text,
            "chunks": chunks,
            "image_url": art.get("top_image_url") or art.get("image_url")
        })

        # D. Prepare Text for Embedding
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
            
        # E. Process Image (Path Normalization Fix)
        # Check both keys just in case
        raw_img_path = art.get("image_path") or art.get("local_image_path")

        if raw_img_path:
            clean_filename = str(raw_img_path).replace("\\", "/").split("/")[-1]
             
            clean_image_path = DATA_DIR / "images" / clean_filename
            
            # 3. Verify existence
            if clean_image_path.exists():
                try:
                    # Validate image by opening it
                    img = Image.open(clean_image_path).convert("RGB")
                    
                    # Store valid path for batch embedding later
                    image_paths_to_embed.append(str(clean_image_path))
                    
                    metadata.append({
                        "id": doc_id_counter,
                        "type": "image",
                        "url": art.get("url", ""),
                        "title": art.get("title", ""),
                        "image_path": str(clean_image_path),
                        "content": "[Image Associated with Article]"
                    })
                    doc_id_counter += 1
                except Exception as e:
                    logger.warning(f"Corrupt image {clean_filename}: {e}")
            else:
                logger.warning(f"Image missing at expected path: {clean_image_path}")

    # 4. Save Intermediate JSON
    save_clean_data(processed_articles_for_json, CLEAN_DATA_PATH)

    # 5. Embed & Index
    if not text_chunks and not image_paths_to_embed:
        logger.error("No content (text or images) to index!")
        return

    # Embed Text
    logger.info(f"Embedding {len(text_chunks)} text chunks...")
    text_embeddings = model.encode(text_chunks, convert_to_numpy=True, show_progress_bar=True)
    all_embeddings = text_embeddings

    # Embed Images (if any)
    if image_paths_to_embed:
        logger.info(f"Embedding {len(image_paths_to_embed)} images...")
        images = [Image.open(p) for p in image_paths_to_embed]
        image_embeddings = model.encode(images, convert_to_numpy=True, show_progress_bar=True)
        
        # Stack them: Text first, then Images
        all_embeddings = np.vstack([text_embeddings, image_embeddings])
    
    # Normalize for Cosine Similarity
    faiss.normalize_L2(all_embeddings)
    
    # Create Index
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(all_embeddings)
    
    # 6. Save Index
    faiss.write_index(index, str(INDEX_DIR / "multimodal.index"))
    with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Indexing complete! Total Vectors: {index.ntotal}")
    logger.info(f"Saved index to {INDEX_DIR}")

if __name__ == "__main__":
    build_multimodal_index()