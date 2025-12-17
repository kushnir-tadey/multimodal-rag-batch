from pathlib import Path
import json
import re
from typing import List, Dict

from src.config import RAW_DIR, PROCESSED_DIR

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_articles(file_path: Path) -> List[Dict]:
    """Load articles from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    return articles

def clean_text(text: str) -> str:
    """Simple cleaning: remove extra whitespace, newlines, and HTML remnants."""
    text = re.sub(r'\s+', ' ', text)  # collapse multiple whitespace/newlines
    text = text.strip()
    return text

def create_chunks(text: str, chunk_size: int = 50) -> List[str]:
    """
    Splits text into chunks of ~50 words.
    CLIP has a limit of 77 tokens (approx 50-60 words).
    """
    words = text.split()
    chunks = []
    
    # Simple sliding window (non-overlapping for now to keep it simple)
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]
        chunk_str = " ".join(chunk_words)
        if len(chunk_str) > 10:  # Ignore tiny artifacts
            chunks.append(chunk_str)
            
    return chunks

def process_articles(articles: List[Dict]) -> List[Dict]:
    """Clean and chunk articles."""
    processed = []
    for art in articles:
        cleaned_text = clean_text(art["text"])
        
        # Create Chunks
        chunks = create_chunks(cleaned_text, chunk_size=50)
        
        processed.append({
            "url": art["url"],
            "title": art["title"],
            "text": cleaned_text,       # Keep full text for reference
            "chunks": chunks,           # NEW: List of searchable segments
            "top_image_url": art.get("top_image_url"),
            "local_image_path": art.get("local_image_path")
        })
    return processed

def save_processed_articles(articles: List[Dict], out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

# ---------------------- Main ----------------------
if __name__ == "__main__":
    raw_file = RAW_DIR / "articles.json"
    processed_file = PROCESSED_DIR / "articles_clean.json"

    if not raw_file.exists():
        print(f"Error: {raw_file} not found. Run the scraper first.")
    else:
        articles = load_articles(raw_file)
        articles_clean = process_articles(articles)
        save_processed_articles(articles_clean, processed_file)

        print(f"Saved {len(articles_clean)} cleaned articles (with chunks) to {processed_file}")