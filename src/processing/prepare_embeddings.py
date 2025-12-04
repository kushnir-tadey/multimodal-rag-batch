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
    """Simple cleaning: remove extra whitespace, newlines, and HTML remnants if any."""
    text = re.sub(r'\s+', ' ', text)  # collapse multiple whitespace/newlines
    text = text.strip()
    return text

def process_articles(articles: List[Dict]) -> List[Dict]:
    """Clean and prepare articles."""
    processed = []
    for art in articles:
        text = clean_text(art["text"])
        processed.append({
            "url": art["url"],
            "title": art["title"],
            "text": text,
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

    articles = load_articles(raw_file)
    articles_clean = process_articles(articles)
    save_processed_articles(articles_clean, processed_file)

    print(f"Saved {len(articles_clean)} cleaned articles to {processed_file}")
