from pathlib import Path
import json
from typing import List
from src.config import PROCESSED_DIR

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of chunk_size words with overlap.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def split_articles(input_file: Path = PROCESSED_DIR / "articles_clean.json") -> List[dict]:
    """
    Load articles and split each article text into chunks.
    Returns a list of dicts with 'url', 'title', 'chunk'.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    all_chunks = []
    for article in articles:
        chunks = chunk_text(article["text"])
        for chunk in chunks:
            all_chunks.append({
                "url": article["url"],
                "title": article["title"],
                "chunk": chunk
            })
    return all_chunks

if __name__ == "__main__":
    chunks = split_articles()
    print(f"Created {len(chunks)} text chunks")
