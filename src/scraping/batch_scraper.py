from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path
import json
import time
import logging
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from urllib.parse import urljoin

from src.config import BATCH_ARCHIVE_URL, BASE_URL, RAW_DIR, IMAGES_DIR

# ----------------------
# Logging
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "MultimodalRAG/0.1 (contact: your_email@example.com)"}

@dataclass
class BatchArticle:
    url: str
    title: str
    text: str
    top_image_url: Optional[str] = None
    local_image_path: Optional[str] = None

# ----------------------
# Utility: Fetch HTML
# ----------------------
def fetch_html(url: str, timeout: int = 15) -> Optional[str]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None

# ----------------------
# Extract Links
# ----------------------
def parse_archive_page(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    # Select articles. The selector might need adjustment if site layout changes
    for a in soup.select("article a[href^='/the-batch/issue']"):
        href = a.get("href")
        if href:
            full_url = urljoin(BASE_URL, href)
            links.append(full_url)
    return list(set(links)) # Deduplicate

# ----------------------
# Download Image
# ----------------------
def download_image(url: str, dest_dir: Path = IMAGES_DIR) -> Optional[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1].split("?")[0] or "image.jpg"
    dst = dest_dir / filename
    if dst.exists(): return dst

    try:
        r = requests.get(url, headers=HEADERS, stream=True, timeout=20)
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        return dst
    except Exception:
        return None

# ----------------------
# Scrape Single Article
# ----------------------
def scrape_article(url: str) -> Optional[BatchArticle]:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return BatchArticle(
            url=url,
            title=article.title or "",
            text=article.text or "",
            top_image_url=article.top_image,
        )
    except Exception as exc:
        logger.warning("Failed to scrape %s: %s", url, exc)
        return None

# ----------------------
# Main Loop (With Pagination)
# ----------------------
def scrape_batch_archive(limit: int = 10) -> None:
    articles: List[BatchArticle] = []
    page = 1
    
    while len(articles) < limit:
        # Handle pagination logic
        current_url = f"{BATCH_ARCHIVE_URL}?page={page}" if page > 1 else BATCH_ARCHIVE_URL
        logger.info(f"Checking archive page {page}: {current_url}")
        
        html = fetch_html(current_url)
        if not html:
            break
            
        page_links = parse_archive_page(html)
        if not page_links:
            logger.info("No more articles found.")
            break
            
        logger.info(f"Found {len(page_links)} links on page {page}")
        
        for url in page_links:
            if len(articles) >= limit:
                break
                
            # Skip if we already have this URL (basic check)
            if any(a.url == url for a in articles):
                continue

            logger.info(f"Scraping [{len(articles)+1}/{limit}]: {url}")
            art = scrape_article(url)
            
            if art:
                if art.top_image_url:
                    local = download_image(art.top_image_url)
                    art.local_image_path = str(local) if local else None
                articles.append(art)
                time.sleep(0.5) # Be polite

        page += 1

    # Save
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "articles.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in articles], f, ensure_ascii=False, indent=2)
    
    logger.info(f"Completed. Saved {len(articles)} articles to {out_path}")

if __name__ == "__main__":
    # KEEP THIS LOW FOR NOW
    scrape_batch_archive(limit=5)