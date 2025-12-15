from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path
import json
import time
import random  # Added for random delays
import logging
import requests
from bs4 import BeautifulSoup
from newspaper import Article, Config # Added Config
from urllib.parse import urljoin

from src.config import BATCH_ARCHIVE_URL, BASE_URL, RAW_DIR, IMAGES_DIR

# ----------------------
# Logging
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fake browser User-Agent to avoid detection
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

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
def fetch_html(url: str, timeout: int = 20) -> Optional[str]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        if response.status_code == 429:
            logger.warning("Rate limit hit (429). Waiting 60 seconds...")
            time.sleep(60)
            return None # Skip this page request
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
    # Updated selector based on The Batch structure
    for a in soup.select("article a[href^='/the-batch/issue']"):
        href = a.get("href")
        if href:
            full_url = urljoin(BASE_URL, href)
            links.append(full_url)
    return list(set(links))

# ----------------------
# Download Image
# ----------------------
def download_image(url: str, dest_dir: Path = IMAGES_DIR) -> Optional[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Basic cleanup of filename
    filename = url.split("/")[-1].split("?")[0]
    if not filename or len(filename) > 50: filename = "image.jpg"
    
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
        # Pass headers to Newspaper so we look like a browser
        config = Config()
        config.browser_user_agent = HEADERS["User-Agent"]
        config.request_timeout = 20
        
        article = Article(url, config=config)
        article.download()
        article.parse()
        
        # Validation: Ignore empty articles
        if not article.text or len(article.text) < 100:
            return None

        return BatchArticle(
            url=url,
            title=article.title or "",
            text=article.text or "",
            top_image_url=article.top_image,
        )
    except Exception as exc:
        # Don't crash on individual article failures
        return None

# ----------------------
# Main Loop
# ----------------------
def scrape_batch_archive(limit: int = 10) -> None:
    articles: List[BatchArticle] = []
    page = 1
    
    # Safety: Stop if we check too many pages without finding articles
    empty_pages_count = 0 
    
    while len(articles) < limit and empty_pages_count < 5:
        current_url = f"{BATCH_ARCHIVE_URL}?page={page}" if page > 1 else BATCH_ARCHIVE_URL
        logger.info(f"Checking archive page {page}...")
        
        html = fetch_html(current_url)
        if not html:
            empty_pages_count += 1
            page += 1
            continue
            
        page_links = parse_archive_page(html)
        if not page_links:
            logger.info("No links found on this page.")
            empty_pages_count += 1
            break
            
        logger.info(f"Found {len(page_links)} links. Scraping...")
        
        articles_added_on_page = 0
        for url in page_links:
            if len(articles) >= limit: break
            if any(a.url == url for a in articles): continue

            art = scrape_article(url)
            if art:
                # Download image
                if art.top_image_url:
                    local = download_image(art.top_image_url)
                    art.local_image_path = str(local) if local else None
                
                articles.append(art)
                articles_added_on_page += 1
                logger.info(f"Saved [{len(articles)}/{limit}]: {art.title}")
                
                # IMPORTANT: Random sleep to behave like a human
                time.sleep(random.uniform(2.0, 5.0)) 
        
        if articles_added_on_page == 0:
            empty_pages_count += 1
        else:
            empty_pages_count = 0 # Reset if we found something

        page += 1

    # Save
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "articles.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in articles], f, ensure_ascii=False, indent=2)
    
    logger.info(f"Completed. Saved {len(articles)} articles to {out_path}")

if __name__ == "__main__":
    # Limit set to 50, but it will respect current file if we were appending (simple overwrite here)
    scrape_batch_archive(limit=50)