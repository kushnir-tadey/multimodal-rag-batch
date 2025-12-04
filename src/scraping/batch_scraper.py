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

# ----------------------
# Constants
# ----------------------
HEADERS = {"User-Agent": "MultimodalRAG/0.1 (contact: your_email@example.com)"}


# ----------------------
# Dataclass for Articles
# ----------------------
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
    """Fetch raw HTML from a given URL."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


# ----------------------
# Extract Article Links (Updated!)
# ----------------------
def parse_archive_page(html: str) -> List[str]:
    """
    Parse the Batch archive page and extract article URLs.

    Based on your HTML snippet:
    Each <article> contains a link like:
       <a href="/the-batch/issue-330/">

    We select: article a[href^='/the-batch/issue']
    """
    soup = BeautifulSoup(html, "html.parser")

    links = []
    for a in soup.select("article a[href^='/the-batch/issue']"):
        href = a.get("href")
        if href:
            # Convert relative links → absolute URLs
            full_url = urljoin(BASE_URL, href)
            links.append(full_url)

    logger.info("Extracted %d article URLs", len(links))
    return links


# ----------------------
# Download Article Images
# ----------------------
def download_image(url: str, dest_dir: Path = IMAGES_DIR) -> Optional[Path]:
    """Download image to local folder."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1].split("?")[0] or "image.jpg"
    dst = dest_dir / filename

    if dst.exists():
        return dst

    try:
        r = requests.get(url, headers=HEADERS, stream=True, timeout=20)
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        return dst
    except Exception as exc:
        logger.warning("Failed to download image %s: %s", url, exc)
        return None


# ----------------------
# Scrape Single Article
# ----------------------
def scrape_article(url: str) -> Optional[BatchArticle]:
    """Download and parse a single article using newspaper3k."""
    try:
        article = Article(url)
        article.download()
        article.parse()

        top_img = article.top_image if article.top_image else None

        return BatchArticle(
            url=url,
            title=article.title or "",
            text=article.text or "",
            top_image_url=top_img,
        )

    except Exception as exc:
        logger.warning("Failed to scrape %s: %s", url, exc)
        return None


# ----------------------
# Main Batch Scraper
# ----------------------
def scrape_batch_archive(limit: int = 10) -> None:
    """Scrape The Batch archive and save articles to JSON."""
    html = fetch_html(BATCH_ARCHIVE_URL)

    if not html:
        logger.error("Could not fetch archive page.")
        return

    article_urls = parse_archive_page(html)[:limit]
    logger.info("Found %d article URLs", len(article_urls))

    articles: List[BatchArticle] = []

    for url in article_urls:
        logger.info("Scraping: %s", url)
        art = scrape_article(url)

        if not art:
            continue

        # Download image
        if art.top_image_url:
            local = download_image(art.top_image_url)
            art.local_image_path = str(local) if local else None

        articles.append(art)
        time.sleep(1.0)  # polite delay

    # Save JSON
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "articles.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in articles], f, ensure_ascii=False, indent=2)

    logger.info("Saved %d articles to %s", len(articles), out_path)


# ----------------------
# Run Script
# ----------------------
if __name__ == "__main__":
    scrape_batch_archive(limit=5)
