# Multimodal AI News RAG

A production-ready **Retrieval-Augmented Generation (RAG)** system that aggregates AI news, processes it with smart recursive chunking, and enables users to search across both text and images using natural language.

## Key Features

* **Multimodal Search:** Retrieves both relevant text paragraphs and associated images for a query.
* **Recursive Chunking:** Uses advanced text splitting (NLTK/LangChain) to preserve semantic context.
* **Analytics Dashboard:**
    * **N-Gram Analysis:** Visualizes top keywords, bi-grams, and tri-grams.
    * **Health Metrics:** Monitors chunk sizes and distribution.
* **Automated Pipeline:** Single-command processing from raw scraping to vector indexing.
* **GPT-4o-mini Integration:** Generates grounded, accurate answers citing specific news sources.

---

##  Quick Start (Docker) 

The easiest way to run the app. No Python installation required.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kushnir-tadey/multimodal-rag-batch.git
    cd multimodal-rag-batch
    ```

2.  **Configure Environment:**
    Create a `.env` file in the root folder. You must add your OpenAI key, but you can also configure the models used:
    ```ini
    # Required
    OPENAI_API_KEY="sk-..."

    # Optional (Defaults)
    EMBEDDING_MODEL="clip-ViT-B-32"  # Multimodal model for Image+Text embedding
    LLM_MODEL="gpt-4o-mini"          # Main LLM for answer generation
    ```

3.  **Open Docker and run:**
    ```bash
    docker compose up --build
    ```
    *The app will automatically index the data and launch.*

4.  **Open:** [http://localhost:8501](http://localhost:8501)

##  Architecture

1.  **Data Collection (Scraper):**
    * Fetches the latest AI news articles and associated images.
    * Saves raw data to `data/raw/` for auditing.

2.  **Indexing Engine:**
    * **Preprocessing:** Cleans raw HTML text and normalizes content.
    * **Recursive Chunking:** Uses `RecursiveCharacterTextSplitter` to break text into semantic units (preserving paragraphs and sentences) rather than arbitrary character counts.
    * **Embedding:** Converts text and images into vector embeddings using `SentenceTransformers`.
    * **Storage:** Saves vectors into a local **FAISS** index for millisecond-latency search.

3.  **RAG Engine:**
    * **Retrieval:** Finds the top-k most relevant chunks based on vector similarity to the user's query.
    * **Generation:** Feeds the retrieved context to **GPT-4o-mini**, which generates a grounded, accurate answer citing specific sources.

4.  **User Interface:**
    * A Streamlit-based web app for interactive search.
    * Dedicated Analytics Dashboard for system monitoring.

---

##  Analytics & Evaluation

The system includes a production-grade **Analytics Dashboard** to monitor data health and content trends.

### 1. Content Analysis (NLP)
* **N-Gram Extraction:** visualizes top **Bi-grams** (2-word pairs) and **Tri-grams** (3-word phrases) to understand trending topics.
* **Lemmatization:** Uses NLTK to normalize words for accurate frequency counts.

### 2. Structural Health Metrics
* **Words per Chunk:** Monitors the distribution of chunk sizes.
* **Chunks per Article:** Visualizes the depth of indexed content (short news vs. long-form analysis).

---

## Local Development (Manual Setup)

If you prefer to run components individually without Docker, follow these steps:

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
# For Windows:
venv\Scripts\activate
# For Mac/Linux:
source venv/bin/activate
```

# Install dependencies
```bash
pip install -r requirements.txt
```

2. Usage Pipeline
Step 1: Scrape Data Fetches articles and downloads images to data/images/.

```bash
python -m src.scraping.batch_scraper
```

Step 2: Index Data Cleans text, embeds content, and builds the FAISS vector database.

```bash
python -m src.indexing.indexer
```

Step 3: Run the App 
Launches the web interface.

```bash
streamlit run src/ui/app.py
```

---

##  Project Structure

```text
multimodal-rag-batch/
├── data/                   # Storage for data artifacts
│   ├── faiss_index/        # Vector database files
│   ├── images/             # Scraped images
│   ├── processed/          # Cleaned JSON data for Analytics
│   └── raw/                # Raw scraped data
├── src/
│   ├── indexing/           # ETL & Indexing logic
│   │   ├── chunker.py      # Recursive text splitting
│   │   └── indexer.py      # Main indexing pipeline
│   ├── rag/                # RAG Core Modules
│   │   ├── generator.py    # OpenAI generation logic
│   │   └── retriever.py    # FAISS search logic
│   ├── scraping/           # Data collection
│   │   └── batch_scraper.py
│   ├── ui/                 # Frontend application
│   │   ├── pages/
│   │   │   └── Analytics.py # Dashboard
│   │   └── app.py          # Main entry point
│   └── config.py           # Global settings
├── venv/                   # Virtual Environment
├── .env                    # API Keys (Git-ignored)
├── .gitignore              # Git ignore rules
└── requirements.txt        # Python dependencies
```

##  Technologies Used

* **Core Language:** Python 3.10+
* **LLM & AI:** OpenAI GPT-4o-mini
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** SentenceTransformers (clip-ViT-B-32)
* **Web Framework:** Streamlit
* **Data Processing:**
    * **LangChain:** Recursive Text Splitting
    * **NLTK:** Stopword removal & Lemmatization
    * **Scikit-Learn:** N-gram analysis (Bi-grams/Tri-grams)
    * **Pandas:** Data manipulation
    * **BeautifulSoup4:** HTML parsing & scraping
* **Containerization:** Docker & Docker Compose
