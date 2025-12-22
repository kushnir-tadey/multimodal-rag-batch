import streamlit as st
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from pathlib import Path
from collections import Counter
import re

# --- NLP Libraries ---
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Page Config
st.set_page_config(page_title="RAG Analytics", layout="wide")
st.title("📊 System Analytics")
st.markdown("Deep dive into **Content Trends** and **Structural Health**.")

# --- 1. NLP Setup (Auto-Download) ---
@st.cache_resource
def setup_nltk():
    """Ensure necessary NLTK data is downloaded."""
    resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt']
    for r in resources:
        try:
            nltk.data.find(f'corpora/{r}')
        except LookupError:
            try:
                nltk.data.find(f'tokenizers/{r}')
            except LookupError:
                nltk.download(r, quiet=True)

setup_nltk()

# --- 2. Load Data ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent 
DATA_PATH = project_root / 'data' / 'processed' / 'articles_clean.json'

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

data = load_data()

if not data:
    st.error(f"❌ Could not find data at: `{DATA_PATH}`")
    st.stop()

# --- 3. Compute Metrics ---
# Content Processing
@st.cache_data
def get_processed_corpus(data):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    stop_words.update(['said', 'also', 'would', 'could', 'news', 'batch'])

    processed_docs = []
    for article in data:
        text = article.get('text', "")
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        clean_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        processed_docs.append(" ".join(clean_words))
    return processed_docs

# Structural Processing
chunk_counts_per_article = [len(a.get('chunks', [])) for a in data]
total_chunks = sum(chunk_counts_per_article)

# Calculate Word Counts per Chunk
all_chunk_word_counts = []
for article in data:
    for chunk in article.get('chunks', []):
        # Split by whitespace to count words
        word_count = len(chunk.split())
        all_chunk_word_counts.append(word_count)

mean_chunk_size = np.mean(all_chunk_word_counts) if all_chunk_word_counts else 0

# --- 4. Display Top-Level Metrics ---
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Articles", len(data))
with c2:
    st.metric("Total Chunks", total_chunks)
with c3:
    st.metric("Avg Chunks/Article", f"{np.mean(chunk_counts_per_article):.1f}")
with c4:
    st.metric("Mean Words/Chunk", f"{mean_chunk_size:.0f} words")

st.divider()

# --- 5. Visualization Tabs ---
tab_content, tab_structure = st.tabs(["📈 Content Trends", "🏗️ Structural Health"])

# === TAB 1: Content (N-Grams) ===
with tab_content:
    st.subheader("Keyword Analysis")
    
    ngram_type = st.radio(
        "Select N-Gram Type:",
        ["Unigrams (Single Words)", "Bigrams (Two Words)", "Trigrams (Three Words)"],
        horizontal=True
    )
    
    processed_docs = get_processed_corpus(data)
    
    if "Unigrams" in ngram_type:
        n, color, title = 1, "viridis", "Top 20 Keywords"
    elif "Bigrams" in ngram_type:
        n, color, title = 2, "magma", "Top 20 Bigrams"
    else:
        n, color, title = 3, "plasma", "Top 20 Trigrams"

    def get_top_ngrams(corpus, n=1, top_k=20):
        vec = CountVectorizer(ngram_range=(n, n), stop_words='english', max_features=10000)
        bag_of_words = vec.fit_transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        return sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]

    top_grams = get_top_ngrams(processed_docs, n=n, top_k=20)
    df_freq = pd.DataFrame(top_grams, columns=['Term', 'Count'])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_freq, x='Count', y='Term', palette=color, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("")
    st.pyplot(fig)

# === TAB 2: Structure (New Charts) ===
with tab_structure:
    st.subheader("Chunking Health Check")
    
    col_chart1, col_chart2 = st.columns(2)
    
    # Chart 1: Chunks per Article
    with col_chart1:
        st.markdown("**1. Distribution of Chunks per Article**")
        
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(chunk_counts_per_article, bins=20, kde=True, color='skyblue', ax=ax1)
        ax1.set_xlabel("Number of Chunks")
        ax1.set_ylabel("Article Count")
        st.pyplot(fig1)

    # Chart 2: Words per Chunk
    with col_chart2:
        st.markdown("**2. Distribution of Words per Chunk**")
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(all_chunk_word_counts, bins=30, kde=True, color='salmon', ax=ax2)
        
        # Add a vertical line for the mean
        ax2.axvline(mean_chunk_size, color='red', linestyle='--', label=f'Mean ({mean_chunk_size:.0f})')
        ax2.legend()
        
        ax2.set_xlabel("Word Count per Chunk")
        ax2.set_ylabel("Chunk Frequency")
        st.pyplot(fig2)

# --- 6. Data Inspector ---
st.divider()
st.subheader("🔍 Data Inspector")

items_per_page = 10
total_pages = (len(data) + items_per_page - 1) // items_per_page
c_page, c_info = st.columns([1, 4])
with c_page:
    current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

start_idx = (current_page - 1) * items_per_page
end_idx = start_idx + items_per_page
batch = data[start_idx:end_idx]

for item in batch:
    with st.expander(f"📄 {item.get('title', 'No Title')}", expanded=False):
        st.markdown(f"**Source:** [{item.get('url', '#')}]({item.get('url', '#')})")
        
        # Show chunk stats for this specific article
        chunks = item.get('chunks', [])
        avg_len = np.mean([len(c.split()) for c in chunks]) if chunks else 0
        st.caption(f"Chunks: {len(chunks)} | Avg Size: {avg_len:.0f} words")
        
        st.markdown("**Preview (First 2 Chunks):**")
        for i, c in enumerate(chunks[:2]):
            st.text(f"[{i+1}] {c}")