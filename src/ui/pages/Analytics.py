import streamlit as st
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
from collections import Counter
import re

# Page Config
st.set_page_config(page_title="RAG Analytics", layout="wide")
st.title("📊 System Analytics")
st.markdown("Deep dive into the **Content Distribution** and **Retrieval Health**.")

# --- Load Data ---
current_file = Path(__file__).resolve()
# Go up 4 levels to reach project root (src/ui/pages -> src/ui -> src -> root)
project_root = current_file.parent.parent.parent.parent 
DATA_PATH = project_root / 'data' / 'processed' / 'articles_clean.json'

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    # Fix encoding for Windows
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

data = load_data()

if not data:
    st.error(f"❌ Could not find data at: `{DATA_PATH}`")
    st.stop()

# --- 1. Compute Stats ---
all_text = ""
chunk_counts = []
total_chunks = 0

for article in data:
    chunks = article.get('chunks', [])
    chunk_counts.append(len(chunks))
    total_chunks += len(chunks)
    # Combine text for keyword analysis
    all_text += " ".join(chunks) + " "

# Simple Keyword Extraction (Top 20)
# We exclude common "stopwords" to make the chart meaningful
stopwords = set(['the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'for', 'it', 'on', 'with', 'as', 'are', 'at', 'this', 'by', 'an', 'be', 'from', 'or', 'was', 'have', 'not', 'but', 'can', 'which', 'they', 'their', 'has', 'more', 'about', 'also', 'will', 'new', 'one', 'its', 'up', 'out', 'all', 'into', 'some', 'news', 'than', 'other', 'we', 'what', 'when', 'there', 'use', 'using', 'how', 'may', 'like', 'such', 'fine'])
words = re.findall(r'\w+', all_text.lower())
filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
word_freq = Counter(filtered_words).most_common(20)

# --- 2. Display Top-Level Metrics ---
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total Articles", len(data))
with c2:
    st.metric("Total Chunks", total_chunks)
with c3:
    st.metric("Unique Keywords", len(set(filtered_words)))

st.divider()

# --- 3. Visualizations ---
st.subheader("📈 Content Insights")

tab1, tab2 = st.tabs(["Top Keywords", "Article Lengths"])

with tab1:
    st.markdown("Most frequent terms across the entire database (excluding common stopwords).")
    
    # Prepare data for plotting
    df_freq = pd.DataFrame(word_freq, columns=['Term', 'Count'])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_freq, x='Count', y='Term', palette='viridis', ax=ax)
    ax.set_title("Top 20 Keywords in The Batch")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("")
    
    st.pyplot(fig)

with tab2:
    st.markdown("Distribution of **Chunks per Article**. This shows if articles are generally short snippets or long-form deep dives.")
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.histplot(chunk_counts, bins=20, kde=True, color='orange', ax=ax2)
    ax2.set_title("Chunks per Article")
    ax2.set_xlabel("Number of 50-word Chunks")
    
    st.pyplot(fig2)

st.divider()

# --- 4. Paginated Data Inspector ---
st.subheader("🔍 Data Inspector")

# Pagination Settings
items_per_page = 10
total_pages = (len(data) + items_per_page - 1) // items_per_page

c_page, c_info = st.columns([1, 4])
with c_page:
    current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
with c_info:
    st.caption(f"Showing items {(current_page-1)*items_per_page + 1} - {min(current_page*items_per_page, len(data))} of {len(data)}")

# Slice Data
start_idx = (current_page - 1) * items_per_page
end_idx = start_idx + items_per_page
batch = data[start_idx:end_idx]

# Display as Expanders (Cleaner than JSON)
for item in batch:
    title = item.get('title', 'No Title')
    url = item.get('url', '#')
    chunks = item.get('chunks', [])
    
    with st.expander(f"📄 {title}", expanded=False):
        st.markdown(f"**Source:** [{url}]({url})")
        st.markdown(f"**Total Chunks:** {len(chunks)}")
        
        # Show first 3 chunks only to save space
        st.markdown("**Preview (First 3 Chunks):**")
        for i, chunk in enumerate(chunks[:3]):
            st.text(f"[{i+1}] {chunk}")
        
        if len(chunks) > 3:
            st.caption(f"... and {len(chunks)-3} more chunks.")