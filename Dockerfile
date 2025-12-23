# 1. Use a lightweight, official Python base image
FROM python:3.10-slim

# 2. Set environment variables to keep Python logs unbuffered
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3. Install system dependencies (needed for building some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory inside the container
WORKDIR /app

# 5. Copy requirements first (to leverage Docker caching)
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Download NLTK data (required for your Analytics dashboard)
RUN python -m nltk.downloader stopwords wordnet omw-1.4 punkt

# 8. Copy the rest of your application code
COPY . .

# 9. Expose the port Streamlit runs on
EXPOSE 8501

# 10. Default command (can be overridden by docker-compose)
CMD ["streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0"]