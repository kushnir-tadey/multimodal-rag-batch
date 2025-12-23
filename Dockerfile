FROM python:3.10-slim

# Set environment variables to keep Python logs unbuffered
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet omw-1.4 punkt

COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0"]