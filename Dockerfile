# Use Python 3.10 slim image for better compatibility
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app/analyzer.py .
COPY app/download_models.py .
COPY app/web_ui.py .
COPY app/templates/ ./templates/

# Create directories for data persistence
RUN mkdir -p /app/chroma_db /app/codebase

# Create a script to download models
RUN echo '#!/usr/bin/env python3\n\
from sentence_transformers import SentenceTransformer\n\
from transformers import AutoTokenizer, AutoModelForCausalLM\n\
import torch\n\
\n\
print("Downloading default embedding model...")\n\
SentenceTransformer("all-MiniLM-L6-v2")\n\
\n\
print("Downloading lightweight models...")\n\
AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")\n\
AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")\n\
\n\
print("Base models downloaded successfully!")' > /app/download_base_models.py

# Pre-download only lightweight models by default
# Larger models will be downloaded on-demand
RUN python3 /app/download_base_models.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create a non-root user for security
RUN useradd -m -u 1000 analyzer && \
    chown -R analyzer:analyzer /app
USER analyzer

# Default command
CMD ["python3", "web_ui.py"]