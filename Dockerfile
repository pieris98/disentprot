# Dockerfile for protein disentanglement experiments
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    htop \
    tree \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data experiments logs

# Create sample data
RUN python3 -c "
from src.data.utils import create_sample_fasta
import os
os.makedirs('data', exist_ok=True)
create_sample_fasta('data/sample_proteins.fasta', num_samples=100)
print('Sample FASTA created')
"

# Set permissions
RUN chmod +x train.py evaluate.py

# Expose port for Jupyter notebook (optional)
EXPOSE 8888

# Default command
CMD ["python3", "train.py", "--help"]
