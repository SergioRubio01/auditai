FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    gnupg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add Python 3.11 repository and install it
RUN apt-get update && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3.11-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/backend 

# Copy requirements first for better caching
COPY backend/requirements.txt /app/backend/
RUN python3.11 -m pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy the .env file
COPY backend/.env /app/backend/

# Copy the code
COPY backend/ /app/backend/

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Switch to backend directory
WORKDIR /app/backend

# Start FastAPI server
CMD ["python3.11", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 