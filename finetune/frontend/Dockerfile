FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/frontend/pages /app/frontend/utils

# Copy requirements first for better caching
COPY frontend/requirements.txt /app/frontend/
RUN pip install --no-cache-dir -r /app/frontend/requirements.txt

# Copy the code
COPY frontend/ /app/frontend/

# Set environment variables
ENV PYTHONPATH=/app
ENV BACKEND_URL=http://backend:8000

# Expose Streamlit port
EXPOSE 8501

# Switch to frontend directory
WORKDIR /app/frontend

# Start Streamlit
CMD ["streamlit", "run", "Train.py", "--server.address", "0.0.0.0"]