# Use multi-stage build for smaller final image
FROM python:3.11-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Final stage
FROM python:3.11-slim

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    dos2unix \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads/PDFAll uploads/PDFValid uploads/Images database

# Ensure start.sh has correct line endings and permissions
RUN dos2unix start.sh && \
    chmod +x start.sh

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports for both backend and frontend
EXPOSE 8000 8501

# Run both services using absolute path
CMD ["/app/start.sh"] 