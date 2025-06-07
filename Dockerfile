# Enhanced multi-stage Dockerfile with performance optimizations
ARG PYTHON_VERSION=3.11

# Stage 1: Dependencies builder with caching optimization
FROM python:${PYTHON_VERSION}-slim AS python-deps

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    libpq-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for better isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel for faster builds
RUN pip install --upgrade pip wheel setuptools

# Copy only requirements for better caching
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 2: System dependencies
FROM python:${PYTHON_VERSION}-slim AS system-deps

# Install runtime dependencies with specific versions
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    curl \
    dos2unix \
    ca-certificates \
    tini \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Stage 3: Application builder
FROM system-deps AS app-builder

# Copy virtual environment from python-deps
COPY --from=python-deps /opt/venv /opt/venv

# Set up non-root user for security
RUN groupadd -r autoaudit && useradd -r -g autoaudit autoaudit

# Set working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=autoaudit:autoaudit . .

# Create necessary directories with proper permissions
RUN mkdir -p \
    uploads/PDFAll \
    uploads/PDFValid \
    uploads/Images \
    database \
    logs \
    cache \
    /tmp/textract \
    && chown -R autoaudit:autoaudit /app \
    && chmod -R 755 /app

# Fix line endings and set executable permissions
RUN dos2unix start.sh && chmod +x start.sh

# Stage 4: Final optimized image
FROM python:${PYTHON_VERSION}-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    tini \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy user/group from builder
COPY --from=app-builder /etc/passwd /etc/passwd
COPY --from=app-builder /etc/group /etc/group

# Copy virtual environment
COPY --from=app-builder /opt/venv /opt/venv

# Copy application with proper ownership
COPY --from=app-builder --chown=autoaudit:autoaudit /app /app

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    TINI_SUBREAPER=1

# Set working directory
WORKDIR /app

# Switch to non-root user
USER autoaudit

# Add health check with better parameters
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8501

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run application
CMD ["/app/start.sh"]