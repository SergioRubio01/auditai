# Copyright 2025-present Sergio García Arrojo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Use multi-stage build for smaller final image
FROM python:3.11-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
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
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY flow/ ./flow/
COPY .env .
COPY start.sh .

# Create necessary directories
RUN mkdir -p uploads/PDFAll uploads/PDFValid uploads/Images

# Ensure start.sh has correct line endings and permissions
RUN dos2unix start.sh && \
    chmod +x start.sh

# Expose ports for both backend and frontend
EXPOSE 8000 8501

# Run both services using absolute path
CMD ["/app/start.sh"] 