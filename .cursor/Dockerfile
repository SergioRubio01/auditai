# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install sudo and git, and other system dependencies.
RUN apt-get update && apt-get install -y \
    sudo \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Create a non-root user and add it to the sudo group
RUN useradd -m -s /bin/bash -g sudo appuser && \
    echo "appuser:appuser" | chpasswd && \
    echo "appuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Set the user and working directory
USER appuser
WORKDIR /home/appuser

# Switch back to root to grant permissions to the working directory
USER root
RUN chown -R appuser:appuser /home/appuser
USER appuser

# Set a command to keep the container running
CMD ["tail", "-f", "/dev/null"]
