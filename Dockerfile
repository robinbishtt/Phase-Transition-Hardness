# Dockerfile for Phase-Transition-Hardness
# Provides a reproducible environment for running experiments

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create results directory
RUN mkdir -p results figures

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command: run validation
CMD ["python", "src/validation.py", "--results_dir", "results"]

# Labels for metadata
LABEL maintainer="Robin Bisht <bishtrobin75@gmail.com>"
LABEL description="Phase-Transition Structure as Foundation for Cryptographic Hardness"
LABEL version="1.0.0"
LABEL repository="https://github.com/robinbishtt/Phase-Transition-Hardness"
