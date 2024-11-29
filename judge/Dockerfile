FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    iputils-ping \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all python files
COPY *.py .

# Create a non-root user for running submissions
RUN useradd -m judge
RUN chown -R judge:judge /app

# Switch to non-root user
USER judge

# Create data directory with permissions
RUN mkdir -p /app/data
