FROM python:3.12

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    apt-get install -y iputils-ping vim

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the judge server code
COPY . .

# Create a non-root user for running submissions
RUN useradd -m judge
RUN chown -R judge:judge /app

# Switch to non-root user
USER judge

# Expose the port the judge server will run on
EXPOSE 5000

CMD ["python", "judge_server.py"] 