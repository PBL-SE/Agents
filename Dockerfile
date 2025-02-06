# 1️⃣ Base image (Minimal Python)
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install only required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements file
COPY requirements.txt .

# Install dependencies (NO CACHE, CPU-only PyTorch)
RUN pip install --no-cache-dir -r requirements.txt torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Remove pip cache
RUN rm -rf /root/.cache/pip

# 2️⃣ Final lightweight image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY . .

# Set environment variables
ENV GOOGLE_API_KEY=${GEMINI_API_KEY}
ENV PINECONE_API_KEY=${PINECONE_API_KEY}
ENV PINECONE_ENVIRONMENT=${PINECONE_ENVIRONMENT}
ENV PINECONE_INDEX_NAME=${PINECONE_INDEX_NAME}
ENV NEONDB_URL=${NEONDB_URL}

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "agents.query_refiner:app", "--host", "0.0.0.0", "--port", "8000"]
