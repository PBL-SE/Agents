# Use an official Python runtime as a parent image
FROM python:3.11-slim as builder

# Set environment variables to avoid interactive prompts during package installs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install only necessary dependencies for building the application
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies and cache them
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Now create the final slim image
FROM python:3.11-slim

# Set environment variables to avoid interactive prompts during package installs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy only necessary files from the builder stage to avoid unnecessary files
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

# Set environment variables (optional but recommended for security reasons)
ENV GOOGLE_API_KEY=${GEMINI_API_KEY}
ENV PINECONE_API_KEY=${PINECONE_API_KEY}
ENV PINECONE_ENVIRONMENT=${PINECONE_ENVIRONMENT}
ENV PINECONE_INDEX_NAME=${PINECONE_INDEX_NAME}
ENV NEONDB_URL=${NEONDB_URL}

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "agents.query_refiner:app", "--host", "0.0.0.0", "--port", "8000"]
