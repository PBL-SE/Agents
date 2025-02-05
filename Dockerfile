# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to avoid interactive prompts during package installs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for pip, gcc, and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies and cache them
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

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
