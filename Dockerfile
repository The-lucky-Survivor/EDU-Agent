FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Hugging Face default port
EXPOSE 7860

# Run FastAPI via Uvicorn on port 7860
CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]
