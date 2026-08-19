FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies for C/Rust extensions if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and pre-indexed vectorstore
COPY . .

EXPOSE 8000

# Run uvicorn on $PORT provided by Railway / container platform (fallback to 8000)
CMD uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}
