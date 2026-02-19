# ===== Base image =====
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Create app directory
WORKDIR /app

# System deps (if needed by numpy/scikit-learn/xgboost)
# Uncomment if you hit build/runtime issues:
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential g++ \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirement first for layer caching
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code (including models/)
COPY . /app

# Expose FastAPI port
EXPOSE 8000

# Start the API (no --reload in production container)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]