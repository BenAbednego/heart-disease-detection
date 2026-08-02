FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including libomp for xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Expose port 7860 (Hugging Face default port)
EXPOSE 7860

# Run Flask app on port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
