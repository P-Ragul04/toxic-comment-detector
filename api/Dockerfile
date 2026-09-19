# Base image with Python. Using slim to keep image size down.
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for torch/transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch explicitly first (much smaller than the
# default CUDA build, and the container almost certainly has no GPU
# on free hosting tiers). requirements.txt's torch entry will then be
# satisfied already and skipped.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py model.py schemas.py ./

# The model is pulled from Hugging Face Hub at container startup
# rather than baked into the image, so MODEL_SOURCE must be set to
# your HF repo id. This keeps the image small and lets you update the
# model without rebuilding the image.
ENV MODEL_SOURCE="raaagul/toxic-comment-detector"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
