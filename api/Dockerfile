# Base image with Python. Using slim to keep image size down.
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for torch/transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (onnxruntime instead of full torch -
# much smaller and lighter on memory for inference-only serving)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py model.py tanglish_model.py schemas.py ./

# Models are pulled from Hugging Face Hub at container startup rather
# than baked into the image, so these must point to your HF repos.
ENV MODEL_SOURCE="raaagul/toxic-comment-detector-onnx"
ENV TANGLISH_MODEL_SOURCE="raaagul/tanglish-offensive-detector-onnx"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
