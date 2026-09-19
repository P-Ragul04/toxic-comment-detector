"""
Model loading and inference logic for the toxic comment detector.

Uses ONNX Runtime instead of full PyTorch for inference. This is
specifically to fit in memory-constrained free hosting tiers (e.g.
Render free tier's 512MB limit) - importing full PyTorch + Transformers'
modeling code alone can eat 300MB+ before a single request is served,
which doesn't leave enough headroom. ONNX Runtime is a much lighter,
inference-only engine (no autograd, no training machinery), and
combined with int8 quantization (done ahead of time in
training/export_onnx.py) the deployed footprint is dramatically
smaller.

Training still uses full PyTorch (see training/train.py) - this
lighter runtime is only for serving.

MODEL_SOURCE should be a Hugging Face Hub repo id containing:
  - model.onnx (the quantized ONNX model)
  - tokenizer files (tokenizer.json, vocab.txt, etc.)
  - thresholds.json (optional - falls back to 0.5 per label if absent)
"""

import json
import os

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "raaagul/toxic-comment-detector-onnx")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_LENGTH = 128
DEFAULT_THRESHOLD = 0.5


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ToxicityModel:
    def __init__(self):
        self.tokenizer = None
        self.session = None
        self.thresholds = {label: DEFAULT_THRESHOLD for label in LABELS}
        self._loaded = False

    def load(self):
        print(f"Loading ONNX model from: {MODEL_SOURCE}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)

        # Resolve the model.onnx file, whether MODEL_SOURCE is a local
        # directory or a Hugging Face Hub repo id
        if os.path.isdir(MODEL_SOURCE):
            model_path = os.path.join(MODEL_SOURCE, "model.onnx")
        else:
            model_path = hf_hub_download(repo_id=MODEL_SOURCE, filename="model.onnx")

        # Single-threaded session options - keeps memory/CPU overhead
        # minimal on constrained hosts, since we're not throughput-bound
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            model_path, sess_options=session_options, providers=["CPUExecutionProvider"]
        )

        self._load_thresholds()
        self._loaded = True
        print("ONNX model loaded successfully.")

    def _load_thresholds(self):
        thresholds_path = None

        if os.path.isdir(MODEL_SOURCE):
            local_path = os.path.join(MODEL_SOURCE, "thresholds.json")
            if os.path.exists(local_path):
                thresholds_path = local_path
        else:
            try:
                thresholds_path = hf_hub_download(
                    repo_id=MODEL_SOURCE, filename="thresholds.json"
                )
            except Exception:
                thresholds_path = None

        if thresholds_path:
            with open(thresholds_path) as f:
                self.thresholds.update(json.load(f))
            print(f"Loaded tuned thresholds: {self.thresholds}")
        else:
            print(
                "WARNING: thresholds.json not found - using default 0.5 "
                "threshold for all labels."
            )

    def is_loaded(self):
        return self._loaded

    def predict(self, text: str) -> dict:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="np",
        )

        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        logits = self.session.run(["logits"], onnx_inputs)[0]
        probs = sigmoid(logits[0])

        scores = []
        any_flagged = False
        for idx, label in enumerate(LABELS):
            prob = float(probs[idx])
            flagged = prob >= self.thresholds[label]
            any_flagged = any_flagged or flagged
            scores.append({"label": label, "probability": prob, "flagged": flagged})

        return {"text": text, "is_toxic": any_flagged, "scores": scores}


# Single shared instance, loaded once at API startup
toxicity_model = ToxicityModel()

