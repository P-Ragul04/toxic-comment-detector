"""
Model loading and inference logic for the toxic comment detector.

MODEL_SOURCE can be either:
  - a local directory (e.g. "../toxic-detector-model") - used for local dev
  - a Hugging Face Hub repo id (e.g. "your-username/toxic-comment-detector")
    - used once you've pushed the model, for portable deployment (Docker,
      Hugging Face Spaces, etc. don't have your local training output)

Falls back to a 0.5 threshold for any label missing from thresholds.json
(e.g. if you're running this against a model that hasn't had threshold
tuning run on it yet).
"""

import json
import os

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "../toxic-detector-model")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_LENGTH = 128
DEFAULT_THRESHOLD = 0.5


class ToxicityModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.thresholds = {label: DEFAULT_THRESHOLD for label in LABELS}
        self._loaded = False

    def load(self):
        print(f"Loading model from: {MODEL_SOURCE} (device: {self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE)
        self.model.to(self.device)
        self.model.eval()

        self._load_thresholds()
        self._loaded = True
        print("Model loaded successfully.")

    def _load_thresholds(self):
        """Load tuned per-label thresholds if available, else fall back
        to 0.5 for everything (with a warning, since that's what caused
        the weaker rare-label performance we saw during evaluation)."""
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
                "threshold for all labels. Run tune_thresholds.py and "
                "include the output file with your model for better "
                "results on rare labels."
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
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]

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
