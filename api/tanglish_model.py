"""
Model loading and inference for the Tanglish (romanized Tamil-English)
offensive language classifier. Binary single-label classification
(offensive / not offensive), same ONNX Runtime approach as the English
model - see model.py for design notes.
"""

import os

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MODEL_SOURCE = os.environ.get(
    "TANGLISH_MODEL_SOURCE", "raaagul/tanglish-offensive-detector-onnx"
)
MAX_LENGTH = 128
THRESHOLD = 0.5  # kept as a standard argmax-equivalent cutoff (see project notes on why tuning wasn't applied here)

# Deterministic safety net, layered under the ML model. The model's
# coverage of severe slurs is limited by what survived the Latin-script
# filter in training (see training/train_tanglish.py) - some severe
# terms were likely written more often in native Tamil script in the
# source corpus and underrepresented in what the model actually
# trained on. Real moderation systems commonly pair a probabilistic
# model with a hardcoded blocklist for exactly this reason: the model
# handles nuance and context, the blocklist guarantees known-severe
# terms are never missed regardless of what the model predicts.
# Extend this list as needed - it's intentionally a small, high-confidence
# set rather than an attempt at exhaustive coverage.
SEVERE_TERM_BLOCKLIST = {
    "thevidiya",
    "otha",
    "punda",
    "soothu",
    "ommala",
}


def contains_blocked_term(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in SEVERE_TERM_BLOCKLIST)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class TanglishModel:
    def __init__(self):
        self.tokenizer = None
        self.session = None
        self._loaded = False

    def load(self):
        print(f"Loading Tanglish ONNX model from: {MODEL_SOURCE}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)

        if os.path.isdir(MODEL_SOURCE):
            model_path = os.path.join(MODEL_SOURCE, "model.onnx")
        else:
            model_path = hf_hub_download(repo_id=MODEL_SOURCE, filename="model.onnx")

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            model_path, sess_options=session_options, providers=["CPUExecutionProvider"]
        )

        self._loaded = True
        print("Tanglish ONNX model loaded successfully.")

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
        logits = self.session.run(["logits"], onnx_inputs)[0][0]
        probs = softmax(logits)
        offensive_prob = float(probs[1])

        # Blocklist override: if a known severe term is present, flag
        # it regardless of what the model's probability says. This
        # never un-flags something the model already caught - it only
        # adds coverage, never removes it.
        if contains_blocked_term(text):
            offensive_prob = max(offensive_prob, 0.99)

        return {
            "text": text,
            "is_offensive": offensive_prob >= THRESHOLD,
            "probability": offensive_prob,
        }


tanglish_model = TanglishModel()
