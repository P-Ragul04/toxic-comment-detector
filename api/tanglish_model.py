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

        return {
            "text": text,
            "is_offensive": offensive_prob >= THRESHOLD,
            "probability": offensive_prob,
        }


tanglish_model = TanglishModel()
