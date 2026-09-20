"""
Export the fine-tuned model to ONNX format, then apply dynamic int8
quantization on top of that.

Why: the full PyTorch + Transformers runtime has a large memory
footprint just from being imported (hundreds of MB), which doesn't fit
in memory-constrained free hosting tiers (e.g. Render free tier's
512MB limit). ONNX Runtime is a much lighter, inference-only engine -
no autograd, no training machinery - so swapping to it for serving
(while still using full PyTorch for training) solves this.

Usage:
    python export_onnx.py

Output:
    ./onnx-model/model.onnx           (fp32 ONNX model)
    ./onnx-model/model_quantized.onnx (int8 quantized - this is what we deploy)
    ./onnx-model/ (tokenizer files, thresholds.json copied over too)
"""

import json
import os
import shutil

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "./toxic-detector-model"
ONNX_DIR = "./onnx-model"
MAX_LENGTH = 128


def main():
    os.makedirs(ONNX_DIR, exist_ok=True)

    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # Dummy input matching what the model expects, used to trace the
    # model's computation graph for export
    dummy_input = tokenizer(
        "This is a dummy sentence for ONNX export.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    onnx_path = os.path.join(ONNX_DIR, "model.onnx")
    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,  # legacy exporter - more compatible with onnxruntime's quantization tooling
    )

    quantized_path = os.path.join(ONNX_DIR, "model_quantized.onnx")
    print(f"Applying dynamic int8 quantization: {quantized_path}")
    quantize_dynamic(
        onnx_path,
        quantized_path,
        weight_type=QuantType.QInt8,
    )

    orig_size = os.path.getsize(onnx_path) / (1024 * 1024)
    quant_size = os.path.getsize(quantized_path) / (1024 * 1024)
    print(f"\nOriginal ONNX size:  {orig_size:.1f} MB")
    print(f"Quantized ONNX size: {quant_size:.1f} MB")

    # Save tokenizer files alongside the ONNX model
    tokenizer.save_pretrained(ONNX_DIR)

    # Copy thresholds.json over too, if it exists
    thresholds_src = os.path.join(MODEL_DIR, "thresholds.json")
    if os.path.exists(thresholds_src):
        shutil.copy(thresholds_src, os.path.join(ONNX_DIR, "thresholds.json"))
        print("Copied thresholds.json")

    # Remove the unquantized fp32 ONNX file - we only deploy the
    # quantized version, no need to upload both
    os.remove(onnx_path)
    os.rename(quantized_path, os.path.join(ONNX_DIR, "model.onnx"))

    print(f"\nDone. ONNX deployment files are in: {ONNX_DIR}")
    print("Next: push this folder to Hugging Face Hub.")


if __name__ == "__main__":
    main()
