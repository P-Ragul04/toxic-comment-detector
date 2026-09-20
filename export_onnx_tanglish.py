"""
Export the Tanglish offensive language classifier to ONNX format with
int8 quantization, same approach as export_onnx.py for the English
model.

Usage:
    python export_onnx_tanglish.py

Output:
    ./onnx-model-tanglish/model.onnx (quantized)
    ./onnx-model-tanglish/ (tokenizer files)
"""

import os

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "./tanglish-detector-model"
ONNX_DIR = "./onnx-model-tanglish"
MAX_LENGTH = 128


def main():
    os.makedirs(ONNX_DIR, exist_ok=True)

    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    dummy_input = tokenizer(
        "இது ஒரு dummy sentence for ONNX export.",
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
        dynamo=False,
    )

    quantized_path = os.path.join(ONNX_DIR, "model_quantized.onnx")
    print(f"Applying dynamic int8 quantization: {quantized_path}")
    quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QInt8)

    orig_size = os.path.getsize(onnx_path) / (1024 * 1024)
    quant_size = os.path.getsize(quantized_path) / (1024 * 1024)
    print(f"\nOriginal ONNX size:  {orig_size:.1f} MB")
    print(f"Quantized ONNX size: {quant_size:.1f} MB")

    tokenizer.save_pretrained(ONNX_DIR)

    os.remove(onnx_path)
    os.rename(quantized_path, os.path.join(ONNX_DIR, "model.onnx"))

    print(f"\nDone. ONNX deployment files are in: {ONNX_DIR}")
    print("Next: push this folder to Hugging Face Hub.")


if __name__ == "__main__":
    main()
