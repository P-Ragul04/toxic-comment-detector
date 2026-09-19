"""
Per-label threshold tuning for the toxic comment classifier.

The default 0.5 threshold hurts rare labels (severe_toxic, threat) the
most, since the model's predicted probabilities for those labels tend
to sit lower even when it's "right" in a ranking sense (which is why
ROC-AUC stays high while F1 at threshold=0.5 doesn't).

Methodology: to avoid overfitting thresholds to the same data we
report final metrics on, we split the test set in half:
  - "tune" half  -> used to find the best threshold per label
  - "holdout" half -> used only to report final, honest metrics

Usage:
    python tune_thresholds.py
"""

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "./toxic-detector-model"
DATA_DIR = "./jigsaw_data"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_LENGTH = 128
BATCH_SIZE = 32
THRESHOLDS_OUT = "./toxic-detector-model/thresholds.json"


def load_test_set():
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    test_labels_df = pd.read_csv(f"{DATA_DIR}/test_labels.csv")
    test_df = test_df.merge(test_labels_df, on="id")
    test_df = test_df[test_df["toxic"] != -1].reset_index(drop=True)
    return test_df


def get_predictions(model, tokenizer, comments, device):
    all_probs = []
    for i in range(0, len(comments), BATCH_SIZE):
        batch = comments[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        if i % (BATCH_SIZE * 20) == 0:
            print(f"  {i}/{len(comments)}")
    return np.concatenate(all_probs, axis=0)


def find_best_threshold(y_true, y_prob):
    """Sweep thresholds along the precision-recall curve and return
    the one that maximizes F1 for this label."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thresholds) = len(precisions) - 1
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])  # drop last point (no threshold)
    return thresholds[best_idx], f1_scores[best_idx]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    test_df = load_test_set()

    tune_df, holdout_df = train_test_split(
        test_df, test_size=0.5, random_state=42
    )
    tune_df = tune_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)
    print(f"Tune split: {len(tune_df)}, Holdout split: {len(holdout_df)}")

    print("\nRunning inference on tune split...")
    tune_probs = get_predictions(model, tokenizer, tune_df["comment_text"].tolist(), device)

    print("\nFinding best threshold per label...")
    best_thresholds = {}
    for idx, label in enumerate(LABELS):
        y_true = tune_df[label].values
        y_prob = tune_probs[:, idx]
        threshold, f1 = find_best_threshold(y_true, y_prob)
        best_thresholds[label] = float(threshold)
        print(f"  {label}: threshold={threshold:.3f} (tune F1={f1:.3f}, default 0.5 F1={f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0):.3f})")

    print("\nRunning inference on holdout split...")
    holdout_probs = get_predictions(model, tokenizer, holdout_df["comment_text"].tolist(), device)
    holdout_labels = holdout_df[LABELS].values

    print("\n=== Final honest metrics on holdout split ===")
    default_preds = (holdout_probs >= 0.5).astype(int)
    tuned_preds = np.zeros_like(default_preds)
    for idx, label in enumerate(LABELS):
        tuned_preds[:, idx] = (holdout_probs[:, idx] >= best_thresholds[label]).astype(int)

    print(f"\nMacro F1 @ threshold=0.5:  {f1_score(holdout_labels, default_preds, average='macro', zero_division=0):.4f}")
    print(f"Macro F1 @ tuned thresholds: {f1_score(holdout_labels, tuned_preds, average='macro', zero_division=0):.4f}")
    print(f"\nMicro F1 @ threshold=0.5:  {f1_score(holdout_labels, default_preds, average='micro', zero_division=0):.4f}")
    print(f"Micro F1 @ tuned thresholds: {f1_score(holdout_labels, tuned_preds, average='micro', zero_division=0):.4f}")

    print("\nPer-label F1 comparison (holdout, honest):")
    for idx, label in enumerate(LABELS):
        f1_default = f1_score(holdout_labels[:, idx], default_preds[:, idx], zero_division=0)
        f1_tuned = f1_score(holdout_labels[:, idx], tuned_preds[:, idx], zero_division=0)
        print(f"  {label}: 0.5 -> {f1_default:.3f}   tuned -> {f1_tuned:.3f}   (threshold={best_thresholds[label]:.3f})")

    with open(THRESHOLDS_OUT, "w") as f:
        json.dump(best_thresholds, f, indent=2)
    print(f"\nSaved tuned thresholds to {THRESHOLDS_OUT}")
    print("The API in Week 2 will load this file and use these thresholds instead of a blanket 0.5.")


if __name__ == "__main__":
    main()
