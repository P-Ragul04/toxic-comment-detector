"""
Detailed evaluation of the fine-tuned toxic comment classifier.

Produces per-label precision/recall/F1 and confusion matrices, which
are much more useful to report in your README/writeup than a single
overall accuracy number (especially since this dataset is heavily
imbalanced — most comments are non-toxic).

Usage:
    python evaluate.py
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "./toxic-detector-model"
DATA_DIR = "./jigsaw_data"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_LENGTH = 128
BATCH_SIZE = 32


def load_test_set():
    """Same CSV-loading + -1 filtering logic as train.py, kept in sync
    manually since these are two standalone scripts."""
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    test_labels_df = pd.read_csv(f"{DATA_DIR}/test_labels.csv")
    test_df = test_df.merge(test_labels_df, on="id")
    test_df = test_df[test_df["toxic"] != -1].reset_index(drop=True)
    return test_df


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    test_df = load_test_set()
    comments = test_df["comment_text"].tolist()

    all_probs = []
    all_labels = []

    print(f"Running inference on {len(comments)} test examples...")
    for i in range(0, len(comments), BATCH_SIZE):
        batch_comments = comments[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch_comments,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        batch_labels = test_df[LABELS].iloc[i : i + BATCH_SIZE].values
        all_labels.append(batch_labels)

        if i % (BATCH_SIZE * 20) == 0:
            print(f"  {i}/{len(comments)}")

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = (all_probs >= 0.5).astype(int)

    print("\n=== Per-label classification report ===")
    print(
        classification_report(
            all_labels, all_preds, target_names=LABELS, zero_division=0
        )
    )

    print("=== Per-label confusion matrices ===")
    for idx, label in enumerate(LABELS):
        cm = confusion_matrix(all_labels[:, idx], all_preds[:, idx])
        print(f"\n{label}:")
        print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
        print(f"  FN={cm[1][0]}  TP={cm[1][1]}")


if __name__ == "__main__":
    main()
