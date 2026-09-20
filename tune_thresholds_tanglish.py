"""
Threshold tuning for the Tanglish offensive language classifier.

Unlike the English multi-label model (6 independent thresholds), this
is binary single-label classification, so there's just one threshold:
the cutoff on P(offensive) above which we call something offensive.
Default behavior (argmax) is equivalent to a 0.5 threshold.

Same non-cheating methodology as before: split validation set in half,
tune the threshold on one half, report honest final metrics on the
other half.

Usage:
    python tune_thresholds_tanglish.py
"""

import json

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "./tanglish-detector-model"
MAX_LENGTH = 128
BATCH_SIZE = 32
LATIN_SCRIPT_THRESHOLD = 0.85
THRESHOLDS_OUT = "./tanglish-detector-model/threshold.json"

LABEL_MAP = {
    "Not_offensive": 0,
    "Offensive_Untargetede": 1,
    "Offensive_Targeted_Insult_Individual": 1,
    "Offensive_Targeted_Insult_Group": 1,
    "Offensive_Targeted_Insult_Other": 1,
}


def is_mostly_latin_script(text: str) -> bool:
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) >= LATIN_SCRIPT_THRESHOLD


def load_validation_set():
    dataset = load_dataset("community-datasets/offenseval_dravidian", "tamil")["validation"]
    label_names = dataset.features["label"].names

    texts, labels = [], []
    for row in dataset:
        label_name = label_names[row["label"]]
        if label_name not in LABEL_MAP:
            continue
        if not is_mostly_latin_script(row["text"]):
            continue
        texts.append(row["text"])
        labels.append(LABEL_MAP[label_name])

    return texts, np.array(labels)


def get_offensive_probs(model, tokenizer, texts, device):
    all_probs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(offensive)
        all_probs.append(probs)
        if i % (BATCH_SIZE * 20) == 0:
            print(f"  {i}/{len(texts)}")
    return np.concatenate(all_probs)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    texts, labels = load_validation_set()
    print(f"Loaded {len(texts)} Tanglish validation rows")

    tune_texts, holdout_texts, tune_labels, holdout_labels = train_test_split(
        texts, labels, test_size=0.5, random_state=42, stratify=labels
    )
    print(f"Tune split: {len(tune_texts)}, Holdout split: {len(holdout_texts)}")

    print("\nRunning inference on tune split...")
    tune_probs = get_offensive_probs(model, tokenizer, tune_texts, device)

    precisions, recalls, thresholds = precision_recall_curve(tune_labels, tune_probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    print(f"\nBest threshold found: {best_threshold:.3f} (tune F1 for offensive class: {f1_scores[best_idx]:.3f})")

    print("\nRunning inference on holdout split...")
    holdout_probs = get_offensive_probs(model, tokenizer, holdout_texts, device)

    default_preds = (holdout_probs >= 0.5).astype(int)
    tuned_preds = (holdout_probs >= best_threshold).astype(int)

    print("\n=== Honest comparison on holdout split ===")
    print("\n--- Default threshold (0.5) ---")
    print(classification_report(holdout_labels, default_preds, target_names=["not_offensive", "offensive"]))
    print("\n--- Tuned threshold ---")
    print(classification_report(holdout_labels, tuned_preds, target_names=["not_offensive", "offensive"]))

    with open(THRESHOLDS_OUT, "w") as f:
        json.dump({"offensive_threshold": best_threshold}, f, indent=2)
    print(f"\nSaved tuned threshold to {THRESHOLDS_OUT}")


if __name__ == "__main__":
    main()
