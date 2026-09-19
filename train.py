"""
Fine-tune DistilBERT for multi-label toxic comment classification.

Dataset: Jigsaw Toxic Comment Classification (via Hugging Face)
Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate

Designed to run on a 6GB VRAM GPU (e.g. RTX 3050 Ti) using fp16 mixed
precision and a small batch size with gradient accumulation.

Usage:
    python train.py
"""

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

MODEL_NAME = "distilbert-base-uncased"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_LENGTH = 128
OUTPUT_DIR = "./results"
FINAL_MODEL_DIR = "./toxic-detector-model"
DATA_DIR = "./jigsaw_data"


def load_and_prepare_dataset():
    """Load the Jigsaw toxic comment CSVs directly with pandas (the old
    HF dataset script for this dataset is no longer supported by newer
    `datasets` versions) and format labels as a single multi-hot vector
    column, which is what the model expects for multi-label
    classification.

    Note: test_labels.csv contains -1 for rows that were excluded from
    the original Kaggle competition scoring. We drop those rows since
    they have no real ground truth."""
    print("Loading dataset from CSV...")

    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")

    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    test_labels_df = pd.read_csv(f"{DATA_DIR}/test_labels.csv")
    test_df = test_df.merge(test_labels_df, on="id")
    # Drop rows with -1 labels (excluded from original scoring)
    test_df = test_df[test_df["toxic"] != -1].reset_index(drop=True)

    def add_labels_column(df):
        df = df.copy()
        df["labels"] = df[LABELS].astype(float).values.tolist()
        return df

    train_df = add_labels_column(train_df)
    test_df = add_labels_column(test_df)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df[["comment_text", "labels"]]),
            "test": Dataset.from_pandas(test_df[["comment_text", "labels"]]),
        }
    )
    print(f"Train: {len(dataset['train'])} rows, Test: {len(dataset['test'])} rows")
    return dataset


def tokenize_dataset(dataset, tokenizer):
    def tokenize_fn(batch):
        return tokenizer(
            batch["comment_text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    # Keep only the columns the model needs
    keep_cols = ["input_ids", "attention_mask", "labels"]
    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names if c not in keep_cols]
    )
    tokenized.set_format("torch")
    return tokenized


def compute_metrics(eval_pred):
    """Multi-label metrics: per-label sigmoid + threshold at 0.5,
    then macro F1 and macro ROC-AUC across the 6 labels."""
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)

    try:
        auc_macro = roc_auc_score(labels, probs, average="macro")
    except ValueError:
        # Can happen early in training if a batch has only one class present
        auc_macro = 0.0

    return {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "roc_auc_macro": auc_macro,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: No GPU detected. Training will be very slow.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_and_prepare_dataset()
    tokenized = tokenize_dataset(dataset, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
    )

    # Batch size / accumulation tuned for ~6GB VRAM at max_length=128.
    # Effective batch size = 16 * 2 = 32. Lower per_device_batch_size
    # further if you hit CUDA OOM errors.
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=0.1,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training...")
    trainer.train()

    print("Final evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"Saving final model to {FINAL_MODEL_DIR}")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)


if __name__ == "__main__":
    main()
