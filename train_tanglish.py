"""
Fine-tune DistilBERT for Tanglish (romanized Tamil-English code-mixed)
offensive language detection.

Dataset: offenseval_dravidian (Tamil subset), from the DravidianLangTech
shared task (Chakravarthi et al.) - real YouTube comments.

Why DistilBERT still works here: Tanglish is written in Latin script
(e.g. "nee romba mokka nu solren" rather than native Tamil script), so
DistilBERT's existing English-vocabulary tokenizer can still process it
as subword tokens. It won't have any pretrained "understanding" of
Tamil words, but fine-tuning on labeled Tanglish examples teaches it
the relevant vocabulary/patterns directly - same approach as before,
just a different fine-tuning dataset. No architecture change needed.

Filtering: the source dataset mixes native Tamil script, romanized
Tanglish, and pure English comments. We keep only rows that are
predominantly Latin-script, since that's what this classifier targets
and what the tokenizer can meaningfully process. We also drop rows
labeled "not-Tamil" (off-topic/other-language noise in the original
dataset).

Labels simplified to binary (Offensive / Not Offensive) - the original
dataset has finer subtypes (targeted at individual/group/other), which
could be a future extension, but binary keeps this focused for now.

Usage:
    python train_tanglish.py
"""

import re

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
OUTPUT_DIR = "./results_tanglish"
FINAL_MODEL_DIR = "./tanglish-detector-model"

# Rows must be at least this fraction ASCII characters to be kept as
# "Tanglish" (romanized) rather than native-script Tamil
LATIN_SCRIPT_THRESHOLD = 0.85

# Original dataset labels -> binary offensive/not-offensive.
# "not-Tamil" rows (off-topic noise) are dropped entirely.
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


def load_and_prepare_dataset():
    print("Loading offenseval_dravidian (Tamil subset)...")
    dataset = load_dataset("community-datasets/offenseval_dravidian", "tamil")

    def filter_and_map(example):
        label_name = example["label"] if isinstance(example["label"], str) else None
        return label_name in LABEL_MAP if label_name else False

    def process(split):
        df = dataset[split]
        # Dataset stores labels as class indices with names accessible
        # via features - resolve to string names first
        label_names = df.features["label"].names
        texts, labels = [], []
        for row in df:
            label_name = label_names[row["label"]]
            if label_name not in LABEL_MAP:
                continue  # drops "not-Tamil" rows
            text = row["text"]
            if not is_mostly_latin_script(text):
                continue  # drops native Tamil-script rows
            texts.append(text)
            labels.append(LABEL_MAP[label_name])
        return texts, labels

    splits = {}
    for split_name in ["train", "validation" if "validation" in dataset else "test"]:
        if split_name not in dataset:
            continue
        texts, labels = process(split_name)
        splits[split_name] = {"text": texts, "label": labels}
        print(f"{split_name}: {len(texts)} Tanglish rows kept "
              f"({sum(labels)} offensive, {len(labels) - sum(labels)} not offensive)")

    from datasets import Dataset, DatasetDict
    return DatasetDict({
        name: Dataset.from_dict(data) for name, data in splits.items()
    })


def tokenize_dataset(dataset, tokenizer):
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    keep_cols = ["input_ids", "attention_mask", "labels"]
    for split in tokenized.keys():
        tokenized[split] = tokenized[split].remove_columns(
            [c for c in tokenized[split].column_names if c not in keep_cols]
        )
    tokenized.set_format("torch")
    return tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    f1 = f1_score(labels, preds, average="macro")
    return {"f1_macro": f1}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_and_prepare_dataset()
    tokenized = tokenize_dataset(dataset, tokenizer)

    eval_split = "validation" if "validation" in tokenized else "test"

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label={0: "not_offensive", 1: "offensive"},
        label2id={"not_offensive": 0, "offensive": 1},
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=4,  # smaller dataset than the English one, a few more epochs is fine
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=0.1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[eval_split],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training...")
    trainer.train()

    print("Final evaluation:")
    predictions = trainer.predict(tokenized[eval_split])
    preds = np.argmax(predictions.predictions, axis=1)
    print(classification_report(
        predictions.label_ids, preds, target_names=["not_offensive", "offensive"]
    ))

    print(f"Saving final model to {FINAL_MODEL_DIR}")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)


if __name__ == "__main__":
    main()