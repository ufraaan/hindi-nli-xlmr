"""
Evaluate a trained XNLI Hindi model.

Usage:
    python src/eval_nli.py                          # uses output/model
    python src/eval_nli.py --model-dir output/model # custom path
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

BATCH_SIZE = 32
MAX_LENGTH = 128
OUTPUT_DIR = Path("results")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained XNLI Hindi model")
    parser.add_argument("--model-dir", type=str, default="output/model")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()

    MODEL_DIR = args.model_dir
    OUT = Path(args.output_dir)
    OUT.mkdir(exist_ok=True)

    if not Path(MODEL_DIR, "config.json").exists():
        print(f"ERROR: no model found at {MODEL_DIR}")
        print("Run: python src/train.py")
        sys.exit(1)

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"device: {device}")
    print(f"loading tokenizer and model from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)

    print("loading xnli hindi test set")
    dataset = load_dataset("facebook/xnli", "hi", split="test")

    def tokenize_fn(batch):
        enc = tokenizer(
            batch["premise"], batch["hypothesis"],
            truncation=True, padding="max_length", max_length=args.max_length,
        )
        enc["labels"] = batch["label"]
        return enc

    enc = dataset.map(
        tokenize_fn, batched=True,
        remove_columns=dataset.column_names,
    )

    t_args = TrainingArguments(
        output_dir="./eval-tmp",
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        disable_tqdm=True,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(model=model, args=t_args, processing_class=tokenizer)

    print("running predictions")
    preds = trainer.predict(enc)
    y_pred = preds.predictions.argmax(axis=-1)
    labels = preds.label_ids

    acc = evaluate.load("accuracy")
    f1m = evaluate.load("f1")
    pr = evaluate.load("precision")
    rc = evaluate.load("recall")

    accuracy = acc.compute(predictions=y_pred, references=labels)["accuracy"]
    precision = pr.compute(predictions=y_pred, references=labels, average="macro")["precision"]
    recall = rc.compute(predictions=y_pred, references=labels, average="macro")["recall"]
    f1 = f1m.compute(predictions=y_pred, references=labels, average="macro")["f1"]

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
        "num_samples": len(labels),
    }

    print(f"\n{'─'*50}")
    print(f"accuracy:   {metrics['accuracy']:.4f}")
    print(f"f1 (macro): {metrics['f1_macro']:.4f}")
    print(f"precision:  {metrics['precision_macro']:.4f}")
    print(f"recall:     {metrics['recall_macro']:.4f}")
    print(f"{'─'*50}")

    with open(OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved {OUT}/metrics.json")

    np.save(OUT / "predictions.npy", y_pred)
    np.save(OUT / "labels.npy", labels)
    print(f"saved {OUT}/predictions.npy, labels.npy")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        labels, y_pred,
        display_labels=["Entailment", "Neutral", "Contradiction"],
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title("Confusion Matrix — XNLI Hindi Test Set")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150)
    print(f"saved {OUT}/confusion_matrix.png")


if __name__ == "__main__":
    main()
