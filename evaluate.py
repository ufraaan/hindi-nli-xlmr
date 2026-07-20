import json
import sys
from pathlib import Path

import torch
import evaluate
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

MODEL_DIR = "./timing-run/model"
BATCH_SIZE = 32
MAX_LENGTH = 128
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"device: {device}")

print("loading tokenizer and model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)

print("loading xnli hindi test set")
dataset = load_dataset("facebook/xnli", "hi", split="test")

def tokenize_fn(batch):
    return tokenizer(batch["premise"], batch["hypothesis"],
                     truncation=True, padding="max_length", max_length=MAX_LENGTH)

enc = dataset.map(tokenize_fn, batched=True, remove_columns=["premise", "hypothesis"])

args = TrainingArguments(
    output_dir="./eval-tmp",
    per_device_eval_batch_size=BATCH_SIZE,
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

trainer = Trainer(model=model, args=args, processing_class=tokenizer)

print("running predictions")
preds = trainer.predict(enc)
logits = preds.predictions
labels = preds.label_ids

y_pred = logits.argmax(axis=-1)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

accuracy = accuracy_metric.compute(predictions=y_pred, references=labels)["accuracy"]
precision = precision_metric.compute(predictions=y_pred, references=labels, average="macro")["precision"]
recall = recall_metric.compute(predictions=y_pred, references=labels, average="macro")["recall"]
f1 = f1_metric.compute(predictions=y_pred, references=labels, average="macro")["f1"]

metrics = {
    "accuracy": round(accuracy, 4),
    "precision_macro": round(precision, 4),
    "recall_macro": round(recall, 4),
    "f1_macro": round(f1, 4),
    "num_samples": len(labels),
}

print(f"accuracy:  {metrics['accuracy']:.4f}")
print(f"f1 (macro): {metrics['f1_macro']:.4f}")
print(f"precision: {metrics['precision_macro']:.4f}")
print(f"recall:    {metrics['recall_macro']:.4f}")

with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"saved metrics.json")

np.save(OUTPUT_DIR / "predictions.npy", y_pred)
np.save(OUTPUT_DIR / "labels.npy", labels)
print(f"saved predictions.npy, labels.npy")

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    labels, y_pred,
    display_labels=["Entailment", "Neutral", "Contradiction"],
    cmap="Blues", ax=ax, colorbar=False,
)
ax.set_title("Confusion Matrix — XNLI Hindi Test Set")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
print(f"saved confusion_matrix.png")
