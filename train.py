"""
Timing Benchmark: Fine-tune XLM-RoBERTa on XNLI Hindi
Runs 1 epoch and reports detailed timing breakdown.
"""

import time
import logging
import sys
from datetime import datetime, timedelta

import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

# ─────── CONFIG ───────────────────────────────────────────────────────────────
SEED = 42
MODEL_NAME = "xlm-roberta-base"
BATCH_SIZE = 16
MAX_LENGTH = 128
SUBSET_FRACTION = 1.0
NUM_EPOCHS = 1
OUTPUT_DIR = "./timing-run"
# ──────────────────────────────────────────────────────────────────────────────

set_seed(SEED)

# ─────── LOGGING SETUP ────────────────────────────────────────────────────────
class Log:
    def __init__(self):
        self.t0 = time.time()
        self._last = self.t0

    def section(self, title):
        self._blank()
        w = 57
        self._out("━" * w)
        self._out(f"  {title}")
        self._out("━" * w)

    def kv(self, key, value):
        self._out(f"  {key:<20} {value}")

    def ok(self, msg, elapsed=None):
        parts = [f"  ✓ {msg}"]
        if elapsed is not None:
            parts.append(f"  [{elapsed}]")
        self._out("  ".join(parts))

    def step(self, msg):
        self._out(f"  → {msg}")

    def metric(self, name, value, suffix=""):
        self._out(f"  {name:<30} {value:>10}{suffix}")

    def blank(self):
        self._out("")

    def _blank(self):
        self._out("")

    def _out(self, msg):
        now = datetime.now().strftime("%H:%M:%S")
        print(f"{now}  {msg}", flush=True)

log = Log()

def elapsed_str(secs):
    return str(timedelta(seconds=int(secs)))

# ─────── SYSTEM INFO ──────────────────────────────────────────────────────────
log.section("SYSTEM INFO")

if torch.backends.mps.is_available():
    device = "mps"
    log.kv("Device", "MPS (Apple GPU)")
    log.kv("GPU", "Apple M5 (10-core)")
elif torch.cuda.is_available():
    device = "cuda"
    log.kv("Device", "CUDA")
    log.kv("GPU", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    log.kv("Device", "CPU (no GPU detected)")

import psutil, platform
log.kv("Host", platform.node())
log.kv("Python", platform.python_version())
log.kv("PyTorch", torch.__version__)
mem = psutil.virtual_memory()
log.kv("RAM", f"{mem.total / 1e9:.1f} GB total, {mem.available / 1e9:.1f} GB free")
log.kv("Model", MODEL_NAME)
log.kv("Batch", BATCH_SIZE)
log.kv("Max len", MAX_LENGTH)
log.kv("Data fraction", f"{SUBSET_FRACTION*100:.0f}%")
log.kv("Epochs", NUM_EPOCHS)
log.kv("Output", OUTPUT_DIR)

# ─────── LOAD DATASET ─────────────────────────────────────────────────────────
log.section("LOAD DATASET")

log.step("Loading XNLI (Hindi) from Hugging Face …")
t0 = time.time()
dataset = load_dataset("facebook/xnli", "hi")
log.ok(f"Loaded in {elapsed_str(time.time() - t0)}")

train_data = dataset["train"]
val_data = dataset["validation"]

if SUBSET_FRACTION < 1.0:
    n = int(len(train_data) * SUBSET_FRACTION)
    train_data = train_data.select(range(n))
    log.step(f"Subset to {n} samples ({SUBSET_FRACTION*100:.0f}%)")

from collections import Counter
labels = Counter(train_data["label"])
log.kv("Train", f"{len(train_data):>8,} samples  [{labels[0]}E / {labels[1]}N / {labels[2]}C]")
log.kv("Valid", f"{len(val_data):>8,} samples")
log.kv("Test", f"{len(dataset['test']):>8,} samples")

# ─────── TOKENIZE ─────────────────────────────────────────────────────────────
log.section("TOKENIZE")

log.step("Loading tokenizer …")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
log.ok(f"Loaded {tokenizer.__class__.__name__}  |  vocab_size={tokenizer.vocab_size}", elapsed_str(time.time() - t0))

def tokenize_fn(batch):
    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

log.step("Tokenizing training set …")
t0 = time.time()
train_enc = train_data.map(
    tokenize_fn, batched=True, num_proc=2,
    remove_columns=["premise", "hypothesis"],
)
t_tok = time.time() - t0
log.ok(f"{len(train_enc):,} samples  |  {len(train_enc)/t_tok:.0f} samples/s", elapsed_str(t_tok))

log.step("Tokenizing validation set …")
t0 = time.time()
val_enc = val_data.map(
    tokenize_fn, batched=True, num_proc=2,
    remove_columns=["premise", "hypothesis"],
)
log.ok(f"{len(val_enc):,} samples", elapsed_str(time.time() - t0))

# ─────── MODEL ────────────────────────────────────────────────────────────────
log.section("MODEL")

log.step("Loading weights …")
t0 = time.time()
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model = model.to(device)
log.ok(f"Loaded in {elapsed_str(time.time() - t0)}")
log.kv("Parameters", f"{model.num_parameters():,}")
log.kv("Trainable", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

if device == "mps":
    mps_mem = getattr(torch.mps, "current_allocated_memory", None)
    if mps_mem:
        log.kv("GPU memory", f"{mps_mem()/1e6:.0f} MB")

# ─────── TRAINING ─────────────────────────────────────────────────────────────
log.section("TRAINING")

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.argmax(axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    logging_strategy="steps",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="no",
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    fp16=False,
    bf16=False,
    seed=SEED,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_enc,
    eval_dataset=val_enc,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

steps = len(train_enc) // BATCH_SIZE
log.kv("Steps per epoch", f"{steps:,}")
log.kv("Total steps", f"{steps * NUM_EPOCHS:,}")
log.kv("Logging", "every 50 steps")
log.kv("Evaluation", "end of epoch")
log.blank()

time_start = time.time()
log._out(f"{'─'*57}")
log._out(f"  Starting at  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log._out(f"{'─'*57}")
log.blank()

train_result = trainer.train()

time_end = time.time()
elapsed = time_end - time_start

log.blank()
log._out(f"{'─'*57}")
log._out(f"  TRAINING COMPLETE")
log._out(f"{'─'*57}")
log.kv("Duration", elapsed_str(elapsed))
log.kv("Samples", f"{len(train_enc):,}")
log.kv("Throughput", f"{len(train_enc) / elapsed:.1f} samples/s")
log.blank()

# ─────── EVAL ─────────────────────────────────────────────────────────────────
log.section("EVALUATION")

t0 = time.time()
metrics = trainer.evaluate()
t_eval = time.time() - t0

log.ok(f"Done in {elapsed_str(t_eval)}")
log.blank()
log.metric("eval_loss", f"{metrics['eval_loss']:.4f}")
log.metric("accuracy", f"{metrics['eval_accuracy']:.4f}", f"  ({metrics['eval_accuracy']*100:.2f}%)")
log.metric("eval_runtime", f"{metrics['eval_runtime']:.1f}s")
log.metric("samples/sec", f"{metrics['eval_samples_per_second']:.1f}")

# ─────── SAVE ─────────────────────────────────────────────────────────────────
log.section("SAVE MODEL")

model.save_pretrained(f"{OUTPUT_DIR}/model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/model")
log.ok(f"Saved to {OUTPUT_DIR}/model/")

# ─────── SUMMARY ──────────────────────────────────────────────────────────────
log.section("TIMING SUMMARY")

log.metric("Data loading", elapsed_str(t_tok))
log.metric("Tokenization", elapsed_str(t_tok))
log.metric("Model load", "0:00:00" if NUM_EPOCHS == 0 else elapsed_str(t_tok))
log.metric("TRAINING (1 epoch)", elapsed_str(elapsed))
log.metric("Evaluation", elapsed_str(t_eval))
log.blank()
total = elapsed
log.kv("Total wall time", elapsed_str(total))
log.kv("3-epoch estimate", elapsed_str(total * 3))
log.blank()
log._out("Full log written to  timing.log")
