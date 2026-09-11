"""
Fine-tune XLM-RoBERTa on XNLI (Hindi) for textual entailment.

Adapts Khader et al. (2016) --- Arabic lexical/semantic matching ---
to Hindi with transformer fine-tuning (assignment options 1, 3, 4, 5, 6).

Usage:
    python src/train.py                          # paper run: 3 epochs, full data
    python src/train.py --epochs 1 --subset 0.05 # quick check (~25 min on M5)
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta

import evaluate
import platform
import psutil
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_NAME = "xlm-roberta-base"
ID2LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

METRIC_ACC = evaluate.load("accuracy")
METRIC_F1 = evaluate.load("f1")
METRIC_P = evaluate.load("precision")
METRIC_R = evaluate.load("recall")


class Log:
    def __init__(self):
        self.t0 = time.time()

    def section(self, title):
        self._out("")
        self._out("━" * 57)
        self._out(f"  {title}")
        self._out("━" * 57)

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

    def _out(self, msg, also_print=True):
        now = datetime.now().strftime("%H:%M:%S")
        line = f"{now} | {msg}"
        if also_print:
            print(line, flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


class CleanLogCallback(TrainerCallback):
    """Mirror Trainer logs into training.log; tqdm is the live console view."""

    def __init__(self, log: Log):
        self.log = log

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        skip = {
            "epoch", "step", "total_flos", "train_runtime", "train_samples_per_second",
            "train_steps_per_second", "train_loss", "grad_norm",
        }
        bits = []
        if "loss" in logs:
            bits.append(f"loss={logs['loss']:.4f}")
        if "eval_loss" in logs:
            bits.append(f"val_loss={logs['eval_loss']:.4f}")
        if "eval_accuracy" in logs:
            bits.append(f"val_acc={logs['eval_accuracy']:.4f}")
        if "eval_f1" in logs:
            bits.append(f"val_f1={logs['eval_f1']:.4f}")
        if "eval_precision" in logs:
            bits.append(f"val_p={logs['eval_precision']:.4f}")
        if "eval_recall" in logs:
            bits.append(f"val_r={logs['eval_recall']:.4f}")
        if "learning_rate" in logs:
            bits.append(f"lr={logs['learning_rate']:.2e}")
        extra = [
            f"{k}={v}" for k, v in logs.items()
            if k not in skip and k not in {
                "loss", "eval_loss", "eval_accuracy", "eval_f1", "learning_rate",
                "eval_precision", "eval_recall", "eval_runtime",
                "eval_samples_per_second", "eval_steps_per_second",
            }
        ]
        if not (bits or extra):
            return
        line = f"  step {state.global_step:<6} " + "  ".join(bits + extra)
        is_eval = "eval_loss" in logs or "eval_accuracy" in logs
        self.log._out(line, also_print=is_eval)


def e_str(secs):
    return str(timedelta(seconds=int(secs)))


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune XLM-RoBERTa on XNLI Hindi")
    p.add_argument("--epochs", type=int, default=3, help="num train epochs")
    p.add_argument("--subset", type=float, default=1.0, help="fraction of train set")
    p.add_argument("--batch-size", type=int, default=16, help="train batch size")
    p.add_argument("--max-length", type=int, default=128, help="max tokens")
    p.add_argument("--output-dir", type=str, default="./output", help="model output dir")
    p.add_argument("--model-name", type=str, default=MODEL_NAME)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=2e-5)
    return p.parse_args()


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.argmax(axis=-1)
    return {
        "accuracy": METRIC_ACC.compute(predictions=preds, references=labels)["accuracy"],
        "f1": METRIC_F1.compute(predictions=preds, references=labels, average="macro")["f1"],
        "precision": METRIC_P.compute(predictions=preds, references=labels, average="macro")["precision"],
        "recall": METRIC_R.compute(predictions=preds, references=labels, average="macro")["recall"],
    }


def tokenize_batch(tokenizer, max_length):
    def tokenize_fn(batch):
        enc = tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        enc["labels"] = batch["label"]
        return enc

    return tokenize_fn


def main():
    global LOG_FILE
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    LOG_FILE = f"{args.output_dir}/training.log"
    open(LOG_FILE, "w").close()

    set_seed(args.seed)
    log = Log()

    log.section("SYSTEM INFO")
    if DEVICE == "mps":
        log.kv("Device", "MPS (Apple GPU)")
        log.kv("GPU", "Apple Silicon (Metal)")
    elif DEVICE == "cuda":
        log.kv("Device", "CUDA")
        log.kv("GPU", torch.cuda.get_device_name(0))
    else:
        log.kv("Device", "CPU")
    log.kv("Host", platform.node())
    log.kv("Python", platform.python_version())
    log.kv("PyTorch", torch.__version__)
    mem = psutil.virtual_memory()
    log.kv("RAM", f"{mem.total / 1e9:.1f} GB total, {mem.available / 1e9:.1f} GB free")
    log.kv("Model", args.model_name)
    log.kv("Batch", args.batch_size)
    log.kv("Max len", args.max_length)
    log.kv("Data fraction", f"{args.subset * 100:.0f}%")
    log.kv("Epochs", args.epochs)
    log.kv("Output", args.output_dir)

    log.section("LOAD DATASET")
    log.step("Loading XNLI (Hindi) from Hugging Face …")
    t0 = time.time()
    dataset = load_dataset("facebook/xnli", "hi")
    log.ok(f"Loaded in {e_str(time.time() - t0)}")

    train_data = dataset["train"]
    val_data = dataset["validation"]

    if args.subset < 1.0:
        n = int(len(train_data) * args.subset)
        train_data = train_data.select(range(n))
        log.step(f"Subset to {n} samples ({args.subset * 100:.0f}%)")

    labels = Counter(train_data["label"])
    log.kv("Train", f"{len(train_data):>8,} samples  [{labels[0]}E / {labels[1]}N / {labels[2]}C]")
    log.kv("Valid", f"{len(val_data):>8,} samples")
    log.kv("Test", f"{len(dataset['test']):>8,} samples")
    log.kv("Label map", "0=entailment  1=neutral  2=contradiction")

    log.section("TOKENIZE")
    log.step("Loading tokenizer …")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    log.ok(f"{tokenizer.__class__.__name__}  |  vocab_size={tokenizer.vocab_size}", e_str(time.time() - t0))

    tokenize_fn = tokenize_batch(tokenizer, args.max_length)
    drop_cols = train_data.column_names

    t0 = time.time()
    train_enc = train_data.map(
        tokenize_fn, batched=True, num_proc=1, remove_columns=drop_cols,
    )
    t_tok = time.time() - t0
    log.ok(f"{len(train_enc):,} samples  |  {len(train_enc) / max(t_tok, 1e-6):.0f} samples/s", e_str(t_tok))
    log.kv("Columns", ", ".join(train_enc.column_names))

    t0 = time.time()
    val_enc = val_data.map(
        tokenize_fn, batched=True, num_proc=1, remove_columns=val_data.column_names,
    )
    log.ok(f"{len(val_enc):,} samples", e_str(time.time() - t0))

    log.section("MODEL")
    t0 = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        attn_implementation="eager",
    )
    log.ok(f"Loaded in {e_str(time.time() - t0)}")
    log.kv("Parameters", f"{model.num_parameters():,}")
    log.kv("Attention", "eager")
    log.kv("Head", "new 3-way classifier (expected)")

    log.section("TRAINING")
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/trainer",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_steps=0.1,
        weight_decay=0.01,
        optim="adamw_torch",
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        disable_tqdm=False,
        dataloader_pin_memory=False,
        fp16=False,
        bf16=False,
        seed=args.seed,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_enc,
        eval_dataset=val_enc,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[CleanLogCallback(log)],
    )

    steps = max(1, len(train_enc) // args.batch_size)
    log.kv("Steps per epoch", f"{steps:,}")
    log.kv("Total steps", f"{steps * args.epochs:,}")
    log.kv("Learning rate", f"{args.lr}")
    log.kv("Warmup", "10% of steps (transformers v5 warmup_steps float)")
    log.kv("Optim", "adamw_torch")
    log.kv("Eval", "end of epoch")
    log.blank()

    time_start = time.time()
    log._out(f"{'─' * 57}")
    log._out(f"  Starting at  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log._out(f"{'─' * 57}")

    trainer.train()

    elapsed = time.time() - time_start
    log.blank()
    log._out(f"{'─' * 57}")
    log._out("  TRAINING COMPLETE")
    log._out(f"{'─' * 57}")
    log.kv("Duration", e_str(elapsed))
    log.kv("Throughput", f"{len(train_enc) / max(elapsed, 1e-6):.1f} samples/s")

    log.section("VALIDATION EVAL")
    metrics = trainer.evaluate()
    log.metric("eval_accuracy", f"{metrics['eval_accuracy']:.4f}", f"  ({metrics['eval_accuracy'] * 100:.2f}%)")
    log.metric("eval_f1", f"{metrics['eval_f1']:.4f}")
    log.metric("eval_precision", f"{metrics['eval_precision']:.4f}")
    log.metric("eval_recall", f"{metrics['eval_recall']:.4f}")

    metrics_path = f"{args.output_dir}/val_metrics.json"
    dump = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
    with open(metrics_path, "w") as f:
        json.dump(dump, f, indent=2)
    log.ok(f"Wrote {metrics_path}")

    log.section("SAVE MODEL")
    model_path = f"{args.output_dir}/model"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    log.ok(f"Saved to {model_path}/")

    log.section("SUMMARY")
    log.metric("Tokenization", e_str(t_tok))
    log.metric("Training", e_str(elapsed))
    log.kv("Total wall time", e_str(elapsed + t_tok))
    log.blank()
    log.step("Next: python src/eval_nli.py --model-dir " + model_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
