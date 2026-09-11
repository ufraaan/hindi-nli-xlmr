"""
Inference demo for the trained XLM-RoBERTa XNLI Hindi model.

Loads the saved model and predicts entailment on sample Hindi pairs.

Usage:
    python src/inference.py                          # use default model + samples
    python src/inference.py --model-dir output/model # custom model path
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABEL_MAP = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ─────── SAMPLE PAIRS ────────────────────────────────────────────────────────
# Hand-crafted Hindi pairs to demonstrate all three labels.

SAMPLES = [
    # Entailment
    (
        "बिल्ली घर के अंदर सो रही है।",
        "घर में एक जानवर सो रहा है।",
        "Entailment",
    ),
    (
        "राम और श्याम दोनों स्कूल गए।",
        "राम स्कूल गया।",
        "Entailment",
    ),
    # Neutral
    (
        "बारिश हो रही है।",
        "सड़क पर गाड़ियाँ चल रही हैं।",
        "Neutral",
    ),
    (
        "किशोरी ने एक किताब पढ़ी।",
        "किताब में कहानियाँ थीं।",
        "Neutral",
    ),
    # Contradiction
    (
        "आज मौसम बहुत अच्छा है।",
        "आज बहुत भारी बारिश हो रही है।",
        "Contradiction",
    ),
    (
        "बच्चे स्कूल में हैं।",
        "सब बच्चे घर पर हैं।",
        "Contradiction",
    ),
]


def predict(model, tokenizer, premise: str, hypothesis: str) -> str:
    """Return predicted label string."""
    enc = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)
    return LABEL_MAP[probs.argmax().item()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="output/model")
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        print("Run: python src/train.py")
        sys.exit(1)

    print(f"device : {DEVICE}")
    print(f"loading model from {model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()
    print("ready\n")

    print(f"{'PREMISE':<42} {'HYPOTHESIS':<42} {'TRUE':<15} {'PREDICTED'}")
    print("─" * 115)
    correct = 0
    total = len(SAMPLES)
    for premise, hypothesis, gold in SAMPLES:
        pred = predict(model, tokenizer, premise, hypothesis)
        match = "✓" if pred == gold else "✗"
        correct += int(pred == gold)
        print(f"{premise:<42} {hypothesis:<42} {gold:<15} {pred}  {match}")

    print("─" * 115)
    print(f"\nAccuracy on samples: {correct}/{total} ({100 * correct / total:.0f}%)")


if __name__ == "__main__":
    main()