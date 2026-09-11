"""
Plot training curve (train loss + per-epoch val accuracy) from a train.py log.

Usage:
    python src/plot_curve.py --log output/training.log --out paper/figures/training_curve.png
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE_RE = re.compile(r"^\d\d:\d\d:\d\d \|\s+step (\d+)\s+(.*)$")
PAIR_RE = re.compile(r"(\w+)=([\d.eE-]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default="output/training.log")
    ap.add_argument("--out", type=str, default="paper/figures/training_curve.png")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    steps, losses, val_acc = [], [], []
    val_points = []  # (step, val_acc, color)

    for line in Path(args.log).read_text().splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        step = int(m.group(1))
        kvs = dict(PAIR_RE.findall(m.group(2)))
        if "loss" in kvs:
            steps.append(step)
            losses.append(float(kvs["loss"]))
        if "val_acc" in kvs:
            val_points.append((step, float(kvs["val_acc"])))

    if not steps:
        raise SystemExit(f"no loss steps parsed from {args.log}")

    fig, ax1 = plt.subplots(figsize=(6, 3.8))
    ax1.plot(steps, losses, lw=1.1, color="#1f6fb4", label="train loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("train loss", color="#1f6fb4")
    ax1.tick_params(axis="y", labelcolor="#1f6fb4")
    ax1.legend(loc="center right")

    ax2 = ax1.twinx()
    if val_points:
        vp = list(zip(*val_points))
        ax2.plot(vp[0], vp[1], "o-", lw=1.4, color="#e05252",
                 label="val accuracy")
        ax2.set_ylabel("val accuracy", color="#e05252")
        ax2.tick_params(axis="y", labelcolor="#e05252")
        last = val_points[-1]
        ax2.annotate(f"{last[1]*100:.1f}%", (last[0], last[1]),
                     textcoords="offset points", xytext=(8, -6), fontsize=9,
                     color="#e05252")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right")

    ax1.set_title("Fine-tuning XLM-RoBERTa-base on XNLI Hindi (9% × 2 epochs)")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()