# Agent notes — Hindi NLI paper metrics

The IEEE paper `paper/main.tex` contains **measured** results from the reported run:
`python src/train.py --epochs 2 --subset 0.09 --output-dir output` (35,343 pairs, ~1h42m on Apple M5).

## Current macros (preamble of `paper/main.tex`)

| Macro | Value | Source |
|---|---|---|
| `\ValAcc` `\ValP` `\ValR` `\ValF` | 63.86 / 66.25 / 63.86 / 63.67% | `output/val_metrics.json` → `eval_accuracy`, `eval_precision`, `eval_recall`, `eval_f1` |
| `\TestAcc` `\TestP` `\TestR` `\TestF` | 63.15 / 64.73 / 63.15 / 62.85% | `results/metrics.json` → `accuracy`, `precision_macro`, `recall_macro`, `f1_macro` |
| `\TrainWall` | 1 h 42 min | `output/training.log` Duration / Total wall time |
| `\TrainRate` | 11.6 samples/s | 2×35,343 / 6,109 s (log prints per-epoch, so recompute for 2 epochs) |
| `\TrainLoss` | 0.850 | epoch-2 mean train loss in the log (final logged 0.789) |
| `\EpOneAcc` `\EpTwoAcc` | 57.35 / 63.86% | per-epoch `val_acc` in the log (2 epochs — no third) |

## Figures (real, already in place)

- `paper/figures/confusion_matrix.png` ← `results/confusion_matrix.png`
- `paper/figures/training_curve.png` ← `src/plot_curve.py --log output/training.log`

To regenerate the curve: `python src/plot_curve.py --log output/training.log --out paper/figures/training_curve.png`

## Do not replace (already real)

- Sanity 5%/1-epoch: val 54.94% / test 56.11% (and P/R/F1 in Table `tab:main`)
- Split sizes 392,702 / 2,490 / 5,010
- 278,045,955 parameters; batch 16; max length 128; lr `2e-5`
- Khader et al. 61% P/R; chance 33.3%; `ln 3` ≈ 1.099
- MPS fused-AdamW/SDPA collapse story

## Remaining before submission

- Optional: delete `Results summary` section (`sec:checklist`) and the end-note appendix