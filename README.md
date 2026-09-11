# Hindi textual entailment (XLM-RoBERTa on XNLI)

Course assignment adapting Khader et al. (2016) Arabic lexical RTE to Hindi via XLM-RoBERTa fine-tuning.

## Layout

```
src/train.py          fine-tune
src/eval_nli.py       test metrics + confusion matrix
src/inference.py      sample Hindi pairs
paper/main.tex        IEEE two-column paper
paper/references.bib
plan/                 original project-plan LaTeX
results/              metrics.json after eval
output/               saved model (created by training)
```

## Setup

```bash
bash setup.sh
source .venv/bin/activate
```

## Train and evaluate

Run from the repo root.

```bash
python src/train.py --epochs 1 --subset 0.05 --output-dir ./timing-run   # ~25 min check
python src/train.py --output-dir ./output                               # 3 epochs, full data
python src/eval_nli.py --model-dir output/model
python src/inference.py --model-dir output/model
```

Then fill every red `[TBD]` in `paper/main.tex` from `output/val_metrics.json`, `results/metrics.json`, and `output/training.log`.

## Compile the IEEE paper

```bash
make paper    # -> paper/main.pdf
make plan     # project plan PDF
```
