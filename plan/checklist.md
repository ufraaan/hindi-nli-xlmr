# Project Plan — Master Checklist

Two-phase deep learning project (8 weeks).

- Phase 1 (Weeks 1–3): Fine-tune XLM-RoBERTa on XNLI Hindi for textual entailment.
  Adapts Khader et al. (2016) Arabic TE paper using modern transformers.
- Phase 2 (Weeks 4–8): Technical review paper on KG construction for Indic languages,
  using Ahmed et al. (2022) Arabic KG survey as template.

## Status Legend

- [ ] TODO
- [x] DONE
- [~] IN PROGRESS

---

## 0. Setup

- [x] Create Python venv (`.venv`)
- [x] Install pinned deps (`requirements.txt`)
- [x] Verify MPS (Apple GPU) device is available
- [x] Confirm XNLI Hindi loads from Hugging Face

## 1. Phase 1 — NLI Assignment

### 1.1 Code

- [x] Refactor `train.py` — arg-driven fine-tuning pipeline (epochs/subset/output)
- [x] Write `inference.py` — demo prediction on sample Hindi premise/hypothesis pairs
- [x] Write `eval_nli.py` — test evaluation + metrics + confusion matrix (replaces old evaluate.py which shadowed HF evaluate)
- [x] `requirements.txt` / `setup.sh` final review
- [x] `src/plot_curve.py` — training-curve figure from `output/training.log`

### 1.2 Training

- [x] Timing run (1 epoch) → record real throughput on Apple M5
- [x] Reported run (2 epochs, 9% of XNLI-Hindi train = 35,343 pairs) → val 63.86%, test 63.15%
- [x] Save best model + tokenizer to `output/`

### 1.3 Evaluation

- [x] Evaluate on XNLI Hindi test set (5,010 pairs)
- [x] Save `results/metrics.json` (accuracy, precision, recall, F1)
- [x] Save `results/confusion_matrix.png`
- [x] Save predictions/labels (`results/predictions.npy`, `results/labels.npy`)
- [x] Compare against Khader et al. (61% P/R) → 63.15% test acc (beats baseline; below published XLM-R-base XNLI-hi ~72--76% because of 9% data budget)

### 1.4 Inference Demo

- [x] Sample Hindi pairs showing Entailment / Neutral / Contradiction
- [x] Run `inference.py` end-to-end, capture sample output (4/6 correct)

### 1.5 Assignment Paper (`paper/main.tex` → `paper/main.pdf`)

- [x] Abstract (150–200 words)
- [x] Introduction — task, motivation for Hindi, gaps in original work
- [x] Related Work — Khader et al., XLM-RoBERTa, XNLI, cross-lingual NLI
- [x] Dataset — XNLI Hindi: size, sources, label distribution, sample pairs
- [x] Methodology — XLM-RoBERTa architecture, fine-tuning, hyperparameters
- [x] Experiments & Results — metrics, comparison table with original
- [x] Conclusion & Future Work
- [x] References (6–8 key citations)
- [x] Compile to `paper/main.pdf` (8 pages)
- [x] Fill university/college name in `\IEEEauthorblockA` (Lords Institute of Engineering & Technology)

### 1.6 Docs

- [x] `README.md` — setup/usage (exists, verify accuracy after code changes)
- [x] Project plan LaTeX → `main.pdf` (exists, house the plan)

## 2. Phase 2 — KG Survey Paper

- [ ] Read Ahmed et al. (2022) Arabic KG survey as template
- [ ] Collect 40+ Indic KG papers (ACL/EMNLP/NAACL/COLING/LREC/TALLIP)
- [ ] Annotate & categorize papers by language / approach / construction phase
- [ ] Build literature matrix (`lit-matrix.xlsx`)
- [ ] Write draft following template structure
- [ ] Iterate with professor feedback; revise
- [ ] Finalize `kg-survey-paper.pdf` + `bibliography.bib`
- [ ] Submit to target venue

---

## Key Files

| File | Purpose |
|---|---|
| `plan/checklist.md` | Master checklist |
| `src/train.py` | Fine-tuning pipeline |
| `src/eval_nli.py` | Test evaluation + metrics + confusion matrix |
| `src/inference.py` | Demo inference |
| `src/plot_curve.py` | Training-curve figure generator |
| `paper/main.tex` | IEEE assignment paper |
| `output/` | Trained model + tokenizer |
| `results/` | metrics.json, confusion_matrix.png, predictions |