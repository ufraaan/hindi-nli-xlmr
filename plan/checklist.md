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

- [ ] Create Python venv (`.venv`)
- [ ] Install pinned deps (`requirements.txt`)
- [ ] Verify MPS (Apple GPU) device is available
- [ ] Confirm XNLI Hindi loads from Hugging Face

## 1. Phase 1 — NLI Assignment

### 1.1 Code

- [x] Refactor `train.py` — arg-driven fine-tuning pipeline (epochs/subset/output)
- [ ] Write `inference.py` — demo prediction on sample Hindi premise/hypothesis pairs
- [ ] Write `eval_nli.py` — test evaluation + metrics + confusion matrix (replaces old evaluate.py which shadowed HF evaluate)
- [ ] `requirements.txt` / `setup.sh` final review

### 1.2 Training

- [ ] Timing run (1 epoch) → record real throughput on Apple M5
- [ ] Full run (3 epochs) on XNLI Hindi train split
- [ ] Save best model + tokenizer to `output/`

### 1.3 Evaluation

- [ ] Evaluate on XNLI Hindi test set (5,010 pairs)
- [ ] Save `results/metrics.json` (accuracy, precision, recall, F1)
- [ ] Save `results/confusion_matrix.png`
- [ ] Save predictions/labels (`results/predictions.npy`, `results/labels.npy`)
- [ ] Compare against Khader et al. (61% P/R) → XLM-R-base XNLI-hi ~72--76%

### 1.4 Inference Demo

- [ ] Sample Hindi pairs showing Entailment / Neutral / Contradiction
- [ ] Run `inference.py` end-to-end, capture sample output

### 1.5 Assignment Paper (`nli-paper.tex` → `nli-paper.pdf`)

- [ ] Abstract (150–200 words)
- [ ] Introduction — task, motivation for Hindi, gaps in original work
- [ ] Related Work — Khader et al., XLM-RoBERTa, XNLI, cross-lingual NLI
- [ ] Dataset — XNLI Hindi: size, sources, label distribution, sample pairs
- [ ] Methodology — XLM-RoBERTa architecture, fine-tuning, hyperparameters
- [ ] Experiments & Results — metrics, comparison table with original
- [ ] Conclusion & Future Work
- [ ] References (6–8 key citations)
- [ ] Compile to `nli-paper.pdf`

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
| `paper/main.tex` | IEEE assignment paper |
| `output/` | Trained model + tokenizer |
| `results/` | metrics.json, confusion_matrix.png, predictions |