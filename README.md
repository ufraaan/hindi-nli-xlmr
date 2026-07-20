# dl textual entailment project

two-phase plan for a deep learning assignment:

**phase 1:** fine-tune xlm-roberta on xnli hindi for textual entailment. adapts the 2016 khader et al. arabic paper using modern transformers.

**phase 2:** write a technical review paper on knowledge graph construction for indic languages, using ahmed et al. (2022) arabic kg survey as template.

## structure

- `train.py` — training script
- `main.tex` + `sections/` — project plan (27 page pdf via `make`)
- `plan.pdf` — compiled plan

## make

```bash
make        # build main.pdf
make clean  # remove build artifacts
```

## tech

pytorch, transformers, datasets, xlm-roberta, xnli, metal (m5 gpu)
