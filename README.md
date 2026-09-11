# hindi textual entailment (xlm-roberta on xnli)

fine-tune `xlm-roberta-base` on xnli-hindi, adapting khader et al. (2016) arabic lexical matching to hindi via transformer fine-tuning.

## results (measured)

| run | split | acc | macro-f1 |
|---|---|---|---|
| sanity 5% / 1 epoch | val | 54.94% | 54.47% |
| sanity 5% / 1 epoch | test | 56.11% | 55.53% |
| reported 9% / 2 epochs | val | 63.86% | 63.67% |
| reported 9% / 2 epochs | test | 63.15% | 62.85% |

zero mock numbers. everything above comes from `output/` and `results/` after the runs.

## layout

```
src/train.py        fine-tune
src/eval_nli.py     test metrics + confusion matrix
src/inference.py    try a saved model on sample pairs
src/plot_curve.py   training curve from a log
paper/main.tex      ieee two-column paper
paper/references.bib
papers/             downloaded reference pdfs (ref-XX.pdf by ref number)
results/            eval outputs
output/             saved model
```

## setup

```bash
bash setup.sh
source .venv/bin/activate
```

## run

```bash
python src/train.py --subset 0.05 --epochs 1 --output-dir output   # quick sanity (~30 min on m5)
python src/train.py --subset 0.09 --epochs 2 --output-dir output   # reported run (~1h42m on m5)
python src/eval_nli.py --model-dir output/model
python src/inference.py --model-dir output/model
```

on apple silicon, use eager attention + unfused adamw (the defaults are set in the code). fused adamw + sdpa collapse to chance on mps - documented in the paper.

## paper

```bash
make paper    # -> paper/main.pdf
```

## references

| # | first author | paper | year | where | pdf |
|---|---|---|---|---|---|
| 1 | khader | textual entailment for arabic (lexical + semantic matching) | 2016 | ijcis | [x] ref-01.pdf |
| 2 | bar-haim | the second pascal recognising textual entailment challenge | 2006 | pascal workshop | [x] ref-02.pdf |
| 3 | bowman | snli: large annotated corpus for nli | 2015 | emnlp | [x] ref-03.pdf |
| 4 | williams | multinli: broad-coverage challenge corpus | 2018 | naacl-hlt | [x] ref-04.pdf |
| 5 | alabbas | a dataset for arabic textual entailment (arbteds) | 2013 | ranlp | [x] ref-05.pdf |
| 6 | al-khawaldeh | negation + sentiment in arabic rte | 2015 | wcsit | [x] ref-06.pdf |
| 7 | conneau | xlm-roberta: cross-lingual representation learning at scale | 2020 | acl | [x] ref-07.pdf |
| 8 | conneau | xnli: cross-lingual sentence representations | 2018 | emnlp | [x] ref-08.pdf |
| 9 | pakray | textual entailment via lexical + syntactic similarity | 2011 | ijaia | [x] ref-09.pdf |
| 10 | devlin | bert | 2019 | naacl-hlt | [x] ref-10.pdf |
| 11 | vaswani | attention is all you need | 2017 | neurips | [x] ref-11.pdf |
| 12 | liu | roberta | 2019 | arxiv | [x] ref-12.pdf |
| 13 | alabbas | arbte: arabic textual entailment | 2011 | ranlp | [x] ref-13.pdf |
| 14 | alabbas + ramsay | arabic nli with tree edit distance | 2013 | jair | [x] ref-14.pdf |
| 15 | lample + conneau | cross-lingual language model pretraining (xlm) | 2019 | neurips | [x] ref-15.pdf |
| 16 | kudo + richardson | sentencepiece | 2018 | emnlp | [x] ref-16.pdf |
| 17 | wolf | transformers library | 2020 | emnlp | [x] ref-17.pdf |
| 18 | jajee | indicmmlu-pro | 2026 | gem workshop | [ ] |
| 19 | brahma | multilingual tokenization for indic languages | 2026 | findings of acl | [ ] |
| 20 | loshchilov + hutter | decoupled weight decay (adamw) | 2019 | iclr | [ ] |
| 21 | kingma + ba | adam | 2015 | iclr | [ ] |