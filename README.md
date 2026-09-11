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

| # | first author | paper | year | where |
|---|---|---|---|---|
| 1 | khader | textual entailment for arabic (lexical + semantic matching) | 2016 | ijcis |
| 2 | dagan | the pascal recognising textual entailment challenge | 2006 | ml challenges |
| 3 | bowman | snli: large annotated corpus for nli | 2015 | emnlp |
| 4 | williams | multinli: broad-coverage challenge corpus | 2018 | naacl-hlt |
| 5 | alabbas | a dataset for arabic textual entailment (arbteds) | 2013 | ranlp |
| 6 | al-khawaldeh | negation + sentiment in arabic rte | 2015 | wcsit |
| 7 | conneau | xlm-roberta: cross-lingual representation learning at scale | 2020 | acl |
| 8 | conneau | xnli: cross-lingual sentence representations | 2018 | emnlp |
| 9 | pakray | textual entailment via lexical + syntactic similarity | 2011 | ijaia |
| 10 | devlin | bert | 2019 | naacl-hlt |
| 11 | vaswani | attention is all you need | 2017 | neurips |
| 12 | liu | roberta | 2019 | arxiv |
| 13 | alabbas | arbte: arabic textual entailment | 2011 | ranlp |
| 14 | alabbas + ramsay | arabic nli with tree edit distance | 2013 | jair |
| 15 | lample + conneau | cross-lingual language model pretraining (xlm) | 2019 | neurips |
| 16 | kudo + richardson | sentencepiece | 2018 | emnlp |
| 17 | wolf | transformers library | 2020 | emnlp |
| 18 | chitale | updesh: instruction data for 13 indic languages | 2026 | acl |
| 19 | jajee | indicmmlu-pro | 2026 | gem workshop |
| 20 | brahma | multilingual tokenization for indic languages | 2026 | findings of acl |
| 21 | loshchilov + hutter | decoupled weight decay (adamw) | 2019 | iclr |
| 22 | kingma + ba | adam | 2015 | iclr |