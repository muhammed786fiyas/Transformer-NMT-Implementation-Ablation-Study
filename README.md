# DA6401 Assignment 3 — Transformer NMT (German → English)

**Course:** DA6401 — Deep Learning Applications  
**Student:** Muhammed Fiyas  
**W&B Report:** [View Report](<PASTE_YOUR_WANDB_REPORT_URL_HERE>)

---

## Overview

A from-scratch implementation of the Transformer architecture (Vaswani et al., "Attention Is All You Need", 2017) for German→English neural machine translation on the Multi30k dataset.

**Phase 1 (Autograded):** 45/50  
**Phase 2 (W&B Report):** 5 controlled ablation experiments

---

## Model

Light configuration used for all experiments:

| Hyperparameter | Value |
|---|---|
| d_model | 256 |
| N (layers) | 3 |
| num_heads | 8 |
| d_ff | 1024 |
| dropout | 0.1 |
| label smoothing | 0.1 |
| warmup steps | 4000 |
| batch size | 128 |
| epochs | 15 |

**Baseline test BLEU: 38.17** (sacrebleu, beam=5, length penalty=1.0)

---

## Repository Structure

```
da6401_assignment3/
├── model.py            # Transformer, MHA, PositionalEncoding, LearnedPE
├── dataset.py          # Multi30kDataset, Vocab, collate_batch
├── train.py            # Training loop, LabelSmoothingLoss, evaluate_bleu
├── lr_scheduler.py     # NoamScheduler
├── experiments/
│   └── attention_rollout.py  # Phase 2 Experiment 2.3
├── sweep.py            # Decode-config sweep (beam size, length penalty)
├── checkpoints/
│   └── best.pt         # Trained checkpoint (auto-downloaded in inference)
├── requirements.txt
└── README.md
```

---

## Phase 1 — Implementation

Four graded modules:

- **`lr_scheduler.py`** — Noam scheduler: linear warmup + inverse-sqrt decay
- **`model.py`** — Scaled dot-product attention, Multi-Head Attention (no `nn.MultiheadAttention`), sinusoidal/learned positional encoding, Pre-LN encoder/decoder stacks, full Transformer
- **`dataset.py`** — Multi30k HuggingFace loader, Vocab class (UNK/PAD/SOS/EOS), spaCy tokenization, collate with padding
- **`train.py`** — Label smoothing loss (KL form), teacher-forced training, greedy + beam search decode, sacrebleu evaluation

### Inference

The `Transformer` class supports zero-argument construction for the autograder:

```python
from model import Transformer
m = Transformer()  # downloads checkpoint from Drive, ready for inference
m.infer("Ein Mann sitzt auf einer Bank.")
# → "A man is sitting on a bench."
```

---

## Phase 2 — Experiments

Five controlled ablations, each changing exactly one variable from the baseline. Full analysis in the W&B report linked above.

| Experiment | Test BLEU | vs Baseline | Key Finding |
|---|---|---|---|
| Baseline (Noam, sin PE, ls=0.1) | 38.17 | — | Reference |
| 2.1 Fixed LR (no warmup) | 32.52 | −5.65 | Warmup critical for stable attention training |
| 2.2 No √dₖ scaling | 34.56 | −3.61 | Scaling prevents attention gradient vanishing |
| 2.3 Head specialization | — | — | 3 functional head types; low redundancy (mean sim −0.12) |
| 2.4 Learned PE | 33.37 | −4.80 | Sinusoidal inductive bias wins on small data |
| 2.5 No label smoothing | 37.02 | −1.15 | Smoothing reduces overconfidence |

---

## Setup

```bash
git clone https://github.com/muhammed786fiyas/da6401_assignment3.git
cd da6401_assignment3
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

### Train

```python
from train import run_training_experiment
run_training_experiment(
    num_epochs=15, d_model=256, N=3, num_heads=8,
    d_ff=1024, dropout=0.1, smoothing=0.1,
    warmup_steps=4000, batch_size=128,
)
```

### Inference

```python
from model import Transformer
m = Transformer()
print(m.infer("Eine Frau geht im Park."))
# → "A woman is walking in the park."
```

---

## Dataset

[bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k) — 29,000 train / 1,014 val / 1,000 test sentence pairs (German → English, image captions from Flickr30k).

---

## References

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need.* NeurIPS 2017.