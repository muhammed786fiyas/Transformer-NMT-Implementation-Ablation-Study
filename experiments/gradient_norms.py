"""
experiments/gradient_norms.py  —  Phase 2, Experiment 2.2
Logs the L2 gradient norms of the Query (W_q) and Key (W_k) weight
matrices for the first 1,000 training steps, for both:
  - scaled attention   (use_scaling=True,  standard)
  - unscaled attention (use_scaling=False, ablation)

This is the empirical evidence for Section 3.2.1 of the paper: without
1/√dₖ scaling, dot products grow large, softmax saturates, and the
gradients flowing back through W_q and W_k become very small
(vanishing gradient in the attention layer).

Run from repo root on Kaggle:
    python experiments/gradient_norms.py

Takes ~10-15 min for 1000 steps on GPU.
"""

import os
import sys

import torch
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset, collate_batch, PAD_IDX, SOS_IDX, EOS_IDX
from train import LabelSmoothingLoss
from lr_scheduler import NoamScheduler
from torch.utils.data import DataLoader

WANDB_PROJECT = "da6401-a3"
MAX_STEPS = 1000
D_MODEL = 256
N = 3
NUM_HEADS = 8
D_FF = 1024
DROPOUT = 0.1
SMOOTHING = 0.1
WARMUP = 4000
BATCH_SIZE = 128


def collect_grad_norms(use_scaling: bool, run_name: str):
    """Train for MAX_STEPS, logging Q/K grad norms every step."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = Multi30kDataset("train")
    sv, tv = train_ds.build_vocab()
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_batch)

    model_config = {
        "src_vocab_size": len(sv), "tgt_vocab_size": len(tv),
        "d_model": D_MODEL, "N": N, "num_heads": NUM_HEADS,
        "d_ff": D_FF, "dropout": DROPOUT,
    }
    model = Transformer(**model_config, use_scaling=use_scaling).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0,
                                 betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=D_MODEL,
                              warmup_steps=WARMUP)
    loss_fn = LabelSmoothingLoss(len(tv), PAD_IDX, SMOOTHING)

    run = wandb.init(project=WANDB_PROJECT, job_type="grad_norm",
                     name=run_name, reinit=True)
    wandb.config.update({"use_scaling": use_scaling, "max_steps": MAX_STEPS,
                         **model_config})

    step = 0
    model.train()
    outer_break = False

    while not outer_break:
        for src, tgt in loader:
            if step >= MAX_STEPS:
                outer_break = True
                break

            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            sm = make_src_mask(src, PAD_IDX)
            tm = make_tgt_mask(tgt_in, PAD_IDX)

            logits = model(src, tgt_in, sm, tm)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)),
                           tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()

            # Collect Q and K grad norms from ALL encoder + decoder layers
            q_norms, k_norms = [], []
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                if "W_q.weight" in name:
                    q_norms.append(param.grad.norm().item())
                elif "W_k.weight" in name:
                    k_norms.append(param.grad.norm().item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if q_norms and k_norms:
                wandb.log({
                    "step": step,
                    "loss": loss.item(),
                    "mean_q_grad_norm": sum(q_norms) / len(q_norms),
                    "mean_k_grad_norm": sum(k_norms) / len(k_norms),
                    "use_scaling": int(use_scaling),
                })

            step += 1

    print(f"{run_name}: finished {step} steps")
    run.finish()


def main():
    print("Running SCALED attention (use_scaling=True)...")
    collect_grad_norms(use_scaling=True,
                       run_name="exp-2.2:grad_norms_with_scaling")

    print("\nRunning UNSCALED attention (use_scaling=False)...")
    collect_grad_norms(use_scaling=False,
                       run_name="exp-2.2:grad_norms_no_scaling")

    print("\nDone. Add the mean_q_grad_norm and mean_k_grad_norm line plots "
          "to the 2.2 section of the report, overlaying both runs.")


if __name__ == "__main__":
    main()