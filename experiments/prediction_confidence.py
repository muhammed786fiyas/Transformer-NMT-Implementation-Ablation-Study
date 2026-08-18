"""
experiments/prediction_confidence.py  —  Phase 2, Experiment 2.5
Logs prediction confidence (softmax prob of correct token) for both
the smoothed (ε=0.1) and unsmoothed (ε=0.0) checkpoints.

Run from repo root:
    python experiments/prediction_confidence.py \
        --smoothed  checkpoints/best.pt \
        --unsmoothed checkpoints/exp25.pt

Both checkpoints are evaluated on the validation set. The mean
confidence per epoch-equivalent batch is logged to W&B as a line plot,
which directly shows label smoothing's effect on model overconfidence.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset, collate_batch, PAD_IDX
from torch.utils.data import DataLoader

WANDB_PROJECT = "da6401-a3"


def compute_confidence(model, loader, device, label):
    """
    For every non-pad target token, compute the softmax probability
    the model assigns to the CORRECT token. Returns list of per-batch
    mean confidences.
    """
    model.eval()
    confidences = []
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            sm = make_src_mask(src, PAD_IDX)
            tm = make_tgt_mask(tgt_in, PAD_IDX)
            logits = model(src, tgt_in, sm, tm)  # [B, T, V]

            probs = F.softmax(logits, dim=-1)    # [B, T, V]
            # gather prob of the correct token at each position
            correct_probs = probs.gather(
                2, tgt_out.unsqueeze(-1).clamp(min=0)).squeeze(-1)  # [B, T]

            # mask out pad positions
            mask = tgt_out != PAD_IDX
            batch_conf = correct_probs[mask].mean().item()
            confidences.append(batch_conf)

    mean_conf = sum(confidences) / len(confidences)
    print(f"{label}: mean prediction confidence = {mean_conf:.4f}")
    return confidences, mean_conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoothed",   default="checkpoints/best.pt",
                        help="checkpoint trained with label smoothing ε=0.1")
    parser.add_argument("--unsmoothed", default="checkpoints/exp25.pt",
                        help="checkpoint trained with label smoothing ε=0.0")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cpu"

    run = wandb.init(project=WANDB_PROJECT, job_type="analysis",
                     name="prediction-confidence-2.5")

    results = {}
    for label, ckpt_path, smoothing in [
        ("smoothed_0.1", args.smoothed,   0.1),
        ("unsmoothed_0.0", args.unsmoothed, 0.0),
    ]:
        if not os.path.exists(ckpt_path):
            print(f"[skip] {ckpt_path} not found")
            continue

        c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sv, tv = c["src_vocab"], c["tgt_vocab"]

        val_ds = Multi30kDataset("validation", sv, tv)
        loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_batch)

        m = Transformer(**c["model_config"])
        m.load_state_dict(c["model_state_dict"])
        m.eval()

        confidences, mean_conf = compute_confidence(m, loader, device, label)
        results[label] = mean_conf

        # Log per-batch confidence as a W&B line
        conf_tbl = wandb.Table(columns=["batch", "confidence", "config"])
        for i, v in enumerate(confidences):
            conf_tbl.add_data(i, v, f"ε={smoothing}")
        wandb.log({f"confidence_per_batch_{label}": conf_tbl})
        wandb.log({f"mean_confidence_{label}": mean_conf})

    # Summary comparison table
    if len(results) == 2:
        tbl = wandb.Table(columns=["config", "mean_confidence"])
        for k, v in results.items():
            tbl.add_data(k, round(v, 4))
        wandb.log({"confidence_summary": tbl})
        diff = results.get("unsmoothed_0.0", 0) - results.get("smoothed_0.1", 0)
        print(f"\nOverconfidence gap (ε=0.0 minus ε=0.1): {diff:+.4f}")
        wandb.log({"overconfidence_gap": diff})

    run.finish()
    print("\nLogged to W&B — add confidence panels to the 2.5 report section.")


if __name__ == "__main__":
    main()