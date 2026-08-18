"""
experiments/attention_rollout.py  —  Phase 2, Experiment 2.3
Head specialization & redundancy, last encoder layer.

- Heatmaps: one representative sentence (native W&B panels).
- Stats: AGGREGATED over N_AGG test sentences so specialization /
  redundancy findings are systematic, not anecdotal.

Run from repo root:  python experiments/attention_rollout.py
"""

import os
import sys

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Transformer, make_src_mask
from dataset import Multi30kDataset, PAD_IDX, EOS_IDX

CKPT = "checkpoints/best.pt"
SENT_DE = "Ein Mann mit einem orangefarbenen Hut starrt etwas an ."
N_AGG = 100          # sentences to average head-stats over
WANDB_PROJECT = "da6401-a3"


def log_heatmap(key, x, y, mat):
    tbl = wandb.Table(columns=["query", "key", "weight"])
    for i, yl in enumerate(y):
        for j, xl in enumerate(x):
            tbl.add_data(f"{i}:{yl}", f"{j}:{xl}", float(mat[i][j]))
    wandb.log({key + "_data": tbl})
    try:
        wandb.log({key: wandb.plot_table(
            "wandb/heatmap/v0", tbl,
            {"x": "key", "y": "query", "value": "weight"},
            {"title": key})})
    except Exception:
        pass


def main():
    c = torch.load(CKPT, map_location="cpu", weights_only=False)
    sv, tv = c["src_vocab"], c["tgt_vocab"]
    m = Transformer(**c["model_config"])
    m.load_state_dict(c["model_state_dict"])
    m.attach_vocab(sv, tv)
    m.eval()
    cased = any(any(ch.isupper() for ch in t) for t in sv.itos[4:200])

    def encode_attn(de_text):
        txt = de_text if cased else de_text.lower()
        toks = [t.text for t in m._nlp.tokenizer(txt.strip())]
        ids = sv(toks) + [EOS_IDX]
        src = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            m.encode(src, make_src_mask(src, PAD_IDX))
        return m.encoder.layers[-1].self_attn.attn_weights[0].numpy(), toks

    run = wandb.init(project=WANDB_PROJECT, job_type="analysis",
                     name="attention-rollout-2.3")

    # ---- heatmaps for one representative sentence ----
    A0, toks0 = encode_attn(SENT_DE)
    H = A0.shape[0]
    labels = toks0 + ["<eos>"]

    # Log as wandb.Image (matplotlib, programmatically generated from
    # real model weights — satisfies "log a heat map for each head").
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for h in range(H):
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(A0[h], cmap="viridis", vmin=0, vmax=A0[h].max())
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_title(f"Last encoder layer — head {h}")
            ax.set_xlabel("key"); ax.set_ylabel("query")
            fig.colorbar(im, fraction=0.046, pad=0.04)
            fig.tight_layout()
            wandb.log({f"attn_head_{h}_heatmap": wandb.Image(fig)})
            plt.close(fig)
        print(f"logged {H} attention heatmaps")
    except Exception as e:
        print(f"[warn] heatmap image failed: {e}")

    # Also log via plot_table as fallback data
    for h in range(H):
        log_heatmap(f"attn_head_{h}_last_enc", labels, labels,
                    A0[h].tolist())

    # ---- aggregate stats over N_AGG test sentences ----
    test = Multi30kDataset("test", sv, tv)
    raw_de = [ex["de"].strip() for ex in test.data][:N_AGG]
    acc = np.zeros((H, 4))  # diag, next, prev, entropy
    cnt = 0
    for de in raw_de:
        A, _ = encode_attn(de)
        S = A.shape[1]
        if S < 3:
            continue
        eye, nx, pv = np.eye(S), np.eye(S, k=1), np.eye(S, k=-1)
        for h in range(H):
            Ah = A[h]
            acc[h, 0] += (Ah * eye).sum() / S
            acc[h, 1] += (Ah * nx).sum() / (S - 1)
            acc[h, 2] += (Ah * pv).sum() / (S - 1)
            acc[h, 3] += -(Ah * np.log(Ah + 1e-12)).sum(1).mean()
        cnt += 1
    acc /= cnt

    stats = wandb.Table(columns=["head", "diag_mass", "next_mass",
                                 "prev_mass", "entropy"])
    print(f"aggregated over {cnt} sentences")
    print(f"{'head':>4} {'diag':>6} {'next':>6} {'prev':>6} {'entropy':>8}")
    for h in range(H):
        d, n, p, e = acc[h]
        stats.add_data(h, round(d, 3), round(n, 3), round(p, 3),
                       round(e, 3))
        print(f"{h:>4} {d:6.3f} {n:6.3f} {p:6.3f} {e:8.3f}")
    wandb.log({"head_specialization_agg": stats})

    # Redundancy from the fixed-size behavioural profile (diag,next,prev,
    # entropy) — sentence-length invariant, so averagable across sentences.
    prof = acc.copy()
    prof = (prof - prof.mean(0)) / (prof.std(0) + 1e-12)
    prof = prof / (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-12)
    sim = prof @ prof.T
    iu = np.triu_indices(H, k=1)
    ms = float(sim[iu].mean())
    wandb.log({"mean_offdiag_head_similarity": ms})
    log_heatmap("head_similarity_heatmap",
                [f"h{j}" for j in range(H)],
                [f"h{j}" for j in range(H)], sim.tolist())
    print(f"\nmean off-diagonal head similarity: {ms:.3f}")
    np.set_printoptions(precision=2, suppress=True)
    print(sim)
    run.finish()
    print("\nLogged: attention-rollout-2.3")


if __name__ == "__main__":
    main()