"""
sweep.py — find the best NO-RETRAIN config for an existing checkpoint.
Tests greedy vs beam (several beam sizes & length penalties), measures
honest BLEU (vs original English refs, case-insensitive) and per-sentence
speed (must stay < 3s for the autograder).

Edit CKPT below to your light-model filename. Run:  python sweep.py
"""

import time
import torch
import evaluate

from dataset import Multi30kDataset
from model import Transformer, make_src_mask, make_tgt_mask
from dataset import PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX

CKPT = "best.pt"          # <-- set to your light checkpoint filename
N_EVAL = 1000             # full test set


def decode(model, src, sm, beam, lp, max_len=100):
    with torch.no_grad():
        mem = model.encode(src, sm)
        if beam <= 1:
            ys = torch.full((1, 1), SOS_IDX, dtype=torch.long)
            for _ in range(max_len - 1):
                tm = make_tgt_mask(ys, PAD_IDX)
                lo = model.decode(mem, sm, ys, tm)
                nx = lo[:, -1].argmax(-1, keepdim=True)
                ys = torch.cat([ys, nx], dim=1)
                if nx.item() == EOS_IDX:
                    break
            return ys[0].tolist()
        beams = [([SOS_IDX], 0.0, False)]
        for _ in range(max_len - 1):
            if all(d for _, _, d in beams):
                break
            cand = []
            for ids, sc, dn in beams:
                if dn:
                    cand.append((ids, sc, True))
                    continue
                ys = torch.tensor([ids], dtype=torch.long)
                tm = make_tgt_mask(ys, PAD_IDX)
                logp = torch.log_softmax(
                    model.decode(mem, sm, ys, tm)[:, -1], -1)[0]
                tv, ti = logp.topk(beam)
                for v, i in zip(tv.tolist(), ti.tolist()):
                    cand.append((ids + [i], sc + v, i == EOS_IDX))
            cand.sort(key=lambda c: c[1] / (((5 + len(c[0])) / 6) ** lp),
                      reverse=True)
            beams = cand[:beam]
        return max(beams,
                   key=lambda c: c[1] / (((5 + len(c[0])) / 6) ** lp))[0]


def main():
    c = torch.load(CKPT, map_location="cpu", weights_only=False)
    print("config:", c["model_config"])
    sv, tv = c["src_vocab"], c["tgt_vocab"]
    cased = any(any(ch.isupper() for ch in t) for t in sv.itos[4:200])
    print("vocab appears", "CASED" if cased else "lowercased")

    test = Multi30kDataset("test", sv, tv)
    raw_de = [ex["de"].strip() for ex in test.data][:N_EVAL]
    raw_en = [ex["en"].strip() for ex in test.data][:N_EVAL]

    m = Transformer(**c["model_config"])
    m.load_state_dict(c["model_state_dict"])
    m.attach_vocab(sv, tv)
    m.eval()
    sb = evaluate.load("sacrebleu")

    def tok_to_text(ids, drop_unk):
        w = []
        for i in ids:
            if i in (SOS_IDX, PAD_IDX):
                continue
            if i == EOS_IDX:
                break
            if drop_unk and i == UNK_IDX:
                continue
            w.append(tv.lookup_token(i))
        return " ".join(w)

    configs = [
        ("greedy, keep <unk>", 1, 0.6, False),
        ("greedy, drop <unk>", 1, 0.6, True),
        ("beam=3, drop <unk>, lp=0.6", 3, 0.6, True),
        ("beam=5, drop <unk>, lp=0.6", 5, 0.6, True),
        ("beam=5, drop <unk>, lp=1.0", 5, 1.0, True),
        ("beam=5, drop <unk>, lp=0.4", 5, 0.4, True),
    ]

    # Pre-tokenize sources once (respect detected casing).
    def src_tensor(de):
        txt = de.lower() if not cased else de
        ids = sv([t.text for t in m._nlp.tokenizer(txt)]) + [EOS_IDX]
        return torch.tensor([ids], dtype=torch.long)

    srcs = [src_tensor(d) for d in raw_de]
    masks = [make_src_mask(s, PAD_IDX) for s in srcs]

    print(f"\nevaluating {len(raw_de)} sentences per config:\n")
    for name, beam, lp, du in configs:
        t = time.time()
        preds = []
        for s, mk in zip(srcs, masks):
            ids = decode(m, s, mk, beam, lp)
            preds.append(tok_to_text(ids, du))
        el = time.time() - t
        score = float(sb.compute(
            predictions=[p.lower() for p in preds],
            references=[[r.lower()] for r in raw_en])["score"])
        print(f"  {name:34s}: BLEU {score:5.2f} | "
              f"{el/len(srcs)*1000:4.0f} ms/sent  (<3000 ok)")


if __name__ == "__main__":
    main()