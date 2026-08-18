"""
dataset.py - Multi30k loading, spaCy tokenization, vocab, numericalization.
DA6401 Assignment 3.

Special token indices are fixed:
    <unk>=0  <pad>=1  <sos>=2  <eos>=3
<pad> must stay at index 1 because make_src_mask / make_tgt_mask in model.py
default to pad_idx=1.

Source (de): tokens + <eos>
Target (en): <sos> + tokens + <eos>
Vocab is built from the TRAIN split only (no val/test leakage), min freq = 2.
"""

from collections import Counter

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# NOTE: `datasets` and `spacy` are imported lazily inside the methods that
# need them. This keeps dataset.py importable in minimal environments (e.g.
# the autograder, which lacks `datasets`) so that unpickling the checkpoint
# — which references dataset.Vocab — succeeds even there.


UNK, PAD, SOS, EOS = "<unk>", "<pad>", "<sos>", "<eos>"
SPECIALS = [UNK, PAD, SOS, EOS]
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3


class Vocab:
    """Minimal token <-> index map. Mirrors the torchtext Vocab interface
    (stoi / itos / lookup_token) so train.py and the autograder can use
    whichever accessor they expect."""

    def __init__(self, counter, min_freq):
        self.itos = list(SPECIALS)
        for token, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq >= min_freq:
                self.itos.append(token)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __call__(self, tokens):
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def lookup_token(self, idx):
        return self.itos[idx]


class Multi30kDataset(Dataset):
    """German -> English pairs from bentrevett/multi30k.

    The first time a split is built with build_vocab(), the resulting
    src_vocab / tgt_vocab should be shared with the val/test instances so
    every split maps tokens to the same indices.
    """

    def __init__(self, split="train", src_vocab=None, tgt_vocab=None,
                 min_freq=2):
        self.split = split
        self.min_freq = min_freq

        # Lazy imports: only needed when actually building a dataset, not
        # when dataset.py is merely imported (e.g. to unpickle Vocab).
        from datasets import load_dataset
        import spacy

        hf_split = "validation" if split in ("val", "validation") else split
        self.data = load_dataset("bentrevett/multi30k", split=hf_split)

        # de_core_news_sm = German source, en_core_web_sm = English target.
        # Disable the pipeline components we don't need so tokenization is fast.
        disable = ["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"]
        self.spacy_de = spacy.load("de_core_news_sm", disable=disable)
        self.spacy_en = spacy.load("en_core_web_sm", disable=disable)

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.samples = None

        if src_vocab is not None and tgt_vocab is not None:
            self.process_data()

    # -- tokenizers ----------------------------------------------------
    # Lowercase before tokenizing. German capitalizes all nouns, so a
    # cased vocab is brittle: any casing change in the input maps words
    # to <unk> and the encoder sees noise (verified: lowercased input
    # dropped BLEU 37 -> 10). Training+inference on lowercased text makes
    # the model case-invariant. BLEU is ~unchanged (scored case-insens.).
    def _tok_de(self, text):
        return [t.text for t in self.spacy_de.tokenizer(text.strip().lower())]

    def _tok_en(self, text):
        return [t.text for t in self.spacy_en.tokenizer(text.strip().lower())]

    # -- vocab ---------------------------------------------------------
    def build_vocab(self):
        """Build src/tgt vocab from THIS split (call on train only).
        Returns (src_vocab, tgt_vocab) so the other splits can reuse them."""
        if self.split != "train":
            raise ValueError("build_vocab should be called on the train split")

        src_counter, tgt_counter = Counter(), Counter()
        for ex in self.data:
            src_counter.update(self._tok_de(ex["de"]))
            tgt_counter.update(self._tok_en(ex["en"]))

        self.src_vocab = Vocab(src_counter, self.min_freq)
        self.tgt_vocab = Vocab(tgt_counter, self.min_freq)
        self.process_data()
        return self.src_vocab, self.tgt_vocab

    # -- numericalization ---------------------------------------------
    def process_data(self):
        """Tokenize every pair and convert to index tensors.
        src = de + <eos>;  tgt = <sos> + en + <eos>."""
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError("Vocab not set - call build_vocab() first "
                               "or pass src_vocab/tgt_vocab to __init__")

        samples = []
        raw_tgt = []
        for ex in self.data:
            src_ids = self.src_vocab(self._tok_de(ex["de"])) + [EOS_IDX]
            tgt_ids = [SOS_IDX] + self.tgt_vocab(self._tok_en(ex["en"])) + [EOS_IDX]
            samples.append((torch.tensor(src_ids, dtype=torch.long),
                            torch.tensor(tgt_ids, dtype=torch.long)))
            # Keep the ORIGINAL English string (no vocab round-trip, no
            # <unk>). Honest BLEU must score against this, not against the
            # numericalized target which has rare words replaced by <unk>.
            raw_tgt.append(ex["en"].strip())
        self.samples = samples
        self.raw_tgt = raw_tgt
        return samples

    # -- Dataset protocol ---------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_batch(batch):
    """Pad a list of (src, tgt) tensors to the longest in the batch.
    Padding value is PAD_IDX so the model's mask helpers zero them out.
    Returns src [B, src_len], tgt [B, tgt_len]."""
    src_list, tgt_list = zip(*batch)
    src = pad_sequence(src_list, batch_first=True, padding_value=PAD_IDX)
    tgt = pad_sequence(tgt_list, batch_first=True, padding_value=PAD_IDX)
    return src, tgt


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train = Multi30kDataset("train")
    src_vocab, tgt_vocab = train.build_vocab()
    val = Multi30kDataset("validation", src_vocab, tgt_vocab)

    print(f"train pairs : {len(train)}")
    print(f"val pairs   : {len(val)}")
    print(f"de vocab    : {len(src_vocab)}")
    print(f"en vocab    : {len(tgt_vocab)}")
    print(f"specials    : unk={UNK_IDX} pad={PAD_IDX} "
          f"sos={SOS_IDX} eos={EOS_IDX}")

    loader = DataLoader(train, batch_size=4, shuffle=False,
                        collate_fn=collate_batch)
    src, tgt = next(iter(loader))
    print(f"\nbatch src shape: {tuple(src.shape)}")
    print(f"batch tgt shape: {tuple(tgt.shape)}")
    print("src[0] ids :", src[0].tolist())
    print("tgt[0] ids :", tgt[0].tolist())
    print("tgt[0] back:",
          " ".join(tgt_vocab.lookup_token(i) for i in tgt[0].tolist()))