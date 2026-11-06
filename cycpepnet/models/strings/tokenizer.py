import json
import re
from pathlib import Path
import numpy as np

# SMILES tokenization regex
_SMILES_REGEX = re.compile(
    r"(\[[^\]]+\]"      # Bracket atoms: [NH], [C@@H], [nH+], ...
    r"|Br?|Cl?"         # Two-char atoms: Br, Cl  (also B, C)
    r"|N|O|S|P|F|I"     # Single-char atoms
    r"|b|c|n|o|s|p"     # Aromatic atoms
    r"|\(|\)"           # Branches
    r"|\.|=|#"          # Disconnection, double/triple bond
    r"|-|\+"            # Single bond / charge signs
    r"|\\|\/"           # Stereo bonds (E/Z)
    r"|:|@"             # Aromatic bond, stereo fallback
    r"|\%[0-9]{2}"      # Ring-closure %10 … %99
    r"|[0-9])"          # Single-digit ring closures
)

# fragSMILES tokenization regex
_FRAGSMILES_REGEX = re.compile(
    r"(<\d+>"           # Connector tokens: <0>, <1>, <12>, ...
    r"|\[[^\]]+\]"      # Bracket atoms: [NH], [C@@H], [nH+], ...
    r"|Br?|Cl?"         # Two-char atoms: Br, Cl  (also B, C)
    r"|N|O|S|P|F|I"     # Single-char atoms
    r"|b|c|n|o|s|p"     # Aromatic atoms
    r"|\(|\)"           # Branches
    r"|\.|=|#"          # Disconnection, double/triple bond
    r"|-|\+"            # Single bond / charge signs
    r"|\\|\/"           # Stereo bonds (E/Z)
    r"|:|@"             # Aromatic bond, stereo fallback
    r"|\%[0-9]{2}"      # Ring-closure %10 … %99
    r"|[0-9])"          # Single-digit ring closures
)

PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"

_SPECIAL = [PAD, UNK, BOS, EOS]

# Atom-level SMILES tokenizer
class SMILESTokenizer:

    def __init__(self, max_len: int = 1024):
        self.max_len = max_len
        self.vocab: dict[str, int] = {t: i for i, t in enumerate(_SPECIAL)}

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def build_vocab(self, seqs: list[str]) -> None:

        tokens: set[str] = set()

        for s in seqs:
            tokens.update(self._tokenize(s))

        for tok in sorted(tokens):
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_idx(self) -> int:
        return self.vocab[PAD]

    @property
    def bos_idx(self) -> int:
        return self.vocab[BOS]

    @property
    def eos_idx(self) -> int:
        return self.vocab[EOS]

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        data = {"max_len": self.max_len, "vocab": self.vocab}

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SMILESTokenizer":
        with open(path) as f:
            data = json.load(f)
        tok       = cls(max_len=data["max_len"])
        tok.vocab = data["vocab"]
        return tok

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def _tokenize(self, smiles: str) -> list[str]:
        return _SMILES_REGEX.findall(smiles)

    def encode(self, smiles: str) -> list[int]:
        tokens = [BOS] + self._tokenize(smiles) + [EOS]
        ids    = [self.vocab.get(t, self.vocab[UNK]) for t in tokens]

        if len(ids) > self.max_len:
            ids = ids[: self.max_len - 1] + [self.vocab[EOS]]

        ids += [self.vocab[PAD]] * (self.max_len - len(ids))
        return ids

    def encode_batch(self, smiles_list: list[str]) -> np.ndarray:
        return np.array([self.encode(s) for s in smiles_list], dtype=np.int64)

    def decode(self, ids: list[int]) -> str:
        inv    = {v: k for k, v in self.vocab.items()}
        tokens = [inv.get(i, UNK) for i in ids]
        tokens = [t for t in tokens if t not in _SPECIAL]
        return "".join(tokens)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def seq_len_stats(self, smiles_list: list[str]) -> dict:
        lengths = np.array([len(self._tokenize(s)) + 2 for s in smiles_list])
        n_truncated = int(np.sum(lengths > self.max_len))
        return {
            "mean"      : round(float(np.mean(lengths)), 1),
            "median"    : round(float(np.median(lengths)), 1),
            "max"       : int(np.max(lengths)),
            "p95"       : int(np.percentile(lengths, 95)),
            "p99"       : int(np.percentile(lengths, 99)),
            "truncated" : n_truncated,
        }

# fragSMILES tokenizer
class FragSMILESTokenizer(SMILESTokenizer):

    def _tokenize(self, fragsmiles: str) -> list[str]:
        return _FRAGSMILES_REGEX.findall(fragsmiles)
