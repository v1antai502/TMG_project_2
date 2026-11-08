"""Clickbait classifier on top of PhoW2V pre-trained word embeddings.

PhoW2V (https://github.com/datquocnguyen/PhoW2V) ships plain word2vec text files such as
`word2vec_vi_words_300dims.txt`. The word-level files expect word-segmented input
("học_sinh"), the same form PhoBERT uses, so both models share `segment_words`.

    vectors = load_phow2v(path, vocab=corpus_words(df))
    tokenizer = PhoW2VTokenizer(vectors.words)
    model = PhoW2VClassifier(vectors.matrix)
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"
SPECIALS = [PAD, UNK, SEP]


@dataclass
class PhoW2VVectors:
    words: list[str]  # specials first, then the kept vocabulary
    matrix: torch.Tensor  # (len(words), dim)
    coverage: float  # share of requested vocabulary found in the file


def find_phow2v(path: str | Path = "", level: str = "words") -> Path:
    """Locate a PhoW2V file. `path` may be the file, a folder, or empty (search Kaggle inputs).

    Prefers the 300-dimensional file when several are present.
    """
    roots = [Path(path)] if path else [Path("/kaggle/input"), Path("data/phow2v")]
    for root in roots:
        if root.is_file():
            return root
        if root.is_dir():
            hits = [
                p
                for p in root.rglob(f"word2vec_vi_{level}_*dims*")
                if p.suffix in (".txt", ".zip")
            ]
            if hits:
                return sorted(hits, key=lambda p: ("300" not in p.name, p.suffix != ".txt"))[0]
    raise FileNotFoundError(
        f"no word2vec_vi_{level}_*dims.txt found under {[str(r) for r in roots]}; "
        "attach a PhoW2V dataset or pass the file path"
    )


def _open_text(path: Path):
    if path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        inner = next(n for n in archive.namelist() if n.endswith(".txt"))
        return io.TextIOWrapper(archive.open(inner), encoding="utf-8", errors="ignore")
    return open(path, encoding="utf-8", errors="ignore")


def corpus_words(*frames, columns: tuple[str, ...] = ("title", "lead_paragraph")) -> set[str]:
    """Every whitespace token of the (word-segmented) text columns."""
    words: set[str] = set()
    for df in frames:
        for col in columns:
            for text in df[col]:
                words.update(text.split())
    return words


def load_phow2v(path: str | Path, vocab: set[str], seed: int = 0) -> PhoW2VVectors:
    """Read only the vectors needed for `vocab` (the full file holds over a million words).

    A word is also kept under its lower-cased form, so "Hà_Nội" can fall back to "hà_nội".
    The special tokens get small random vectors (zeros for padding).
    """
    wanted = vocab | {w.lower() for w in vocab}
    found: dict[str, np.ndarray] = {}
    with _open_text(Path(path)) as f:
        header = f.readline().split()
        dim = int(header[1]) if len(header) == 2 else len(header) - 1
        if len(header) != 2:  # no "count dim" header line: the first line is already a vector
            f.seek(0)
        for line in f:
            word, _, rest = line.partition(" ")
            if word in wanted and word not in found:
                vec = np.array(rest.split(), dtype=np.float32)
                if vec.shape[0] == dim:
                    found[word] = vec

    words = SPECIALS + sorted(found)
    matrix = np.zeros((len(words), dim), dtype=np.float32)
    rng = np.random.default_rng(seed)
    scale = float(np.std(np.stack(list(found.values())))) if found else 0.1
    matrix[1 : len(SPECIALS)] = rng.normal(0, scale, (len(SPECIALS) - 1, dim))
    for i, word in enumerate(words[len(SPECIALS) :], start=len(SPECIALS)):
        matrix[i] = found[word]
    covered = sum(w in found or w.lower() in found for w in vocab)
    return PhoW2VVectors(words, torch.from_numpy(matrix), covered / max(len(vocab), 1))


class PhoW2VTokenizer:
    """Maps word-segmented text to PhoW2V ids. Callable like a Hugging Face tokenizer.

    Title and lead are joined as `title <sep> lead`. Returns `w2v_ids` and `w2v_mask`,
    named so they can sit in the same batch as PhoBERT's `input_ids`.
    """

    def __init__(self, words: list[str]):
        self.words = words
        self.index = {w: i for i, w in enumerate(words)}
        self.pad_id, self.unk_id, self.sep_id = (self.index[t] for t in SPECIALS)

    def __len__(self) -> int:
        return len(self.words)

    def encode(self, text: str) -> list[int]:
        index, unk = self.index, self.unk_id
        return [index.get(w) or index.get(w.lower(), unk) for w in text.split()]

    def __call__(self, titles, leads=None, max_length: int = 128, **_) -> dict:
        rows = []
        for i, title in enumerate(titles):
            ids = self.encode(title)
            if leads is not None and leads[i]:
                ids = ids + [self.sep_id] + self.encode(leads[i])
            rows.append(ids[:max_length] or [self.unk_id])
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = torch.tensor(row)
        return {"w2v_ids": ids, "w2v_mask": (ids != self.pad_id).long()}

    def save_pretrained(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "phow2v_vocab.txt").write_text("\n".join(self.words), encoding="utf-8")


class JointTokenizer:
    """Runs several tokenizers on the same text and merges their outputs into one batch."""

    def __init__(self, *tokenizers):
        self.tokenizers = tokenizers

    def __call__(self, titles, leads=None, **kwargs) -> dict:
        batch = {}
        for tokenizer in self.tokenizers:
            batch.update(tokenizer(titles, leads, **kwargs))
        return batch

    def save_pretrained(self, out_dir: str | Path) -> None:
        for tokenizer in self.tokenizers:
            tokenizer.save_pretrained(out_dir)


class PhoW2VEncoder(nn.Module):
    """PhoW2V embeddings -> BiLSTM -> masked mean + max pooling."""

    def __init__(
        self, vectors: torch.Tensor, hidden: int = 128, dropout: float = 0.3, freeze: bool = True
    ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(vectors.clone(), freeze=freeze, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(vectors.shape[1], hidden, batch_first=True, bidirectional=True)
        self.output_size = 4 * hidden

    def forward(self, w2v_ids: torch.Tensor, w2v_mask: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(w2v_ids))
        lengths = w2v_mask.sum(1).clamp(min=1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=w2v_ids.size(1))
        mask = w2v_mask.unsqueeze(-1).to(out.dtype)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        peak = out.masked_fill(mask == 0, -1e4).max(1).values
        return torch.cat([mean, peak], dim=-1)


class PhoW2VClassifier(nn.Module):
    """Text-only classifier on PhoW2V. `encode` returns the pooled feature for reuse in fusion."""

    def __init__(
        self,
        vectors: torch.Tensor,
        num_labels: int = 2,
        hidden: int = 128,
        dropout: float = 0.3,
        freeze: bool = True,
    ):
        super().__init__()
        self.encoder = PhoW2VEncoder(vectors, hidden, dropout, freeze)
        self.hidden_size = self.encoder.output_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def encode(self, w2v_ids: torch.Tensor, w2v_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(w2v_ids, w2v_mask)

    def forward(self, w2v_ids: torch.Tensor, w2v_mask: torch.Tensor, **_) -> torch.Tensor:
        return self.classifier(self.dropout(self.encode(w2v_ids, w2v_mask)))
