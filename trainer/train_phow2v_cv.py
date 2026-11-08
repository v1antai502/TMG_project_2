"""K-fold cross-validation of the PhoW2V models, on the same folds as `train_phobert_cv`.

    python -m trainer.train_phow2v_cv --mode phow2v --w2v-path path/to/word2vec_vi_words_300dims.txt
    python -m trainer.train_phow2v_cv --mode phobert_phow2v --lr 2e-5 --epochs 5

`mode`:
    phow2v           PhoW2V embeddings + BiLSTM only (`lr` applies to the whole model)
    phobert_phow2v   PhoBERT and PhoW2V features concatenated; PhoBERT trains at `lr`,
                     the PhoW2V branch and the head at `w2v_lr`
"""

from __future__ import annotations

from dataclasses import dataclass

from architecture.phobert import load_tokenizer
from architecture.phobert_phow2v import NEW_MODULES, PhoBERTPhoW2VClassifier
from architecture.phow2v import (
    JointTokenizer,
    PhoW2VClassifier,
    PhoW2VTokenizer,
    corpus_words,
    find_phow2v,
    load_phow2v,
)
from trainer import train_phobert_cv
from trainer.train_phobert import ROOT, parse_config
from trainer.train_phobert_cv import CVConfig


@dataclass
class W2VConfig(CVConfig):
    out_dir: str = str(ROOT / "outputs" / "phow2v_cv")
    mode: str = "phow2v"
    w2v_path: str = ""  # file or folder; empty = search /kaggle/input and data/phow2v
    w2v_hidden: int = 128
    w2v_dropout: float = 0.3
    w2v_freeze: bool = True  # keep the pre-trained vectors fixed
    w2v_lr: float = 1e-3
    lr: float = 1e-3
    epochs: int = 15
    monitor: str = "loss"


def setup(cfg: W2VConfig, pool, test):
    """Tokenizer, model factory and extra Trainer arguments for `train_phobert_cv.run`."""
    path = find_phow2v(cfg.w2v_path)
    vectors = load_phow2v(path, corpus_words(pool, test), seed=cfg.seed)
    print(
        f"PhoW2V: {path.name} | {len(vectors.words)} vectors of dim {vectors.matrix.shape[1]} "
        f"| covers {vectors.coverage:.1%} of the dataset vocabulary"
    )
    w2v_tokenizer = PhoW2VTokenizer(vectors.words)

    if cfg.mode == "phow2v":
        def make_model():
            return PhoW2VClassifier(
                vectors.matrix, hidden=cfg.w2v_hidden, dropout=cfg.w2v_dropout, freeze=cfg.w2v_freeze
            )

        return w2v_tokenizer, make_model, {}

    if cfg.mode == "phobert_phow2v":
        def make_model():
            return PhoBERTPhoW2VClassifier(
                vectors.matrix,
                cfg.model_name,
                dropout=cfg.dropout,
                w2v_hidden=cfg.w2v_hidden,
                w2v_dropout=cfg.w2v_dropout,
                w2v_freeze=cfg.w2v_freeze,
            )

        tokenizer = JointTokenizer(load_tokenizer(cfg.model_name), w2v_tokenizer)
        return tokenizer, make_model, {"lr_overrides": {p: cfg.w2v_lr for p in NEW_MODULES}}

    raise ValueError(f"unknown mode {cfg.mode!r}; use 'phow2v' or 'phobert_phow2v'")


def run(cfg: W2VConfig) -> dict:
    return train_phobert_cv.run(cfg, setup=setup)


if __name__ == "__main__":
    run(parse_config(W2VConfig, __doc__.strip().splitlines()[0]))
