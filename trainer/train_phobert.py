"""Fine-tune PhoBERT-base-v2 on title + lead paragraph (text only).

The defaults are the final configuration chosen after the cross-validation experiments
(out-of-fold F1 0.771): weighted cross-entropy, 5 epochs, layer-wise learning rate decay
0.9, checkpoint on validation loss. The best checkpoint, tokenizer and config are saved
to `out_dir` and can be loaded with `utils.predict.ClickbaitPredictor`.

    python -m trainer.train_phobert --data-dir data/processed --out-dir outputs/phobert
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

from architecture.phobert import PHOBERT_NAME, PhoBERTClassifier, load_tokenizer
from trainer.trainer import Trainer
from utils.dataloader import (
    ID2LABEL,
    ROOT,
    class_weights,
    load_processed,
    make_dataloaders,
    segment_words,
)
from utils.loss_function import ClassificationLoss


@dataclass
class Config:
    data_dir: str = str(ROOT / "data" / "processed")
    out_dir: str = str(ROOT / "outputs" / "phobert_final")
    model_name: str = PHOBERT_NAME
    max_length: int = 128
    use_lead: bool = True
    batch_size: int = 32
    epochs: int = 5
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    weighted_loss: bool = True
    patience: int = 3
    monitor: str = "loss"  # validation metric for checkpoint selection; "loss" is minimised
    rdrop_alpha: float = 0.0  # > 0 enables R-Drop (doubles the training time)
    layer_decay: float = 0.9  # < 1 lowers the learning rate layer by layer towards the embeddings
    # loss variants, see utils.loss_function.ClassificationLoss (all off = weighted cross-entropy)
    label_smoothing: float = 0.0
    focal_gamma: float = 0.0
    gce_q: float = 0.0
    supcon_weight: float = 0.0
    supcon_temperature: float = 0.3
    num_workers: int = 2
    segment: bool = True  # word-segment the text with pyvi, as PhoBERT was pre-trained
    lead_dropout: float = 0.5  # share of training examples shown without their lead
    seed: int = 42
    limit: int = 0  # use only the first N rows of each split (smoke tests)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_criterion(cfg: Config, weights: torch.Tensor | None):
    """(training loss, evaluation loss) for the loss options in `cfg`, or (None, None).

    Validation and test loss always use plain weighted cross-entropy, so checkpoint
    selection and reported losses are comparable across training losses.
    """
    if not (cfg.label_smoothing or cfg.focal_gamma or cfg.gce_q or cfg.supcon_weight):
        return None, None
    criterion = ClassificationLoss(
        weights,
        label_smoothing=cfg.label_smoothing,
        gamma=cfg.focal_gamma,
        gce_q=cfg.gce_q,
        supcon_weight=cfg.supcon_weight,
        supcon_temperature=cfg.supcon_temperature,
    )
    return criterion, ClassificationLoss(weights)


def run(cfg: Config) -> dict:
    """Train, then evaluate the best checkpoint on the test split.

    Returns the trainer, history, test metrics and a per-article test prediction table.
    """
    # the slow PhoBERT tokenizer logs a truncation notice on every sentence-pair batch
    transformers.logging.set_verbosity_error()
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    splits = load_processed(cfg.data_dir)
    if cfg.limit:
        splits = {k: v.head(cfg.limit) for k, v in splits.items()}
    if cfg.segment:
        splits = {k: segment_words(v) for k, v in splits.items()}

    tokenizer = load_tokenizer(cfg.model_name)
    loaders = make_dataloaders(
        splits,
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        use_lead=cfg.use_lead,
        use_image=False,
        num_workers=cfg.num_workers,
        lead_dropout=cfg.lead_dropout,
    )
    model = PhoBERTClassifier(
        cfg.model_name, dropout=cfg.dropout, return_features=cfg.supcon_weight > 0
    )
    weights = class_weights(splits["train"]) if cfg.weighted_loss else None
    criterion, eval_criterion = build_criterion(cfg, weights)
    trainer = Trainer(
        model,
        loaders["train"],
        loaders["val"],
        out_dir=out_dir,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        class_weights=weights,
        criterion=criterion,
        eval_criterion=eval_criterion,
        patience=cfg.patience,
        monitor=cfg.monitor,
        rdrop_alpha=cfg.rdrop_alpha,
        layer_decay=cfg.layer_decay,
    )
    print(f"device: {trainer.device} | train batches: {len(loaders['train'])}")
    history = trainer.fit()

    test_metrics = trainer.evaluate(loaders["test"])
    print("test | " + " | ".join(f"{k} {v:.4f}" for k, v in test_metrics.items()))
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    out = trainer.predict(loaders["test"])
    preds = pd.DataFrame({"id": out["id"], "prob_clickbait": out["prob"]})
    preds["pred"] = (preds["prob_clickbait"] >= 0.5).astype(int).map(ID2LABEL)
    preds = preds.merge(load_processed(cfg.data_dir)["test"], on="id")[
        ["id", "source", "category", "title", "label", "pred", "prob_clickbait"]
    ]
    # the same test articles with the headline alone, as the extension often sees them
    title_only = make_dataloaders(
        {"test": splits["test"].assign(lead_paragraph="")},
        tokenizer=tokenizer, batch_size=cfg.batch_size, max_length=cfg.max_length,
        use_lead=cfg.use_lead, use_image=False,
    )["test"]
    title_metrics = trainer.evaluate(title_only)
    print("test, title only | " + " | ".join(f"{k} {v:.4f}" for k, v in title_metrics.items()))
    (out_dir / "test_metrics_title_only.json").write_text(json.dumps(title_metrics, indent=2))
    preds["prob_clickbait_title_only"] = pd.Series(
        trainer.predict(title_only)["prob"], index=out["id"]
    ).reindex(preds["id"]).to_numpy()

    preds.to_csv(out_dir / "test_predictions.csv", index=False)
    tokenizer.save_pretrained(out_dir / "tokenizer")

    return {
        "trainer": trainer,
        "history": history,
        "test_metrics": test_metrics,
        "test_metrics_title_only": title_metrics,
        "test_predictions": preds,
    }


def parse_config(config_cls, description: str):
    """Build a config from command-line flags, one flag per dataclass field."""
    parser = argparse.ArgumentParser(description=description)
    for name, default in asdict(config_cls()).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)
    return config_cls(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_config(Config, __doc__.strip().splitlines()[0]))
