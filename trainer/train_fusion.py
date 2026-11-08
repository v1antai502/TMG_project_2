"""Fine-tune the PhoBERT + CLIP fusion model (text + image clickbait detection).

Produces a production-ready model directory with `best.pt`, `config.json` and
`tokenizer/` that can be loaded by `utils.predict.FusionPredictor`.

    python -m trainer.train_fusion --out-dir outputs/fusion_final
    python -m trainer.train_fusion --unfreeze-image-layers 4 --epochs 6
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

from architecture.clip_image import CLIP_NAME, image_transforms
from architecture.fusion import NEW_MODULES, FusionClassifier
from architecture.phobert import PHOBERT_NAME, load_tokenizer
from trainer.train_phobert import Config, parse_config, set_seed
from trainer.trainer import Trainer
from utils.dataloader import (
    ID2LABEL,
    ROOT,
    class_weights,
    load_processed,
    make_dataloaders,
    segment_words,
)
from utils.loss_function import FusionLoss


@dataclass
class FusionConfig(Config):
    out_dir: str = str(ROOT / "outputs" / "fusion_final")
    image_model: str = CLIP_NAME
    image_size: int = 224
    augment: bool = True
    unfreeze_image_layers: int = 0  # 0 = CLIP frozen, N = top N layers, -1 = all
    fusion_hidden: int = 512
    fusion_dropout: float = 0.2
    text_aux_weight: float = 0.3   # auxiliary text-only head weight
    image_aux_weight: float = 0.3  # auxiliary image-only head weight
    head_lr: float = 1e-3          # learning rate for new heads
    image_lr: float = 1e-5         # learning rate for unfrozen CLIP layers
    lr: float = 2e-5              # learning rate for PhoBERT
    layer_decay: float = 0.9
    epochs: int = 5
    monitor: str = "loss"
    num_workers: int = 2
    lead_dropout: float = 0.5


def run(cfg: FusionConfig) -> dict:
    """Train, then evaluate the best checkpoint on the test split."""
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
    train_tfm, eval_tfm = image_transforms(cfg.image_size, cfg.augment)
    aux = bool(cfg.text_aux_weight or cfg.image_aux_weight)

    loaders = make_dataloaders(
        splits,
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        use_lead=cfg.use_lead,
        use_image=True,
        image_transform=train_tfm,
        eval_image_transform=eval_tfm,
        num_workers=cfg.num_workers,
        lead_dropout=cfg.lead_dropout,
    )

    model = FusionClassifier(
        cfg.model_name,
        cfg.image_model,
        dropout=cfg.fusion_dropout,
        fusion_hidden=cfg.fusion_hidden,
        unfreeze_image_layers=cfg.unfreeze_image_layers,
        aux_heads=aux,
    )

    weights = class_weights(splits["train"]) if cfg.weighted_loss else None
    criterion = FusionLoss(weights, cfg.text_aux_weight, cfg.image_aux_weight, gamma=cfg.focal_gamma)
    eval_criterion = FusionLoss(weights, 0.0, 0.0)  # validation uses fused CE only

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
        lr_overrides={
            **{p: cfg.head_lr for p in NEW_MODULES},
            "image.": cfg.image_lr,
        },
    )

    print(f"device: {trainer.device} | train batches: {len(loaders['train'])}")
    print(f"unfreeze_image_layers: {cfg.unfreeze_image_layers} | aux_heads: {aux}")
    history = trainer.fit()

    # Test evaluation
    test_metrics = trainer.evaluate(loaders["test"])
    print("test | " + " | ".join(f"{k} {v:.4f}" for k, v in test_metrics.items()))
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    # Save predictions
    out = trainer.predict(loaders["test"])
    preds = pd.DataFrame({"id": out["id"], "prob_clickbait": out["prob"]})
    preds["pred"] = (preds["prob_clickbait"] >= 0.5).astype(int).map(ID2LABEL)
    preds = preds.merge(load_processed(cfg.data_dir)["test"][["id", "label"]], on="id")
    preds.to_csv(out_dir / "test_predictions.csv", index=False)

    # Save tokenizer
    tokenizer.save_pretrained(out_dir / "tokenizer")

    print(f"\n✓ Model saved to {out_dir}")
    print(f"  To serve: copy {out_dir} to models/fusion_final/ and restart the server")

    return {"trainer": trainer, "history": history, "test_metrics": test_metrics}


if __name__ == "__main__":
    run(parse_config(FusionConfig, __doc__.strip().splitlines()[0]))

