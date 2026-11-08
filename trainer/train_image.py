"""Fine-tune CLIP ViT-B/16 on thumbnail images for clickbait detection.

Produces a production-ready model directory with `best.pt` and `config.json`
that can be loaded by `utils.predict.ImagePredictor`.

    python -m trainer.train_image --out-dir outputs/clip_final
    python -m trainer.train_image --unfreeze-layers 4 --epochs 8   # fine-tune top layers
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

from architecture.clip_image import CLIP_NAME, CLIPImageClassifier, image_transforms
from trainer.train_phobert import Config, parse_config, set_seed
from trainer.trainer import Trainer
from utils.dataloader import (
    ID2LABEL,
    ROOT,
    class_weights,
    load_processed,
    make_dataloaders,
)


@dataclass
class ImageConfig(Config):
    out_dir: str = str(ROOT / "outputs" / "clip_final")
    image_model: str = CLIP_NAME
    image_size: int = 224
    unfreeze_layers: int = 0  # 0 = frozen encoder (linear probe), N = top N layers, -1 = all
    augment: bool = True
    head_lr: float = 1e-3    # learning rate for the classifier head
    lr: float = 1e-5         # learning rate for unfrozen encoder layers
    batch_size: int = 64
    epochs: int = 10
    monitor: str = "loss"
    num_workers: int = 2
    # text fields are ignored for image-only training
    use_lead: bool = False
    segment: bool = False
    lead_dropout: float = 0.0


def run(cfg: ImageConfig) -> dict:
    """Train, then evaluate the best checkpoint on the test split."""
    transformers.logging.set_verbosity_error()
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    splits = load_processed(cfg.data_dir)
    if cfg.limit:
        splits = {k: v.head(cfg.limit) for k, v in splits.items()}

    train_tfm, eval_tfm = image_transforms(cfg.image_size, cfg.augment)

    loaders = make_dataloaders(
        splits,
        tokenizer=None,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        use_lead=False,
        use_image=True,
        image_transform=train_tfm,
        eval_image_transform=eval_tfm,
        num_workers=cfg.num_workers,
    )

    model = CLIPImageClassifier(
        cfg.image_model,
        dropout=cfg.dropout,
        unfreeze_layers=cfg.unfreeze_layers,
    )

    weights = class_weights(splits["train"]) if cfg.weighted_loss else None
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
        patience=cfg.patience,
        monitor=cfg.monitor,
        lr_overrides={"classifier.": cfg.head_lr},
    )

    print(f"device: {trainer.device} | train batches: {len(loaders['train'])}")
    print(f"unfreeze_layers: {cfg.unfreeze_layers} | augment: {cfg.augment}")
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

    # Save CLIP encoder config for offline loading
    try:
        from transformers import CLIPVisionConfig
        model.encoder.config.save_pretrained(out_dir / "image_encoder_config")
    except Exception:
        pass

    print(f"\n✓ Model saved to {out_dir}")
    print(f"  To serve: copy {out_dir} to models/clip_final/ and restart the server")

    return {"trainer": trainer, "history": history, "test_metrics": test_metrics}


if __name__ == "__main__":
    run(parse_config(ImageConfig, __doc__.strip().splitlines()[0]))

