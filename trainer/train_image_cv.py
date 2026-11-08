"""K-fold cross-validation of the image-only CLIP model, on the same folds as the text models.

    python -m trainer.train_image_cv --unfreeze-layers 0               # linear probe
    python -m trainer.train_image_cv --unfreeze-layers 4 --epochs 6    # fine-tune the top 4 layers

The head always trains at `head_lr`; unfrozen encoder layers train at `lr`.
"""

from __future__ import annotations

from dataclasses import dataclass

from architecture.clip_image import CLIP_NAME, CLIPImageClassifier, image_transforms
from trainer import train_phobert_cv
from trainer.train_phobert import ROOT, parse_config
from trainer.train_phobert_cv import CVConfig


@dataclass
class ImageConfig(CVConfig):
    out_dir: str = str(ROOT / "outputs" / "clip_cv")
    image_model: str = CLIP_NAME
    image_size: int = 224
    unfreeze_layers: int = 0  # 0 = frozen encoder, N = top N layers, -1 = all
    augment: bool = True
    head_lr: float = 1e-3
    lr: float = 1e-5  # learning rate of the unfrozen encoder layers
    batch_size: int = 64
    epochs: int = 10
    monitor: str = "loss"
    num_workers: int = 2


def setup(cfg: ImageConfig, pool, test):
    """Model factory, Trainer arguments and loader arguments for `train_phobert_cv.run`."""
    train_tfm, eval_tfm = image_transforms(cfg.image_size, cfg.augment)

    def make_model():
        return CLIPImageClassifier(
            cfg.image_model, dropout=cfg.dropout, unfreeze_layers=cfg.unfreeze_layers
        )

    trainer_kwargs = {"lr_overrides": {"classifier.": cfg.head_lr}}
    loader_kwargs = {
        "use_image": True,
        "image_transform": train_tfm,
        "eval_image_transform": eval_tfm,
    }
    return None, make_model, trainer_kwargs, loader_kwargs


def run(cfg: ImageConfig) -> dict:
    return train_phobert_cv.run(cfg, setup=setup)


if __name__ == "__main__":
    run(parse_config(ImageConfig, __doc__.strip().splitlines()[0]))
