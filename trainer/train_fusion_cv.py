"""K-fold cross-validation of the text + image fusion model, on the same folds as the others.

    python -m trainer.train_fusion_cv
    python -m trainer.train_fusion_cv --text-aux-weight 0 --image-aux-weight 0   # no auxiliary heads
    python -m trainer.train_fusion_cv --unfreeze-image-layers 4                  # also tune CLIP's top layers

Learning rates: PhoBERT at `lr` (with `layer_decay`), unfrozen CLIP layers at `image_lr`,
the fusion head and auxiliary heads at `head_lr`.
"""

from __future__ import annotations

from dataclasses import dataclass

from architecture.clip_image import CLIP_NAME, image_transforms
from architecture.fusion import NEW_MODULES, FusionClassifier
from architecture.phobert import load_tokenizer
from trainer import train_phobert_cv
from trainer.train_phobert import ROOT, parse_config
from trainer.train_phobert_cv import CVConfig
from utils.loss_function import FusionLoss


@dataclass
class FusionConfig(CVConfig):
    out_dir: str = str(ROOT / "outputs" / "fusion_cv")
    image_model: str = CLIP_NAME
    image_size: int = 224
    augment: bool = True
    unfreeze_image_layers: int = 0  # 0 = CLIP frozen, N = top N layers, -1 = all
    fusion_hidden: int = 512
    fusion_dropout: float = 0.2
    text_aux_weight: float = 0.3  # both 0 = no auxiliary heads
    image_aux_weight: float = 0.3
    head_lr: float = 1e-3
    image_lr: float = 1e-5
    lr: float = 2e-5  # PhoBERT
    layer_decay: float = 0.9
    epochs: int = 5
    monitor: str = "loss"
    num_workers: int = 2


def setup(cfg: FusionConfig, pool, test):
    """Tokenizer, model factory, Trainer arguments and loader arguments for `train_phobert_cv.run`."""
    train_tfm, eval_tfm = image_transforms(cfg.image_size, cfg.augment)
    aux = bool(cfg.text_aux_weight or cfg.image_aux_weight)

    def make_model():
        return FusionClassifier(
            cfg.model_name,
            cfg.image_model,
            dropout=cfg.fusion_dropout,
            fusion_hidden=cfg.fusion_hidden,
            unfreeze_image_layers=cfg.unfreeze_image_layers,
            aux_heads=aux,
        )

    def make_criterion(weights):
        # validation/test loss is the fused cross-entropy alone, comparable with the text model
        return (
            FusionLoss(weights, cfg.text_aux_weight, cfg.image_aux_weight, gamma=cfg.focal_gamma),
            FusionLoss(weights, 0.0, 0.0),
        )

    trainer_kwargs = {
        "lr_overrides": {**{p: cfg.head_lr for p in NEW_MODULES}, "image.": cfg.image_lr},
        "make_criterion": make_criterion,
    }
    loader_kwargs = {
        "use_image": True,
        "image_transform": train_tfm,
        "eval_image_transform": eval_tfm,
    }
    return load_tokenizer(cfg.model_name), make_model, trainer_kwargs, loader_kwargs


def run(cfg: FusionConfig) -> dict:
    return train_phobert_cv.run(cfg, setup=setup)


if __name__ == "__main__":
    run(parse_config(FusionConfig, __doc__.strip().splitlines()[0]))
