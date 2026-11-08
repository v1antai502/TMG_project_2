"""Image-only clickbait classifier on top of the CLIP ViT-B/16 vision encoder."""

from __future__ import annotations

import torch
from torch import nn
from transformers import CLIPVisionModel

CLIP_NAME = "openai/clip-vit-base-patch16"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def image_transforms(size: int = 224, augment: bool = True):
    """(train, eval) torchvision transforms with CLIP's normalisation.

    Eval resizes the shorter side to `size` and centre-crops, as CLIP was trained.
    Train uses a random crop covering at least 60% of the image and a horizontal flip.
    """
    from torchvision import transforms as T

    bicubic = T.InterpolationMode.BICUBIC
    to_tensor = [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)]
    eval_tfm = T.Compose([T.Resize(size, interpolation=bicubic), T.CenterCrop(size), *to_tensor])
    if not augment:
        return eval_tfm, eval_tfm
    train_tfm = T.Compose(
        [
            T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(3 / 4, 4 / 3), interpolation=bicubic),
            T.RandomHorizontalFlip(),
            *to_tensor,
        ]
    )
    return train_tfm, eval_tfm


def set_clip_trainable(encoder: nn.Module, unfreeze_layers: int) -> None:
    """Freeze a CLIP vision encoder except its top `unfreeze_layers` layers (-1 = train all)."""
    # older transformers nest the transformer under `.vision_model`
    vision = getattr(encoder, "vision_model", encoder)
    train_all = unfreeze_layers < 0
    for param in encoder.parameters():
        param.requires_grad = train_all
    if unfreeze_layers > 0:
        for module in (*vision.encoder.layers[-unfreeze_layers:], vision.post_layernorm):
            for param in module.parameters():
                param.requires_grad = True


class CLIPImageClassifier(nn.Module):
    """CLIP vision encoder + linear head on the pooled image feature.

    `unfreeze_layers` controls how much of the encoder is trained:
        0   encoder frozen, only the head learns (linear probe)
        N   the top N transformer layers are trained as well
        -1  the whole encoder is trained

    `encode` returns the pooled feature so the encoder can be reused as the image
    branch of the fusion model.
    """

    def __init__(
        self,
        name: str = CLIP_NAME,
        num_labels: int = 2,
        dropout: float = 0.1,
        unfreeze_layers: int = 0,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else CLIPVisionModel.from_pretrained(name)
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.set_trainable(unfreeze_layers)

    def set_trainable(self, unfreeze_layers: int) -> None:
        set_clip_trainable(self.encoder, unfreeze_layers)

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values=pixel_values).pooler_output

    def forward(self, pixel_values: torch.Tensor, **_) -> torch.Tensor:
        return self.classifier(self.dropout(self.encode(pixel_values)))
