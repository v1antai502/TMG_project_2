"""Text + image fusion: PhoBERT and the CLIP vision encoder, features concatenated."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, CLIPVisionModel

from architecture.clip_image import CLIP_NAME, set_clip_trainable
from architecture.phobert import PHOBERT_NAME

# Parameter-name prefixes of the parts trained from scratch (see `lr_overrides` in the Trainer).
NEW_MODULES = ("text_norm.", "image_norm.", "fusion.", "text_head.", "image_head.")


class FusionClassifier(nn.Module):
    """PhoBERT <s> feature + CLIP pooled image feature -> MLP head.

    Each feature is layer-normalised before concatenation because the two encoders
    produce very different scales. With `aux_heads` the model also classifies from each
    branch alone and returns

        {"logits": fused, "text_logits": ..., "image_logits": ...}

    which `utils.loss_function.FusionLoss` turns into auxiliary loss terms, so the weaker
    branch keeps learning instead of being ignored by the fusion head.

    `unfreeze_image_layers`: 0 keeps CLIP frozen, N trains its top N layers, -1 all.
    """

    def __init__(
        self,
        text_name: str = PHOBERT_NAME,
        image_name: str = CLIP_NAME,
        num_labels: int = 2,
        dropout: float = 0.2,
        fusion_hidden: int = 512,
        unfreeze_image_layers: int = 0,
        aux_heads: bool = True,
    ):
        super().__init__()
        self.text = AutoModel.from_pretrained(text_name, add_pooling_layer=False)
        self.image = CLIPVisionModel.from_pretrained(image_name)
        set_clip_trainable(self.image, unfreeze_image_layers)

        text_dim, image_dim = self.text.config.hidden_size, self.image.config.hidden_size
        self.text_norm = nn.LayerNorm(text_dim)
        self.image_norm = nn.LayerNorm(image_dim)
        self.hidden_size = text_dim + image_dim
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_labels),
        )
        self.text_head = nn.Linear(text_dim, num_labels) if aux_heads else None
        self.image_head = nn.Linear(image_dim, num_labels) if aux_heads else None

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalised (text, image) features."""
        text = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        image = self.image(pixel_values=pixel_values).pooler_output
        return self.text_norm(text), self.image_norm(image)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        **_,
    ) -> dict:
        text, image = self.encode(input_ids, attention_mask, pixel_values)
        out = {"logits": self.fusion(torch.cat([text, image], dim=-1))}
        if self.text_head is not None:
            out["text_logits"] = self.text_head(text)
            out["image_logits"] = self.image_head(image)
        return out
