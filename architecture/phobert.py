"""Text-only clickbait classifier on top of PhoBERT-base-v2."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

PHOBERT_NAME = "vinai/phobert-base-v2"
PHOBERT_MAX_LENGTH = 256  # hard limit of the model's position embeddings


def load_tokenizer(name: str = PHOBERT_NAME):
    """PhoBERT's tokenizer. Input text must be word-segmented (see `segment_words`)."""
    return AutoTokenizer.from_pretrained(name)


class PhoBERTClassifier(nn.Module):
    """PhoBERT encoder + linear head on the <s> token.

    `encode` returns the pooled text feature so the encoder can later be reused
    as the text branch of a multimodal model.
    """

    def __init__(
        self,
        name: str = PHOBERT_NAME,
        num_labels: int = 2,
        dropout: float = 0.1,
        encoder: nn.Module | None = None,
        return_features: bool = False,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(
            name, add_pooling_layer=False
        )
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        # also return the pooled feature, for losses that act on it (supervised contrastive)
        self.return_features = return_features

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_
    ) -> torch.Tensor | dict:
        features = self.encode(input_ids, attention_mask)
        logits = self.classifier(self.dropout(features))
        if self.return_features:
            return {"logits": logits, "features": features}
        return logits
