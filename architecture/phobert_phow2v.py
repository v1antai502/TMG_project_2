"""PhoBERT + PhoW2V: the two text encoders run side by side and their features are concatenated."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from architecture.phobert import PHOBERT_NAME
from architecture.phow2v import PhoW2VEncoder

# Parameter-name prefixes of the parts trained from scratch; they need a much higher
# learning rate than the pre-trained PhoBERT weights (see `lr_overrides` in the Trainer).
NEW_MODULES = ("w2v.", "classifier.")


class PhoBERTPhoW2VClassifier(nn.Module):
    """Concatenates PhoBERT's <s> feature with the pooled PhoW2V BiLSTM feature.

    Expects both tokenizers' outputs in the batch (see `architecture.phow2v.JointTokenizer`).
    """

    def __init__(
        self,
        vectors: torch.Tensor,
        name: str = PHOBERT_NAME,
        num_labels: int = 2,
        dropout: float = 0.1,
        w2v_hidden: int = 128,
        w2v_dropout: float = 0.3,
        w2v_freeze: bool = True,
    ):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(name, add_pooling_layer=False)
        self.w2v = PhoW2VEncoder(vectors, w2v_hidden, w2v_dropout, w2v_freeze)
        self.hidden_size = self.phobert.config.hidden_size + self.w2v.output_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        w2v_ids: torch.Tensor,
        w2v_mask: torch.Tensor,
    ) -> torch.Tensor:
        bert = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        return torch.cat([bert.last_hidden_state[:, 0], self.w2v(w2v_ids, w2v_mask)], dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        w2v_ids: torch.Tensor,
        w2v_mask: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        return self.classifier(
            self.dropout(self.encode(input_ids, attention_mask, w2v_ids, w2v_mask))
        )
