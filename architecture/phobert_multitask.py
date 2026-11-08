"""PhoBERT with two heads: clickbait (the task) and article topic (an auxiliary task).

Both heads read the same <s> feature. Training on the topic as well is meant to shape that
feature with an extra signal; at prediction time only the clickbait head is needed, so the
extension would not need a topic label.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from architecture.phobert import PHOBERT_NAME


class PhoBERTMultiTask(nn.Module):
    """Returns {"logits": clickbait, "topic_scores": topic, "topic_loss": ...}.

    `topic_loss` (cross-entropy of the topic head) is only present when the batch carries
    `topic_labels`; `utils.loss_function.MultiTaskLoss` adds it to the clickbait loss.
    """

    def __init__(
        self,
        num_topics: int,
        name: str = PHOBERT_NAME,
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name, add_pooling_layer=False)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.topic_head = nn.Linear(hidden, num_topics)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        topic_labels: torch.Tensor | None = None,
        **_,
    ) -> dict:
        features = self.dropout(self.encode(input_ids, attention_mask))
        out = {"logits": self.classifier(features), "topic_scores": self.topic_head(features)}
        if topic_labels is not None:
            out["topic_loss"] = F.cross_entropy(out["topic_scores"].float(), topic_labels)
        return out
