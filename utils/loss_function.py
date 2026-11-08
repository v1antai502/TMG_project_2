"""Losses for the clickbait classifiers.

`ClassificationLoss` is the single-model loss with optional label smoothing, focal,
generalised cross-entropy and supervised-contrastive terms. `FusionLoss` is for the
text + image fusion model:

The fusion model may return either the fused logits alone, or a dict

    {"logits": fused, "text_logits": ..., "image_logits": ...}

where the two extra entries come from optional auxiliary heads on each branch.
The auxiliary terms keep a branch learning on its own, so the weaker modality
(here usually the image) is not simply ignored by the fusion head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FusionLoss(nn.Module):
    """Class-weighted cross-entropy on the fused logits plus weighted auxiliary terms.

        loss = CE(fused) + text_weight * CE(text_only) + image_weight * CE(image_only)

    An auxiliary term is skipped when its weight is 0 or the model does not return it.
    `gamma > 0` turns every term into focal loss, which down-weights easy examples.
    """

    AUX_KEYS = {"text": "text_logits", "image": "image_logits"}

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        text_weight: float = 0.3,
        image_weight: float = 0.3,
        label_smoothing: float = 0.0,
        gamma: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.aux_weights = {"text": text_weight, "image": image_weight}
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        self.last_terms: dict[str, float] = {}

    def _term(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if self.gamma == 0:
            return F.cross_entropy(
                logits, labels, weight=self.class_weights, label_smoothing=self.label_smoothing
            )
        ce = F.cross_entropy(
            logits, labels, weight=self.class_weights,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        p_true = logits.softmax(-1).gather(1, labels[:, None]).squeeze(1)
        return ((1 - p_true) ** self.gamma * ce).mean()

    def forward(self, outputs: torch.Tensor | dict, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            outputs = {"logits": outputs}
        loss = self._term(outputs["logits"], labels)
        self.last_terms = {"fused": loss.item()}
        for name, key in self.AUX_KEYS.items():
            weight = self.aux_weights[name]
            if weight and outputs.get(key) is not None:
                term = self._term(outputs[key], labels)
                self.last_terms[name] = term.item()
                loss = loss + weight * term
        return loss


def supervised_contrastive(
    features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.3
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020) over one batch.

    Pulls each example's feature towards the other examples of its class and away from
    the rest. Examples with no same-class partner in the batch are skipped.
    """
    z = F.normalize(features.float(), dim=-1)
    sim = z @ z.T / temperature
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = (labels[:, None] == labels[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim.masked_fill(eye, -1e9), dim=1, keepdim=True)
    n_pos = positives.sum(1)
    valid = n_pos > 0
    if not valid.any():
        return sim.sum() * 0.0
    per_anchor = -torch.where(positives, log_prob, torch.zeros_like(log_prob)).sum(1)
    return (per_anchor[valid] / n_pos[valid]).mean()


class MultiTaskLoss(nn.Module):
    """`base` loss on the clickbait logits plus `topic_weight` x the model's own `topic_loss`.

    The topic cross-entropy is computed inside the model (it needs the topic labels, which
    the Trainer passes to the model, not to the loss).
    """

    def __init__(self, base: nn.Module, topic_weight: float = 0.5):
        super().__init__()
        self.base = base
        self.topic_weight = topic_weight
        self.last_terms: dict[str, float] = {}

    def forward(self, outputs: dict, labels: torch.Tensor) -> torch.Tensor:
        loss = self.base(outputs, labels)
        self.last_terms = {"clickbait": loss.item()}
        if self.topic_weight and outputs.get("topic_loss") is not None:
            self.last_terms["topic"] = outputs["topic_loss"].item()
            loss = loss + self.topic_weight * outputs["topic_loss"]
        return loss


class ClassificationLoss(nn.Module):
    """Class-weighted cross-entropy with optional variants, for a single (non-fusion) model.

    With every option at its default this equals `nn.CrossEntropyLoss(weight=class_weights)`.

    label_smoothing  spread this much probability mass over the other class
    gamma            > 0: focal loss, down-weights examples the model already gets right
    gce_q            > 0: generalised cross-entropy (Zhang & Sabuncu 2018), (1 - p^q) / q.
                     Bounded per example, so cases the model cannot fit stop dominating.
                     Replaces the cross-entropy term; label_smoothing and gamma are ignored.
    supcon_weight    > 0: loss = (1 - w) * classification + w * supervised contrastive.
                     Needs the model to return {"logits": ..., "features": ...}.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        gamma: float = 0.0,
        gce_q: float = 0.0,
        supcon_weight: float = 0.0,
        supcon_temperature: float = 0.3,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        self.gce_q = gce_q
        self.supcon_weight = supcon_weight
        self.supcon_temperature = supcon_temperature
        self.last_terms: dict[str, float] = {}

    def _classification(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if self.class_weights is not None:
            sample_weight = self.class_weights[labels]
        else:
            sample_weight = torch.ones_like(labels, dtype=logits.dtype)
        p_true = logits.softmax(-1).gather(1, labels[:, None]).squeeze(1)
        if self.gce_q > 0:
            per_sample = sample_weight * (1 - p_true.clamp_min(1e-7) ** self.gce_q) / self.gce_q
        else:
            # already multiplied by the class weight of each example
            per_sample = F.cross_entropy(
                logits, labels, weight=self.class_weights,
                label_smoothing=self.label_smoothing, reduction="none",
            )
            if self.gamma > 0:
                per_sample = (1 - p_true) ** self.gamma * per_sample
        return per_sample.sum() / sample_weight.sum()

    def forward(self, outputs: torch.Tensor | dict, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            outputs = {"logits": outputs}
        loss = self._classification(outputs["logits"], labels)
        self.last_terms = {"classification": loss.item()}
        if self.supcon_weight > 0:
            if outputs.get("features") is None:
                raise ValueError("supcon_weight > 0 needs the model to return 'features'")
            contrastive = supervised_contrastive(
                outputs["features"], labels, self.supcon_temperature
            )
            self.last_terms["supcon"] = contrastive.item()
            loss = (1 - self.supcon_weight) * loss + self.supcon_weight * contrastive
        return loss
