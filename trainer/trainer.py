"""Model-agnostic training loop for the clickbait classifiers.

The model is called as `model(**batch)` with every tensor in the batch except
`labels`, and must return logits of shape (batch, num_labels), or a dict with the
main logits under "logits" (extra entries are passed on to the criterion).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup


def classification_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Accuracy, precision/recall/F1 of the clickbait class (id 1) and macro F1."""
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    def f1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    return {
        "accuracy": (tp + tn) / max(len(labels), 1),
        "precision": tp / (tp + fp) if tp else 0.0,
        "recall": tp / (tp + fn) if tp else 0.0,
        "f1": f1(tp, fp, fn),
        "macro_f1": (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2,
    }


def parameter_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    layer_decay: float = 1.0,
    lr_overrides: dict[str, float] | None = None,
) -> list[dict]:
    """AdamW groups: no weight decay on biases/LayerNorm, optional layer-wise lr decay.

    With `layer_decay` < 1 the head keeps `lr` and each transformer layer below it gets
    `layer_decay` times the rate of the layer above (embeddings lowest). Layers are
    recognised by ".layer.<n>." or ".layers.<n>." in the parameter name, as in
    BERT/RoBERTa/ViT/CLIP encoders.

    `lr_overrides` maps a parameter-name prefix to a fixed learning rate, for modules
    trained from scratch next to a pre-trained encoder, e.g. {"classifier.": 1e-3}.
    """
    lr_overrides = lr_overrides or {}
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    layer_ids = [int(m.group(1)) for n, _ in named if (m := re.search(r"\.layers?\.(\d+)\.", n))]
    num_layers = max(layer_ids) + 1 if layer_ids else 0

    def depth(name: str) -> int:  # 0 = embeddings ... num_layers + 1 = head
        m = re.search(r"\.layers?\.(\d+)\.", name)
        if m:
            return int(m.group(1)) + 1
        return 0 if "embeddings" in name else num_layers + 1

    groups: dict[tuple, dict] = {}
    for name, param in named:
        decays = not any(k in name for k in ("bias", "LayerNorm.weight"))
        d = depth(name) if layer_decay != 1.0 else num_layers + 1
        override = next((p for p in lr_overrides if name.startswith(p)), None)
        group = groups.setdefault(
            (d, decays, override),
            {
                "params": [],
                "weight_decay": weight_decay if decays else 0.0,
                "lr": lr_overrides[override]
                if override is not None
                else lr * layer_decay ** (num_layers + 1 - d),
            },
        )
        group["params"].append(param)
    return list(groups.values())


class Trainer:
    """`monitor` is a validation metric name; "loss" is minimised, anything else maximised.

    `rdrop_alpha` > 0 enables R-Drop: every batch is passed through the model twice with
    different dropout masks and a symmetric KL term pulls the two predictions together.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        out_dir: str | Path,
        epochs: int = 10,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        class_weights: torch.Tensor | None = None,
        criterion: nn.Module | None = None,
        eval_criterion: nn.Module | None = None,
        grad_clip: float = 1.0,
        patience: int = 3,
        monitor: str = "f1",
        rdrop_alpha: float = 0.0,
        layer_decay: float = 1.0,
        lr_overrides: dict[str, float] | None = None,
        amp: bool = True,
        device: str | torch.device | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.patience = patience
        self.monitor = monitor
        self.rdrop_alpha = rdrop_alpha
        self.amp = amp and self.device.type == "cuda"

        self.optimizer = torch.optim.AdamW(
            parameter_groups(model, lr, weight_decay, layer_decay, lr_overrides), lr=lr
        )
        total_steps = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, int(total_steps * warmup_ratio), total_steps
        )
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        # default: class-weighted cross-entropy; pass e.g. a FusionLoss to override
        self.criterion = (criterion or nn.CrossEntropyLoss(weight=class_weights)).to(self.device)
        # loss reported on val/test (and used when monitor="loss"); keeping it fixed makes
        # runs with different training losses comparable
        self.eval_criterion = (eval_criterion or self.criterion).to(self.device)
        self.history: list[dict] = []
        self.best_path = self.out_dir / "best.pt"

    def _forward(self, batch: dict) -> tuple[torch.Tensor | dict, torch.Tensor]:
        inputs = {
            k: v.to(self.device, non_blocking=True)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }
        labels = inputs.pop("labels")
        with torch.autocast(self.device.type, enabled=self.amp):
            outputs = self.model(**inputs)
        if isinstance(outputs, dict):
            outputs = {k: v.float() for k, v in outputs.items() if v is not None}
        else:
            outputs = outputs.float()
        return outputs, labels

    def _train_loss(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, labels = self._forward(batch)
        loss = self.criterion(outputs, labels)
        if self.rdrop_alpha > 0:
            outputs2, _ = self._forward(batch)  # second pass, different dropout mask
            loss = 0.5 * (loss + self.criterion(outputs2, labels))
            log_p, log_q = (
                (o["logits"] if isinstance(o, dict) else o).log_softmax(-1)
                for o in (outputs, outputs2)
            )
            kl = 0.5 * (
                F.kl_div(log_p, log_q, reduction="batchmean", log_target=True)
                + F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)
            )
            loss = loss + self.rdrop_alpha * kl
        return loss, labels

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total, seen = 0.0, 0
        bar = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False)
        for batch in bar:
            loss, labels = self._train_loss(batch)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            total += loss.item() * len(labels)
            seen += len(labels)
            bar.set_postfix(loss=f"{total / seen:.4f}")
        return total / seen

    @torch.no_grad()
    def predict(self, loader) -> dict:
        """Ids, labels, clickbait probabilities and mean loss over a loader.

        For models with auxiliary heads, `branch_probs` holds the clickbait probability
        of every extra "<name>_logits" output, keyed by <name>, and `class_preds` the
        predicted class of every "<name>_scores" output (e.g. an auxiliary topic head).
        """
        self.model.eval()
        ids, labels_all, probs_all, total = [], [], [], 0.0
        branches: dict[str, list] = {}
        class_preds: dict[str, list] = {}
        for batch in loader:
            outputs, labels = self._forward(batch)
            total += self.eval_criterion(outputs, labels).item() * len(labels)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            ids.extend(batch["id"])
            labels_all.append(labels.cpu().numpy())
            probs_all.append(logits.softmax(-1)[:, 1].cpu().numpy())
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if key.endswith("_logits"):
                        branches.setdefault(key[: -len("_logits")], []).append(
                            value.softmax(-1)[:, 1].cpu().numpy()
                        )
                    elif key.endswith("_scores"):
                        class_preds.setdefault(key[: -len("_scores")], []).append(
                            value.argmax(-1).cpu().numpy()
                        )
        labels_all, probs_all = np.concatenate(labels_all), np.concatenate(probs_all)
        return {
            "id": ids,
            "label": labels_all,
            "prob": probs_all,
            "loss": total / len(labels_all),
            "branch_probs": {k: np.concatenate(v) for k, v in branches.items()},
            "class_preds": {k: np.concatenate(v) for k, v in class_preds.items()},
        }

    def evaluate(self, loader, threshold: float = 0.5) -> dict[str, float]:
        out = self.predict(loader)
        metrics = classification_metrics(out["label"], (out["prob"] >= threshold).astype(int))
        return {"loss": out["loss"], **metrics}

    def fit(self) -> list[dict]:
        """Train with early stopping on `monitor`; the best weights are reloaded at the end."""
        sign = -1.0 if self.monitor == "loss" else 1.0
        best, bad_epochs = -float("inf"), 0
        for epoch in range(1, self.epochs + 1):
            start = time.time()
            train_loss = self.train_epoch(epoch)
            val = self.evaluate(self.val_loader)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val.items()},
                "seconds": round(time.time() - start, 1),
            }
            self.history.append(row)
            print(
                f"epoch {epoch:>2} | train loss {train_loss:.4f} | val loss {val['loss']:.4f} "
                f"| acc {val['accuracy']:.3f} | P {val['precision']:.3f} R {val['recall']:.3f} "
                f"F1 {val['f1']:.3f} | macro F1 {val['macro_f1']:.3f} | {row['seconds']}s"
            )
            if sign * val[self.monitor] > best:
                best, bad_epochs = sign * val[self.monitor], 0
                torch.save(self.model.state_dict(), self.best_path)
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    print(f"early stop: no val {self.monitor} improvement for {self.patience} epochs")
                    break
        self.model.load_state_dict(torch.load(self.best_path, map_location=self.device))
        (self.out_dir / "history.json").write_text(json.dumps(self.history, indent=2))
        print(f"best val {self.monitor}: {sign * best:.4f} -> {self.best_path}")
        return self.history
