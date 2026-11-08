"""K-fold cross-validation of PhoBERT-base-v2 with an averaged ensemble on the test split.

    python -m trainer.train_phobert_cv --n-folds 5 --out-dir outputs/phobert_cv

The train and val splits are pooled and re-divided into `n_folds` stratified folds.
Each fold model is trained on the other folds, early-stopped on its own fold, and
then predicts (a) its held-out fold and (b) the untouched test split.

  - out-of-fold predictions give a cross-validated score over every pooled article
  - the test probabilities of the fold models are averaged into the ensemble
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

from architecture.phobert import PhoBERTClassifier, load_tokenizer
from trainer.train_phobert import Config, build_criterion, parse_config, set_seed
from trainer.trainer import Trainer, classification_metrics
from utils.dataloader import (
    ID2LABEL,
    ROOT,
    add_bodies,
    class_weights,
    kfold_ids,
    load_processed,
    make_dataloaders,
    segment_words,
)


@dataclass
class CVConfig(Config):
    out_dir: str = str(ROOT / "outputs" / "phobert_cv")
    num_workers: int = 0
    n_folds: int = 5
    keep_checkpoints: bool = False  # each fold's best.pt is ~540 MB
    lead_dropout: float = 0.0  # the cross-validation experiments always showed the lead
    body_path: str = ""  # bodies.jsonl from utils.collect_bodies; empty = title + lead only
    body_words: int = 250  # how much of the body to append (raise max_length to 256 with it)


def _metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    return classification_metrics(labels, (probs >= threshold).astype(int))


def phobert_setup(cfg: CVConfig, pool: pd.DataFrame, test: pd.DataFrame):
    """Tokenizer, model factory and extra Trainer arguments for the plain PhoBERT run."""
    return (
        load_tokenizer(cfg.model_name),
        lambda: PhoBERTClassifier(
            cfg.model_name, dropout=cfg.dropout, return_features=cfg.supcon_weight > 0
        ),
        {},
    )


def run(cfg: CVConfig, setup=phobert_setup) -> dict:
    """Returns fold metrics, out-of-fold and test prediction tables and a summary.

    `setup(cfg, pool, test)` supplies the tokenizer, a function building a fresh model
    for each fold, extra Trainer keyword arguments and, optionally, extra
    `make_dataloaders` arguments, so other models can reuse the same folds and
    reporting (see `trainer.train_phow2v_cv` and `trainer.train_image_cv`).
    """
    transformers.logging.set_verbosity_error()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    raw = load_processed(cfg.data_dir)
    if cfg.limit:
        raw = {k: v.head(cfg.limit) for k, v in raw.items()}
    if cfg.body_path:
        raw = {k: add_bodies(v, cfg.body_path, cfg.body_words) for k, v in raw.items()}
        n_body = sum(int(v["has_body"].sum()) for v in raw.values())
        print(f"article body added for {n_body} of {sum(map(len, raw.values()))} articles")
    pool_raw = pd.concat([raw["train"], raw["val"]], ignore_index=True)
    test_raw = raw["test"]
    pool, test = (segment_words(pool_raw), segment_words(test_raw)) if cfg.segment else (pool_raw.copy(), test_raw.copy())
    folds = kfold_ids(pool, cfg.n_folds, seed=cfg.seed)

    tokenizer, make_model, trainer_kwargs, *rest = setup(cfg, pool, test)
    loader_kwargs = {"use_image": False, **(rest[0] if rest else {})}
    make_criterion = trainer_kwargs.pop("make_criterion", None)
    class_names = loader_kwargs.pop("class_names", {})  # auxiliary classification heads
    oof_class: dict[str, np.ndarray] = {}
    test_class: dict[str, np.ndarray] = {}
    oof = np.full(len(pool), np.nan)
    test_probs = np.zeros((cfg.n_folds, len(test)))
    oof_branch: dict[str, np.ndarray] = {}  # auxiliary heads of fusion models
    test_branch: dict[str, np.ndarray] = {}
    fold_rows = []

    for fold in range(cfg.n_folds):
        print(f"\n===== fold {fold + 1}/{cfg.n_folds} =====")
        set_seed(cfg.seed + fold)
        held_out = folds == fold
        splits = {
            "train": pool[~held_out].reset_index(drop=True),
            "val": pool[held_out].reset_index(drop=True),
            "test": test,
        }
        loaders = make_dataloaders(
            splits,
            tokenizer=tokenizer,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
            use_lead=cfg.use_lead,
            num_workers=cfg.num_workers,
            lead_dropout=cfg.lead_dropout,
            **loader_kwargs,
        )
        weights = class_weights(splits["train"]) if cfg.weighted_loss else None
        if make_criterion is not None:
            criterion, eval_criterion = make_criterion(weights)
        else:
            criterion, eval_criterion = build_criterion(cfg, weights)
        trainer = Trainer(
            make_model(),
            loaders["train"],
            loaders["val"],
            out_dir=out_dir / f"fold{fold}",
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            class_weights=weights,
            criterion=criterion,
            eval_criterion=eval_criterion,
            patience=cfg.patience,
            monitor=cfg.monitor,
            rdrop_alpha=cfg.rdrop_alpha,
            layer_decay=cfg.layer_decay,
            **trainer_kwargs,
        )
        history = trainer.fit()

        # loaders for val/test are not shuffled, so predictions come back in row order
        val_out, test_out = trainer.predict(loaders["val"]), trainer.predict(loaders["test"])
        oof[held_out] = val_out["prob"]
        test_probs[fold] = test_out["prob"]
        for name, preds in val_out["class_preds"].items():
            oof_class.setdefault(name, np.full(len(pool), -1))[held_out] = preds
            test_class.setdefault(name, np.zeros((cfg.n_folds, len(test)), dtype=int))[fold] = test_out[
                "class_preds"
            ][name]
        for name, probs in val_out["branch_probs"].items():
            oof_branch.setdefault(name, np.full(len(pool), np.nan))[held_out] = probs
            test_branch.setdefault(name, np.zeros((cfg.n_folds, len(test))))[fold] = test_out[
                "branch_probs"
            ][name]
        pick = min if cfg.monitor == "loss" else max
        best_epoch = pick(history, key=lambda r: r[f"val_{cfg.monitor}"])["epoch"]
        fold_rows.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                **{f"val_{k}": v for k, v in _metrics(val_out["label"], val_out["prob"]).items()},
                **{f"test_{k}": v for k, v in _metrics(test_out["label"], test_out["prob"]).items()},
            }
        )

        if not cfg.keep_checkpoints:
            trainer.best_path.unlink(missing_ok=True)
        del trainer
        torch.cuda.empty_cache()

    pool_labels = pool["label_id"].to_numpy()
    test_labels = test["label_id"].to_numpy()
    ensemble = test_probs.mean(axis=0)
    fold_table = pd.DataFrame(fold_rows).set_index("fold")
    metric_names = list(_metrics(test_labels, ensemble))
    summary = {
        "oof": _metrics(pool_labels, oof),
        "fold_val_mean": {m: fold_table[f"val_{m}"].mean() for m in metric_names},
        "fold_val_std": {m: fold_table[f"val_{m}"].std() for m in metric_names},
        "single_model_test_mean": {m: fold_table[f"test_{m}"].mean() for m in metric_names},
        "single_model_test_std": {m: fold_table[f"test_{m}"].std() for m in metric_names},
        "ensemble_test": _metrics(test_labels, ensemble),
    }

    keep = ["id", "source", "category", "title", "label"] + (["has_body"] if cfg.body_path else [])
    oof_table = pool_raw[keep].assign(fold=folds, prob_clickbait=oof)
    oof_table["pred"] = (oof_table["prob_clickbait"] >= 0.5).astype(int).map(ID2LABEL)
    test_table = test_raw[keep].assign(
        **{f"prob_fold{f}": test_probs[f] for f in range(cfg.n_folds)}, prob_clickbait=ensemble
    )
    test_table["pred"] = (test_table["prob_clickbait"] >= 0.5).astype(int).map(ID2LABEL)
    for name, preds in oof_class.items():
        names = np.array(class_names.get(name, []))
        oof_table[f"pred_{name}"] = names[preds] if len(names) else preds
        # test: majority vote of the fold models
        votes = np.apply_along_axis(np.bincount, 0, test_class[name], minlength=max(len(names), 1))
        test_table[f"pred_{name}"] = names[votes.argmax(0)] if len(names) else votes.argmax(0)
        if f"{name}_id" in pool:
            summary[f"oof_{name}_accuracy"] = float((preds == pool[f"{name}_id"].to_numpy()).mean())
            print(f"auxiliary {name} head, out-of-fold accuracy: {summary[f'oof_{name}_accuracy']:.4f}")
    for name in oof_branch:
        oof_table[f"prob_{name}_head"] = oof_branch[name]
        test_table[f"prob_{name}_head"] = test_branch[name].mean(axis=0)
        summary[f"oof_{name}_head"] = _metrics(pool_labels, oof_branch[name])

    fold_table.to_csv(out_dir / "fold_metrics.csv")
    oof_table.to_csv(out_dir / "oof_predictions.csv", index=False)
    test_table.to_csv(out_dir / "test_predictions.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir / "tokenizer")

    print("\n===== summary (clickbait F1) =====")
    print(f"out-of-fold, {len(pool)} articles : {summary['oof']['f1']:.4f}")
    print(
        f"single model on test          : {summary['single_model_test_mean']['f1']:.4f} "
        f"+/- {summary['single_model_test_std']['f1']:.4f}"
    )
    print(f"{cfg.n_folds}-model ensemble on test      : {summary['ensemble_test']['f1']:.4f}")
    return {
        "fold_metrics": fold_table,
        "oof_predictions": oof_table,
        "test_predictions": test_table,
        "summary": summary,
    }


if __name__ == "__main__":
    run(parse_config(CVConfig, __doc__.split("\n")[0]))
