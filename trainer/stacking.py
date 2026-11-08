"""Features + stacking experiment, after the ablation in Michaluk et al. (arXiv 2602.18171).

    python -m trainer.stacking
    python -m trainer.stacking --text-run outputs/text_baseline --no-bodies

Runs on CPU in about a minute: no neural network is trained here. PhoBERT enters through
the out-of-fold and test probabilities already saved by `trainer.train_phobert_cv`.

Compared on the same 5 folds as every earlier run (out-of-fold) and on the test split:

    tfidf                TF-IDF (word + character n-grams) + logistic regression
    features             the hand-crafted features alone (gradient-boosted trees)
    phobert              the saved PhoBERT predictions as they are
    phobert + features   gradient-boosted trees on PhoBERT's probability and the features

The paper used XGBoost; sklearn's HistGradientBoostingClassifier is the same model family
and avoids an extra dependency. Each stacker is trained on the pooled train + val
articles, using PhoBERT's out-of-fold probabilities so that it never sees a probability
from a model trained on that article. On test it receives the 5-model ensemble probability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline, make_union

from trainer.trainer import classification_metrics
from utils.dataloader import ROOT, load_processed
from utils.text_features import text_features

BODY_PATH = ROOT / "data" / "bodies" / "bodies.jsonl"


def gbt() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=30,
        l2_regularization=1.0, class_weight="balanced", random_state=42,
    )


def tfidf_model():
    vectors = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True),
    )
    return make_pipeline(vectors, LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000))


def scores(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    return {**classification_metrics(y, (prob >= 0.5).astype(int)), "auc": roc_auc_score(y, prob)}


def paired_f1_interval(y, base, other, n=2000, seed=0) -> tuple[float, float, float]:
    """F1(other) - F1(base) on the same articles, with a 95% bootstrap interval."""
    def f1(t, p):
        tp = (t & p).sum()
        return 2 * tp / max(2 * tp + (~t & p).sum() + (t & ~p).sum(), 1)

    y, base, other = y.astype(bool), base >= 0.5, other >= 0.5
    idx = np.random.default_rng(seed).integers(0, len(y), (n, len(y)))
    diffs = np.array([f1(y[i], other[i]) - f1(y[i], base[i]) for i in idx])
    return f1(y, other) - f1(y, base), *np.percentile(diffs, [2.5, 97.5])


def load_bodies(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    bodies = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                bodies[rec["id"]] = rec["body"] if rec["status"] == "ok" else ""
    return bodies


def run(text_run: Path, out_dir: Path, use_bodies: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = load_processed()
    pool = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    test = splits["test"].reset_index(drop=True)

    # PhoBERT predictions and the fold of every pooled article (same folds as every run)
    oof = pd.read_csv(text_run / "oof_predictions.csv")[["id", "fold", "prob_clickbait"]]
    tst = pd.read_csv(text_run / "test_predictions.csv")[["id", "prob_clickbait"]]
    pool = pool.merge(oof, on="id", validate="one_to_one")
    test = test.merge(tst, on="id", validate="one_to_one")
    assert pool["prob_clickbait"].notna().all() and len(pool) == len(oof)

    bodies = load_bodies(BODY_PATH) if use_bodies else None
    print("title-body similarity:", "included" if bodies else "not used (no bodies file)")
    X_pool, X_test = text_features(pool, bodies), text_features(test, bodies)
    feature_names = list(X_pool.columns)
    y_pool, y_test = pool["label_id"].to_numpy(), test["label_id"].to_numpy()
    folds = pool["fold"].to_numpy()
    text_pool = (pool["title"] + " . " + pool["lead_paragraph"]).to_numpy()
    text_test = (test["title"] + " . " + test["lead_paragraph"]).to_numpy()

    def stacked(X, probs):
        return X.assign(phobert_logit=logit(np.clip(probs, 1e-6, 1 - 1e-6)))

    models = {
        "tfidf": (tfidf_model, lambda idx: text_pool[idx], lambda: text_test),
        "features": (gbt, lambda idx: X_pool.iloc[idx], lambda: X_test),
        "phobert + features": (
            gbt,
            lambda idx: stacked(X_pool.iloc[idx], pool["prob_clickbait"].to_numpy()[idx]),
            lambda: stacked(X_test, test["prob_clickbait"].to_numpy()),
        ),
    }
    oof_probs = {"phobert": pool["prob_clickbait"].to_numpy()}
    test_probs = {"phobert": test["prob_clickbait"].to_numpy()}
    importances = []
    for name, (make, pool_input, test_input) in models.items():
        prob = np.full(len(pool), np.nan)
        for k in np.unique(folds):
            train_idx, held_idx = np.where(folds != k)[0], np.where(folds == k)[0]
            model = make().fit(pool_input(train_idx), y_pool[train_idx])
            prob[held_idx] = model.predict_proba(pool_input(held_idx))[:, 1]
            if name != "tfidf":
                imp = permutation_importance(
                    model, pool_input(held_idx), y_pool[held_idx], scoring="f1", n_repeats=5, random_state=k
                )
                importances.append(pd.Series(imp.importances_mean, index=pool_input(held_idx).columns, name=name))
        oof_probs[name] = prob
        test_probs[name] = make().fit(pool_input(np.arange(len(pool))), y_pool).predict_proba(test_input())[:, 1]
        print(f"done: {name}")

    order = ["tfidf", "features", "phobert", "phobert + features"]
    table = pd.DataFrame(
        {
            **{("out-of-fold", m): scores(y_pool, oof_probs[m]) for m in order},
            **{("test", m): scores(y_test, test_probs[m]) for m in order},
        }
    ).T
    vs_phobert = pd.DataFrame(
        {
            m: dict(zip(["oof_f1_diff", "oof_low", "oof_high"], paired_f1_interval(y_pool, oof_probs["phobert"], oof_probs[m])))
            | dict(zip(["test_f1_diff", "test_low", "test_high"], paired_f1_interval(y_test, test_probs["phobert"], test_probs[m])))
            for m in order if m != "phobert"
        }
    ).T
    importance = (
        pd.concat(importances, axis=1).T.groupby(level=0).mean().T
        .reindex(columns=["features", "phobert + features"])
        .sort_values("features", ascending=False)
    )
    by_label = X_pool.groupby(pool["label"]).mean().T[["clickbait", "non-clickbait"]]

    table.to_csv(out_dir / "scores.csv")
    vs_phobert.to_csv(out_dir / "vs_phobert.csv")
    importance.to_csv(out_dir / "feature_importance.csv")
    by_label.to_csv(out_dir / "feature_means_by_label.csv")
    pd.DataFrame({"id": pool["id"], "label": pool["label"], "fold": folds, **{f"prob_{m}": oof_probs[m] for m in order}}).to_csv(
        out_dir / "oof_predictions.csv", index=False
    )
    pd.DataFrame({"id": test["id"], "label": test["label"], **{f"prob_{m}": test_probs[m] for m in order}}).to_csv(
        out_dir / "test_predictions.csv", index=False
    )

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\n===== scores (clickbait class) =====")
        print(table.round(4).to_string())
        print("\n===== F1 difference from PhoBERT alone, 95% bootstrap interval =====")
        print(vs_phobert.round(4).to_string())
        print("\n===== permutation importance (drop in F1 when shuffled, averaged over folds) =====")
        print(importance.round(4).to_string())
        print("\n===== feature means by label (pooled train + val) =====")
        print(by_label.round(3).to_string())
    print(f"\nsaved to {out_dir}")
    return {"scores": table, "vs_phobert": vs_phobert, "importance": importance, "by_label": by_label}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--text-run", type=Path, default=ROOT / "outputs" / "text_baseline")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "stacking")
    parser.add_argument("--no-bodies", action="store_true", help="skip the title-body similarity feature")
    args = parser.parse_args()
    run(args.text_run, args.out_dir, use_bodies=not args.no_bodies)
