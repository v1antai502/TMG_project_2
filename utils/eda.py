"""Exploratory analysis helpers for the Vietnamese clickbait dataset.

Every function takes the cleaned DataFrame from `utils.dataloader.load_dataset`.
Table functions return a DataFrame; `plot_*` / `show_*` functions return the figure.
"""

from __future__ import annotations

import os
import re
import textwrap
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

LABELS = ["non-clickbait", "clickbait"]
COLORS = {"non-clickbait": "#4C78A8", "clickbait": "#E45756"}

# Surface cues commonly associated with clickbait headlines.
TITLE_CUES = {
    "question mark": r"\?",
    "exclamation mark": r"!",
    "ellipsis": r"\.\.\.|…",
    "quotation marks": r"[\"'“”‘’]",
    "colon": r":",
    "contains a number": r"\d",
    "starts with a number": r"^\d",
    "demonstrative (này/đây/thế này)": r"\b(?:này|đây|thế này|điều này)\b",
    "ALL-CAPS word (3+ letters)": r"\b[A-ZĐ]{3,}\b",
}

_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


def overview(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, missing/empty count and number of unique values."""
    cols = [c for c in df.columns if c not in ("title_raw", "lead_raw")]
    empty = {
        c: int((df[c].isna() | (df[c].astype(str).str.strip() == "")).sum()) for c in cols
    }
    return pd.DataFrame(
        {
            "dtype": [str(df[c].dtype) for c in cols],
            "missing": [empty[c] for c in cols],
            "unique": [df[c].nunique() for c in cols],
        },
        index=cols,
    )


def label_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
    return pd.DataFrame({"count": counts, "share": (counts / len(df)).round(3)})


def label_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Label counts and clickbait rate per value of `col`, sorted by rate."""
    table = pd.crosstab(df[col], df["label"]).reindex(columns=LABELS, fill_value=0)
    table["total"] = table.sum(axis=1)
    table["clickbait_rate"] = (table["clickbait"] / table["total"]).round(3)
    return table.sort_values("clickbait_rate", ascending=False)


def plot_label_distribution(df: pd.DataFrame):
    counts = label_counts(df)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(counts.index, counts["count"], color=[COLORS[l] for l in counts.index])
    ax.bar_label(bars, labels=[f"{c} ({s:.0%})" for c, s in zip(counts["count"], counts["share"])])
    ax.set_ylabel("articles")
    ax.set_title("Label distribution")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_label_by(df: pd.DataFrame, col: str):
    """Stacked counts (left) and clickbait rate (right) per value of `col`."""
    table = label_by(df, col).iloc[::-1]
    overall = (df["label"] == "clickbait").mean()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 0.4 * len(table) + 1.5), sharey=True
    )
    left = np.zeros(len(table))
    for label in LABELS:
        ax1.barh(table.index, table[label], left=left, color=COLORS[label], label=label)
        left += table[label].to_numpy()
    ax1.set_xlabel("articles")
    ax1.set_title(f"Articles per {col}")
    ax1.legend(frameon=False)

    bars = ax2.barh(table.index, table["clickbait_rate"], color=COLORS["clickbait"])
    ax2.bar_label(bars, labels=[f"{r:.0%}" for r in table["clickbait_rate"]], padding=2)
    ax2.axvline(overall, color="black", linestyle="--", linewidth=1)
    ax2.text(overall, len(table) - 0.4, f" overall {overall:.0%}", va="bottom", fontsize=8)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("clickbait rate")
    ax2.set_title(f"Clickbait rate per {col}")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def text_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Word and character counts of title and lead, one row per article."""
    return pd.DataFrame(
        {
            "label": df["label"],
            "title_words": df["title"].str.split().str.len(),
            "title_chars": df["title"].str.len(),
            "lead_words": df["lead_paragraph"].str.split().str.len(),
            "lead_chars": df["lead_paragraph"].str.len(),
        }
    )


def text_length_summary(df: pd.DataFrame) -> pd.DataFrame:
    lengths = text_lengths(df)
    return (
        lengths.groupby("label")
        .agg(["mean", "median", "min", "max"])
        .reindex(LABELS)
        .round(1)
        .T
    )


def plot_text_lengths(df: pd.DataFrame):
    lengths = text_lengths(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, col in zip(axes, ["title_words", "lead_words"]):
        bins = np.arange(0, lengths[col].max() + 2) - 0.5
        if len(bins) > 60:
            bins = 50
        for label in LABELS:
            ax.hist(
                lengths.loc[lengths["label"] == label, col],
                bins=bins,
                density=True,
                alpha=0.55,
                color=COLORS[label],
                label=label,
            )
        ax.set_xlabel(col.replace("_", " "))
        ax.set_ylabel("density")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Text length by label (whitespace-separated words)")
    fig.tight_layout()
    return fig


def title_cue_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Share of titles per label that contain each surface cue in `TITLE_CUES`."""
    rows = {}
    for name, pattern in TITLE_CUES.items():
        hit = df["title"].str.contains(pattern, regex=True)
        rows[name] = hit.groupby(df["label"]).mean().reindex(LABELS)
    table = pd.DataFrame(rows).T
    table["difference"] = table["clickbait"] - table["non-clickbait"]
    return table.sort_values("difference", ascending=False).round(3)


def plot_title_cues(df: pd.DataFrame):
    table = title_cue_rates(df).iloc[::-1]
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(table) + 1.2))
    for offset, label in zip((-0.2, 0.2), LABELS):
        ax.barh(y + offset, table[label], height=0.4, color=COLORS[label], label=label)
    ax.set_yticks(y, table.index)
    ax.set_xlabel("share of titles")
    ax.set_title("Surface cues in titles")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _ngrams(text: str, n: int) -> set[str]:
    tokens = _TOKEN.findall(text.lower())
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def distinctive_ngrams(
    df: pd.DataFrame, col: str = "title", n: int = 2, top: int = 20, min_count: int = 8
) -> pd.DataFrame:
    """N-grams most over-represented in each label.

    Counts are document frequencies. Vietnamese words are not segmented, so an n-gram
    is n syllables. Ranked by smoothed log-odds of appearing in a clickbait text.
    """
    doc_freq = {label: Counter() for label in LABELS}
    for text, label in zip(df[col], df["label"]):
        doc_freq[label].update(_ngrams(text, n))
    n_docs = df["label"].value_counts()
    rows = []
    for gram in set(doc_freq["clickbait"]) | set(doc_freq["non-clickbait"]):
        cb, non = doc_freq["clickbait"][gram], doc_freq["non-clickbait"][gram]
        if cb + non < min_count:
            continue
        log_odds = np.log((cb + 0.5) / (n_docs["clickbait"] - cb + 0.5)) - np.log(
            (non + 0.5) / (n_docs["non-clickbait"] - non + 0.5)
        )
        rows.append((gram, cb, non, round(log_odds, 2)))
    table = pd.DataFrame(rows, columns=["ngram", "clickbait", "non-clickbait", "log_odds"])
    table = table.sort_values("log_odds", ascending=False)
    return pd.concat(
        {"clickbait": table.head(top), "non-clickbait": table.tail(top).iloc[::-1]}
    ).reset_index(level=0, names="leans").reset_index(drop=True)


def plot_publish_dates(df: pd.DataFrame, since: str = "2025-05-01"):
    """Articles per day from `since`; older articles are counted in the title."""
    dates = df.dropna(subset=["publish_dt"])
    recent = dates[dates["publish_dt"] >= pd.Timestamp(since, tz=dates["publish_dt"].dt.tz)]
    daily = (
        pd.crosstab(recent["publish_dt"].dt.date, recent["label"])
        .reindex(columns=LABELS, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for label in LABELS:
        ax.plot(
            daily.index, daily[label], marker="o", markersize=3, color=COLORS[label], label=label
        )
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("articles per day (log)")
    ax.set_title(
        f"Publish date since {since} "
        f"({len(dates) - len(recent)} older, {df['publish_dt'].isna().sum()} undated not shown)"
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def image_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Real format, mode, size and file size of every thumbnail (reads headers only)."""
    rows = []
    for article_id, label, path in zip(df["id"], df["label"], df["image_path"]):
        row = {"id": article_id, "label": label}
        try:
            with Image.open(path) as img:
                row.update(
                    format=img.format,
                    mode=img.mode,
                    width=img.width,
                    height=img.height,
                    frames=getattr(img, "n_frames", 1),
                )
            row["size_kb"] = round(os.path.getsize(path) / 1024, 1)
        except (OSError, ValueError) as err:
            row["error"] = type(err).__name__
        rows.append(row)
    stats = pd.DataFrame(rows)
    if "width" in stats:
        stats["aspect"] = (stats["width"] / stats["height"]).round(3)
        stats["megapixels"] = (stats["width"] * stats["height"] / 1e6).round(2)
    return stats


def plot_image_stats(stats: pd.DataFrame):
    stats = stats.dropna(subset=["width"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for label in LABELS:
        part = stats[stats["label"] == label]
        axes[0].scatter(
            part["width"], part["height"], s=6, alpha=0.35, color=COLORS[label], label=label
        )
        axes[1].hist(
            part["aspect"].clip(0.4, 2.6),
            bins=np.linspace(0.4, 2.6, 45),
            density=True,
            alpha=0.55,
            color=COLORS[label],
        )
        axes[2].hist(
            np.log10(part["size_kb"].clip(lower=1)),
            bins=40,
            density=True,
            alpha=0.55,
            color=COLORS[label],
        )
    axes[0].set(xlabel="width (px)", ylabel="height (px)", title="Image dimensions")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, markerscale=3)
    axes[1].set(xlabel="width / height (clipped to 0.4-2.6)", title="Aspect ratio")
    axes[2].set(xlabel="log10 file size (KB)", title="File size")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def show_samples(
    df: pd.DataFrame, label: str | None = None, n: int = 8, cols: int = 4, seed: int = 0
):
    """Grid of random thumbnails with source and title underneath."""
    pool = df if label is None else df[df["label"] == label]
    sample = pool.sample(min(n, len(pool)), random_state=seed)
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), sample.iterrows()):
        with Image.open(row["image_path"]) as img:
            img = img.convert("RGB")
            img.thumbnail((400, 400))
            ax.imshow(img)
        ax.set_title(
            f"[{row['source']}] " + "\n".join(textwrap.wrap(row["title"], 38)[:3]),
            fontsize=8,
            color=COLORS[row["label"]],
            loc="left",
        )
    fig.suptitle(label or "random sample", fontsize=11)
    fig.tight_layout()
    return fig


def cross_source_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Titles that appear more than once after lower-casing (e.g. syndicated stories)."""
    key = df["title"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    dup = df[key.duplicated(keep=False)].assign(_key=key)
    return dup.sort_values("_key")[["id", "source", "label", "title"]].reset_index(drop=True)
