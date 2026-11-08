"""Loading, cleaning, splitting and batching of the Vietnamese clickbait dataset.

Typical use:

    df = load_dataset()                      # cleaned DataFrame, one row per article
    splits = split_dataset(df)               # {"train": df, "val": df, "test": df}
    loaders = make_dataloaders(splits, tokenizer=tok, image_transform=tfm)

Only the batching part (`Collator`, `make_dataloaders`, `class_weights`) needs torch,
so the DataFrame helpers can be used from notebooks without it.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
JSONL_PATH = RAW_DIR / "clickbait_dataset_vietnamese.jsonl"
IMAGE_DIR = RAW_DIR / "images"

LABEL2ID = {"non-clickbait": 0, "clickbait": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Báo Mới is an aggregator and appends the original publisher to every title,
# e.g. "... - Chuyên trang An Ninh Thế Giới - Báo Công an Nhân dân".
_PUBLISHER_SUFFIX = re.compile(
    r"(\s+-\s+(Báo|Tạp chí|Chuyên trang|Trang|Đài|Cổng|Kênh|Thời báo|Ấn phẩm|VOV|VTV|VTC"
    r"|Vietnam\+|VietnamPlus|Bnews|VietTimes|TheLEADER|Giác ngộ|Công An|Đại Biểu)[^-]*)+$",
    re.IGNORECASE,
)
# CamelCase brand names that look like a glued dateline ("TikTok Việt Nam gỡ...").
_BRANDS = ("TikTok", "ChatGPT", "OpenAI", "YouTube", "VinFast", "PayPal", "DeepSeek")
_VN_DATE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})(?:\s*-\s*(\d{1,2}):(\d{2}))?")
_NAIVE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+$")


def load_raw(path: str | Path = JSONL_PATH) -> pd.DataFrame:
    """Read the JSONL file as-is."""
    return pd.read_json(path, lines=True, convert_dates=False)


def strip_publisher_suffix(title: str) -> str:
    return _PUBLISHER_SUFFIX.sub("", title).strip()


def split_location_prefix(lead: str) -> str:
    """VnExpress leads have the dateline glued to the text ("Nghệ AnCảng hàng không...")."""
    if lead.startswith("TP HCM") and lead[6:7].isupper():
        return "TP HCM - " + lead[6:]
    for i in range(1, min(len(lead), 30)):
        if lead[i - 1].islower() and lead[i].isupper():
            words = lead[:i].split()
            if (words[-1] + lead[i:].split()[0]).startswith(_BRANDS):
                break
            if len(words) <= 4 and all(w.isalpha() and w[0].isupper() for w in words):
                return f"{lead[:i]} - {lead[i:]}"
            break
    return lead


def parse_datetime(value) -> pd.Timestamp:
    """Parse the mixed date formats into a tz-aware timestamp (Asia/Ho_Chi_Minh)."""
    if not isinstance(value, str) or not value.strip():
        return pd.NaT
    value = value.strip()
    try:
        if value[0].isdigit():
            if _NAIVE_ISO.match(value):
                value += "+07:00"
            ts = pd.to_datetime(value, utc=True, format="ISO8601")
        else:  # "Chủ Nhật, 22/06/2025 - 12:23"
            m = _VN_DATE.search(value)
            if not m:
                return pd.NaT
            day, month, year, hour, minute = (int(g) if g else 0 for g in m.groups())
            ts = pd.Timestamp(year, month, day, hour, minute, tz="Asia/Ho_Chi_Minh")
        return ts.tz_convert("Asia/Ho_Chi_Minh")
    except (ValueError, TypeError):
        return pd.NaT


def clean(
    df: pd.DataFrame,
    image_dir: str | Path = IMAGE_DIR,
    clean_text: bool = True,
    drop_missing_images: bool = True,
) -> pd.DataFrame:
    """Normalise text, resolve image paths and encode labels.

    Adds `title_raw`, `lead_raw`, `image_path`, `has_image`, `publish_dt`, `label_id`.
    """
    df = df.copy()
    image_dir = Path(image_dir)

    df["title_raw"] = df["title"]
    df["lead_raw"] = df["lead_paragraph"]
    df["title"] = df["title"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    df["lead_paragraph"] = (
        df["lead_paragraph"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    )
    if clean_text:
        is_baomoi = df["source"] == "Báo Mới"
        df.loc[is_baomoi, "title"] = df.loc[is_baomoi, "title"].map(strip_publisher_suffix)
        is_vne = df["source"] == "VnExpress"
        df.loc[is_vne, "lead_paragraph"] = df.loc[is_vne, "lead_paragraph"].map(
            split_location_prefix
        )

    # `thumbnail_url` points at data/images/, the files actually live in `image_dir`.
    paths = [image_dir / f"{i}_image.png" for i in df["id"]]
    df["image_path"] = [str(p) for p in paths]
    df["has_image"] = [p.is_file() and p.stat().st_size > 0 for p in paths]

    df["publish_dt"] = pd.Series(
        [parse_datetime(v) for v in df["publish_datetime"]],
        index=df.index,
        dtype="datetime64[ns, Asia/Ho_Chi_Minh]",
    )
    df["label_id"] = df["label"].map(LABEL2ID).astype(int)

    if drop_missing_images:
        df = df[df["has_image"]]
    return df.reset_index(drop=True)


def load_dataset(path: str | Path = JSONL_PATH, **clean_kwargs) -> pd.DataFrame:
    return clean(load_raw(path), **clean_kwargs)


def load_processed(processed_dir: str | Path = ROOT / "data" / "processed") -> dict[str, pd.DataFrame]:
    """Read the splits written by `utils.processing` as {"train": df, "val": df, "test": df}."""
    processed_dir = Path(processed_dir)
    splits = {}
    for name in ("train", "val", "test"):
        part = pd.read_json(processed_dir / f"{name}.jsonl", lines=True, convert_dates=False)
        part["lead_paragraph"] = part["lead_paragraph"].fillna("")
        part["image_path"] = [str(processed_dir / f) for f in part["image_file"]]
        splits[name] = part
    return splits


def add_bodies(
    df: pd.DataFrame, bodies_path: str | Path, max_words: int = 250, min_title_overlap: float = 0.5
) -> pd.DataFrame:
    """Append the start of each article's body (see `utils.collect_bodies`) to its lead.

    Only bodies with status "ok" whose page title matches the article title are used;
    other rows keep their lead unchanged. Adds a boolean `has_body` column.
    The text model has no separate body input, so the body rides along in `lead_paragraph`.
    """
    bodies = {}
    with open(bodies_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                usable = rec["status"] == "ok" and rec.get("title_overlap", 0) >= min_title_overlap
                bodies[rec["id"]] = rec["body"] if usable else ""
    # the same opening on three or more articles is site boilerplate, not an article
    openings = pd.Series([b[:300] for b in bodies.values() if b]).value_counts()
    boilerplate = set(openings[openings >= 3].index)
    bodies = {i: ("" if b[:300] in boilerplate else b) for i, b in bodies.items()}

    df = df.copy()
    starts = []
    for article_id, lead in zip(df["id"], df["lead_paragraph"]):
        paragraphs = bodies.get(article_id, "").split("\n")
        lead_tokens = set(lead.lower().split())
        first = set(paragraphs[0].lower().split())
        if lead_tokens and len(first & lead_tokens) >= 0.8 * len(first):  # body opens with the lead
            paragraphs = paragraphs[1:]
        starts.append(" ".join(" ".join(paragraphs).split()[:max_words]))
    df["has_body"] = [bool(s) for s in starts]
    df["lead_paragraph"] = [f"{lead} {s}".strip() for lead, s in zip(df["lead_paragraph"], starts)]
    return df


def segment_words(df: pd.DataFrame, columns: tuple[str, ...] = ("title", "lead_paragraph")) -> pd.DataFrame:
    """Word-segment the text columns ("học sinh" -> "học_sinh"), as PhoBERT expects."""
    from pyvi import ViTokenizer

    df = df.copy()
    for col in columns:
        df[col] = [ViTokenizer.tokenize(t) if t else "" for t in df[col]]
    return df


def split_dataset(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    stratify_by: tuple[str, ...] = ("label", "source"),
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test split."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for _, group in df.groupby(list(stratify_by), sort=True):
        idx = rng.permutation(group.index.to_numpy())
        n_test = int(round(len(idx) * test_size))
        n_val = int(round(len(idx) * val_size))
        parts["test"].append(idx[:n_test])
        parts["val"].append(idx[n_test : n_test + n_val])
        parts["train"].append(idx[n_test + n_val :])
    return {
        name: df.loc[rng.permutation(np.concatenate(chunks))].reset_index(drop=True)
        for name, chunks in parts.items()
    }


def kfold_ids(
    df: pd.DataFrame,
    n_splits: int = 5,
    stratify_by: tuple[str, ...] = ("label", "source"),
    seed: int = 42,
) -> np.ndarray:
    """Stratified fold id (0..n_splits-1) for every row of `df`, in row order."""
    rng = np.random.default_rng(seed)
    folds = np.empty(len(df), dtype=int)
    offset = 0  # carry the round-robin position across strata so fold sizes stay even
    for _, group in df.reset_index(drop=True).groupby(list(stratify_by), sort=True):
        idx = rng.permutation(group.index.to_numpy())
        folds[idx] = (np.arange(len(idx)) + offset) % n_splits
        offset += len(idx)
    return folds


def load_image(path: str | Path) -> Image.Image:
    """Open a thumbnail as RGB. Files are JPEG/PNG/WebP/GIF regardless of the .png name."""
    with Image.open(path) as img:
        return img.convert("RGB")


class ClickbaitDataset:
    """Map-style dataset yielding one article as a dict.

    `image_transform` is applied to the PIL image (e.g. a torchvision transform);
    leave it None to get PIL images and let the collator's image processor handle them.
    """

    def __init__(
        self, df: pd.DataFrame, image_transform=None, use_image: bool = True, lead_dropout: float = 0.0
    ):
        self.df = df.reset_index(drop=True)
        self.image_transform = image_transform
        self.use_image = use_image
        # chance of hiding the lead, so the model also learns to judge a headline alone
        self.lead_dropout = lead_dropout

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        item = {
            "id": row["id"],
            "title": row["title"],
            "lead": "" if self.lead_dropout and random.random() < self.lead_dropout else row["lead_paragraph"],
            "label": int(row["label_id"]),
        }
        if "topic_id" in row:  # auxiliary topic label for multi-task training
            item["topic"] = int(row["topic_id"])
        if self.use_image:
            image = load_image(row["image_path"])
            if self.image_transform is not None:
                image = self.image_transform(image)
            item["image"] = image
        return item


class Collator:
    """Turns a list of items into a batch of tensors.

    `tokenizer` is a Hugging Face tokenizer; title and lead are encoded as a sentence
    pair (title only when `use_lead=False`). `image_processor` is an optional Hugging
    Face image processor for datasets that yield PIL images.
    """

    def __init__(self, tokenizer=None, image_processor=None, max_length: int = 128,
                 use_lead: bool = True):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.use_lead = use_lead

    def __call__(self, items: list[dict]) -> dict:
        import torch

        batch = {
            "id": [it["id"] for it in items],
            "labels": torch.tensor([it["label"] for it in items], dtype=torch.long),
        }
        if "topic" in items[0]:
            batch["topic_labels"] = torch.tensor([it["topic"] for it in items], dtype=torch.long)
        titles = [it["title"] for it in items]
        leads = [it["lead"] for it in items]
        if self.tokenizer is None:
            batch["title"], batch["lead"] = titles, leads
        else:
            batch.update(
                self.tokenizer(
                    titles,
                    leads if self.use_lead else None,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
            )
        if "image" in items[0]:
            images = [it["image"] for it in items]
            if self.image_processor is not None:
                batch["pixel_values"] = self.image_processor(images, return_tensors="pt")[
                    "pixel_values"
                ]
            elif isinstance(images[0], torch.Tensor):
                batch["pixel_values"] = torch.stack(images)
            else:
                batch["images"] = images
        return batch


def class_weights(df: pd.DataFrame):
    """Inverse-frequency class weights for the loss, ordered by label id."""
    import torch

    counts = df["label_id"].value_counts().reindex(range(len(LABEL2ID)), fill_value=0)
    weights = len(df) / (len(LABEL2ID) * counts.clip(lower=1))
    return torch.tensor(weights.to_numpy(), dtype=torch.float)


def make_dataloaders(
    splits: dict[str, pd.DataFrame],
    tokenizer=None,
    image_processor=None,
    image_transform=None,
    eval_image_transform=None,
    batch_size: int = 32,
    max_length: int = 128,
    use_lead: bool = True,
    use_image: bool = True,
    num_workers: int = 0,
    lead_dropout: float = 0.0,
) -> dict:
    """One DataLoader per split; only `train` is shuffled and uses `image_transform`.

    `eval_image_transform` defaults to `image_transform`; pass a separate one when the
    training transform contains augmentation.

    With `num_workers` > 0 the workers are started with "spawn" and kept alive between
    epochs. Forked workers inherit the other loaders' iterators and print
    "can only test a child process" errors when cleaning them up.
    """
    import torch
    from torch.utils.data import DataLoader

    if eval_image_transform is None:
        eval_image_transform = image_transform
    collate = Collator(tokenizer, image_processor, max_length, use_lead)
    worker_kwargs = (
        {"multiprocessing_context": "spawn", "persistent_workers": True} if num_workers else {}
    )
    loaders = {}
    for name, part in splits.items():
        is_train = name == "train"
        loaders[name] = DataLoader(
            ClickbaitDataset(
                part,
                image_transform if is_train else eval_image_transform,
                use_image,
                lead_dropout if is_train else 0.0,
            ),
            batch_size=batch_size,
            shuffle=is_train,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            **worker_kwargs,
        )
    return loaders
