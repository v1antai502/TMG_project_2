"""Build the processed dataset in data/processed from data/raw.

    python -m utils.processing

Writes:
    data/processed/images/<id>.jpg     RGB JPEG, longer side at most `max_side` px
    data/processed/{train,val,test}.jsonl

Text is cleaned by `utils.dataloader.clean`; rows whose image is missing or cannot be
decoded are dropped. `image_file` is stored relative to data/processed so the folder
can be uploaded as-is (e.g. as a Kaggle dataset).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from utils.dataloader import JSONL_PATH, ROOT, load_dataset, split_dataset

PROCESSED_DIR = ROOT / "data" / "processed"
COLUMNS = [
    "id", "title", "lead_paragraph", "category", "source", "url",
    "publish_dt", "label", "label_id", "image_file",
]


def process_image(src: str | Path, dst: Path, max_side: int = 512, quality: int = 90) -> bool:
    """Convert one thumbnail to an RGB JPEG. Returns False if it cannot be decoded."""
    try:
        with Image.open(src) as img:
            img.seek(0)  # first frame of animated GIF/WebP
            if img.mode in ("RGBA", "LA", "P"):
                # flatten transparency onto white instead of the default black
                img = img.convert("RGBA")
                background = Image.new("RGBA", img.size, "white")
                img = Image.alpha_composite(background, img)
            img = img.convert("RGB")
            img.thumbnail((max_side, max_side), Image.LANCZOS)
            img.save(dst, "JPEG", quality=quality)
        return True
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


def build(
    raw_path: str | Path = JSONL_PATH,
    out_dir: str | Path = PROCESSED_DIR,
    max_side: int = 512,
    quality: int = 90,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(raw_path, drop_missing_images=False)
    n_raw = len(df)
    df = df[df["has_image"]].reset_index(drop=True)
    n_missing = n_raw - len(df)

    df["image_file"] = "images/" + df["id"] + ".jpg"
    ok = [
        process_image(src, out_dir / dst, max_side, quality)
        for src, dst in zip(df["image_path"], df["image_file"])
    ]
    failed = df.loc[[not x for x in ok], "id"].tolist()
    df = df[ok].reset_index(drop=True)

    df["publish_dt"] = df["publish_dt"].map(lambda t: t.isoformat() if pd.notna(t) else None)
    splits = split_dataset(df, val_size=val_size, test_size=test_size, seed=seed)
    for name, part in splits.items():
        part[COLUMNS].to_json(
            out_dir / f"{name}.jsonl", orient="records", lines=True, force_ascii=False
        )

    print(f"raw rows: {n_raw} | no image: {n_missing} | undecodable: {len(failed)} {failed}")
    for name, part in splits.items():
        print(f"{name}: {len(part)} rows, clickbait rate {part['label_id'].mean():.3f}")
    print(f"written to {out_dir}")
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", default=PROCESSED_DIR)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(
        out_dir=args.out_dir,
        max_side=args.max_side,
        quality=args.quality,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
