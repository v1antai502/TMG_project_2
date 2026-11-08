"""Build the desktop package another person can install: extension + model service + model.

    python release/build_desktop.py

Writes dist/clickbait-detector-desktop-<version>/ and a .zip of it. The package runs fully
offline once its Python packages are installed (see release/INSTALL.md, copied into it).
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "phobert_final"

# (source, destination inside the package)
FILES = [
    ("service/app.py", "service/app.py"),
    ("architecture/phobert.py", "architecture/phobert.py"),
    ("utils/dataloader.py", "utils/dataloader.py"),
    ("utils/predict.py", "utils/predict.py"),
    ("release/requirements.txt", "requirements.txt"),
    ("release/start_service.bat", "start_service.bat"),
    ("release/start_service.sh", "start_service.sh"),
    ("release/INSTALL.md", "INSTALL.md"),
]
MODEL_FILES = ["best.pt", "config.json", "encoder_config", "tokenizer"]
CRLF, LF = bytes([13, 10]), bytes([10])


def save_half_precision(src: Path, dst: Path) -> None:
    """Store the weights as 16-bit floats: half the size (about 270 MB instead of 540 MB).

    Loading casts them back into the 32-bit model, so no code changes are needed. Measured on
    the test split: probabilities move by at most 0.0015 and no decision changes.
    """
    import torch

    state = torch.load(src, map_location="cpu")
    torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}, dst)


def build() -> Path:
    version = json.loads((ROOT / "extension" / "manifest.json").read_text(encoding="utf-8"))["version"]
    name = f"clickbait-detector-desktop-{version}"
    out = ROOT / "dist" / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for src, dst in FILES:
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src, out / dst)
        if dst.endswith(".bat"):  # cmd.exe needs Windows line endings for goto/labels
            data = (out / dst).read_bytes().replace(CRLF, LF).replace(LF, CRLF)
            (out / dst).write_bytes(data)
    shutil.copytree(ROOT / "extension", out / "extension", ignore=shutil.ignore_patterns("README.md"))

    if not (MODEL / "encoder_config").exists():
        from utils.predict import save_encoder_config

        save_encoder_config(MODEL)
    for item in MODEL_FILES:
        src, dst = MODEL / item, out / "models" / "phobert_final" / item
        dst.parent.mkdir(parents=True, exist_ok=True)
        if item == "best.pt":
            save_half_precision(src, dst)
        else:
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)

    archive = out.parent / f"{name}.zip"  # not with_suffix: the version contains dots
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                z.write(path, (Path(name) / path.relative_to(out)).as_posix())
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{out}  ({size / 1e6:.0f} MB)\n{archive}  ({archive.stat().st_size / 1e6:.0f} MB)")
    return archive


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    build()
