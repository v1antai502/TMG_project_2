"""Load trained clickbait models and score headlines and images.

    from utils.predict import ClickbaitPredictor, ImagePredictor, FusionPredictor

    # Text-only (existing)
    text = ClickbaitPredictor.from_dir("outputs/phobert_final")
    text.predict("Sốc: 5 bí quyết này ai cũng nên biết!")
    # [{'prob_clickbait': 0.93, 'label': 'clickbait'}]

    # Image-only
    img = ImagePredictor.from_dir("outputs/clip_final")
    img.predict("path/to/thumbnail.jpg")
    # [{'prob_clickbait': 0.78, 'label': 'clickbait'}]

    # Fusion (text + image)
    fusion = FusionPredictor.from_dir("outputs/fusion_final")
    fusion.predict("Sốc!", images="path/to/image.jpg")
    # [{'prob_clickbait': 0.95, 'label': 'clickbait'}]

Each predictor's directory is what its trainer writes: `config.json`, `best.pt`,
and (for text models) `tokenizer/`. Input goes through the same cleaning and word
segmentation as in training.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pandas as pd
import torch
import transformers
from PIL import Image
from transformers import AutoConfig, AutoModel

from architecture.phobert import PhoBERTClassifier, load_tokenizer
from utils.dataloader import ID2LABEL, load_image, segment_words, split_location_prefix, strip_publisher_suffix


# ---------------------------------------------------------------------------
# Image loading from various sources
# ---------------------------------------------------------------------------

def load_image_from_source(source) -> Image.Image:
    """Load an image from a file path, URL, base64 string, bytes, or PIL Image.

    Accepted inputs:
        - ``PIL.Image.Image`` — used directly
        - ``bytes`` / ``bytearray`` — decoded with Pillow
        - ``str`` starting with ``data:`` — data-URI with base64 payload
        - ``str`` starting with ``http://`` or ``https://`` — downloaded (timeout 15 s)
        - ``str`` (other) or ``pathlib.Path`` — treated as a local file path
    """
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (bytes, bytearray)):
        return Image.open(io.BytesIO(source)).convert("RGB")
    if isinstance(source, (str, Path)):
        source_str = str(source)
        # Data-URI (e.g. "data:image/png;base64,iVBOR...")
        if source_str.startswith("data:"):
            _, encoded = source_str.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        # URL
        if source_str.startswith(("http://", "https://")):
            import urllib.request
            req = urllib.request.Request(source_str, headers={"User-Agent": "ClickbaitDetector/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        # Local file path
        return load_image(source_str)
    raise ValueError(f"Cannot load image from {type(source).__name__}")


class ClickbaitPredictor:
    def __init__(self, model: PhoBERTClassifier, tokenizer, config: dict, threshold: float = 0.5):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config
        self.threshold = threshold
        self.device = next(model.parameters()).device

    @classmethod
    def from_dir(cls, path: str | Path, device: str | None = None, threshold: float = 0.5):
        transformers.logging.set_verbosity_error()  # base-model load report is expected noise
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = load_tokenizer(str(path / "tokenizer"))
        # Only the architecture is needed from the base model: every weight comes from best.pt.
        # `encoder_config/` (written by `save_encoder_config`) makes loading fully offline;
        # without it the base model's config is fetched from the Hugging Face hub.
        local = path / "encoder_config"
        encoder_config = AutoConfig.from_pretrained(str(local) if local.exists() else config["model_name"])
        encoder = AutoModel.from_config(encoder_config, add_pooling_layer=False)
        model = PhoBERTClassifier(encoder=encoder, dropout=config.get("dropout", 0.1))
        model.load_state_dict(torch.load(path / "best.pt", map_location="cpu"))
        return cls(model.to(device), tokenizer, config, threshold)

    @staticmethod
    def clean(title: str, lead: str = "") -> tuple[str, str]:
        """The same light cleaning as `utils.dataloader.clean`, for text scraped from a page."""
        title = strip_publisher_suffix(" ".join(title.split()))
        lead = split_location_prefix(" ".join(lead.split())) if lead else ""
        return title, lead

    @torch.no_grad()
    def predict(self, titles: str | list[str], leads: str | list[str] | None = None, batch_size: int = 32) -> list[dict]:
        titles = [titles] if isinstance(titles, str) else list(titles)
        if leads is None:
            leads = [""] * len(titles)
        leads = [leads] if isinstance(leads, str) else list(leads)
        rows = pd.DataFrame([self.clean(t, l) for t, l in zip(titles, leads)], columns=["title", "lead_paragraph"])
        if self.config.get("segment", True):
            rows = segment_words(rows)
        use_lead = self.config.get("use_lead", True)
        results = []
        for start in range(0, len(rows), batch_size):
            part = rows.iloc[start : start + batch_size]
            leads_part = list(part["lead_paragraph"]) if use_lead else None
            batch = self.tokenizer(
                list(part["title"]), leads_part, padding=True, truncation=True,
                max_length=self.config.get("max_length", 128), return_tensors="pt",
            )
            logits = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
            )
            for p in logits.float().softmax(-1)[:, 1].tolist():
                results.append({"prob_clickbait": round(p, 4), "label": ID2LABEL[int(p >= self.threshold)]})
        return results


def save_encoder_config(model_dir: str | Path) -> None:
    """Store the base model's architecture config next to the weights, for offline loading."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    AutoConfig.from_pretrained(config["model_name"]).save_pretrained(model_dir / "encoder_config")


# ---------------------------------------------------------------------------
# Image-only predictor (CLIP ViT-B/16)
# ---------------------------------------------------------------------------

class ImagePredictor:
    """Load a trained CLIP image model and score thumbnail images for clickbait.

    The directory must contain ``config.json`` and ``best.pt`` written by
    ``trainer.train_image`` or an equivalent script.
    """

    def __init__(self, model, transform, config: dict, threshold: float = 0.5):
        self.model = model.eval()
        self.transform = transform
        self.config = config
        self.threshold = threshold
        self.device = next(model.parameters()).device

    @classmethod
    def from_dir(cls, path: str | Path, device: str | None = None, threshold: float = 0.5):
        from architecture.clip_image import CLIPImageClassifier, image_transforms

        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        image_name = config.get("image_model", "openai/clip-vit-base-patch16")
        # Prefer local encoder config for offline loading
        local = path / "image_encoder_config"
        if local.exists():
            from transformers import CLIPVisionConfig, CLIPVisionModel
            encoder = CLIPVisionModel(CLIPVisionConfig.from_pretrained(str(local)))
        else:
            encoder = None

        model = CLIPImageClassifier(
            name=image_name,
            dropout=config.get("dropout", 0.1),
            unfreeze_layers=0,
            encoder=encoder,
        )
        model.load_state_dict(torch.load(path / "best.pt", map_location="cpu"))
        _, eval_tfm = image_transforms(config.get("image_size", 224), augment=False)
        return cls(model.to(device), eval_tfm, config, threshold)

    @torch.no_grad()
    def predict(self, images, batch_size: int = 16) -> list[dict]:
        """Score one or more images.

        Each element of *images* can be anything accepted by
        :func:`load_image_from_source`: a file path, URL, base64 string,
        raw bytes, or a PIL Image.
        """
        if not isinstance(images, (list, tuple)):
            images = [images]
        results: list[dict] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            tensors = [self.transform(load_image_from_source(img)) for img in chunk]
            pixel_values = torch.stack(tensors).to(self.device)
            logits = self.model(pixel_values=pixel_values)
            for p in logits.float().softmax(-1)[:, 1].tolist():
                results.append({"prob_clickbait": round(p, 4), "label": ID2LABEL[int(p >= self.threshold)]})
        return results


# ---------------------------------------------------------------------------
# Text + Image fusion predictor (PhoBERT + CLIP)
# ---------------------------------------------------------------------------

class FusionPredictor:
    """Load a trained fusion model and score headlines *with* their thumbnails.

    The directory must contain ``config.json``, ``best.pt`` and ``tokenizer/``,
    written by ``trainer.train_fusion`` or an equivalent script.
    """

    def __init__(self, model, tokenizer, transform, config: dict, threshold: float = 0.5):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.transform = transform
        self.config = config
        self.threshold = threshold
        self.device = next(model.parameters()).device

    @classmethod
    def from_dir(cls, path: str | Path, device: str | None = None, threshold: float = 0.5):
        from architecture.clip_image import image_transforms
        from architecture.fusion import FusionClassifier

        transformers.logging.set_verbosity_error()
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        model = FusionClassifier(
            text_name=config.get("model_name", "vinai/phobert-base-v2"),
            image_name=config.get("image_model", "openai/clip-vit-base-patch16"),
            dropout=config.get("fusion_dropout", config.get("dropout", 0.2)),
            fusion_hidden=config.get("fusion_hidden", 512),
            unfreeze_image_layers=0,
            aux_heads=False,  # no auxiliary heads at inference
        )
        # `strict=False` because the saved model may include aux heads from training
        model.load_state_dict(torch.load(path / "best.pt", map_location="cpu"), strict=False)

        tokenizer = load_tokenizer(str(path / "tokenizer"))
        _, eval_tfm = image_transforms(config.get("image_size", 224), augment=False)
        return cls(model.to(device), tokenizer, eval_tfm, config, threshold)

    @torch.no_grad()
    def predict(
        self,
        titles: str | list[str],
        leads: str | list[str] | None = None,
        images=None,
        batch_size: int = 16,
    ) -> list[dict]:
        """Score headlines together with their thumbnail images.

        *images* must be the same length as *titles*; each element is anything
        accepted by :func:`load_image_from_source`.
        """
        titles = [titles] if isinstance(titles, str) else list(titles)
        if leads is None:
            leads = [""] * len(titles)
        leads = [leads] if isinstance(leads, str) else list(leads)
        if images is None:
            raise ValueError("images are required for fusion prediction")
        if not isinstance(images, (list, tuple)):
            images = [images]
        if len(images) != len(titles):
            raise ValueError(f"Expected {len(titles)} images, got {len(images)}")

        rows = pd.DataFrame(
            [ClickbaitPredictor.clean(t, l) for t, l in zip(titles, leads)],
            columns=["title", "lead_paragraph"],
        )
        if self.config.get("segment", True):
            rows = segment_words(rows)

        use_lead = self.config.get("use_lead", True)
        results: list[dict] = []
        for start in range(0, len(rows), batch_size):
            part = rows.iloc[start : start + batch_size]
            leads_part = list(part["lead_paragraph"]) if use_lead else None
            text_batch = self.tokenizer(
                list(part["title"]),
                leads_part,
                padding=True,
                truncation=True,
                max_length=self.config.get("max_length", 128),
                return_tensors="pt",
            )
            img_chunk = images[start : start + batch_size]
            tensors = [self.transform(load_image_from_source(img)) for img in img_chunk]
            pixel_values = torch.stack(tensors).to(self.device)

            out = self.model(
                input_ids=text_batch["input_ids"].to(self.device),
                attention_mask=text_batch["attention_mask"].to(self.device),
                pixel_values=pixel_values,
            )
            logits = out["logits"] if isinstance(out, dict) else out
            for p in logits.float().softmax(-1)[:, 1].tolist():
                results.append({"prob_clickbait": round(p, 4), "label": ID2LABEL[int(p >= self.threshold)]})
        return results
