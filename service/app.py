"""Local web service that scores headlines (and thumbnails) with the clickbait models.

    python -m service.app                       # http://127.0.0.1:8000, this computer only
    MODEL_DIR=models/phobert_final python -m service.app --port 8000

    # reachable from a phone on the same Wi-Fi (e.g. to test Edge for Android):
    python -m service.app --host 0.0.0.0 --api-key <a long random string>

Endpoints
    GET  /health          model status (text, image, fusion)
    POST /predict         text-only: {"headlines": [{"id": "a1", "title": "...", "lead": "..."}]}
    POST /predict-image   image-only: {"images": [{"id": "i1", "url": "https://..."}]}
    POST /predict-fusion  text+image:  {"items": [{"id": "f1", "title": "...", "image_url": "..."}]}

By default it listens on 127.0.0.1 only, so it is reachable from this computer (and the
browser extension) but not from the network. With `--host 0.0.0.0` it accepts connections
from the local network, and then refuses to start without an access key: every request
must carry it in the `X-API-Key` header.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import secrets

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.dataloader import ROOT
from utils.predict import ClickbaitPredictor, ImagePredictor, FusionPredictor, load_image_from_source

MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models" / "phobert_final"))
IMAGE_MODEL_DIR = Path(os.environ.get("IMAGE_MODEL_DIR", ROOT / "models" / "clip_final"))
FUSION_MODEL_DIR = Path(os.environ.get("FUSION_MODEL_DIR", ROOT / "models" / "fusion_final"))
API_KEY = os.environ.get("CLICKBAIT_API_KEY", "")  # set by --api-key; empty = no key needed
MAX_HEADLINES = 256
MAX_IMAGES = 64
CACHE_SIZE = 20_000


class Headline(BaseModel):
    id: str = Field(..., max_length=200)
    title: str = Field(..., min_length=1, max_length=500)
    lead: str | None = Field(None, max_length=2000)


class PredictRequest(BaseModel):
    headlines: list[Headline] = Field(..., max_length=MAX_HEADLINES)


class ImageItem(BaseModel):
    id: str = Field(..., max_length=200)
    url: str | None = Field(None, max_length=4000, description="Image URL to download")
    base64: str | None = Field(None, description="Base64-encoded image (with or without data-URI prefix)")


class ImagePredictRequest(BaseModel):
    images: list[ImageItem] = Field(..., max_length=MAX_IMAGES)


class FusionItem(BaseModel):
    id: str = Field(..., max_length=200)
    title: str = Field(..., min_length=1, max_length=500)
    lead: str | None = Field(None, max_length=2000)
    image_url: str | None = Field(None, max_length=4000)
    image_base64: str | None = Field(None)


class FusionPredictRequest(BaseModel):
    items: list[FusionItem] = Field(..., max_length=MAX_IMAGES)


class Prediction(BaseModel):
    id: str
    prob_clickbait: float
    label: str


class PredictResponse(BaseModel):
    results: list[Prediction]
    cached: int = 0
    seconds: float


state: dict = {}


def check_key(x_api_key: str = Header("")) -> None:
    if API_KEY and not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "missing or wrong X-API-Key")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not (MODEL_DIR / "best.pt").exists():
        raise RuntimeError(f"no model at {MODEL_DIR} (expected best.pt, tokenizer/, config.json)")
    started = time.time()
    state["model"] = ClickbaitPredictor.from_dir(MODEL_DIR)
    state["cache"] = OrderedDict()  # (title, lead) -> probability
    print(f"text model loaded from {MODEL_DIR} on {state['model'].device} in {time.time() - started:.1f}s")

    # Optional: image-only model
    if (IMAGE_MODEL_DIR / "best.pt").exists():
        t = time.time()
        state["image_model"] = ImagePredictor.from_dir(IMAGE_MODEL_DIR)
        print(f"image model loaded from {IMAGE_MODEL_DIR} in {time.time() - t:.1f}s")
    else:
        print(f"image model not found at {IMAGE_MODEL_DIR} (image endpoint disabled)")

    # Optional: fusion model (text + image)
    if (FUSION_MODEL_DIR / "best.pt").exists():
        t = time.time()
        state["fusion_model"] = FusionPredictor.from_dir(FUSION_MODEL_DIR)
        print(f"fusion model loaded from {FUSION_MODEL_DIR} in {time.time() - t:.1f}s")
    else:
        print(f"fusion model not found at {FUSION_MODEL_DIR} (fusion endpoint disabled)")

    yield
    state.clear()


app = FastAPI(title="Vietnamese clickbait detector", lifespan=lifespan)
# the extension calls from its background worker, which host_permissions already allow;
# this also lets extension pages (the popup) call the service directly
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome|moz)-extension://.*$",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.get("/health", dependencies=[Depends(check_key)])
def health() -> dict:
    model: ClickbaitPredictor | None = state.get("model")
    image_model: ImagePredictor | None = state.get("image_model")
    fusion_model: FusionPredictor | None = state.get("fusion_model")
    return {
        "status": "ok" if model else "loading",
        "model_dir": str(MODEL_DIR.name),
        "device": str(model.device) if model else None,
        "threshold": model.threshold if model else None,
        "cached_headlines": len(state.get("cache", {})),
        "models": {
            "text": model is not None,
            "image": image_model is not None,
            "fusion": fusion_model is not None,
        },
    }


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(check_key)])
def predict(request: PredictRequest) -> PredictResponse:
    model: ClickbaitPredictor | None = state.get("model")
    if model is None:
        raise HTTPException(503, "model is still loading")
    started = time.time()
    cache: OrderedDict = state["cache"]
    keys = [(h.title.strip(), (h.lead or "").strip()) for h in request.headlines]

    missing = list(dict.fromkeys(k for k in keys if k not in cache))
    if missing:
        scored = model.predict([t for t, _ in missing], [l for _, l in missing])
        for key, result in zip(missing, scored):
            cache[key] = result["prob_clickbait"]
            if len(cache) > CACHE_SIZE:
                cache.popitem(last=False)

    results = []
    for headline, key in zip(request.headlines, keys):
        prob = cache[key]
        cache.move_to_end(key)
        results.append(Prediction(
            id=headline.id,
            prob_clickbait=prob,
            label="clickbait" if prob >= model.threshold else "non-clickbait",
        ))
    return PredictResponse(
        results=results, cached=len(keys) - len(missing), seconds=round(time.time() - started, 3)
    )


# ---------------------------------------------------------------------------
# Image-only endpoint
# ---------------------------------------------------------------------------

def _resolve_image_source(item):
    """Return an image source (URL or base64 string) from an ImageItem or FusionItem."""
    if getattr(item, "url", None):
        return item.url
    if getattr(item, "base64", None):
        return item.base64
    if getattr(item, "image_url", None):
        return item.image_url
    if getattr(item, "image_base64", None):
        return item.image_base64
    return None


@app.post("/predict-image", response_model=PredictResponse, dependencies=[Depends(check_key)])
def predict_image(request: ImagePredictRequest) -> PredictResponse:
    """Score thumbnail images for clickbait (CLIP vision encoder)."""
    image_model: ImagePredictor | None = state.get("image_model")
    if image_model is None:
        raise HTTPException(503, "image model is not loaded (place best.pt + config.json in models/clip_final/)")
    started = time.time()

    sources = []
    for item in request.images:
        src = _resolve_image_source(item)
        if src is None:
            raise HTTPException(422, f"image '{item.id}': provide either 'url' or 'base64'")
        sources.append(src)

    try:
        scored = image_model.predict(sources)
    except Exception as exc:
        raise HTTPException(422, f"image processing error: {exc}")

    results = [
        Prediction(id=item.id, prob_clickbait=s["prob_clickbait"], label=s["label"])
        for item, s in zip(request.images, scored)
    ]
    return PredictResponse(results=results, seconds=round(time.time() - started, 3))


# ---------------------------------------------------------------------------
# Text + Image fusion endpoint
# ---------------------------------------------------------------------------

@app.post("/predict-fusion", response_model=PredictResponse, dependencies=[Depends(check_key)])
def predict_fusion(request: FusionPredictRequest) -> PredictResponse:
    """Score headlines together with their thumbnail images (PhoBERT + CLIP fusion)."""
    fusion_model: FusionPredictor | None = state.get("fusion_model")
    if fusion_model is None:
        raise HTTPException(503, "fusion model is not loaded (place best.pt + config.json + tokenizer/ in models/fusion_final/)")
    started = time.time()

    titles, leads, images = [], [], []
    for item in request.items:
        titles.append(item.title)
        leads.append(item.lead or "")
        src = _resolve_image_source(item)
        if src is None:
            raise HTTPException(422, f"item '{item.id}': provide either 'image_url' or 'image_base64'")
        images.append(src)

    try:
        scored = fusion_model.predict(titles, leads, images)
    except Exception as exc:
        raise HTTPException(422, f"fusion processing error: {exc}")

    results = [
        Prediction(id=item.id, prob_clickbait=s["prob_clickbait"], label=s["label"])
        for item, s in zip(request.items, scored)
    ]
    return PredictResponse(results=results, seconds=round(time.time() - started, 3))


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------

# TMG301 WEB UI ROUTE (added by install_web.py)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

@app.get("/", include_in_schema=False)
def web_home():
    return FileResponse(WEB_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = also accept the local network")
    parser.add_argument("--api-key", default=API_KEY, help="required on every request when set")
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost") and not args.api_key:
        parser.error("listening beyond this computer needs --api-key (any long random string)")
    API_KEY = args.api_key
    if args.host not in ("127.0.0.1", "localhost"):
        import socket

        lan_ip = socket.gethostbyname(socket.gethostname())
        print(f"phones on the same Wi-Fi can use http://{lan_ip}:{args.port} with this API key")
    uvicorn.run(app, host=args.host, port=args.port)
