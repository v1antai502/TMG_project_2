"""Collect the article body for every row of the processed dataset from its URL.

    python -m utils.collect_bodies            # everything still missing
    python -m utils.collect_bodies --limit 5  # first 5 articles per source (test run)

Writes data/bodies/bodies.jsonl, one line per article, and can be re-run: articles that
already have a final result are skipped. One worker per source, so every site gets at
most one request per `--delay` seconds.

Báo Mới is an aggregator that no longer serves the article text; its pages still link to
the original publisher, so those are fetched in a second step.

The bodies are the publishers' copyrighted text: data/bodies/ is git-ignored on purpose.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from utils.dataloader import ROOT, load_processed

OUT_PATH = ROOT / "data" / "bodies" / "bodies.jsonl"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    ),
    "Accept-Language": "vi,en;q=0.8",
}
MIN_PARAGRAPH_CHARS = 40
MIN_BODY_WORDS = 80  # shorter results are kept but marked "short"
RETRY_STATUS = {429, 500, 502, 503, 504}
FINAL_STATUS = {"ok", "short", "gone", "not_found", "no_original"}  # not retried on a re-run


def _tokens(text: str) -> set[str]:
    return set(re.sub(r"\W+", " ", text.lower()).split())


def extract_body(html: bytes | str) -> tuple[str, str]:
    """(body, page title). The body is the element holding the most paragraph text."""
    soup = BeautifulSoup(html, "html.parser")
    meta = soup.find("meta", property="og:title")
    title = meta.get("content", "") if meta else (soup.title.get_text(strip=True) if soup.title else "")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "figure"]):
        tag.decompose()

    def paragraphs(node, recursive):
        texts = (p.get_text(" ", strip=True) for p in node.find_all("p", recursive=recursive))
        return [t for t in texts if len(t) >= MIN_PARAGRAPH_CHARS]

    best: list[str] = []
    for node in soup.find_all(["article", "div", "section", "main"]):
        paras = paragraphs(node, recursive=False)
        if sum(map(len, paras)) > sum(map(len, best)):
            best = paras
    if sum(map(len, best)) < 400:  # paragraphs nested deeper than one level
        container = soup.find("article") or soup.find("main") or soup
        nested = paragraphs(container, recursive=True)
        if sum(map(len, nested)) > sum(map(len, best)):
            best = nested
    return "\n".join(best), title or ""


def fetch(session: requests.Session, url: str, delay: float, retries: int = 2):
    """GET with retries on transient failures. Returns the response or raises."""
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=25, allow_redirects=True)
            if resp.status_code not in RETRY_STATUS or attempt == retries:
                return resp
        except requests.RequestException:
            if attempt == retries:
                raise
        time.sleep(delay * (attempt + 2))
    raise RuntimeError("unreachable")


def is_homepage(url: str) -> bool:
    return urlparse(url).path.strip("/") == ""


def collect_one(session: requests.Session, row: dict, delay: float) -> dict:
    out = {"id": row["id"], "source": row["source"], "url": row["url"], "status": "error", "body": ""}
    try:
        resp = fetch(session, row["url"], delay)
        out["http_status"] = resp.status_code
        if resp.status_code in (404, 410):
            out["status"] = "not_found"
            return out
        if not resp.ok:
            return out
        if is_homepage(resp.url) and not is_homepage(row["url"]):
            out["status"] = "gone"  # article removed, redirected to the front page
            return out

        if row["source"] == "Báo Mới":
            match = re.search(r'"originalUrl":"(.*?)"', resp.text)
            if not match:
                out["status"] = "no_original"
                return out
            out["original_url"] = json.loads(f'"{match.group(1)}"')
            time.sleep(delay)
            resp = fetch(session, out["original_url"], delay)
            out["http_status"] = resp.status_code
            if resp.status_code in (404, 410):
                out["status"] = "not_found"
                return out
            if not resp.ok:
                return out
            if is_homepage(resp.url):
                out["status"] = "gone"
                return out

        body, page_title = extract_body(resp.content)
        wanted = _tokens(row["title"])
        out.update(
            final_url=resp.url,
            page_title=page_title[:300],
            title_overlap=round(len(wanted & _tokens(page_title)) / max(len(wanted), 1), 2),
            body=body,
            body_words=len(body.split()),
        )
        out["status"] = "ok" if out["body_words"] >= MIN_BODY_WORDS else "short"
    except (requests.RequestException, ValueError) as err:
        out["error"] = type(err).__name__
    return out


def load_done(path: Path) -> dict[str, str]:
    """Latest status per article id already in the output file."""
    done: dict[str, str] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["id"]] = rec["status"]
    return done


def load_bodies(path: Path = OUT_PATH) -> pd.DataFrame:
    """One row per article (the latest attempt), for use in training."""
    records: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records[rec["id"]] = rec
    return pd.DataFrame(records.values())


def collect(limit: int = 0, delay: float = 1.0, out_path: Path = OUT_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    articles = pd.concat(load_processed().values(), ignore_index=True)[["id", "source", "url", "title"]]
    done = load_done(out_path)
    todo = articles[~articles["id"].map(lambda i: done.get(i) in FINAL_STATUS)]
    print(f"{len(articles)} articles | {len(articles) - len(todo)} already done", end=" | ")
    if limit:
        todo = todo.groupby("source", group_keys=False).head(limit)
    print(f"fetching {len(todo)}")

    lock = threading.Lock()
    counts: dict[str, int] = {}
    started = time.time()

    def work(group: pd.DataFrame) -> None:
        session = requests.Session()
        with open(out_path, "a", encoding="utf-8") as f:
            for row in group.to_dict("records"):
                rec = collect_one(session, row, delay)
                rec["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    counts[rec["status"]] = counts.get(rec["status"], 0) + 1
                    n = sum(counts.values())
                    if n % 100 == 0 or n == len(todo):
                        print(f"{n}/{len(todo)} | {dict(sorted(counts.items()))} | {time.time() - started:.0f}s", flush=True)
                time.sleep(delay)

    groups = [g for _, g in todo.groupby("source")]
    with ThreadPoolExecutor(max_workers=max(len(groups), 1)) as pool:
        list(pool.map(work, groups))

    summary = load_bodies(out_path)
    print("\nstatus by source:")
    print(pd.crosstab(summary["source"], summary["status"], margins=True).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--limit", type=int, default=0, help="articles per source (0 = all)")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests to one site")
    args = parser.parse_args()
    collect(args.limit, args.delay)
