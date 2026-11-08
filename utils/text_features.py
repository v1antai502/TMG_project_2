"""Hand-crafted "informativeness" features for Vietnamese headlines.

A Vietnamese adaptation of the 15-25 measures in Michaluk et al., "Click it or Leave it"
(arXiv 2602.18171): length, capitalisation, punctuation, numbers, pronouns, superlatives,
speculation, bait phrases, lexical diversity and title-text similarity. English-only
measures (POS-tagged superlatives, TextBlob sentiment, Flesch reading ease) are replaced by
word lists or dropped.

The word lists are general Vietnamese clickbait vocabulary written up front. A few of them
also appeared in this dataset's EDA, which looked at all splits, so scores using these
features may be slightly optimistic.

    feats = text_features(df)                # one row per article, all numeric
    feats = text_features(df, bodies=bodies) # adds title-body similarity
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Phrases are matched on lower-cased text with word boundaries; multi-word entries count once.
SECOND_PERSON = ["bạn", "các bạn", "bạn đọc", "mình"]
DEMONSTRATIVES = ["này", "đây", "điều này", "thế này", "như vậy", "đó", "ấy"]
QUESTION_WORDS = ["vì sao", "tại sao", "là gì", "ra sao", "thế nào", "như thế nào", "ai", "gì", "bao nhiêu", "liệu"]
SUPERLATIVES = ["nhất", "cực", "cực kỳ", "siêu", "vô cùng", "quá", "khủng", "đỉnh", "cực đỉnh", "hết sức", "chưa từng", "kỷ lục"]
SPECULATIVES = ["có thể", "liệu", "dường như", "có lẽ", "được cho là", "nghi", "nghi vấn", "lộ diện", "rộ tin"]
BAIT_PHRASES = [
    "sốc", "bất ngờ", "gây sốt", "xôn xao", "choáng", "ngỡ ngàng", "hé lộ", "tiết lộ", "lộ", "dân mạng",
    "cư dân mạng", "khó tin", "không thể tin", "ai cũng", "sự thật", "bí mật", "bí quyết", "mẹo", "đừng",
    "phải biết", "nên biết", "không ngờ", "gây tranh cãi", "tranh cãi", "netizen", "viral", "hot", "cực hot",
    "thót tim", "rùng mình", "điếng người", "ngã ngửa", "dở khóc dở cười", "cười ngất", "đáng sợ", "nóng",
]
FUNCTION_WORDS = [
    "của", "và", "là", "có", "được", "cho", "với", "các", "những", "một", "này", "đã", "sẽ", "không",
    "trong", "khi", "thì", "mà", "để", "từ", "về", "ra", "vào", "lại", "cũng", "như", "đến", "bị", "nhiều",
]
BAIT_PUNCT = set('!?"“”‘’\'(…#')
NON_BAIT_PUNCT = set("$%&,.;:-/–")


def _count_phrases(text: str, phrases: list[str]) -> int:
    return sum(len(re.findall(rf"(?<!\w){re.escape(p)}(?!\w)", text)) for p in phrases)


def _syllables(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.lower())


def _overlap(a: str, b: str) -> float:
    """Share of the title's distinct syllables that also occur in the other text."""
    sa, sb = set(_syllables(a)), set(_syllables(b))
    return len(sa & sb) / len(sa) if sa and sb else np.nan


def headline_features(title: str) -> dict[str, float]:
    letters = [c for c in title if c.isalpha()]
    words = title.split()
    syllables = _syllables(title)
    lower = title.lower()
    n_words = max(len(words), 1)
    return {
        "char_count": len(title),
        "word_count": len(words),
        "mean_word_length": np.mean([len(w) for w in syllables]) if syllables else 0.0,
        "capital_ratio": sum(c.isupper() for c in letters) / max(len(letters), 1),
        "all_caps_words": sum(len(w) >= 3 and w.isupper() for w in re.findall(r"\w+", title)),
        "punctuation_ratio": sum(not c.isalnum() and not c.isspace() for c in title) / max(len(title), 1),
        "bait_punct_count": sum(c in BAIT_PUNCT for c in title),
        "non_bait_punct_count": sum(c in NON_BAIT_PUNCT for c in title),
        "question_mark": title.count("?"),
        "exclamation_mark": title.count("!"),
        "quote_count": sum(c in "\"“”'‘’" for c in title),
        "ellipsis": int("..." in title or "…" in title),
        "has_colon": int(":" in title),
        "numbers_count": len(re.findall(r"\d+(?:[.,]\d+)*", title)),
        "starts_with_number": int(bool(re.match(r"\s*\d", title))),
        "second_person": _count_phrases(lower, SECOND_PERSON),
        "demonstratives": _count_phrases(lower, DEMONSTRATIVES),
        "question_words": _count_phrases(lower, QUESTION_WORDS),
        "superlatives": _count_phrases(lower, SUPERLATIVES),
        "speculatives": _count_phrases(lower, SPECULATIVES),
        "bait_phrases": _count_phrases(lower, BAIT_PHRASES),
        "function_word_ratio": _count_phrases(lower, FUNCTION_WORDS) / n_words,
        "type_token_ratio": len(set(syllables)) / max(len(syllables), 1),
    }


def text_features(df: pd.DataFrame, bodies: dict[str, str] | None = None) -> pd.DataFrame:
    """Feature table indexed like `df`. Uses `title` and `lead_paragraph` (unsegmented text)."""
    rows = []
    for article_id, title, lead in zip(df["id"], df["title"], df["lead_paragraph"]):
        row = headline_features(title)
        row["lead_word_count"] = len(lead.split())
        row["title_lead_overlap"] = _overlap(title, lead)
        if bodies is not None:
            body = " ".join(bodies.get(article_id, "").split()[:400])
            row["title_body_overlap"] = _overlap(title, body) if body else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=df.index)
