from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

STOPWORDS = {
    "a", "an", "and", "are", "around", "at", "be", "can", "could", "do", "for", "from",
    "have", "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "show", "something",
    "that", "the", "this", "to", "what", "when", "with", "you", "your",
}

SYNONYMS = {
    "hotel": "hotel", "hotels": "hotel", "lodging": "hotel", "stay": "hotel", "stays": "hotel",
    "movie": "media", "movies": "media", "music": "media", "show": "media", "shows": "media",
    "meeting": "calendar", "meetings": "calendar", "scheduled": "calendar", "schedule": "calendar",
    "event": "calendar", "events": "calendar", "appointment": "calendar",
    "forecast": "weather", "rain": "weather", "weather": "weather",
    "document": "document", "documents": "document", "file": "document", "files": "document",
    "note": "document", "notes": "document", "policy": "document",
    "transfer": "bank", "payment": "bank", "payments": "bank", "money": "bank", "account": "bank",
}


def _normalize_token(token: str) -> str:
    token = token.lower()
    if token in SYNONYMS:
        return SYNONYMS[token]
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return SYNONYMS.get(token, token)


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in TOKEN_RE.findall(text):
        lowered = raw.lower()
        # Filter grammatical stopwords before domain normalization so an
        # imperative such as "show hotels" is not misread as the media noun.
        if lowered in STOPWORDS:
            continue
        normalized = _normalize_token(lowered)
        if normalized and normalized not in STOPWORDS:
            tokens.append(normalized)
    return tokens


@dataclass
class BM25Retriever:
    docs: list[dict[str, Any]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.doc_tokens = [tokenize(d.get("text", "") + " " + d.get("title", "")) for d in self.docs]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            for term in set(toks):
                self.df[term] += 1
        self.N = len(self.docs)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query: str, idx: int) -> float:
        q_terms = tokenize(query)
        tf = Counter(self.doc_tokens[idx])
        dl = self.doc_lens[idx] or 1
        score = 0.0
        for term in q_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-6))
            score += self.idf(term) * (freq * (self.k1 + 1) / denom)
        return score

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        scored = []
        for i, doc in enumerate(self.docs):
            scored.append((self.score(query, i), doc))
        scored.sort(key=lambda x: (-x[0], str(x[1].get("doc_id", ""))))
        return [{**doc, "score": float(score), "rank": rank + 1} for rank, (score, doc) in enumerate(scored[:top_k])]
