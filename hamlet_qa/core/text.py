"""Small deterministic text helpers used by context assembly prototypes."""

from __future__ import annotations

import re
from typing import Any


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize_terms(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def flatten_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(key) for key, enabled in value.items() if enabled]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return []


def phrase_in_text(text: str, phrase: str) -> bool:
    phrase = phrase.strip().lower()
    if not phrase:
        return False
    tokens = tokenize_terms(phrase)
    if not tokens:
        return False
    pattern = r"(?<![a-z0-9])" + r"\s+".join(re.escape(token) for token in tokens)
    pattern += r"(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None
