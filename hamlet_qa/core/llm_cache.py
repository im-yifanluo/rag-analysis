"""Generic JSON key-value cache for LLM and model outputs.

Generalizes the original SetR selector cache so CRAG query rewrites, MacRAG
chunk summaries, and RECOMP compressor outputs can share one implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class JsonKVCache:
    """JSON-file-backed cache holding one named section of keyed dict values."""

    def __init__(self, path: str | Path | None, section: str):
        self.path = Path(path) if path else None
        self.section = section
        self.data: dict[str, Any] = {section: {}}
        if self.path and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                self.data.update(loaded)

    def _section(self) -> dict[str, Any]:
        section = self.data.setdefault(self.section, {})
        if not isinstance(section, dict):
            section = {}
            self.data[self.section] = section
        return section

    def get(self, cache_key: str) -> dict[str, Any] | None:
        value = self._section().get(cache_key)
        return dict(value) if isinstance(value, dict) else None

    def set(self, cache_key: str, value: dict[str, Any]) -> None:
        self._section()[cache_key] = value

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
