"""Compact candidate catalog from a retrieval trace (shared)."""

from __future__ import annotations

import re
from typing import Any


def build_candidate_catalog(
    retrieval_trace: list[dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]],
    catalog_k: int,
    excerpt_chars: int = 200,
) -> str:
    """Compact catalog of top candidates: id, location, short excerpt.

    Deliberately excludes any answer/gold information — only retrieved source
    metadata and a leading excerpt.
    """
    lines: list[str] = []
    for row in retrieval_trace[:catalog_k]:
        chunk_id = str(row.get("chunk_id"))
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None:
            continue
        act = chunk.get("act")
        scene = chunk.get("scene")
        title = str(chunk.get("scene_title", "")).strip()
        excerpt = re.sub(r"\s+", " ", str(chunk.get("text", "")).strip())[:excerpt_chars]
        lines.append(f"- [{chunk_id}] Act {act} Scene {scene} ({title}): {excerpt}")
    return "\n".join(lines) if lines else "(no candidates)"
