"""Stage 2: source-faithful candidate evidence-unit construction.

Every unit's text is copied verbatim from source chunks — no abstraction. Units
come in several granularities (whole chunk, sentence, speaker-turn line span,
and neighbor-expanded blocks) so the budgeted assembler can trade off coverage
against token cost. Neighbor expansion never crosses a scene boundary.
"""

from __future__ import annotations

import re
from typing import Any

from hamlet_qa.features.macrag.assembly import combine_without_overlap
from hamlet_qa.features.recomp.compressor import split_sentences
from hamlet_qa.features.reader_support.schema import EvidenceUnit

DEFAULT_UNIT_TYPES = ["chunk", "sentence", "line_span", "neighbor_left", "neighbor_right"]
_MIN_UNIT_WORDS = 3


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def word_count(text: str) -> int:
    return len(text.split())


def split_blocks(text: str) -> list[str]:
    """Speaker-turn / stage-direction blocks (blank-line separated).

    For Hamlet's dialogue this keeps the speaker label attached to its lines,
    which a plain sentence split would scatter.
    """
    blocks = re.split(r"\n\s*\n", text or "")
    return [block.strip() for block in blocks if block.strip()]


def _order_key(global_index: int | None, sub: int) -> list[int]:
    return [int(global_index) if global_index is not None else 10**9, sub]


def _make_unit(
    unit_id: str,
    unit_type: str,
    text: str,
    source_chunk_ids: list[str],
    primary_chunk: dict[str, Any],
    token_count: int,
    sub: int,
    global_index_start: int | None,
    global_index_end: int | None,
    metadata: dict[str, Any] | None = None,
) -> EvidenceUnit:
    return EvidenceUnit(
        unit_id=unit_id,
        unit_type=unit_type,
        text=text,
        source_chunk_ids=source_chunk_ids,
        primary_chunk_id=str(primary_chunk["chunk_id"]),
        token_count=token_count,
        source_order_key=_order_key(global_index_start, sub),
        act=primary_chunk.get("act"),
        scene=primary_chunk.get("scene"),
        scene_title=primary_chunk.get("scene_title"),
        scene_id=primary_chunk.get("scene_id"),
        global_index_start=global_index_start,
        global_index_end=global_index_end,
        metadata=metadata or {},
    )


def _neighbor_unit(
    unit_type: str,
    primary_chunk: dict[str, Any],
    members: list[dict[str, Any]],
) -> EvidenceUnit:
    merged = ""
    for member in members:
        merged = combine_without_overlap(merged, str(member["text"]))
    member_ids = [str(member["chunk_id"]) for member in members]
    suffix = {"neighbor_left": "nl", "neighbor_right": "nr", "neighbor_both": "nb"}[unit_type]
    return _make_unit(
        unit_id=f"{primary_chunk['chunk_id']}::{suffix}",
        unit_type=unit_type,
        text=merged,
        source_chunk_ids=member_ids,
        primary_chunk=primary_chunk,
        token_count=word_count(merged),
        sub=0,
        global_index_start=int(members[0]["global_index"]),
        global_index_end=int(members[-1]["global_index"]),
        metadata={"member_chunk_ids": member_ids},
    )


def build_units(
    retrieval_trace: list[dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]],
    candidate_chunks: int,
    unit_types: list[str],
    include_neighbors: bool,
    neighbor_hops: int,
    max_unit_tokens: int,
    max_units_total: int,
) -> dict[str, Any]:
    """Build, dedupe, and cap candidate units. Returns units + a drop log."""
    by_global_index = {
        int(chunk["global_index"]): chunk for chunk in chunk_lookup.values()
    }
    candidate_ids = [
        str(row["chunk_id"])
        for row in retrieval_trace[:candidate_chunks]
        if str(row["chunk_id"]) in chunk_lookup
    ]
    candidates = [chunk_lookup[chunk_id] for chunk_id in candidate_ids]
    want = set(unit_types)
    dropped: dict[str, int] = {"oversize": 0, "too_short": 0, "duplicate": 0, "capped": 0}

    # Generate in rounds so every candidate keeps its whole-chunk unit even when
    # the total cap bites; finer units fill the remainder.
    rounds: list[EvidenceUnit] = []

    if "chunk" in want:
        for chunk in candidates:
            rounds.append(
                _make_unit(
                    unit_id=f"{chunk['chunk_id']}::chunk",
                    unit_type="chunk",
                    text=str(chunk["text"]),
                    source_chunk_ids=[str(chunk["chunk_id"])],
                    primary_chunk=chunk,
                    token_count=int(chunk["token_count"]),
                    sub=0,
                    global_index_start=int(chunk["global_index"]),
                    global_index_end=int(chunk["global_index"]),
                )
            )

    if include_neighbors:
        for chunk in candidates:
            gi = int(chunk["global_index"])
            scene_id = str(chunk["scene_id"])
            left = by_global_index.get(gi - 1)
            right = by_global_index.get(gi + 1)
            left_ok = left is not None and str(left["scene_id"]) == scene_id
            right_ok = right is not None and str(right["scene_id"]) == scene_id
            if "neighbor_left" in want and left_ok:
                rounds.append(_neighbor_unit("neighbor_left", chunk, [left, chunk]))
            if "neighbor_right" in want and right_ok:
                rounds.append(_neighbor_unit("neighbor_right", chunk, [chunk, right]))
            if "neighbor_both" in want and left_ok and right_ok:
                rounds.append(
                    _neighbor_unit("neighbor_both", chunk, [left, chunk, right])
                )

    if "line_span" in want:
        for chunk in candidates:
            gi = int(chunk["global_index"])
            for index, block in enumerate(split_blocks(str(chunk["text"]))):
                rounds.append(
                    _make_unit(
                        unit_id=f"{chunk['chunk_id']}::span{index:02d}",
                        unit_type="line_span",
                        text=block,
                        source_chunk_ids=[str(chunk["chunk_id"])],
                        primary_chunk=chunk,
                        token_count=word_count(block),
                        sub=1000 + index,
                        global_index_start=gi,
                        global_index_end=gi,
                    )
                )

    if "sentence" in want:
        for chunk in candidates:
            gi = int(chunk["global_index"])
            for index, sentence in enumerate(split_sentences(str(chunk["text"]))):
                rounds.append(
                    _make_unit(
                        unit_id=f"{chunk['chunk_id']}::sent{index:02d}",
                        unit_type="sentence",
                        text=sentence,
                        source_chunk_ids=[str(chunk["chunk_id"])],
                        primary_chunk=chunk,
                        token_count=word_count(sentence),
                        sub=1 + index,
                        global_index_start=gi,
                        global_index_end=gi,
                    )
                )

    units: list[EvidenceUnit] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for unit in rounds:
        if unit.token_count > max_unit_tokens:
            dropped["oversize"] += 1
            continue
        if unit.unit_type != "chunk" and word_count(unit.text) < _MIN_UNIT_WORDS:
            dropped["too_short"] += 1
            continue
        key = (normalize_text(unit.text), tuple(sorted(unit.source_chunk_ids)))
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        if len(units) >= max_units_total:
            dropped["capped"] += 1
            continue
        units.append(unit)

    type_counts: dict[str, int] = {}
    for unit in units:
        type_counts[unit.unit_type] = type_counts.get(unit.unit_type, 0) + 1

    return {
        "units": units,
        "candidate_chunk_ids": candidate_ids,
        "num_candidate_chunks": len(candidate_ids),
        "unit_type_counts": type_counts,
        "dropped": dropped,
    }
