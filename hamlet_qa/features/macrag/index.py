"""MacRAG offline artifacts: summary slicing, storage, and loading.

The slicer ports LangChain's RecursiveCharacterTextSplitter behavior used by
`third_party/MacRAG/MacRAG/src/gen_index_macrag.py` step 3 (chunk_size=450
chars, overlap=300, separators ["\\n\\n", "\\n", " ", ""]). Artifacts are plain
JSONL so they stay embedder-agnostic; embeddings are recomputed at run time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hamlet_qa.core.io import load_jsonl

SUMMARIES_FILENAME = "hamlet_macrag_summaries.jsonl"
SLICES_FILENAME = "hamlet_macrag_slices.jsonl"
DEFAULT_SLICE_SEPARATORS = ["\n\n", "\n", " ", ""]


def _split_by_separator(text: str, separator: str) -> list[str]:
    if separator:
        return text.split(separator)
    return list(text)


def recursive_character_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """Port of RecursiveCharacterTextSplitter.split_text (character length)."""
    separators = separators or DEFAULT_SLICE_SEPARATORS

    def split_text(text: str, separators: list[str]) -> list[str]:
        final_chunks: list[str] = []
        separator = separators[-1]
        remaining: list[str] = []
        for index, candidate in enumerate(separators):
            if candidate == "" or candidate in text:
                separator = candidate
                remaining = separators[index + 1 :]
                break
        splits = [s for s in _split_by_separator(text, separator) if s]

        good_splits: list[str] = []
        for split in splits:
            if len(split) < chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    final_chunks.extend(merge_splits(good_splits, separator))
                    good_splits = []
                if not remaining:
                    final_chunks.append(split)
                else:
                    final_chunks.extend(split_text(split, remaining))
        if good_splits:
            final_chunks.extend(merge_splits(good_splits, separator))
        return final_chunks

    def merge_splits(splits: list[str], separator: str) -> list[str]:
        separator_len = len(separator)
        docs: list[str] = []
        current: list[str] = []
        total = 0
        for split in splits:
            split_len = len(split)
            if current and total + split_len + separator_len > chunk_size:
                docs.append(separator.join(current).strip())
                while total > chunk_overlap or (
                    current and total + split_len + separator_len > chunk_size
                ):
                    total -= len(current[0]) + (separator_len if len(current) > 1 else 0)
                    current.pop(0)
            current.append(split)
            total += split_len + (separator_len if len(current) > 1 else 0)
        if current:
            docs.append(separator.join(current).strip())
        return [doc for doc in docs if doc]

    return split_text(text, separators)


def build_slices_for_chunk(
    chunk_id: str,
    summary_record: dict[str, Any],
    slice_size: int,
    slice_overlap: int,
) -> list[dict[str, Any]]:
    """Slice one chunk summary, plus one metadata slice (official step 3)."""
    slices: list[dict[str, Any]] = []
    summary_pieces = recursive_character_split(
        str(summary_record.get("summary", "")),
        chunk_size=slice_size,
        chunk_overlap=slice_overlap,
    )
    for index, piece in enumerate(summary_pieces):
        slices.append(
            {
                "slice_id": f"{chunk_id}_summary{index:02d}",
                "parent_chunk_id": chunk_id,
                "slice_index": index,
                "slice_kind": "summary",
                "text": piece,
            }
        )
    metadata_text = " ".join(
        part
        for part in (
            str(summary_record.get("title", "")).strip(),
            str(summary_record.get("keywords", "")).strip(),
            str(summary_record.get("subheadings", "")).strip(),
        )
        if part
    )
    if metadata_text:
        slices.append(
            {
                "slice_id": f"{chunk_id}_meta",
                "parent_chunk_id": chunk_id,
                "slice_index": len(summary_pieces),
                "slice_kind": "metadata",
                "text": metadata_text,
            }
        )
    return slices


def load_macrag_artifacts(artifacts_dir: str | Path) -> dict[str, Any]:
    artifacts_dir = Path(artifacts_dir)
    slices_path = artifacts_dir / SLICES_FILENAME
    summaries_path = artifacts_dir / SUMMARIES_FILENAME
    if not slices_path.exists():
        raise FileNotFoundError(
            f"MacRAG artifacts not found at {slices_path}. Build them first: "
            "python -m hamlet_qa.cli.build_macrag_index"
        )
    slices = load_jsonl(slices_path)
    summaries = load_jsonl(summaries_path) if summaries_path.exists() else []
    if not slices:
        raise ValueError(f"MacRAG slice file is empty: {slices_path}")
    return {"slices": slices, "summaries": summaries}
