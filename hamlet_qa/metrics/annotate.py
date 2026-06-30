"""Post-hoc metric annotation: sidecar storage and orchestration.

Annotations are written to `metrics_annotations.jsonl` next to the
`results.jsonl` they describe, keyed by (question_id, treatment,
context_budget). `results.jsonl` itself is never rewritten; readers merge the
sidecar fields at load time (`load_results_with_annotations`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from hamlet_qa.core.io import load_jsonl

ANNOTATIONS_FILENAME = "metrics_annotations.jsonl"
KNOWN_METRICS = ("ci", "sufficient_context", "evidence_role", "plan_eval")

# Field whose presence in an annotation marks a metric as already computed for a
# row (used to skip recomputation unless overwrite is set).
METRIC_MARKER_FIELD = {
    "ci": "ci_base_loss",
    "sufficient_context": "sufficient_context",
    "evidence_role": "evidence_role_recall",
    "plan_eval": "plan_eval_applicable",
}


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["question_id"]),
        str(row["treatment"]),
        int(row["context_budget"]),
    )


def annotations_path_for(results_path: str | Path) -> Path:
    return Path(results_path).parent / ANNOTATIONS_FILENAME


def load_annotations(results_path: str | Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    path = annotations_path_for(results_path)
    if not path.exists():
        return {}
    annotations: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in load_jsonl(path):
        annotations[row_key(row)] = dict(row)
    return annotations


def save_annotations(
    results_path: str | Path,
    annotations: dict[tuple[str, str, int], dict[str, Any]],
) -> Path:
    path = annotations_path_for(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in annotations.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def merge_annotations_into_rows(
    rows: list[dict[str, Any]],
    annotations: dict[tuple[str, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row in rows:
        annotation = annotations.get(row_key(row))
        if annotation:
            extra = {
                key: value
                for key, value in annotation.items()
                if key not in {"question_id", "treatment", "context_budget"}
            }
            merged.append(dict(row, **extra))
        else:
            merged.append(dict(row))
    return merged


def annotate_results(
    results_path: str | Path,
    metric_fns: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Compute metric fields for every result row and update the sidecar.

    metric_fns maps a metric name to a function(row) -> annotation fields.
    Existing annotations are kept; a metric is recomputed for a row only when
    any of its fields are missing, or when overwrite is set.
    """
    results_path = Path(results_path)
    rows = load_jsonl(results_path)
    annotations = load_annotations(results_path)

    for index, row in enumerate(rows, start=1):
        key = row_key(row)
        record = annotations.get(
            key,
            {
                "question_id": key[0],
                "treatment": key[1],
                "context_budget": key[2],
            },
        )
        for metric_name, metric_fn in metric_fns.items():
            marker_field = METRIC_MARKER_FIELD.get(metric_name, metric_name)
            if not overwrite and marker_field in record:
                continue
            record.update(metric_fn(row))
        annotations[key] = record
        if progress is not None:
            progress(f"[{index}/{len(rows)}] {key[0]} / {key[1]}")

    return save_annotations(results_path, annotations)
