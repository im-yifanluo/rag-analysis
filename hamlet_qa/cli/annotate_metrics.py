"""Annotate a results.jsonl with oracle CI values and sufficient-context labels.

Post-hoc by design: one reader-model load annotates both metrics over all
rows, and past runs can be annotated without rerunning treatments. Results
are written to a `metrics_annotations.jsonl` sidecar; `results.jsonl` is
never modified.

Example:
    python -m hamlet_qa.cli.annotate_metrics \\
        --results runs/qwen_hamlet_probe/results.jsonl \\
        --metrics ci sufficient_context
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hamlet_qa.metrics.annotate import KNOWN_METRICS, annotate_results
from hamlet_qa.metrics.ci import compute_ci_for_row
from hamlet_qa.metrics.evidence_role import compute_evidence_role_recall_for_row
from hamlet_qa.metrics.sufficient_context import compute_sufficient_context_for_row

# Metrics that need the reader model loaded onto the GPU.
_READER_METRICS = {"ci", "sufficient_context"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute post-hoc metrics over a results.jsonl file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(KNOWN_METRICS),
        default=list(KNOWN_METRICS),
    )
    parser.add_argument(
        "--reader-model",
        default=None,
        help=(
            "Scorer/autorater model. Defaults to the reader_model recorded in "
            "run_config.json beside the results file."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute metrics even for rows that already have annotations.",
    )
    return parser.parse_args()


def resolve_reader_model(args: argparse.Namespace) -> str:
    if args.reader_model:
        return args.reader_model
    run_config_path = Path(args.results).parent / "run_config.json"
    if run_config_path.exists():
        with run_config_path.open("r", encoding="utf-8") as handle:
            run_config = json.load(handle)
        model = run_config.get("reader_model")
        if model:
            return str(model)
    raise SystemExit(
        "No --reader-model given and no run_config.json found beside the "
        "results file."
    )


def main() -> None:
    args = parse_args()

    metric_fns = {}
    if "evidence_role" in args.metrics:
        metric_fns["evidence_role"] = compute_evidence_role_recall_for_row

    if any(metric in _READER_METRICS for metric in args.metrics):
        model_name = resolve_reader_model(args)
        print(f"Loading reader model {model_name} for metric annotation...")

        from hamlet_qa.core.generation import VLLMReader

        reader = VLLMReader(
            model_name,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            device=args.device,
        )
        if "ci" in args.metrics:
            metric_fns["ci"] = lambda row: compute_ci_for_row(row, reader)
        if "sufficient_context" in args.metrics:
            metric_fns["sufficient_context"] = lambda row: (
                compute_sufficient_context_for_row(row, reader)
            )

    path = annotate_results(
        args.results,
        metric_fns,
        overwrite=args.overwrite,
        progress=print,
    )
    print(f"Wrote annotations to {path}")


if __name__ == "__main__":
    main()
