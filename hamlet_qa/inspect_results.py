"""CLI for rendering a markdown inspection report from result JSONL."""

from __future__ import annotations

import argparse

from hamlet_qa.report import write_inspection_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Hamlet QA inspection report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = write_inspection_report(args.results, args.output)
    print(f"Wrote inspection report to {output_path}")


if __name__ == "__main__":
    main()
