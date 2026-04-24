"""CLI for creating the default Hamlet chunk file."""

from __future__ import annotations

import argparse

from hamlet_qa.chunking import build_chunks, load_tokenizer, write_chunks
from hamlet_qa.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOKENIZER_MODEL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build token-window chunks from hamlet.txt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--document", default="hamlet.txt")
    parser.add_argument("--output", default="data/hamlet_chunks.jsonl")
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_model)
    chunks = build_chunks(
        args.document,
        tokenizer,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    write_chunks(args.output, chunks)
    print(f"Wrote {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
