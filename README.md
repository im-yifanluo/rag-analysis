# Hamlet QA Failure Analysis

This repo is a small, inspectable harness for studying why long-document QA
fails on one document: `hamlet.txt`.

The workflow is:

1. Split Hamlet into stable act/scene chunks.
2. Keep a small editable question set with required evidence quotes.
3. Derive gold chunk IDs automatically by quote matching.
4. Run closed-book, gold-evidence, neighbor, and dense-retrieval treatments.
5. Inspect the full prompt, selected chunks, retrieval scores, evidence recall,
   quote recall, and model output for every question.

## Setup

Recommended on the research servers: use Miniforge/conda so the Python version
is explicit and independent of the system `python`.

```bash
bash setup.sh
conda activate hamlet-qa
```

The repo expects Python `3.12`; `.python-version` pins the local pyenv-style
version to `3.12.3`, and `environment.yml` creates a `hamlet-qa` conda env with
Python `3.12`. If conda/mamba is not available, `setup.sh` falls back to a local
`venv/` only when `python3.12` is installed:

```bash
source venv/bin/activate
```

The server run uses vLLM for generation. Implementation and tests do not run
reader-model inference.

## Regenerate Chunks

```bash
python -m hamlet_qa.build_chunks \
  --document hamlet.txt \
  --output data/hamlet_chunks.jsonl
```

Defaults:

- tokenizer: `Qwen/Qwen2.5-7B-Instruct` (kept to preserve the current chunk baseline)
- chunk size: `256` tokens
- overlap: `64` tokens

Chunk IDs are stable for the default settings, for example
`act03_scene02_chunk004`.

The current chunking intentionally stays simple for the first diagnostic run.
TODO: future chunking should use speaker-turn-aware fixed-target chunks with a
hard max token cutoff.

## Run An Experiment

```bash
python -m hamlet_qa.run_experiment \
  --run-name first_hamlet_probe
```

Defaults:

- reader model: `Qwen/Qwen3.5-9B`
- embedding model: `Snowflake/snowflake-arctic-embed-m-v1.5`
- temperature: `0.0`
- treatments: `closed_book gold_evidence gold_evidence_neighbors dense_relevance dense_relevance_neighbors`
- context budgets: `1000`
- neighbor window: `1`

Outputs are written to `runs/<run_name>/results.jsonl`, with copies of the
config, chunks, input questions, and quote-resolved questions used for the run.
The context budget counts context tokens only. Prompt overhead and generated
tokens are logged separately.

To build prompts and traces without vLLM generation:

```bash
python -m hamlet_qa.run_experiment --run-name dry_prompts --prepare-only
```

This still performs dense retrieval for grounded treatments so relevance
ordering is available. For a model- and embedder-free CLI smoke test, restrict
treatments to `closed_book`; tests can inject a cached/stub retriever.

## Inspect Results

```bash
python -m hamlet_qa.inspect_results \
  --results runs/first_hamlet_probe/results.jsonl \
  --output runs/first_hamlet_probe/inspection.md
```

## Editable Data

- `data/hamlet_chunks.jsonl`: generated default chunks.
- `data/hamlet_questions.json`: starter questions and required evidence quotes.

Gold chunk IDs are derived from `required_evidence_quotes` during validation and
written to `hamlet_questions_resolved.json` inside each run directory. If
chunking settings change, regenerate chunks and confirm the quote validation
still resolves the intended evidence.

## Tests

```bash
python -m unittest discover -s tests
```
