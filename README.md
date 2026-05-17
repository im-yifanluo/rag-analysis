# Hamlet QA Failure Analysis

This repository is a small, inspectable harness for studying context assembly
failure modes in long-document question answering. It uses one public-domain
document, `hamlet.txt`, so that retrieval, evidence coverage, prompt ordering,
and model outputs can be inspected by hand.

Essentially, I made a qualitative
failure analysis testing ground that asks the following:

> Given a fixed context budget, what evidence does a RAG system actually place
> in front of the generator, and how do retrieval, budget, and ordering choices
> affect the answer?

## What This Repo Does

The workflow is:

1. Split Hamlet into stable act/scene chunks.
2. Maintain a small editable question set with required evidence quotes.
3. Derive gold chunk IDs automatically by quote matching.
4. Run closed-book, gold-evidence, dense, sparse, and ordering treatments.
5. Inspect prompts, selected chunks, retrieval scores, evidence recall, quote
   recall, and model outputs for each question.

The current default run uses:

- reader model: `Qwen/Qwen3.5-9B`
- tokenizer model: `Qwen/Qwen3.5-9B`
- embedding model: `Qwen/Qwen3-Embedding-8B`
- reranker model: `Qwen/Qwen3-Reranker-8B`
- chunking: `256` tokens with `64` token overlap
- context budget: `1000` context tokens
- treatments: `closed_book`, `gold_evidence`, `dense_reranked`,
  `dense_document_order`, `dense_random_order`, `sparse_bm25`

## Repository Layout

```text
hamlet_qa/              Python package for chunking, retrieval, experiments, and reports
data/                   Default chunks and editable question file
runs/                   Committed sample run artifacts and local generated runs
tests/                  Unit tests for core behavior
hamlet.txt              Source document used by the harness
requirements.txt        Python package dependencies
environment.yml         Conda environment definition
setup.sh                Research-server setup helper
```

## Setup

Recommended on research servers: use Miniforge/conda so the Python version is
explicit and independent of the system `python`.

```bash
bash setup.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hamlet-qa
```

If activation says `Run 'conda init' before 'conda activate'`, the environment
was still created; your shell has not loaded conda's activation hook. Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hamlet-qa
```

The repo expects Python `3.12`. `.python-version` pins the local pyenv-style
version to `3.12.3`, and `environment.yml` creates a `hamlet-qa` conda
environment with Python `3.12`. If conda/mamba is unavailable, `setup.sh` falls
back to a local `venv/` only when `python3.12` is installed:

```bash
source venv/bin/activate
```

The full experiment uses vLLM for reader-model inference. Unit tests and
prompt-preparation runs do not require vLLM generation. If the PyPI `vllm`
package in `requirements.txt` lags the server's CUDA/PyTorch setup, install a
compatible vLLM wheel for that machine before running generation.

## Regenerate Chunks

```bash
python -m hamlet_qa.build_chunks \
  --document hamlet.txt \
  --output data/hamlet_chunks.jsonl
```

Defaults:

- tokenizer: `Qwen/Qwen3.5-9B`
- chunk size: `256` tokens
- overlap: `64` tokens

Chunk IDs are stable for the default settings, for example
`act03_scene02_chunk004`. If chunking settings change, regenerate chunks and
validate that required evidence quotes still resolve to the intended chunks.

## Run An Experiment

```bash
python -m hamlet_qa.run_experiment \
  --run-name qwen_hamlet_probe
```

Outputs are written to `runs/<run_name>/`:

- `results.jsonl`: one row per question, treatment, and context budget
- `run_config.json`: serialized run configuration
- `hamlet_chunks.jsonl`: chunk file copied into the run directory
- `hamlet_questions_input.json`: original question file copied into the run
- `hamlet_questions_resolved.json`: questions after quote-to-chunk matching

The context budget counts selected context tokens only. Prompt overhead and
generated tokens are logged separately in each result row.

On a 3xA40 server, opt into the multi-GPU placement explicitly:

```bash
python -m hamlet_qa.run_experiment \
  --run-name qwen_hamlet_probe \
  --gpu-layout a40-3gpu
```

That preset uses `cuda:0` for the embedder, `cuda:1` for the reranker, and
`cuda:2` for the vLLM reader. It changes only model placement, not retrieval
selection, reranking, prompt ordering, or scoring logic.

To build prompts and retrieval traces without vLLM generation:

```bash
python -m hamlet_qa.run_experiment \
  --run-name dry_prompts \
  --prepare-only
```

For a model-, embedder-, reranker-, and BM25-free CLI smoke test, restrict the
treatments to `closed_book`; the test suite uses injected stub retrievers for
the heavier retrieval paths.

## Inspect Results

Render a standalone browser viewer:

```bash
python -m hamlet_qa.results_html \
  runs/qwen_hamlet_probe/results.jsonl \
  --output runs/qwen_hamlet_probe/results_viewer.html
```

The viewer groups results by question, filters by treatment or text search, and
shows the stored model output, evidence quotes, selected chunks, retrieval
scores, retrieval traces, prompt text, and expandable evidence chunks.

Render a compact Markdown inspection report:

```bash
python -m hamlet_qa.inspect_results \
  --results runs/qwen_hamlet_probe/results.jsonl \
  --output runs/qwen_hamlet_probe/inspection.md
```

## Editable Data

- `data/hamlet_chunks.jsonl`: generated default chunks.
- `data/hamlet_questions.json`: starter questions and required evidence quotes.

Gold chunk IDs are derived from `required_evidence_quotes` during validation and
written to `hamlet_questions_resolved.json` inside each run directory. This
keeps the editable question file independent from a particular chunking run.

## Reproducibility Notes

- Use Python `3.12` for server runs.
- Keep `run_config.json` with any reported result.
- Report the exact context budget and chunking settings with any metric or
  qualitative example.
- Interpret context budgets relative to document length. A fixed token budget
  can mean sparse retrieval for one document and near-full-document coverage
  for another.
- Ordering comparisons are most informative when the required evidence is
  present in the selected context; ordering cannot repair missing evidence.

## Tests

```bash
python -m unittest discover -s tests
```

The tests cover chunk parsing, question validation, retrieval ranking behavior,
experiment row construction, report rendering, and basic repository hygiene.
