# Hamlet QA — A Harness for Post-Retrieval Context Assembly Research

A small, fully inspectable RAG harness for studying **why long-document QA
fails, and which post-retrieval context-assembly method fixes it**, on a single
controlled document: `data/hamlet.txt`.

---

## 1. Goal

Retrieval-augmented generation on a long document has two failure surfaces that
usually get measured together and confused:

1. **Assembly failure** — the right evidence never makes it into the prompt
   (bad retrieval, bad ranking, bad truncation, lost-in-the-middle).
2. **Reader failure** — the evidence is in the prompt, but the model still
   answers wrong.

This repo isolates those surfaces. It fixes the document, fixes a small set of
questions with hand-verified evidence quotes, derives gold chunks automatically,
and then runs many **context-assembly treatments** through one identical
pipeline so the *only* thing that changes between runs is how candidate chunks
are turned into prompt context. Every prompt, selected chunk, retrieval score,
evidence-recall number, and model output is logged for inspection.

The harness is built to answer questions like:

- Does a corrective re-retrieval method (CRAG) actually recover when retrieval
  is lexically trapped onto the wrong scene?
- Does hierarchical/merged retrieval (MacRAG) or set-selection (SetR) help on
  multi-scene character arcs?
- Does context compression (RECOMP) win when the minimal evidence is larger
  than the token budget?
- For each wrong answer, was the context *sufficient* (reader failure) or not
  (assembly failure)? Which chunks actually carried the answer (oracle
  contextual influence)?

---

## 2. Quickstart

### Full experiment (canonical end-to-end sequence)

Run these in order on the GPU server, inside the activated `hamlet-qa` env. Set
`RUN=...` once and `LAYOUT=...` to your GPU preset (`single`, `a40-2gpu`, or
`a40-3gpu`). Each step's details are in the linked sections.

```bash
# 0. Install + activate (one-time; see §5 for GPU notes)
bash setup.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hamlet-qa

RUN=probe_v2
LAYOUT=a40-2gpu

# 1. Build the MacRAG slice index (one-time; REQUIRED — `macrag` is a default
#    treatment and will error without it). See §6.2.
python -m hamlet_qa.cli.build_macrag_index

# 2. Cheap wiring smoke test: builds prompts + traces for all treatments,
#    skips answer generation. See §6.3.
python -m hamlet_qa.cli.run_experiment --run-name "${RUN}_dry" \
  --prepare-only --gpu-layout "$LAYOUT"

# 3. Full experiment: all 12 treatments × 10 questions. See §6.3.
python -m hamlet_qa.cli.run_experiment --run-name "$RUN" --gpu-layout "$LAYOUT"

# 4. Post-hoc metrics (sidecar; never rewrites results.jsonl). See §6.4.
python -m hamlet_qa.cli.annotate_metrics \
  --results "runs/$RUN/results.jsonl" --metrics ci sufficient_context

# 5. Render the interactive viewer. See §6.5.
python -m hamlet_qa.inspection.results_html \
  "runs/$RUN/results.jsonl" --output "runs/$RUN/results_viewer.html"
```

Notes:

- Skip step 1 only if `data/macrag/` artifacts already exist, **or** if you drop
  `macrag` from `--treatments`.
- CRAG thresholds use baked defaults (upper `2.5`, lower `0.875`). To re-derive
  them against this run's candidates, run `python -m hamlet_qa.cli.calibrate_crag`
  after step 3 and re-run (see §4.2).
- Only regenerate chunks (§6.1) if you change the chunker; gold-quote resolution
  must still pass afterward.

### Tests (CPU-only, no GPU)

```bash
python -m unittest discover -s tests
```

---

## 3. Design

### 3.1 The single experiment boundary

Everything in the pipeline is held constant except one stage:
**post-retrieval context assembly** — the function that turns candidate chunks
into the prompt-ready context. The pipeline always follows the same path:

```
load chunks + questions
        │
        ▼
build retrieval traces        (dense FAISS + reranker, BM25, or MacRAG slices)
        │
        ▼
context assembly  ◄────────────  the ONLY stage that varies between treatments
   (features/<method>/assemble_*)
        │
        ▼
build reader prompt           (shared HamletQAPromptBuilder)
        │
        ▼
generate (vLLM)  +  log everything uniformly  →  runs/<name>/results.jsonl
```

Because only the assembly stage changes, differences in evidence recall and
answer quality are attributable to the method, not to incidental pipeline drift.

### 3.2 Repository layout

```
hamlet_qa/
  core/          stable pipeline shared by every method
    evidence/        shared evidence primitives (schema, greedy coverage,
                     candidate catalog, reader-as-judge support teacher) — so
                     features depend on core, never on each other
    config.py        run configuration + CLI defaults + GPU layouts
    chunking.py      act/scene token chunker
    questions.py     question loading + quote→gold-chunk resolution
    retrieval.py     dense (FAISS) + sparse (BM25) + reranker
    context.py       ContextAssemblyRequest / ContextAssemblyResult contract
    prompts.py       shared reader prompt builder
    generation.py    vLLM reader (incl. score_completion for CI)
    experiment.py    orchestration: stages, GPU staging, logging
    llm_cache.py     JSON KV cache reused by SetR/CRAG/MacRAG/RECOMP
  features/      context-assembly methods (one folder each)
    baseline/  ordering/  setr/  domain/  crag/  macrag/  recomp/
    reader_support/  (ours: nodes, units, teacher, assembly, schema)
    evidence_plan/   (ours: decomposition/planner prompts, contract, executor)
    registry.py      maps treatment name → adapter + retrieval needs
  metrics/       post-hoc per-row metrics (CI value, sufficient context,
                 evidence-role recall)
  cli/           build_chunks, run_experiment, build_macrag_index,
                 calibrate_crag, annotate_metrics, inspect_results
  inspection/    Markdown report + standalone HTML viewer
data/            hamlet.txt, chunks, questions, domain KG, macrag artifacts
third_party/     cloned official repos + papers for every ported method
runs/            per-run outputs (results + copies of inputs/config)
tests/           CPU-only unit tests with stub models
```

### 3.3 The plugin contract

A treatment is one adapter plus one registry row. The adapter receives a frozen
`ContextAssemblyRequest` (question, candidate chunks, retrieval trace, budget,
per-method `feature_params`, and live `feature_handles`) and returns a
`ContextAssemblyResult` (selected chunk IDs/chunks, retrieval metadata, prompt
ordering, and a full `context_assembly_trace`). `features/registry.py` declares,
per treatment, whether it needs dense retrieval, sparse retrieval, the MacRAG
slice index, a domain graph, or the reader model at assembly time.

To add a method (e.g. MMR, RankRAG, IRCoT): create
`hamlet_qa/features/<method>/` with an `assemble_*` adapter and add one
`TreatmentSpec` to the registry. No core change is required.

### 3.4 Evaluation signals

- **`evidence_chunk_recall`** — fraction of gold chunk IDs present (ID-based).
- **`evidence_quote_recall`** — fraction of required evidence quotes present,
  checked **both** by selected chunk ID **and** by verbatim text containment in
  the assembled context. This is the primary signal for treatments that emit
  synthetic pseudo-chunks (CRAG refined knowledge, RECOMP summaries, MacRAG
  merged blocks) whose original chunk IDs disappear.
- **Sufficient context** + **oracle CI value** (post-hoc, §4.3) separate
  assembly failures from reader failures and identify which chunks carried the
  answer.

Gold chunk IDs are derived automatically from each question's
`required_evidence_quotes` during validation and written to
`hamlet_questions_resolved.json` in the run directory.

---

## 4. Implementation

### 4.1 Treatments

| Treatment | Family | What it does |
|---|---|---|
| `closed_book` | control | no context |
| `gold_evidence` | control | minimal quote-covering gold chunks, then budget fill |
| `dense_reranked` | baseline | dense FAISS hits, reranked, filled in relevance order |
| `dense_document_order` | ordering | same dense set, reordered by document order |
| `dense_random_order` | ordering | same dense set, deterministic seeded shuffle |
| `sparse_bm25` | baseline | BM25 over chunk text |
| `setr` | selection | LLM set-selection via the SetR `selection_IRI` prompt |
| `domain` | knowledge | Hamlet domain-KG scaffold + KG-guided ordering |
| `crag` | corrective | evaluator-gated refine / re-retrieve / combine |
| `macrag` | hierarchical | summary-slice retrieval + neighbor merge |
| `recomp_extractive` | compression | sentence selection by a Contriever encoder |
| `recomp_abstractive` | compression | T5 (or prompted reader) summary |
| `reader_support` | **ours** | reader-teacher evidence-support selection of source units |
| `plan_fixed` | **ours (experiment)** | swappable decomposition → per-node retrieve+rerank → greedy coverage |
| `plan_dynamic` | **ours (experiment)** | LLM emits a procedure contract; executor runs the planned procedure |

The four SOTA methods (`crag`, `macrag`, `recomp_*`) are ports of the official
code cloned under `third_party/`, with every divergence logged per-row in
`context_assembly_trace.deviations`. Each was reviewed line-by-line against its
upstream source; the deviations below are intentional and documented.
`reader_support` is our own prototype (not a paper reproduction).

### 4.2 Method details

**`setr`** — [arXiv 2507.06838](https://arxiv.org/abs/2507.06838), cloned at
`third_party/SetR/`. Formats the top dense candidates as numbered SetR passages,
sends the original `selection_IRI` system/user prompt to the reader model (the
selector), parses `### Final Selection: [..] [..]` exactly like the official
`data_formatting.py`, maps passage numbers back to chunk IDs, and enforces the
budget without adding unselected fallback chunks. Runtime adaptation of the
prompt — it does not train a SetR checkpoint. Knobs: `--setr-max-passages`,
`--setr-selector-max-tokens` (default `4096`). Selector I/O is cached at
`data/cache/setr_selector_cache.json`.

**`domain`** — deterministic Hamlet domain-knowledge assembler. Loads
`data/hamlet_domain_kg.yaml`, detects aliases in the question (e.g.
`King → Claudius`), expands related graph nodes, builds a compact
`domain_scaffold` pseudo-chunk within budget, then orders dense chunks by graph
matches. Point at an edited graph with `--domain-kg`.

**`crag`** — [arXiv 2401.15884](https://arxiv.org/abs/2401.15884), cloned at
`third_party/CorrectiveRAG/CRAG/`. The evaluator scores the top `--crag-ndocs`
dense candidates and picks one action: **Correct** (refine retrieved docs),
**Incorrect** (corrective re-retrieval), **Ambiguous** (combine both as the
official `Knowledge1: ... [sep] Knowledge2: ...`). Refinement is the official
decompose-then-recompose: strips per `--crag-decompose-mode` (`fixed_num`
50-word windows / `excerption` sentence strips / `selection` whole passage),
evaluator-scored, top strips joined by `; ` in score order, emitted as one
`crag_refined_knowledge` pseudo-chunk.
*Deviations:* the fine-tuned T5-large evaluator → Qwen reranker scores with
thresholds recalibrated by `python -m hamlet_qa.cli.calibrate_crag` (defaults
upper `2.5`, lower `0.875`); web search → in-corpus substitute (reader rewrites
the query with the official popqa keyword prompt, then BM25 re-retrieves
`--crag-external-top-k` chunks); GPT-3.5 keyword extractor → local reader;
refined knowledge word-truncated to budget.

**`macrag`** — [arXiv 2505.06569](https://arxiv.org/abs/2505.06569), cloned at
`third_party/MacRAG/MacRAG/`. Requires a one-time offline index build:

```bash
python -m hamlet_qa.cli.build_macrag_index
```

That summarizes every chunk with the reader model (official persona/instruction
prompt; upstream uses GPT-4o), slices summaries at 450/300 chars with the
official `["\n\n","\n"," ",""]` separators, and writes
`data/macrag/hamlet_macrag_summaries.jsonl` + `hamlet_macrag_slices.jsonl` (the
slice→parent mapping). Summaries are cached, so reruns are incremental. At query
time: retrieve top `--macrag-top-k1` slices, map to parent chunks (keep-best
dedupe), rerank parents, expand each top `--macrag-top-k2` parent by
`--macrag-chunk-ext` neighbor hops *within the same scene*, and merge contiguous
chunks with overlap removal (`--macrag-merge-version 1` = one block per scene,
`2` = one block per contiguous run).
*Deviations:* local reader → GPT-4o; harness chunks → 1500/500-char chunking;
Qwen embedder/reranker → e5/ms-marco-MiniLM; document unit = act+scene; only the
R&B generation strategy; token budget drops whole blocks.

**`recomp_extractive` / `recomp_abstractive`** —
[arXiv 2310.04408](https://arxiv.org/abs/2310.04408), cloned at
`third_party/RECOMP/recomp/`. Compressors run in a precompute stage before the
reader loads; the compressed summary enters the prompt as one pseudo-chunk, and
an **empty summary is valid** (selective augmentation → reader answers
closed-book).
- *Extractive:* the official Contriever-style dual encoder
  (`fangyuan/hotpotqa_extractive_compressor`) scores every sentence of the top
  `--recomp-input-docs` chunks against the question (dot product of mean-pooled
  embeddings) and keeps `--recomp-top-sentences` in score order.
- *Abstractive:* the official T5 checkpoint (`fangyuan/hotpotqa_abstractive`)
  with the official input format `Question: ...\n Document: ...\n Summary: `.
  `--recomp-abstractive-mode prompted_qwen` instead prompts the reader with the
  paper's Table 8 GPT-3.5 compression prompt.

*Deviations:* HotpotQA checkpoints used zero-shot on Shakespeare; harness prompt
builder → paper few-shot QA prompt; 256-token chunks → 100-word Wikipedia
passages; NLTK sentence splitting on newline-collapsed verse; summaries
word-truncated to budget.

**`reader_support` — Reader-Supervised Evidence Support Assembler (ours).** Our
own prototype, not a paper reproduction (`hamlet_qa/features/reader_support/`).
Instead of hand-blending reranker/BM25/entity scores, it uses the reader model
as a *teacher* to judge how well each candidate source unit supports each
specific evidence need, then reconstructs a compact, non-redundant,
source-faithful, ordered sub-document under the token budget. Four stages:

1. **Evidence-node induction** (`nodes.py`): the reader decomposes the question
   into ≤`support_max_nodes` information needs, conditioned on the question plus
   a compact candidate catalog (chunk id + act/scene + excerpt — no answers/gold).
   Node text states the *need*, never the answer. JSON-only with a repair retry
   and a single-node fallback.
2. **Candidate units** (`units.py`): source-faithful units per top candidate —
   whole `chunk`, `sentence`, speaker-turn `line_span`, and `neighbor_*` blocks
   (neighbor expansion never crosses a scene; merged via the MacRAG overlap
   dedupe). Deduplicated, token-capped. No abstractive text.
3. **Reader-teacher support scoring** (`teacher.py`): for each (node, unit) pair
   the reader returns a JSON support judgement (score 0–1, type, a verbatim
   supporting span). Scores are validated and capped — empty-span "complete"
   claims cap at 0.7, non-substring spans cap at 0.5 with a warning,
   "contradictory" → 0. A lexical prior only *prefilters* which pairs are scored
   (`support_teacher_units_per_node`). All teacher labels are logged in the trace
   (training signal for a future learned scorer; `SupportScorer` is the drop-in
   interface, `TrainedSupportScorer` the stub).
4. **Budgeted assembly + ordering** (`assembly.py`): greedy maximization of node
   coverage `1 − Π(1 − support)` minus a redundancy penalty, per-step
   token-normalized (`gain / tokens^τ`); stops on budget, coverage threshold, or
   no positive gain. Whole-chunk picks keep their real chunk id (so they count
   toward chunk recall); sub-units become source-extractive pseudo-chunks.
   Ordering is anchor-first then evidence-node order then document order. If no
   unit clears `support_min_unit_score`, it returns **empty** context
   (`reader_support_empty`) rather than forcing in noise.

All reader calls are cached (`data/cache/reader_support_cache.json`). Constraints
honored: assembly never reads `expected_answer`/`required_evidence_quotes`/gold;
the final context is source-extractive only. Knobs are the `--support-*` flags
(see `--help`).

**`plan_fixed` / `plan_dynamic` — the evidence-planning experiment (ours).**
`reader_support` entangles several innovations; these two treatments
(`hamlet_qa/features/evidence_plan/`) deliberately **isolate one variable: how
the LLM breaks the question down and plans the retrieval procedure.** Everything
downstream is a shared, simple executor (`executor.py`): per-node dense
retrieval → support scoring → budgeted **greedy coverage** (reusing
`reader_support.greedy_select`) → ordering. Selected context is whole source
chunks (real chunk ids).

- **`plan_fixed`** — a swappable **decomposition prompt** splits the question
  into evidence nodes; the procedure is then **fixed by flags**: dense-retrieve
  per node, rerank per node, greedy-cover. The retrieval mode can be `parallel`
  (independent multi-part questions) or `sequential` (`--plan-retrieval-mode
  sequential`), where a dependent node's query is rewritten from gathered
  evidence via a follow-up prompt (true bridge multi-hop). This isolates *which
  decomposition prompt is best*.
- **`plan_dynamic`** — a **planner prompt** emits a JSON **procedure contract**
  (`contract.py`): question type, retrieval mode (parallel/sequential),
  selection policy, ordering policy, support policy, and the nodes. The executor
  validates/normalizes the contract (logging every fallback) and runs exactly
  it. This tests *whether the LLM can also plan the procedure*.

The per-(node, chunk) **support signal** is selectable (`--plan-support-policy`):
`reranker` (the reranker logit squashed through a sigmoid into [0,1]) or
`teacher` (the `reader_support` reader-teacher), giving a reranker-vs-teacher
ablation too. Selection is `greedy_coverage` or `top_per_node`.

**Swapping prompts is the experiment.** Three named registries
(`prompts.py`), each chosen by one flag; every variant's version is in the cache
key so swaps never reuse a stale cache:

| Slot | Flag | Variants |
|---|---|---|
| Decomposition (`plan_fixed`) | `--decomp-prompt` | `subquestions` · `info_requirements` · `strategy` |
| Planner (`plan_dynamic`) | `--planner-prompt` | `contract_v1` · `strategy_contract` |
| Follow-up (sequential) | `--followup-prompt` | `rewrite_with_evidence` |

The `strategy` / `strategy_contract` variants make the LLM write a **step-wise
solving strategy** (subquestions + information to cover + order) before acting.
The full plan — nodes, contract, per-node retrieval, support, coverage,
ordering — is logged in `context_assembly_trace`, and `evidence_role_recall`
(§4.3) measures whether the plan covered the gold roles. New infra:
`feature_handles["node_retriever"]` keeps the embedder+reranker resident for
per-sub-question retrieval (multi-GPU layout recommended).

### 4.3 Post-hoc metrics

Computed after a run with a single reader-model load, written to a sidecar
(`metrics_annotations.jsonl`) next to `results.jsonl` — the results file is
never rewritten. The Markdown report and HTML viewer merge them on read (`suff`
and `ci+` badges, summary columns, per-chunk φ details).

- **Oracle CI value** ([arXiv 2509.21359](https://arxiv.org/abs/2509.21359)) —
  leave-one-out contextual influence `phi_i = loss(C \ c_i) − loss(C)`, where
  loss is the mean token cross-entropy of `expected_answer` under the reader
  (vLLM prompt logprobs). `phi_i > 0` means the chunk helps; rows record
  `ci_values`, `ci_positive_chunk_ids`, and `ci_positive_fraction` (the paper's
  hyperparameter-free keep-set). *Deviations:* one gold answer instead of an
  answer set; free-form sentence answer.
- **Sufficient context** ([arXiv 2411.06037](https://arxiv.org/abs/2411.06037))
  — the Appendix C.1 1-shot autorater prompt over (question, assembled context)
  → binary `sufficient_context` + explanation. *Deviations:* local reader
  replaces Gemini 1.5 Pro; the timestamp sentence is dropped. Combined with
  answer correctness, this separates assembly failures (insufficient context)
  from reader failures (sufficient context, wrong answer).
- **Evidence-role recall** (reader-free) — fraction of the question's evidence
  *roles* covered by the assembled context (a role is covered when any of its
  required quotes is present by chunk id or verbatim text). Unanswerable
  questions return null. Computed from the row alone, so `--metrics evidence_role`
  runs without loading the reader.
- **Plan evaluation** (`plan_eval`, reader-free) — stage-wise evaluation for the
  `plan_fixed` / `plan_dynamic` treatments, scoring the two stages that need no
  LLM: **(1) after the plan is generated** — `plan_num_nodes`,
  `plan_node_fallback`, `plan_num_gold_roles` (were the evidence slots produced;
  *which* slots is left to the LLM judge); **(2) after retrieval** —
  `plan_slot_retrieval_recall` = fraction of required evidence roles whose gold
  chunk was surfaced by the per-node retrieval (plus `plan_gold_chunk_retrieval`
  and per-role `plan_slot_detail`). **Stage 3 (final-answer quality)** is left to
  a stronger-model LLM judge, which attaches to the row's `model_output`. Returns
  `plan_eval_applicable=False` for non-plan treatments.

---

## 5. Install

The repo uses Miniforge/conda so the Python version and package set are explicit
and independent of the system `python`. It expects Python `3.12`
(`.python-version` pins `3.12.3`); `environment.yml` creates/updates the
`hamlet-qa` env with non-GPU deps; `setup.sh` then installs the GPU stack
(PyTorch, vLLM, Transformers, SentenceTransformers, FAISS) via
`uv pip install --torch-backend=cu129`, pinning `vllm-0.22.0+cu129` on Linux
x86_64 so vLLM and PyTorch share a CUDA runtime family.

```bash
bash setup.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hamlet-qa
```

If activation says `Run 'conda init' before 'conda activate'`, the env is fine —
your shell just hasn't loaded conda's hook; run the `source ...` line above.

If `bash setup.sh` reports Miniforge/conda was not found, install it first:

```bash
curl -L -o Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
export PATH="$HOME/miniforge3/bin:$PATH"
bash setup.sh
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate hamlet-qa
```

Run commands without activating the shell via `conda run -n hamlet-qa ...`.

**GPU package repair.** Only the server run needs reader inference; tests and
`--prepare-only` do not. If PyTorch reports `The NVIDIA driver on your system is
too old`, or vLLM fails with `libcudart.so.13: cannot open shared object file`,
repair only the GPU packages inside the activated env:

```bash
uv pip install --torch-backend=cu129 \
  --reinstall-package torch --reinstall-package vllm \
  torch \
  "https://github.com/vllm-project/vllm/releases/download/v0.22.0/vllm-0.22.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
uv pip uninstall -y torchvision torchaudio
```

This project does not use TorchVision/TorchAudio. If Transformers fails with
`operator torchvision::nms does not exist`, remove them:
`uv pip uninstall -y torchvision torchaudio`.

> The `hamlet-qa` env (vLLM/torch) is intended for the GPU server. Locally you
> can still develop and run the test suite CPU-only with stub models.

---

## 6. Run

### 6.1 Regenerate chunks (optional)

```bash
python -m hamlet_qa.cli.build_chunks \
  --document data/hamlet.txt --output data/hamlet_chunks.jsonl
```

Defaults: tokenizer `Qwen/Qwen3.5-9B`, chunk size `256` tokens, overlap `64`.
Chunk IDs are stable (e.g. `act03_scene02_chunk004`). If chunking settings
change, regenerate and confirm quote validation still resolves the intended
evidence.

### 6.2 Build the MacRAG index (only if running `macrag`)

```bash
python -m hamlet_qa.cli.build_macrag_index
```

### 6.3 Run the experiment

```bash
python -m hamlet_qa.cli.run_experiment --run-name qwen_hamlet_probe
```

Defaults: reader & tokenizer `Qwen/Qwen3.5-9B`, embedder `Qwen/Qwen3-Embedding-8B`,
reranker `Qwen/Qwen3-Reranker-8B`, temperature `0.0`, context budget `1000`,
top-k candidates `50`, seed `13`, and all treatments listed in §4.1. Dense
retrieval embeds chunks into FAISS, then the reranker defines the final dense
ranking. Outputs go to `runs/<run_name>/results.jsonl` alongside copies of the
config, chunks, input questions, and quote-resolved questions. The budget counts
**context tokens only**; prompt overhead and generated tokens are logged
separately.

**GPU layouts** change only model placement, never retrieval/ordering/scoring:

```bash
# Default: one device stages embedder, reranker, and reader.
python -m hamlet_qa.cli.run_experiment --run-name qwen_hamlet_probe

# 2×A40: cuda:0 embed+rerank, cuda:1 reader + SetR selector.
python -m hamlet_qa.cli.run_experiment --run-name qwen_hamlet_probe --gpu-layout a40-2gpu

# 3×A40: cuda:0 embed, cuda:1 rerank, cuda:2 reader.
python -m hamlet_qa.cli.run_experiment --run-name qwen_hamlet_probe --gpu-layout a40-3gpu
```

**Prepare-only** (build prompts and traces, skip generation):

```bash
python -m hamlet_qa.cli.run_experiment --run-name dry_prompts --prepare-only
```

Prepare-only still does dense retrieval for grounded treatments, BM25 for
`sparse_bm25`, and (for `setr`, `reader_support`, and `recomp_abstractive
--recomp-abstractive-mode prompted_qwen`) loads the reader for
selection/scoring/compression — it only skips final answer generation. Pass
`--reranker-model none` to disable reranking. For a model-, embedder-,
reranker-, and BM25-free smoke test, restrict treatments to `closed_book`; tests
inject cached/stub retrievers.

To benchmark our method against the others (the `--support-*` flags tune it):

```bash
python -m hamlet_qa.cli.run_experiment --run-name probe_reader_support \
  --gpu-layout a40-2gpu \
  --treatments closed_book gold_evidence dense_reranked sparse_bm25 setr \
    crag macrag recomp_extractive recomp_abstractive reader_support
```

### 6.3.1 Evidence-planning experiment — what to run

The `plan_fixed` / `plan_dynamic` study is **opt-in** (not in the default
treatments). The experimental variable is the **decomposition / planner prompt**;
everything downstream is held constant. Each run varies exactly one prompt flag.

**Requirements:** a **multi-GPU layout** — `plan_*` keeps the embedder+reranker
resident for per-sub-question retrieval, next to the reader. Set `LAYOUT` once.
(If GPU 0 is busy, replace `--gpu-layout $LAYOUT` with explicit device flags:
`--embedding-device cuda:1 --reranker-device cuda:1 --reader-device cuda:2`.)

```bash
# Activate the env on the GPU server (see §5), then:
LAYOUT=a40-2gpu

# 1. Baselines — run once, for reference.
python -m hamlet_qa.cli.run_experiment --run-name plan_baselines --gpu-layout $LAYOUT \
  --treatments closed_book gold_evidence dense_reranked setr

# 2. plan_fixed — one run per decomposition prompt (parallel retrieval).
for P in subquestions info_requirements strategy; do
  python -m hamlet_qa.cli.run_experiment --run-name plan_fixed_$P --gpu-layout $LAYOUT \
    --treatments plan_fixed --decomp-prompt $P --plan-retrieval-mode parallel
done

# 3. plan_fixed — sequential retrieval (bridge/multi-hop), strongest decomposition.
python -m hamlet_qa.cli.run_experiment --run-name plan_fixed_strategy_seq --gpu-layout $LAYOUT \
  --treatments plan_fixed --decomp-prompt strategy --plan-retrieval-mode sequential

# 4. plan_dynamic — one run per planner prompt (the LLM plans the procedure).
for P in contract_v1 strategy_contract; do
  python -m hamlet_qa.cli.run_experiment --run-name plan_dynamic_$P --gpu-layout $LAYOUT \
    --treatments plan_dynamic --planner-prompt $P
done

# 5. Evaluate every run — reader-free, NO CI (no model load at all).
for R in plan_baselines plan_fixed_subquestions plan_fixed_info_requirements \
         plan_fixed_strategy plan_fixed_strategy_seq \
         plan_dynamic_contract_v1 plan_dynamic_strategy_contract; do
  python -m hamlet_qa.cli.annotate_metrics --results runs/$R/results.jsonl \
    --metrics evidence_role plan_eval
done

# 6. Inspect a run (nodes/contracts + per-node retrieval + answers).
python -m hamlet_qa.inspection.results_html runs/plan_fixed_strategy/results.jsonl \
  --output runs/plan_fixed_strategy/results_viewer.html
```

That's **7 runs** + their annotations. Optional extra ablation: append
`--plan-support-policy teacher` to any `plan_*` run to score (node, chunk) with
the reader-teacher instead of the reranker.

**What each stage measures** (`evidence_role` + `plan_eval`, both reader-free):

| Stage | When | Fields to compare across runs |
|---|---|---|
| 1 — plan generated | after decomposition/planning | `plan_num_nodes`, `plan_node_fallback`, `plan_num_gold_roles` (slots produced; *which* slots → LLM judge later) |
| 2 — retrieval | after per-node retrieval | **`plan_slot_retrieval_recall`** (did retrieval find each slot's gold chunk), `plan_gold_chunk_retrieval`, `plan_slot_detail` |
| — context | after assembly | `evidence_role_recall`, `evidence_quote_recall`, `context_tokens` |
| 3 — final answer | after generation | left to the stronger-model **LLM judge** later — reads `model_output` |

CI is deliberately omitted. When the LLM judge is ready, add it as a 4th metric
that reads `model_output` (stage 3) plus the logged nodes (stage 1) — no rerun
needed, the artifacts are already in each row.

### 6.4 Annotate post-hoc metrics

```bash
python -m hamlet_qa.cli.annotate_metrics \
  --results runs/probe_reader_support/results.jsonl \
  --metrics ci sufficient_context evidence_role
```

### 6.5 Inspect results

Interactive HTML viewer (groups by question; filters by treatment/text; shows
model output, evidence quotes, selected chunks, retrieval scores/traces, full
prompt, and expandable evidence chunks):

```bash
python -m hamlet_qa.inspection.results_html \
  runs/qwen_hamlet_probe/results.jsonl \
  --output runs/qwen_hamlet_probe/results_viewer.html
```

By default it embeds chunk text from `hamlet_chunks.jsonl` beside the results
file, then the run config's `chunks_path`, then `data/hamlet_chunks.jsonl`. Pick
a specific chunk file with `--chunks`. The in-page file pickers can load another
results file or chunk file.

Compact Markdown summary:

```bash
python -m hamlet_qa.cli.inspect_results \
  --results runs/qwen_hamlet_probe/results.jsonl \
  --output runs/qwen_hamlet_probe/inspection.md
```

---

## 7. Editable data

- `data/hamlet_chunks.jsonl` — generated default chunks.
- `data/hamlet_questions.json` — 10 questions with required evidence quotes.
  The set includes four method-probe questions:
  `q_trap_ghost_poison` (lexical retrieval trap toward the Mousetrap scene →
  targets CRAG's corrective action), `q_arc_rosencrantz_guildenstern`
  (three-scene character arc → MacRAG/SetR), `q_final_scene_deaths` (minimal
  evidence cover is 1024 tokens > the 1000-token budget by design, so raw-chunk
  treatments cap below 1.0 quote recall while compression can pass → RECOMP),
  and `q_fortinbras_campaign` (three weakly-overlapping requirements → SetR).
  `q_speaker_bitter_cold` is the easy sanity check;
  `q_unanswerable_yorick_wife` is the abstention probe.
- `data/hamlet_domain_kg.yaml` — editable Hamlet domain graph for `domain` (and
  optional role templates for `setr`).
- `data/macrag/` — MacRAG summaries and slices built by `build_macrag_index`.

---

## 8. Tests

```bash
python -m unittest discover -s tests
```

CPU-only, no GPU required: the suite uses stub models and cached/stub
retrievers, and covers SetR parsing, the CRAG/MacRAG/RECOMP ports, the gold
minimal-cover regression, text-containment quote presence, the CI and
sufficient-context metrics, and the report/viewer rendering.
