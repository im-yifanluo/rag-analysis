# Reader-Supervised Evidence Support Assembler (`reader_support`)

Our own context-assembly method (a prototype, not a paper reproduction). This
document is the exact specification of what the code does: every step, the
models and libraries it uses, and the verbatim prompts.

Code: `hamlet_qa/features/reader_support/` — `nodes.py`, `units.py`,
`teacher.py`, `assembly.py`, `schema.py`. Treatment name: **`reader_support`**.

---

## 0. One-paragraph summary

Instead of scoring candidate passages with a hand-tuned blend of reranker / BM25
/ entity-overlap scores, we use the **reader model itself as a teacher** to judge
how well each candidate *source unit* supports each specific *evidence need* in
the question. A budgeted greedy coverage objective then reconstructs a compact,
non-redundant, **source-faithful** sub-document under the token budget. The final
context contains only verbatim source text (no abstractive summaries).

---

## 1. Models and libraries

| Role | What it is | Where |
|---|---|---|
| **Node-induction model** | the configured **reader model** (default `Qwen/Qwen3.5-9B`) served by **vLLM** | `request.selector_model`, a `VLLMReader` |
| **Support-teacher model** | the **same reader model** (the resident reader is the teacher) | `request.selector_model` |
| Sentence splitter | `nltk.sent_tokenize` with a regex fallback | reused from `recomp/compressor.py::split_sentences` |
| Overlap-safe chunk merge | longest suffix/prefix dedupe | reused from `macrag/assembly.py::combine_without_overlap` |
| Prefilter | **pure-Python lexical token overlap** (no model) | `assembly.py::lexical_prior` |
| Greedy selection / coverage / redundancy | **pure Python** (`math`, `re`) | `assembly.py` |
| Caching | `JsonKVCache` (sha256 `stable_hash`) | `core/llm_cache.py` |

**No** embedding model, reranker, BM25, FAISS, or torch is used by this method's
*scoring* path. The only learned model in the loop is the reader (used twice:
once to induce nodes, once per (node, unit) pair as the support teacher). The
candidate pool comes from the existing **dense retrieval trace** (top
`support_candidate_chunks` rows), which is produced upstream by the harness'
Qwen embedder + reranker — but those scores are **not** used as the final
selection signal here.

All reader calls are cached in `data/cache/reader_support_cache.json` (two
sections: `node_induction`, `support_scores`).

---

## 2. The four stages (exact algorithm)

Entry point: `assemble_reader_support(request)` in `assembly.py`. Input is a
`ContextAssemblyRequest` carrying the question, the dense `retrieval_trace`, the
`chunk_lookup`, the `context_budget`, the `selector_model` (reader), and the
`support_*` params in `feature_params`.

### Stage 1 — Evidence-node induction (`nodes.py`)

1. Build a **candidate catalog** from the top `support_node_candidate_catalog_k`
   (default 20) dense-trace rows: one line per chunk = `[chunk_id] Act A Scene S
   (title): <first 200 chars>`. **No** expected answers or gold labels are
   included — topical context only.
2. Call the reader once with the node-induction prompt (§4.1), `max_tokens =
   support_node_induction_max_tokens` (default 1024).
3. Parse strict JSON → a list of `EvidenceNode` (≤ `support_max_nodes`, default
   5). Each node = `{node_id, need, node_query, order_index, depends_on,
   raw_reason}`. The `need` describes *what to find*, never the answer.
4. **Failure handling:** if the first output is not parseable JSON, retry once
   with a stricter repair instruction (§4.2). If still unparseable, fall back to
   a **single node = the whole question**. The parse error and fallback flag are
   recorded in the trace (never silently swallowed).
5. Cached by `sha256(prompt + model_name + prompt_version + max_nodes)`.

### Stage 2 — Candidate evidence-unit construction (`units.py`)

For the top `support_candidate_chunks` (default 30) dense-trace chunks, build
**source-faithful** `EvidenceUnit`s. Unit text is copied verbatim from source
chunks; **no abstraction**. Types (controlled by `support_unit_types`):

- **`chunk`** — the whole candidate chunk. `token_count` = the chunk's real
  tokenizer token count. (A selected whole chunk keeps its real `chunk_id`, so it
  still counts toward chunk-id recall.)
- **`sentence`** — sentences from `split_sentences` (NLTK / regex fallback).
- **`line_span`** — blank-line-separated blocks, which for Hamlet keep a speaker
  label attached to its lines (a sentence split would scatter them).
- **`neighbor_left` / `neighbor_right` / `neighbor_both`** — the chunk merged
  with its previous / next / both neighbors, **only within the same scene**
  (`scene_id` must match), overlap-deduped via `combine_without_overlap`.

Bookkeeping: sub-unit / merged-unit `token_count` = whitespace **word count** (a
budget proxy consistent with the harness' other pseudo-chunks). Units are built
in rounds (all chunk units, then neighbors, then line spans, then sentences) so
every candidate keeps its whole-chunk unit even when the total cap bites, then:

- drop units with `token_count > support_max_unit_tokens` (default 512),
- drop non-chunk units with `< 3` words,
- **deduplicate** by `(normalized_text, sorted(source_chunk_ids))`,
- cap the total at `support_max_units_total` (default 200).

Each unit records `source_chunk_ids`, `primary_chunk_id`, act/scene metadata,
`global_index_start/end`, and a `source_order_key` for document-order sorting.

### Stage 3 — Reader-teacher support scoring (`teacher.py`)

This is the core. For each evidence node `n_j`:

1. **Prefilter** (cost control, not the final score): rank all units by
   `lexical_prior(node, unit)` = `|query_terms ∩ unit_terms| / |query_terms|`
   (query terms = node `need` + `node_query`, stopwords removed) and keep the top
   `support_teacher_units_per_node` (default 12). The prefilter is the **only**
   place reranker/BM25/embedding-style signals could enter, and here it is a pure
   lexical overlap.
2. For each prefiltered `(n_j, unit)` pair, call the reader-teacher with the
   support prompt (§4.3), `max_tokens = support_teacher_max_tokens` (default 384).
3. Parse the JSON judgement and **validate / cap** it (`validate_and_cap`):
   - clamp `support_score` to `[0, 1]`;
   - `support_type == "contradictory"` → score `0.0`;
   - `support_type` in {partial, complete} with an **empty** `supporting_span` →
     cap at `0.7`;
   - non-empty `supporting_span` that is **not an exact substring** of the unit
     text (whitespace-normalized) → cap at `0.5` + a validation warning;
   - unparseable output → score `0.0`.
4. Result is a `SupportScore` per pair; the support matrix is
   `score[node_id][unit_id]`. Pairs that were not scored default to `0.0`.
5. Cached by `sha256(question + node need/query + unit text + model_name +
   prompt_version)`. Every teacher label is also written to
   `context_assembly_trace.teacher_labels` (this is the training signal for a
   future learned scorer; `SupportScorer` is the drop-in interface and
   `TrainedSupportScorer` is the stub).

### Stage 4 — Budgeted assembly + ordering (`assembly.py`)

Pure, side-effect-free `greedy_select`:

- **Selectable** = units whose max support over nodes ≥ `support_min_unit_score`
  (default 0.45). If none qualify → return **empty context**
  (`retrieval_method = "reader_support_empty"`); we do **not** force in noise.
- Per-node coverage with diminishing returns:
  `Coverage_j(S) = 1 − Π_{u∈S} (1 − support[j][u])`.
- Per-step greedy. For a candidate unit `u` (maintaining `miss_j = Π(1−support)`):
  - `coverage_gain = Σ_j miss_j · support[j][u]`
  - `redundancy = max_{s∈S} ( 0.6·jaccard(text_u, text_s) + 0.4·cosine(support_vec_u,
    support_vec_s) + 0.5·[shares a source chunk id] )`, clamped to `[0,1]`
  - `marginal = coverage_gain − β·redundancy` (β = `support_redundancy_beta`, 0.15)
  - `gain_per_token = marginal / token_count^τ` (τ = `support_token_exponent_tau`, 0.7)
  - pick the unit with the largest **positive** `gain_per_token` that fits the
    remaining budget.
- **Stop** when: budget exhausted, no positive gain, every node's coverage ≥
  `support_node_coverage_threshold` (0.85), or `support_max_selected_units` (8)
  reached.
- **Optional context repair:** if a selected `sentence`/`line_span` unit's teacher
  judgement set `needs_more_context = true`, and a same-chunk `neighbor_*` unit
  fits the budget and adds coverage, add it (logged as `context_repair`).
- **Ordering** (`support_prompt_order`, default `anchor_then_node_doc_order`):
  put the **anchor** (selected unit with the highest single support score) first,
  then order the rest by (strongest-supported node's `order_index`, document
  order, token count, unit id). Alternatives: `node_doc_order`, `document_order`.
- **Output chunks:** a selected `chunk` unit → its real chunk dict (keeps the
  real `chunk_id`); any other unit → a **source-extractive pseudo-chunk**
  `reader_support::<unit_id>` carrying `source_chunk_ids`, `member_chunk_ids`,
  `unit_type`, and act/scene metadata. Because the text is verbatim,
  `evidence_quote_recall` still works by text containment.

The `context_assembly_trace` records: the node-induction prompt + raw output +
parse error/fallback, the nodes, unit-type counts and drop log, per-node top
scored units, all teacher labels, coverage progress after each pick, every
selection step (marginal gain / tokens / gain-per-token / redundancy), the final
per-node coverage (mean/min/#covered), the final ordering, and cache hit counts.

---

## 3. Constraints honored

- Assembly never reads `expected_answer`, `required_evidence_quotes`,
  `derived_gold_chunk_ids`, or metric annotations.
- Final context is **source-extractive only** (no generated/abstractive text).
- Every reader call is cached.
- Respects `context_budget`.
- No Hamlet-question-ID-specific logic; uses act/scene metadata only when present
  (None-safe), so it generalizes to other long documents.
- Parser errors are logged in the trace, not swallowed.

---

## 4. Verbatim prompts

The reader is called with a system prompt and a user prompt. Below are the exact
strings from the code (`{...}` are Python `str.format` fields).

### 4.1 Node induction — system prompt (`nodes.py::NODE_INDUCTION_SYSTEM`)

```
You are an expert reading-comprehension analyst. You decompose a question about a long document into the distinct information needs that must each be supported by source text before the question can be answered.
```

### 4.1 Node induction — user prompt (`nodes.py::NODE_INDUCTION_TEMPLATE`)

```
Decompose the QUESTION into at most {max_nodes} evidence nodes. An evidence node is ONE information need that must be supported by source text to answer the question.

Rules:
- Describe the information NEED, do not write or guess the answer.
  Good: "Identify the final fate of Rosencrantz and Guildenstern."
  Bad: "They are dead."
- Prefer fewer, non-overlapping nodes. Split only genuinely separate needs (for example, a multi-part question).
- Use the CANDIDATE CATALOG only to understand what the question is asking about. Do not assume the catalog contains the answers.
- order_index reflects the natural reading/answer order of the needs (1, 2, 3, ...).
- depends_on lists node_ids that must be resolved first, or [] if none.

QUESTION:
{question}

CANDIDATE CATALOG (retrieved passages, for topical context only):
{catalog}

Respond with JSON ONLY in exactly this shape:
{
  "nodes": [
    {"node_id": "n1", "need": "...", "node_query": "...", "order_index": 1, "depends_on": [], "reason": "..."}
  ]
}
```

### 4.2 Node induction — JSON repair suffix (`nodes.py::NODE_REPAIR_SUFFIX`, appended to the user prompt on retry)

```


Your previous response could not be parsed as JSON. Respond again with valid JSON ONLY, no prose, matching the required shape exactly.
```

### 4.3 Support teacher — system prompt (`teacher.py::SUPPORT_TEACHER_SYSTEM`)

```
You are a strict evidence adjudicator. You judge ONLY from the candidate text shown to you. You do not use any outside knowledge, you do not answer the question, and you never invent text that is not present.
```

### 4.3 Support teacher — user prompt (`teacher.py::SUPPORT_TEACHER_TEMPLATE`)

```
Judge how well the CANDIDATE TEXT supports the EVIDENCE NEED.

EVIDENCE NEED:
{need}

CANDIDATE TEXT:
"""
{unit_text}
"""

Scoring rubric (support_score):
- 0.0  = irrelevant, or contradicts the need
- 0.25 = same topic/entities but does NOT answer the need
- 0.5  = partial support, useful but incomplete
- 0.75 = mostly supports the need, may need a little local context
- 1.0  = directly and sufficiently supports the need

Rules:
- Judge only from the CANDIDATE TEXT. Do not use outside knowledge of the work.
- Do not answer the overall question; only assess support for THIS need.
- If you claim partial/complete support, quote the exact supporting substring (copied verbatim from the candidate). If you cannot quote it, the support is not there.
- If the text is only topically related but does not answer the need, use support_type "related" and a score near 0.25.

Respond with JSON ONLY in exactly this shape:
{"support_score": 0.0, "support_type": "none|related|partial|complete|contradictory", "supporting_span": "exact substring or empty", "needs_more_context": false, "explanation": "brief"}
```

Prompt versions are pinned (`reader_support.nodes.v1`, `reader_support.teacher.v1`)
and included in cache keys, so changing a prompt invalidates the cache.

---

## 5. Configuration knobs (defaults)

All are `--support-*` CLI flags / `RunConfig` fields, logged in `run_config.json`.

| Param | Default | Meaning |
|---|---|---|
| `support_candidate_chunks` | 30 | dense candidates that seed units |
| `support_node_candidate_catalog_k` | 20 | chunks shown in the node-induction catalog |
| `support_max_nodes` | 5 | max evidence nodes |
| `support_teacher_units_per_node` | 12 | units prefiltered per node before scoring |
| `support_unit_types` | `chunk,sentence,line_span,neighbor_left,neighbor_right` | unit granularities |
| `support_include_neighbors` | true | build neighbor-expanded units |
| `support_neighbor_hops` | 1 | neighbor window |
| `support_max_units_total` | 200 | global unit cap |
| `support_max_unit_tokens` | 512 | drop oversize units |
| `support_node_coverage_threshold` | 0.85 | per-node coverage stop condition |
| `support_redundancy_beta` | 0.15 | redundancy penalty weight β |
| `support_token_exponent_tau` | 0.7 | token-cost exponent τ |
| `support_min_unit_score` | 0.45 | min support to be selectable |
| `support_max_selected_units` | 8 | max units in the final context |
| `support_node_induction_max_tokens` | 1024 | node-induction generation budget |
| `support_teacher_max_tokens` | 384 | per-pair teacher generation budget |
| `support_prompt_order` | `anchor_then_node_doc_order` | ordering strategy |
| `support_score_cache_path` | `data/cache/reader_support_cache.json` | LLM cache |

---

## 6. Cost note

Per question the reader is called once for node induction plus up to
`(#nodes × support_teacher_units_per_node)` times for support scoring (≈ tens of
calls), all cached. This is the most expensive treatment in the harness; the
prefilter and caches keep it tractable.
