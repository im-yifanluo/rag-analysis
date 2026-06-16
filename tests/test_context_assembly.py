from __future__ import annotations

import unittest
from pathlib import Path

from hamlet_qa.features.domain.kg import (
    DOMAIN_SCAFFOLD_CHUNK_ID,
    DomainKnowledgeGraph,
    select_domain_kg_lite,
)
from hamlet_qa.features.setr.selector import (
    SETR_SELECTION_IRI_PROMPT,
    SETR_SELECTION_SYS_PROMPT,
    SetRSelectionError,
    map_positions_to_chunk_positions,
    parse_setr_final_selection,
    select_setr,
)
from hamlet_qa.features.crag.assembly import CRAG_PSEUDO_CHUNK_ID, assemble_crag
from hamlet_qa.features.crag.corrective import (
    action_from_scores,
    combine_knowledge,
    extract_strips_from_psg,
    select_relevant_strips,
    top_n_for_mode,
)
from hamlet_qa.features.crag.rewrite import CRAG_KEYWORD_PROMPT
from hamlet_qa.features.macrag.assembly import assemble_macrag, combine_without_overlap
from hamlet_qa.features.macrag.index import (
    build_slices_for_chunk,
    recursive_character_split,
)
from hamlet_qa.features.macrag.retrieval import slice_hits_to_parent_candidates
from hamlet_qa.features.macrag.summarize import parse_summary_response, summarize_chunk
from hamlet_qa.features.recomp.assembly import (
    assemble_recomp_abstractive,
    assemble_recomp_extractive,
)
from hamlet_qa.features.recomp.compressor import (
    RECOMP_PROMPTED_ABSTRACTIVE_PROMPT,
    compress_extractive,
    split_sentences,
)
from hamlet_qa.core.questions import Question, RequiredEvidenceQuote
from hamlet_qa.core.context import (
    ContextAssemblyRequest,
    chunks_by_id,
    context_token_count,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def role_question() -> Question:
    return Question(
        id="q_roles",
        question="How do the setup and confirmation answer the question?",
        expected_answer="The setup and confirmation are both needed.",
        evidence_scope="synthetic",
        reasoning_skill="cross_scene_bridge",
        required_evidence_quotes=[
            RequiredEvidenceQuote(
                act=1,
                scene=1,
                speaker="A",
                quote="setup",
                role="setup",
            ),
            RequiredEvidenceQuote(
                act=1,
                scene=1,
                speaker="B",
                quote="confirmation",
                role="confirmation",
            ),
        ],
        derived_gold_chunk_ids=[],
        notes="synthetic",
    )


def role_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "c_setup_1",
            "global_index": 0,
            "act": 1,
            "scene": 1,
            "scene_title": "Synthetic.",
            "token_count": 3,
            "text": "setup duplicate one",
            "roles": ["setup"],
        },
        {
            "chunk_id": "c_setup_2",
            "global_index": 1,
            "act": 1,
            "scene": 1,
            "scene_title": "Synthetic.",
            "token_count": 3,
            "text": "setup duplicate two",
            "roles": ["setup"],
        },
        {
            "chunk_id": "c_confirmation",
            "global_index": 2,
            "act": 1,
            "scene": 1,
            "scene_title": "Synthetic.",
            "token_count": 3,
            "text": "missing confirmation",
            "roles": ["confirmation"],
        },
    ]


class FakeSetRSelector:
    model_name = "fake-reader"

    def __init__(self, output: str):
        self.output = output
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_tokens": max_tokens,
            }
        )
        return self.output


class SetRSelectorTests(unittest.TestCase):
    def test_parse_setr_final_selection_returns_raw_numbers_like_official_code(self):
        parsed = parse_setr_final_selection(
            "reasoning\n### Final Selection: [3] [1] [3] [99]"
        )

        self.assertEqual(parsed, [3, 1, 3, 99])

    def test_parse_setr_final_selection_accepts_bare_digits(self):
        parsed = parse_setr_final_selection("steps...\n### Final Selection: 1 2 3")

        self.assertEqual(parsed, [1, 2, 3])

    def test_map_positions_dedupes_and_filters_invalid_numbers(self):
        mapping = map_positions_to_chunk_positions([3, 1, 3, 99], num_candidates=3)

        self.assertEqual(mapping["selected_positions"], [3, 1])
        self.assertEqual(mapping["dropped_out_of_range"], [99])
        self.assertEqual(mapping["dropped_duplicates"], [3])

    def test_map_positions_raises_when_no_valid_numbers(self):
        with self.assertRaises(SetRSelectionError):
            map_positions_to_chunk_positions([0, 99], num_candidates=3)

    def test_parse_setr_final_selection_requires_final_selection_marker(self):
        with self.assertRaises(SetRSelectionError):
            parse_setr_final_selection("[1] [2]")

    def test_setr_uses_selector_model_output_and_original_prompt(self):
        chunks = role_chunks()
        lookup = chunks_by_id(chunks)
        trace = [
            {"rank": 1, "chunk_id": "c_setup_1", "score": 1.0},
            {"rank": 2, "chunk_id": "c_setup_2", "score": 0.9},
            {"rank": 3, "chunk_id": "c_confirmation", "score": 0.8},
        ]
        candidate_ids = [row["chunk_id"] for row in trace]
        selector = FakeSetRSelector(
            "Step 1. setup and confirmation.\n"
            "Step 2. [1] and [3].\n"
            "### Final Selection: [1] [3]"
        )

        setr = select_setr(
            role_question(),
            candidate_ids,
            lookup,
            context_budget=6,
            selector_model=selector,
            retrieval_trace=trace,
            selector_max_tokens=123,
        )

        self.assertEqual(setr["selected_chunk_ids"], ["c_setup_1", "c_confirmation"])
        self.assertEqual(len(selector.calls), 1)
        self.assertEqual(selector.calls[0]["system_prompt"], SETR_SELECTION_SYS_PROMPT)
        self.assertEqual(selector.calls[0]["max_tokens"], 123)
        self.assertIn("Step 1. Please list up", SETR_SELECTION_IRI_PROMPT)
        self.assertIn("[1] setup duplicate one", str(selector.calls[0]["user_prompt"]))
        self.assertIn("[3] missing confirmation", str(selector.calls[0]["user_prompt"]))
        self.assertEqual(setr["context_assembly_trace"]["selected_positions"], [1, 3])
        self.assertFalse(setr["context_assembly_trace"]["cache_hit"])

    def test_setr_selector_output_is_cached(self):
        chunks = role_chunks()
        lookup = chunks_by_id(chunks)
        trace = [
            {"rank": 1, "chunk_id": "c_setup_1", "score": 1.0},
            {"rank": 2, "chunk_id": "c_setup_2", "score": 0.9},
            {"rank": 3, "chunk_id": "c_confirmation", "score": 0.8},
        ]
        with self.subTest("cache miss then hit"):
            cache_path = REPO_ROOT / "runs" / "tmp_setr_selector_test_cache.json"
            try:
                cache_path.unlink(missing_ok=True)
                selector = FakeSetRSelector("### Final Selection: [1] [3]")
                first = select_setr(
                    role_question(),
                    [row["chunk_id"] for row in trace],
                    lookup,
                    context_budget=6,
                    selector_model=selector,
                    retrieval_trace=trace,
                    cache_path=cache_path,
                )
                second = select_setr(
                    role_question(),
                    [row["chunk_id"] for row in trace],
                    lookup,
                    context_budget=6,
                    selector_model=selector,
                    retrieval_trace=trace,
                    cache_path=cache_path,
                )
            finally:
                cache_path.unlink(missing_ok=True)

        self.assertEqual(first["selected_chunk_ids"], second["selected_chunk_ids"])
        self.assertEqual(len(selector.calls), 1)
        self.assertFalse(first["context_assembly_trace"]["cache_hit"])
        self.assertTrue(second["context_assembly_trace"]["cache_hit"])

    def test_setr_can_select_from_top_50_not_only_top_20(self):
        chunks = [
            {
                "chunk_id": f"c_{index:02d}",
                "global_index": index,
                "act": 1,
                "scene": 1,
                "scene_title": "Synthetic.",
                "token_count": 1,
                "text": f"candidate {index}",
            }
            for index in range(1, 51)
        ]
        lookup = chunks_by_id(chunks)
        trace = [
            {"rank": index, "chunk_id": f"c_{index:02d}", "score": 1.0 / index}
            for index in range(1, 51)
        ]
        selector = FakeSetRSelector("### Final Selection: [25]")

        setr = select_setr(
            role_question(),
            [row["chunk_id"] for row in trace],
            lookup,
            context_budget=10,
            selector_model=selector,
            retrieval_trace=trace,
            max_passages=50,
        )

        self.assertEqual(setr["selected_chunk_ids"], ["c_25"])
        self.assertIn("[25] candidate 25", str(selector.calls[0]["user_prompt"]))
        self.assertEqual(setr["context_assembly_trace"]["max_passages"], 50)

    def test_setr_allows_empty_selection_for_no_evidence_question(self):
        chunks = role_chunks()
        lookup = chunks_by_id(chunks)
        trace = [
            {"rank": 1, "chunk_id": "c_setup_1", "score": 1.0},
            {"rank": 2, "chunk_id": "c_confirmation", "score": 0.8},
        ]
        selector = FakeSetRSelector("### Final Selection: []")
        question = Question(
            id="q_unanswerable",
            question="What is not stated?",
            expected_answer="The text does not state it.",
            evidence_scope="synthetic",
            reasoning_skill="unanswerable",
            required_evidence_quotes=[],
            derived_gold_chunk_ids=[],
            notes="synthetic",
        )

        setr = select_setr(
            question,
            [row["chunk_id"] for row in trace],
            lookup,
            context_budget=6,
            selector_model=selector,
            retrieval_trace=trace,
        )

        self.assertEqual(setr["selected_chunk_ids"], [])
        self.assertEqual(setr["context_tokens"], 0)
        self.assertTrue(setr["context_assembly_trace"]["empty_selection_allowed"])
        self.assertEqual(setr["context_assembly_trace"]["raw_selected_numbers"], [])


class DomainKGLiteTests(unittest.TestCase):
    def setUp(self):
        self.graph = DomainKnowledgeGraph.from_file(
            REPO_ROOT / "data" / "hamlet_domain_kg.yaml"
        )

    def test_domain_kg_lite_detects_character_aliases(self):
        mentions = self.graph.detect_mentions("What do King and Queen know?")

        by_alias = {mention["alias"]: mention["node_id"] for mention in mentions}
        self.assertEqual(by_alias["King"], "character:claudius")
        self.assertEqual(by_alias["Queen"], "character:gertrude")

    def test_domain_kg_lite_expands_event_to_related_graph_nodes(self):
        expanded = self.graph.expand_nodes(["event:mousetrap"], max_depth=2)

        self.assertIn("event:mousetrap", expanded)
        self.assertIn("event:claudius_reaction", expanded)
        self.assertIn("event:horatio_confirmation", expanded)

    def test_domain_kg_lite_creates_scaffold_within_budget(self):
        chunks = [
            {
                "chunk_id": "c_mousetrap",
                "global_index": 0,
                "act": 3,
                "scene": 2,
                "scene_title": "Synthetic.",
                "token_count": 10,
                "text": "The Mousetrap makes the King rise while Horatio notes him.",
            }
        ]
        trace = [{"rank": 1, "chunk_id": "c_mousetrap", "score": 1.0}]
        assembled = select_domain_kg_lite(
            Question(
                id="q_domain",
                question="How does the Mousetrap expose the King?",
                expected_answer="It prompts a reaction.",
                evidence_scope="synthetic",
                reasoning_skill="cross_scene_bridge",
                required_evidence_quotes=[],
                derived_gold_chunk_ids=[],
                notes="synthetic",
            ),
            ["c_mousetrap"],
            chunks_by_id(chunks),
            context_budget=32,
            retrieval_trace=trace,
            domain_kg=self.graph,
        )

        self.assertEqual(assembled["selected_chunk_ids"][0], DOMAIN_SCAFFOLD_CHUNK_ID)
        self.assertLessEqual(assembled["context_tokens"], 32)
        self.assertLessEqual(context_token_count(assembled["selected_chunks"]), 32)
        self.assertIn("Domain scaffold", assembled["selected_chunks"][0]["text"])


class StubSentenceScorer:
    model_name = "stub-extractive-compressor"

    def __init__(self, keyword: str):
        self.keyword = keyword

    def score_sentences(self, query: str, sentences: list[str]) -> list[float]:
        del query
        return [
            1.0 + index * 0.01 if self.keyword in sentence.lower() else 0.0
            for index, sentence in enumerate(sentences)
        ]


class FakeRecompReader:
    model_name = "fake-reader"

    def __init__(self, summary: str):
        self.summary = summary
        self.calls: list[dict[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {"system_prompt": system_prompt, "user_prompt": user_prompt}
        )
        return self.summary


def recomp_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "c1",
            "global_index": 0,
            "act": 1,
            "scene": 1,
            "scene_title": "First.",
            "token_count": 12,
            "text": "The queen drinks poison. The king watches in silence.",
        },
        {
            "chunk_id": "c2",
            "global_index": 1,
            "act": 1,
            "scene": 1,
            "scene_title": "First.",
            "token_count": 10,
            "text": "Laertes falls wounded. The poison works quickly there.",
        },
    ]


def recomp_request(
    treatment: str,
    feature_params: dict | None = None,
    feature_handles: dict | None = None,
    context_budget: int = 50,
    selector_model=None,
) -> ContextAssemblyRequest:
    chunks = recomp_chunks()
    return ContextAssemblyRequest(
        question=role_question(),
        treatment=treatment,
        context_budget=context_budget,
        chunk_lookup=chunks_by_id(chunks),
        doc_order_ids=[chunk["chunk_id"] for chunk in chunks],
        retrieval_trace=[
            {"rank": 1, "chunk_id": "c1", "score": 1.0},
            {"rank": 2, "chunk_id": "c2", "score": 0.9},
        ],
        selector_model=selector_model,
        feature_params=feature_params or {},
        feature_handles=feature_handles or {},
    )


class StubCragEvaluator:
    """Scores strips by keyword presence; deterministic for tests."""

    def __init__(self, keyword: str, hit_score: float = 5.0, miss_score: float = -2.0):
        self.keyword = keyword
        self.hit_score = hit_score
        self.miss_score = miss_score
        self.scored_documents: list[list[str]] = []

    def score(self, query: str, documents: list[str]) -> list[float]:
        del query
        self.scored_documents.append(list(documents))
        return [
            self.hit_score + index * 0.01
            if self.keyword in document.lower()
            else self.miss_score
            for index, document in enumerate(documents)
        ]


class FakeRewriteReader:
    model_name = "fake-reader"

    def __init__(self, rewritten: str = "poison, ear, orchard"):
        self.rewritten = rewritten
        self.calls: list[str] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        self.calls.append(user_prompt)
        return self.rewritten


class StubBM25:
    def __init__(self, hits: list[dict]):
        self.hits = hits
        self.queries: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        self.queries.append((query, top_k))
        return self.hits[:top_k]


def crag_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "c_trap",
            "global_index": 0,
            "act": 3,
            "scene": 2,
            "scene_title": "Mousetrap.",
            "token_count": 30,
            "text": (
                "He poisons him in the garden for his estate. "
                "The story is extant and written in very choice Italian. "
                "You shall see anon how the murderer gets the love of "
                "Gonzago's wife."
            ),
        },
        {
            "chunk_id": "c_true",
            "global_index": 1,
            "act": 1,
            "scene": 5,
            "scene_title": "Ghost.",
            "token_count": 28,
            "text": (
                "With juice of cursed hebenon in a vial he did pour the "
                "leperous distilment in my ears. It holds an enmity with "
                "blood of man. So the whole ear of Denmark is abused."
            ),
        },
    ]


def crag_request(
    rerank_scores: list[float],
    feature_params: dict | None = None,
    handles: dict | None = None,
    selector_model=None,
    context_budget: int = 200,
) -> ContextAssemblyRequest:
    chunks = crag_chunks()
    trace = [
        {
            "rank": index + 1,
            "chunk_id": chunk["chunk_id"],
            "score": score,
            "rerank_score": score,
        }
        for index, (chunk, score) in enumerate(zip(chunks, rerank_scores))
    ]
    params = {
        "crag_ndocs": 10,
        "crag_upper_threshold": 2.5,
        "crag_lower_threshold": 0.875,
        "crag_decompose_mode": "excerption",
        "crag_external_top_k": 5,
    }
    params.update(feature_params or {})
    return ContextAssemblyRequest(
        question=role_question(),
        treatment="crag",
        context_budget=context_budget,
        chunk_lookup=chunks_by_id(chunks),
        doc_order_ids=[chunk["chunk_id"] for chunk in chunks],
        retrieval_trace=trace,
        selector_model=selector_model,
        feature_params=params,
        feature_handles=handles or {},
    )


class CragTests(unittest.TestCase):
    def test_extract_strips_fixed_num_merges_short_remainder(self):
        words = [f"w{i}" for i in range(105)]
        strips = extract_strips_from_psg(" ".join(words), mode="fixed_num")

        self.assertEqual(len(strips), 2)
        self.assertEqual(len(strips[0].split()), 50)
        # 5-word remainder (<10) merges into the last 50-word window.
        self.assertEqual(len(strips[1].split()), 55)

    def test_extract_strips_excerption_concatenates_three_sentence_strips(self):
        psg = (
            "Alpha beta gamma delta epsilon zeta one. "
            "Beta gamma delta epsilon zeta eta two. "
            "Gamma delta epsilon zeta eta theta three. "
            "Delta epsilon zeta eta theta iota four."
        )
        strips = extract_strips_from_psg(psg, mode="excerption")

        self.assertEqual(len(strips), 2)
        self.assertIn("one", strips[0])
        self.assertIn("three", strips[0])
        self.assertIn("four", strips[1])

    def test_extract_strips_selection_keeps_whole_passage(self):
        self.assertEqual(extract_strips_from_psg("whole text", mode="selection"), ["whole text"])
        self.assertEqual(top_n_for_mode("selection"), 3)
        self.assertEqual(top_n_for_mode("excerption"), 6)

    def test_select_relevant_strips_scores_and_joins_by_score_order(self):
        evaluator = StubCragEvaluator(keyword="hebenon")
        result = select_relevant_strips(
            ["short one", "the hebenon poison was poured in the ear", "another strip about nothing relevant"],
            "what poison?",
            evaluator,
            top_n=2,
        )

        self.assertTrue(result["refined_text"].startswith("the hebenon poison"))
        self.assertIn("; ", result["refined_text"])
        # "short one" (<4 words) scored -1.0 without calling the evaluator.
        self.assertEqual(len(evaluator.scored_documents[0]), 2)

    def test_action_thresholds_match_official_flag_logic(self):
        self.assertEqual(action_from_scores([3.0, -1.0], 2.5, 0.875), "correct")
        self.assertEqual(action_from_scores([1.0, -1.0], 2.5, 0.875), "ambiguous")
        self.assertEqual(action_from_scores([0.5, -1.0], 2.5, 0.875), "incorrect")
        self.assertEqual(action_from_scores([2.5], 2.5, 0.875), "correct")
        self.assertEqual(action_from_scores([0.875], 2.5, 0.875), "ambiguous")

    def test_crag_correct_action_refines_internal_knowledge(self):
        evaluator = StubCragEvaluator(keyword="hebenon")
        result = assemble_crag(
            crag_request(
                rerank_scores=[1.0, 5.0],
                handles={"crag_evaluator": evaluator},
            )
        )

        trace = result.context_assembly_trace
        self.assertEqual(trace["action"], "correct")
        self.assertEqual(result.retrieval_method, "crag_correct")
        self.assertEqual(result.selected_chunk_ids, [CRAG_PSEUDO_CHUNK_ID])
        self.assertIn("hebenon", result.selected_chunks[0]["text"])
        self.assertNotIn("external_knowledge", trace)

    def test_crag_incorrect_action_rewrites_query_and_reretrieves(self):
        evaluator = StubCragEvaluator(keyword="hebenon")
        reader = FakeRewriteReader()
        bm25 = StubBM25(
            hits=[{"chunk_id": "c_true", "rank": 1, "score": 9.0}]
        )
        result = assemble_crag(
            crag_request(
                rerank_scores=[0.2, 0.1],
                handles={"crag_evaluator": evaluator, "crag_reretriever": bm25},
                selector_model=reader,
            )
        )

        trace = result.context_assembly_trace
        self.assertEqual(trace["action"], "incorrect")
        self.assertEqual(bm25.queries[0], ("poison, ear, orchard", 5))
        self.assertEqual(
            trace["external_knowledge"]["reretrieved_chunk_ids"],
            ["c_true"],
        )
        self.assertIn("hebenon", result.selected_chunks[0]["text"])
        # The rewrite prompt is the official popqa few-shot template.
        self.assertTrue(reader.calls[0].startswith(CRAG_KEYWORD_PROMPT[:60]))
        self.assertIn(role_question().question, reader.calls[0])

    def test_crag_ambiguous_action_combines_internal_and_external(self):
        evaluator = StubCragEvaluator(keyword="poison")
        reader = FakeRewriteReader()
        bm25 = StubBM25(hits=[{"chunk_id": "c_true", "rank": 1, "score": 9.0}])
        result = assemble_crag(
            crag_request(
                rerank_scores=[1.0, 0.9],
                handles={"crag_evaluator": evaluator, "crag_reretriever": bm25},
                selector_model=reader,
            )
        )

        trace = result.context_assembly_trace
        self.assertEqual(trace["action"], "ambiguous")
        text = result.selected_chunks[0]["text"]
        self.assertIn("Knowledge1:", text)
        self.assertIn("[sep] Knowledge2:", text)
        self.assertEqual(
            combine_knowledge("a", "b"),
            "Knowledge1: a [sep] Knowledge2: b",
        )

    def test_crag_truncates_refined_knowledge_to_budget(self):
        evaluator = StubCragEvaluator(keyword="poison")
        result = assemble_crag(
            crag_request(
                rerank_scores=[5.0, 4.0],
                handles={"crag_evaluator": evaluator},
                context_budget=5,
            )
        )

        self.assertEqual(result.selected_chunks[0]["token_count"], 5)
        self.assertTrue(result.context_assembly_trace["truncated_to_budget"])


def macrag_chunks() -> list[dict]:
    """Two scenes; chunks overlap by one word inside each scene."""

    def chunk(scene: int, index_in_scene: int, global_index: int, words: list[str]) -> dict:
        return {
            "chunk_id": f"act01_scene{scene:02d}_chunk{index_in_scene:03d}",
            "global_index": global_index,
            "act": 1,
            "scene": scene,
            "scene_id": f"act01_scene{scene:02d}",
            "scene_title": f"Scene {scene}.",
            "chunk_in_scene": index_in_scene,
            "start_token": (len(words) - 1) * (index_in_scene - 1),
            "end_token": (len(words) - 1) * (index_in_scene - 1) + len(words),
            "token_count": len(words),
            "text": " ".join(words),
        }

    return [
        chunk(1, 1, 0, ["alpha", "beta", "gamma"]),
        chunk(1, 2, 1, ["gamma", "delta", "epsilon"]),
        chunk(1, 3, 2, ["epsilon", "zeta", "eta"]),
        chunk(2, 1, 3, ["theta", "iota", "kappa"]),
    ]


def macrag_request(
    trace_chunk_ids: list[str],
    feature_params: dict | None = None,
    context_budget: int = 100,
) -> ContextAssemblyRequest:
    chunks = macrag_chunks()
    lookup = chunks_by_id(chunks)
    trace = [
        {
            "chunk_id": chunk_id,
            "rank": rank,
            "score": 1.0 - rank * 0.1,
            "slice_id": f"{chunk_id}_summary00",
            "slice_rank": rank,
            "retrieval_method": "macrag_summary_slices_reranked",
        }
        for rank, chunk_id in enumerate(trace_chunk_ids, start=1)
    ]
    params = {"macrag_top_k2": 7, "macrag_chunk_ext": 1, "macrag_merge_version": 1}
    params.update(feature_params or {})
    return ContextAssemblyRequest(
        question=role_question(),
        treatment="macrag",
        context_budget=context_budget,
        chunk_lookup=lookup,
        doc_order_ids=[chunk["chunk_id"] for chunk in chunks],
        retrieval_trace=trace,
        feature_params=params,
    )


class MacragTests(unittest.TestCase):
    def test_combine_without_overlap_removes_duplicated_boundary(self):
        self.assertEqual(
            combine_without_overlap("alpha beta gamma", "gamma delta"),
            "alpha beta gamma delta",
        )
        self.assertEqual(combine_without_overlap("", "alpha"), "alpha")
        self.assertEqual(
            combine_without_overlap("alpha beta", "gamma delta"),
            "alpha betagamma delta",
        )

    def test_recursive_character_split_respects_size_and_overlap(self):
        text = " ".join(f"word{i:02d}" for i in range(40))
        pieces = recursive_character_split(text, chunk_size=60, chunk_overlap=30)

        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(piece) <= 60 for piece in pieces))
        # Consecutive pieces share overlapping words.
        first_words = pieces[0].split()
        second_words = pieces[1].split()
        self.assertTrue(set(first_words) & set(second_words))

    def test_build_slices_includes_metadata_slice(self):
        slices = build_slices_for_chunk(
            "c1",
            {
                "summary": "short summary",
                "title": "T",
                "keywords": "K",
                "subheadings": "S",
            },
            slice_size=450,
            slice_overlap=300,
        )

        self.assertEqual(slices[0]["slice_id"], "c1_summary00")
        self.assertEqual(slices[-1]["slice_kind"], "metadata")
        self.assertEqual(slices[-1]["text"], "T K S")
        self.assertTrue(all(item["parent_chunk_id"] == "c1" for item in slices))

    def test_parse_summary_response_and_fallback(self):
        parsed = parse_summary_response(
            'noise [ {"Title": "T", "Keywords": "K", "Subheadings": "S", '
            '"Summary": "The summary."} ] trailing'
        )
        self.assertEqual(parsed["summary"], "The summary.")

        class BadSummarizer:
            model_name = "bad"

            def generate(self, system_prompt: str, user_prompt: str) -> str:
                return "not json at all"

        record = summarize_chunk("chunk text " * 100, BadSummarizer(), max_retries=1)
        self.assertTrue(record["fallback"])
        self.assertLessEqual(len(record["summary"]), 500)

    def test_slice_hits_dedupe_keeps_best_slice_per_parent(self):
        candidates = slice_hits_to_parent_candidates(
            [
                {"slice_id": "a_s0", "parent_chunk_id": "a", "slice_rank": 1},
                {"slice_id": "a_s1", "parent_chunk_id": "a", "slice_rank": 2},
                {"slice_id": "b_s0", "parent_chunk_id": "b", "slice_rank": 3},
            ]
        )

        self.assertEqual(
            [candidate["slice_id"] for candidate in candidates],
            ["a_s0", "b_s0"],
        )

    def test_macrag_merges_contiguous_chunks_and_dedups_overlap(self):
        result = assemble_macrag(
            macrag_request(["act01_scene01_chunk002"])
        )

        # chunk_ext=1 pulls chunks 1 and 3 in around the seed; the run merges
        # into one block whose overlaps are removed.
        self.assertEqual(len(result.selected_chunks), 1)
        merged = result.selected_chunks[0]
        self.assertEqual(
            merged["chunk_id"],
            "macrag_merged_act01_scene01_chunk001_to_act01_scene01_chunk003",
        )
        self.assertEqual(merged["text"], "alpha beta gamma delta epsilon zeta eta")
        # Exact token accounting: end_token(last) - start_token(first) = 7.
        self.assertEqual(merged["token_count"], 7)

    def test_macrag_expansion_stays_within_scene(self):
        result = assemble_macrag(
            macrag_request(["act01_scene01_chunk003"])
        )

        merged_ids = result.context_assembly_trace["expanded_chunk_ids"]
        # The neighbor at global_index 3 is in scene 2 and must be excluded.
        self.assertNotIn("act01_scene02_chunk001", merged_ids)
        self.assertEqual(
            merged_ids,
            ["act01_scene01_chunk002", "act01_scene01_chunk003"],
        )

    def test_macrag_merge_version_two_emits_block_per_run(self):
        result = assemble_macrag(
            macrag_request(
                ["act01_scene01_chunk001", "act01_scene01_chunk003"],
                feature_params={"macrag_chunk_ext": 0, "macrag_merge_version": 2},
            )
        )

        # chunks 1 and 3 are not contiguous: two blocks, each a real chunk.
        self.assertEqual(
            result.selected_chunk_ids,
            ["act01_scene01_chunk001", "act01_scene01_chunk003"],
        )

    def test_macrag_drops_blocks_over_budget(self):
        result = assemble_macrag(
            macrag_request(
                ["act01_scene01_chunk002", "act01_scene02_chunk001"],
                context_budget=7,
            )
        )

        self.assertEqual(len(result.selected_chunks), 1)
        self.assertEqual(
            result.context_assembly_trace["dropped_over_budget_blocks"],
            [["act01_scene02_chunk001"]],
        )
        self.assertLessEqual(result.context_tokens, 7)


class RecompTests(unittest.TestCase):
    def test_split_sentences_fallback_handles_verse_newlines(self):
        sentences = split_sentences("To be, or not to be.\nThat is the question. Ay!")
        self.assertEqual(
            sentences,
            ["To be, or not to be.", "That is the question.", "Ay!"],
        )

    def test_compress_extractive_keeps_top_sentences_in_score_order(self):
        result = compress_extractive(
            "What about the poison?",
            recomp_chunks(),
            StubSentenceScorer(keyword="poison"),
            top_sentences=2,
        )

        self.assertEqual(result["num_input_sentences"], 4)
        selected = [record["sentence"] for record in result["selected_sentences"]]
        self.assertEqual(len(selected), 2)
        self.assertTrue(all("poison" in sentence for sentence in selected))
        # Score order, not document order: the later poison sentence scores
        # higher under the stub's index tie-break.
        self.assertEqual(
            result["summary"],
            "The poison works quickly there. The queen drinks poison.",
        )

    def test_assemble_recomp_extractive_builds_summary_pseudo_chunk(self):
        handles = {
            "recomp_summaries": {
                "recomp_extractive:q_roles": {
                    "summary": "The queen drinks poison and dies.",
                    "compressor_model": "stub-extractive-compressor",
                    "selected_sentences": [],
                    "num_input_sentences": 4,
                }
            }
        }
        result = assemble_recomp_extractive(
            recomp_request("recomp_extractive", feature_handles=handles)
        )

        self.assertEqual(result.selected_chunk_ids, ["recomp_extractive_summary"])
        self.assertEqual(
            result.selected_chunks[0]["text"],
            "The queen drinks poison and dies.",
        )
        self.assertEqual(result.selected_chunks[0]["token_count"], 6)
        self.assertEqual(result.original_hit_chunk_ids, ["c1", "c2"])
        self.assertIn("deviations", result.context_assembly_trace)

    def test_assemble_recomp_extractive_truncates_summary_to_budget(self):
        handles = {
            "recomp_summaries": {
                "recomp_extractive:q_roles": {
                    "summary": "one two three four five six seven eight",
                    "compressor_model": "stub",
                }
            }
        }
        result = assemble_recomp_extractive(
            recomp_request(
                "recomp_extractive",
                feature_handles=handles,
                context_budget=3,
            )
        )

        self.assertEqual(result.selected_chunks[0]["text"], "one two three")
        self.assertEqual(result.context_tokens, 3)
        self.assertTrue(
            result.context_assembly_trace["summary_truncated_to_budget"]
        )

    def test_assemble_recomp_abstractive_empty_summary_falls_back_to_no_context(self):
        handles = {
            "recomp_summaries": {
                "recomp_abstractive:q_roles": {
                    "summary": "",
                    "compressor_model": "stub-t5",
                }
            }
        }
        result = assemble_recomp_abstractive(
            recomp_request("recomp_abstractive", feature_handles=handles)
        )

        self.assertEqual(result.selected_chunk_ids, [])
        self.assertEqual(result.context_tokens, 0)
        self.assertTrue(result.context_assembly_trace["empty_summary"])

    def test_assemble_recomp_abstractive_prompted_mode_uses_paper_prompt(self):
        reader = FakeRecompReader("The poison kills the queen and Laertes.")
        result = assemble_recomp_abstractive(
            recomp_request(
                "recomp_abstractive",
                feature_params={"recomp_abstractive_mode": "prompted_qwen"},
                selector_model=reader,
            )
        )

        self.assertEqual(len(reader.calls), 1)
        user_prompt = reader.calls[0]["user_prompt"]
        self.assertTrue(
            user_prompt.startswith(
                RECOMP_PROMPTED_ABSTRACTIVE_PROMPT.split("{question}")[0]
            )
        )
        self.assertIn("How do the setup and confirmation", user_prompt)
        self.assertIn("The queen drinks poison.", user_prompt)
        self.assertEqual(
            result.selected_chunks[0]["text"],
            "The poison kills the queen and Laertes.",
        )
        self.assertEqual(result.context_assembly_trace["mode"], "prompted_qwen")


if __name__ == "__main__":
    unittest.main()
