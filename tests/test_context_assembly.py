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
    parse_setr_final_selection,
    select_setr,
)
from hamlet_qa.core.questions import Question, RequiredEvidenceQuote
from hamlet_qa.core.context import chunks_by_id, context_token_count


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
    def test_parse_setr_final_selection_dedupes_and_filters_invalid_numbers(self):
        parsed = parse_setr_final_selection(
            "reasoning\n### Final Selection: [3] [1] [3] [99]",
            num_candidates=3,
        )

        self.assertEqual(parsed, [3, 1])

    def test_parse_setr_final_selection_requires_final_selection_marker(self):
        with self.assertRaises(SetRSelectionError):
            parse_setr_final_selection("[1] [2]", num_candidates=2)

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


if __name__ == "__main__":
    unittest.main()
