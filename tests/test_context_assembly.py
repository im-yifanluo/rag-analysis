from __future__ import annotations

import unittest
from pathlib import Path

from hamlet_qa.features.domain.kg import (
    DOMAIN_SCAFFOLD_CHUNK_ID,
    DomainKnowledgeGraph,
    select_domain_kg_lite,
)
from hamlet_qa.features.setr.selector import select_setr_lite
from hamlet_qa.core.questions import Question, RequiredEvidenceQuote
from hamlet_qa.core.context import (
    chunks_by_id,
    context_token_count,
    select_chunk_ids_for_budget,
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


def covered_roles(chunks: list[dict], selected_ids: list[str]) -> set[str]:
    lookup = chunks_by_id(chunks)
    roles: set[str] = set()
    for chunk_id in selected_ids:
        roles.update(lookup[chunk_id].get("roles", []))
    return roles


class SetRLiteTests(unittest.TestCase):
    def test_setr_lite_selects_more_distinct_roles_than_plain_top_k(self):
        chunks = role_chunks()
        lookup = chunks_by_id(chunks)
        trace = [
            {"rank": 1, "chunk_id": "c_setup_1", "score": 1.0},
            {"rank": 2, "chunk_id": "c_setup_2", "score": 0.9},
            {"rank": 3, "chunk_id": "c_confirmation", "score": 0.8},
        ]
        candidate_ids = [row["chunk_id"] for row in trace]

        top_k_ids = select_chunk_ids_for_budget(candidate_ids, lookup, context_budget=6)
        setr = select_setr_lite(
            role_question(),
            candidate_ids,
            lookup,
            context_budget=6,
            retrieval_trace=trace,
        )

        self.assertEqual(covered_roles(chunks, top_k_ids), {"setup"})
        self.assertEqual(
            covered_roles(chunks, setr["selected_chunk_ids"]),
            {"setup", "confirmation"},
        )

    def test_setr_lite_avoids_duplicate_role_when_missing_role_is_available(self):
        chunks = role_chunks()
        lookup = chunks_by_id(chunks)
        trace = [
            {"rank": 1, "chunk_id": "c_setup_1", "score": 1.0},
            {"rank": 2, "chunk_id": "c_setup_2", "score": 0.9},
            {"rank": 3, "chunk_id": "c_confirmation", "score": 0.8},
        ]

        setr = select_setr_lite(
            role_question(),
            [row["chunk_id"] for row in trace],
            lookup,
            context_budget=6,
            retrieval_trace=trace,
        )

        self.assertEqual(setr["selected_chunk_ids"], ["c_setup_1", "c_confirmation"])
        self.assertNotIn("c_setup_2", setr["selected_chunk_ids"])


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
