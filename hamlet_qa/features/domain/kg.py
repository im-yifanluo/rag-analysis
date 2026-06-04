"""Domain-knowledge-guided Hamlet context scaffold and selector."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from hamlet_qa.core.context import candidate_rank_map, dedupe_existing_chunk_ids
from hamlet_qa.core.text import flatten_string_list, phrase_in_text, tokenize_terms


DOMAIN_SCAFFOLD_CHUNK_ID = "domain_scaffold"


def normalize_role(role: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", role.lower()).strip("_")
    return normalized or "answer"


def load_domain_kg_data(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml  # type: ignore
        except ImportError as import_error:
            raise ValueError(
                f"{path} is not JSON-subset YAML, and PyYAML is not installed for "
                "general YAML parsing."
            ) from import_error
        try:
            data = yaml.safe_load(text)
        except Exception as yaml_error:  # pragma: no cover - depends on optional PyYAML
            raise ValueError(f"Could not parse domain KG file {path}") from yaml_error
        if not isinstance(data, dict):
            raise ValueError(f"Domain KG file {path} must contain a mapping")
        return data
    if not isinstance(data, dict):
        raise ValueError(f"Domain KG file {path} must contain a mapping")
    return data


class DomainKnowledgeGraph:
    """Small editable Hamlet graph used for deterministic context scaffolding."""

    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.characters: dict[str, dict[str, Any]] = {
            str(key): dict(value)
            for key, value in dict(data.get("characters", {})).items()
        }
        self.events: dict[str, dict[str, Any]] = {
            str(key): dict(value)
            for key, value in dict(data.get("events", {})).items()
        }
        self.relations: list[dict[str, Any]] = [
            dict(item) for item in list(data.get("relations", []))
        ]
        self.evidence_role_templates: dict[str, list[str]] = {
            normalize_role(str(key)): [str(item) for item in flatten_string_list(value)]
            for key, value in dict(data.get("evidence_role_templates", {})).items()
        }
        self.alias_index = self._build_alias_index()

    @classmethod
    def from_file(cls, path: str | Path) -> "DomainKnowledgeGraph":
        return cls(load_domain_kg_data(path))

    def canonical_node_id(self, value: str) -> str:
        if ":" in value:
            return value
        if value in self.characters:
            return f"character:{value}"
        if value in self.events:
            return f"event:{value}"
        return value

    def _build_alias_index(self) -> list[tuple[str, str, str]]:
        aliases: list[tuple[str, str, str]] = []
        for key, record in self.characters.items():
            node_id = f"character:{key}"
            label = str(record.get("name", key))
            aliases.append((label, node_id, label))
            for alias in flatten_string_list(record.get("aliases")):
                aliases.append((alias, node_id, label))
        for alias, target in dict(self.data.get("aliases", {})).items():
            node_id = self.canonical_node_id(str(target))
            aliases.append((str(alias), node_id, self.node_label(node_id)))
        for key, record in self.events.items():
            node_id = f"event:{key}"
            label = str(record.get("name", key))
            aliases.append((label, node_id, label))
            for alias in flatten_string_list(record.get("aliases")):
                aliases.append((alias, node_id, label))
        for alias, target in dict(self.data.get("event_aliases", {})).items():
            node_id = self.canonical_node_id(str(target))
            aliases.append((str(alias), node_id, self.node_label(node_id)))

        unique: dict[tuple[str, str], tuple[str, str, str]] = {}
        for alias, node_id, label in aliases:
            normalized = " ".join(tokenize_terms(alias))
            if not normalized:
                continue
            unique[(normalized, node_id)] = (alias, node_id, label)
        return sorted(
            unique.values(),
            key=lambda item: (-len(tokenize_terms(item[0])), -len(item[0]), item[0]),
        )

    def node_record(self, node_id: str) -> dict[str, Any]:
        kind, _, key = node_id.partition(":")
        if kind == "character":
            return self.characters.get(key, {})
        if kind == "event":
            return self.events.get(key, {})
        return {}

    def node_label(self, node_id: str) -> str:
        record = self.node_record(node_id)
        if record:
            return str(record.get("name") or node_id.partition(":")[2])
        return node_id

    def aliases_for_node(self, node_id: str) -> list[str]:
        aliases = [alias for alias, target, _label in self.alias_index if target == node_id]
        return sorted(set(aliases), key=lambda item: (len(item), item.lower()))

    def detect_mentions(self, text: str) -> list[dict[str, Any]]:
        mentions: dict[str, dict[str, Any]] = {}
        lowered = text.lower()
        for alias, node_id, label in self.alias_index:
            if not phrase_in_text(lowered, alias):
                continue
            first_pos = lowered.find(alias.lower())
            current = mentions.get(node_id)
            if current is None or first_pos < int(current["offset"]):
                mentions[node_id] = {
                    "alias": alias,
                    "node_id": node_id,
                    "canonical": label,
                    "offset": first_pos,
                }
        return sorted(mentions.values(), key=lambda item: int(item["offset"]))

    def expand_nodes(self, seed_node_ids: list[str], max_depth: int = 2) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        frontier = [(self.canonical_node_id(node_id), 0) for node_id in seed_node_ids]
        while frontier:
            node_id, depth = frontier.pop(0)
            if node_id in seen:
                continue
            seen.add(node_id)
            ordered.append(node_id)
            if depth >= max_depth:
                continue
            for relation in self.relations:
                source = self.canonical_node_id(str(relation.get("source", "")))
                target = self.canonical_node_id(str(relation.get("target", "")))
                if source == node_id and target and target not in seen:
                    frontier.append((target, depth + 1))
                elif target == node_id and source and source not in seen:
                    frontier.append((source, depth + 1))
        return ordered

    def relation_lines_for_nodes(self, node_ids: list[str]) -> list[str]:
        selected = set(node_ids)
        lines: list[str] = []
        for relation in self.relations:
            source = self.canonical_node_id(str(relation.get("source", "")))
            target = self.canonical_node_id(str(relation.get("target", "")))
            if source not in selected or target not in selected:
                continue
            relation_name = str(relation.get("relation", "related_to")).replace("_", " ")
            description = str(relation.get("description", "")).strip()
            line = f"- {self.node_label(source)} --{relation_name}-> {self.node_label(target)}"
            if description:
                line += f": {description}"
            lines.append(line)
        return lines

    def node_line(self, node_id: str) -> str:
        kind, _, _key = node_id.partition(":")
        record = self.node_record(node_id)
        label = self.node_label(node_id)
        description = str(record.get("description", "")).strip()
        aliases = ", ".join(self.aliases_for_node(node_id)[:5])
        line = f"- {label} ({kind})"
        if description:
            line += f": {description}"
        if aliases:
            line += f" Aliases: {aliases}."
        return line

    def build_scaffold(
        self,
        question_text: str,
        mentions: list[dict[str, Any]],
        expanded_node_ids: list[str],
        token_budget: int,
    ) -> dict[str, Any]:
        lines: list[str] = []

        def count_tokens(candidate_lines: list[str]) -> int:
            return len("\n".join(candidate_lines).split())

        def add(line: str) -> None:
            if token_budget <= 0:
                return
            candidate = lines + [line]
            if count_tokens(candidate) <= token_budget:
                lines.append(line)

        add("Domain scaffold:")
        summary = str(self.data.get("summary", "")).strip()
        if summary:
            add(f"Story: {summary}")
        if mentions:
            alias_line = "; ".join(
                f"{item['alias']} -> {item['canonical']}" for item in mentions
            )
            add(f"Detected aliases: {alias_line}")
        add(f"Question focus: {question_text}")
        if expanded_node_ids:
            add("Relevant graph nodes:")
            for node_id in expanded_node_ids:
                add(self.node_line(node_id))
            relation_lines = self.relation_lines_for_nodes(expanded_node_ids)
            if relation_lines:
                add("Relevant graph relations:")
                for line in relation_lines:
                    add(line)

        text = "\n".join(lines)
        return {
            "chunk_id": DOMAIN_SCAFFOLD_CHUNK_ID,
            "global_index": -1,
            "act": 0,
            "scene": 0,
            "scene_id": "domain_kg",
            "scene_title": "Domain KG scaffold",
            "chunk_in_scene": 0,
            "start_token": 0,
            "end_token": len(text.split()),
            "token_count": len(text.split()),
            "text": text,
        }


def domain_node_matches_for_chunk(
    graph: DomainKnowledgeGraph,
    chunk: dict[str, Any],
    node_ids: list[str],
) -> list[str]:
    text = str(chunk.get("text", ""))
    matched: list[str] = []
    for node_id in node_ids:
        aliases = graph.aliases_for_node(node_id)
        if any(phrase_in_text(text, alias) for alias in aliases):
            matched.append(node_id)
            continue
        record = graph.node_record(node_id)
        keywords = flatten_string_list(record.get("keywords"))
        if any(phrase_in_text(text, keyword) for keyword in keywords):
            matched.append(node_id)
    return matched


def select_domain_kg_lite(
    question: Any,
    candidate_chunk_ids: list[str],
    chunk_lookup: dict[str, dict[str, Any]],
    context_budget: int,
    retrieval_trace: list[dict[str, Any]] | None = None,
    domain_kg: DomainKnowledgeGraph | None = None,
) -> dict[str, Any]:
    """Assemble a KG scaffold plus post-retrieval chunks boosted by KG matches."""

    if domain_kg is None:
        raise ValueError("domain_kg_lite requires a DomainKnowledgeGraph")

    question_text = getattr(question, "question", str(question))
    mentions = domain_kg.detect_mentions(question_text)
    seed_node_ids = [str(item["node_id"]) for item in mentions]
    expanded_node_ids = domain_kg.expand_nodes(seed_node_ids, max_depth=2)

    scaffold_budget = min(context_budget, max(24, context_budget // 3))
    scaffold = domain_kg.build_scaffold(
        question_text,
        mentions,
        expanded_node_ids,
        token_budget=scaffold_budget,
    )
    scaffold_tokens = int(scaffold["token_count"])
    remaining_budget = max(0, context_budget - scaffold_tokens)

    candidates = dedupe_existing_chunk_ids(candidate_chunk_ids, chunk_lookup)
    rank_map = candidate_rank_map(candidates)
    chunk_node_matches = {
        chunk_id: domain_node_matches_for_chunk(
            domain_kg,
            chunk_lookup[chunk_id],
            expanded_node_ids,
        )
        for chunk_id in candidates
    }
    ranked_candidates = sorted(
        candidates,
        key=lambda chunk_id: (
            -len(chunk_node_matches[chunk_id]),
            rank_map[chunk_id],
            int(chunk_lookup[chunk_id]["global_index"]),
        ),
    )

    selected_ids: list[str] = []
    selected_tokens = 0
    for chunk_id in ranked_candidates:
        token_count = int(chunk_lookup[chunk_id]["token_count"])
        if selected_tokens + token_count > remaining_budget:
            continue
        selected_ids.append(chunk_id)
        selected_tokens += token_count

    final_chunk_ids = [DOMAIN_SCAFFOLD_CHUNK_ID] if scaffold_tokens else []
    final_chunk_ids.extend(selected_ids)
    final_chunks = ([scaffold] if scaffold_tokens else []) + [
        dict(chunk_lookup[chunk_id]) for chunk_id in selected_ids
    ]
    return {
        "selected_chunk_ids": final_chunk_ids,
        "selected_chunks": final_chunks,
        "context_tokens": scaffold_tokens + selected_tokens,
        "prompt_order": "domain_kg_scaffold_then_dense_hits",
        "retrieval_method": (
            f"{retrieval_trace[0].get('retrieval_method', 'dense_faiss')}_domain_kg_lite"
            if retrieval_trace
            else "domain_kg_lite"
        ),
        "context_assembly_trace": {
            "method": "domain_kg_lite",
            "detected_mentions": mentions,
            "expanded_node_ids": expanded_node_ids,
            "chunk_node_matches": chunk_node_matches,
            "scaffold_token_budget": scaffold_budget,
            "scaffold_tokens": scaffold_tokens,
            "selected_retrieval_chunk_ids": selected_ids,
        },
    }
