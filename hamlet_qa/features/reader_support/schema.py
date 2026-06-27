"""Dataclasses for the Reader-Supervised Evidence Support Assembler.

These are plain, JSON-serializable records shared by node induction, unit
construction, support scoring, and assembly. They are kept deliberately simple
so the full pipeline can be dumped into `context_assembly_trace` for analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvidenceNode:
    """One information need that must be supported to answer the question."""

    node_id: str
    need: str
    node_query: str
    order_index: int = 0
    depends_on: list[str] = field(default_factory=list)
    confidence: float | None = None
    raw_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceUnit:
    """A source-faithful candidate span that may enter the final context."""

    unit_id: str
    unit_type: str  # chunk | sentence | line_span | neighbor_left/right/both
    text: str
    source_chunk_ids: list[str]
    primary_chunk_id: str
    token_count: int
    source_order_key: list[int]
    act: int | None = None
    scene: int | None = None
    scene_title: str | None = None
    scene_id: str | None = None
    global_index_start: int | None = None
    global_index_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> dict[str, Any]:
        """Compact view for the trace (omits full text)."""
        return {
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "primary_chunk_id": self.primary_chunk_id,
            "source_chunk_ids": list(self.source_chunk_ids),
            "token_count": self.token_count,
            "text_preview": self.text[:120],
        }


@dataclass
class SupportScore:
    """Reader-teacher judgement of how well a unit supports a node."""

    node_id: str
    unit_id: str
    support_score: float
    support_type: str  # none | related | partial | complete | contradictory
    supporting_span: str = ""
    needs_more_context: bool = False
    explanation: str = ""
    raw_output: str = ""
    parse_error: str | None = None
    validation_warnings: list[str] = field(default_factory=list)
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
