"""Act/scene parsing and token-window chunking for Hamlet."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from hamlet_qa.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOKENIZER_MODEL,
)
from hamlet_qa.io_utils import read_text, write_jsonl


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ...

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        ...


@dataclass(frozen=True)
class SceneRecord:
    act: int
    scene: int
    scene_id: str
    title: str
    heading: str
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    global_index: int
    act: int
    scene: int
    scene_id: str
    scene_title: str
    chunk_in_scene: int
    start_token: int
    end_token: int
    token_count: int
    text: str


ACT_RE = re.compile(r"(?m)^\s*ACT\s+([IVXLCDM]+)\s*$")
SCENE_RE = re.compile(r"(?m)^\s*SCENE\s+([IVXLCDM]+)\.\s*(.*?)\s*$")
PLAY_ACT_I_RE = re.compile(r"(?m)^\s*ACT\s+I\s*$")

ROMAN_VALUES = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}


def roman_to_int(value: str) -> int:
    total = 0
    previous = 0
    for char in reversed(value.upper()):
        current = ROMAN_VALUES[char]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total


def scene_id(act: int, scene: int) -> str:
    return f"act{act:02d}_scene{scene:02d}"


def chunk_id(act: int, scene: int, chunk_number: int) -> str:
    return f"{scene_id(act, scene)}_chunk{chunk_number:03d}"


def extract_play_body(text: str) -> tuple[str, int]:
    """Return the play body starting at the second/play ACT I heading."""
    matches = list(PLAY_ACT_I_RE.finditer(text))
    if not matches:
        raise ValueError("Could not find an ACT I heading in the document.")
    start_match = matches[1] if len(matches) > 1 else matches[0]
    return text[start_match.start() :], start_match.start()


def parse_scenes(text: str) -> list[SceneRecord]:
    body, body_offset = extract_play_body(text)
    act_matches = list(ACT_RE.finditer(body))
    scene_matches = list(SCENE_RE.finditer(body))
    if not scene_matches:
        raise ValueError("No scene headings found after the play ACT I heading.")

    scenes: list[SceneRecord] = []
    act_index = 0
    for index, match in enumerate(scene_matches):
        while (
            act_index + 1 < len(act_matches)
            and act_matches[act_index + 1].start() < match.start()
        ):
            act_index += 1
        act_match = act_matches[act_index]
        act_number = roman_to_int(act_match.group(1))
        scene_number = roman_to_int(match.group(1))
        end = scene_matches[index + 1].start() if index + 1 < len(scene_matches) else len(body)
        heading = match.group(0).strip()
        title = match.group(2).strip()
        raw_scene = body[match.start() : end].strip()
        scenes.append(
            SceneRecord(
                act=act_number,
                scene=scene_number,
                scene_id=scene_id(act_number, scene_number),
                title=title,
                heading=heading,
                text=raw_scene,
                start_char=body_offset + match.start(),
                end_char=body_offset + end,
            )
        )
    return scenes


def load_tokenizer(model_name: str = DEFAULT_TOKENIZER_MODEL):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def count_tokens(tokenizer: TokenizerLike, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def chunk_scenes(
    scenes: list[SceneRecord],
    tokenizer: TokenizerLike,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkRecord]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    records: list[ChunkRecord] = []
    step = chunk_size - chunk_overlap
    global_index = 0

    for scene in scenes:
        token_ids = tokenizer.encode(scene.text, add_special_tokens=False)
        if not token_ids:
            continue

        start = 0
        chunk_number = 1
        while start < len(token_ids):
            end = min(start + chunk_size, len(token_ids))
            chunk_text = tokenizer.decode(
                token_ids[start:end],
                skip_special_tokens=True,
            ).strip()
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id(scene.act, scene.scene, chunk_number),
                    global_index=global_index,
                    act=scene.act,
                    scene=scene.scene,
                    scene_id=scene.scene_id,
                    scene_title=scene.title,
                    chunk_in_scene=chunk_number,
                    start_token=start,
                    end_token=end,
                    token_count=end - start,
                    text=chunk_text,
                )
            )
            global_index += 1
            chunk_number += 1
            if end >= len(token_ids):
                break
            start += step

    return records


def build_chunks(
    document_path: str | Path,
    tokenizer: TokenizerLike,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkRecord]:
    text = read_text(document_path)
    return chunk_scenes(
        parse_scenes(text),
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def write_chunks(path: str | Path, chunks: list[ChunkRecord]) -> None:
    write_jsonl(path, (asdict(chunk) for chunk in chunks))
