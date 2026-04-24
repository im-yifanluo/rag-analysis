from __future__ import annotations

import unittest
from pathlib import Path

from hamlet_qa.chunking import SceneRecord, chunk_scenes, parse_scenes


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    def __init__(self):
        self.current_tokens: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        self.current_tokens = text.split()
        return list(range(len(self.current_tokens)))

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(self.current_tokens[index] for index in token_ids)


class SceneParsingTests(unittest.TestCase):
    def test_parse_real_hamlet_scenes_skips_front_matter(self):
        text = (REPO_ROOT / "hamlet.txt").read_text(encoding="utf-8")
        scenes = parse_scenes(text)

        self.assertEqual(len(scenes), 20)
        self.assertEqual(scenes[0].scene_id, "act01_scene01")
        self.assertEqual(scenes[0].title, "Elsinore. A platform before the Castle.")
        self.assertTrue(scenes[0].text.startswith("SCENE I."))
        self.assertEqual(scenes[-1].scene_id, "act05_scene02")
        self.assertEqual(scenes[-1].title, "A hall in the Castle.")


class ChunkingTests(unittest.TestCase):
    def test_chunk_ids_overlap_and_scene_boundaries_are_stable(self):
        scenes = [
            SceneRecord(
                act=1,
                scene=1,
                scene_id="act01_scene01",
                title="First scene.",
                heading="SCENE I. First scene.",
                text="a b c d e f g h i",
                start_char=0,
                end_char=17,
            ),
            SceneRecord(
                act=1,
                scene=2,
                scene_id="act01_scene02",
                title="Second scene.",
                heading="SCENE II. Second scene.",
                text="j k l",
                start_char=18,
                end_char=23,
            ),
        ]

        chunks = chunk_scenes(scenes, FakeTokenizer(), chunk_size=4, chunk_overlap=2)

        self.assertEqual(
            [chunk.chunk_id for chunk in chunks],
            [
                "act01_scene01_chunk001",
                "act01_scene01_chunk002",
                "act01_scene01_chunk003",
                "act01_scene01_chunk004",
                "act01_scene02_chunk001",
            ],
        )
        self.assertEqual(chunks[0].text, "a b c d")
        self.assertEqual(chunks[1].text, "c d e f")
        self.assertEqual(chunks[0].end_token - chunks[1].start_token, 2)
        self.assertEqual(chunks[-1].text, "j k l")
        self.assertTrue(all(chunk.token_count > 0 for chunk in chunks))
        self.assertEqual(chunks[-1].global_index, 4)


if __name__ == "__main__":
    unittest.main()
