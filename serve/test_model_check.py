"""serve/test_model_check.py - round 13: serve/model_check.py on the byte tokenizer (no pack needed)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import frontend, model_check  # noqa: E402
from serve.server import ByteTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class FullTok(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>", "<tool_call>", "</tool_call>"]


class ModelCheck(unittest.TestCase):
    def test_the_original_template_and_a_full_vocabulary_agree(self):
        tpl = (ROOT / "serve/chat_template.jinja").read_text(encoding="utf-8")
        self.assertEqual(model_check.check(FullTok(), tpl, vision=True), [])

    def test_a_missing_special_and_a_template_without_xml_tool_calls(self):
        tpl = (ROOT / "serve/chat_template.jinja").read_text(encoding="utf-8").replace("<function=", "<fn=")
        lines = model_check.check(ByteTokenizer(), tpl)
        self.assertTrue(any("'<think>' is not one special token" in l for l in lines))
        self.assertTrue(any("does not write '<function='" in l for l in lines))
        off = model_check.apply(lines)
        self.assertIn("web search", off)


if __name__ == "__main__":
    unittest.main()
