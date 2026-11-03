"""serve/model_check.py - round 13: the server's fixed template strings checked against the model's own tokenizer and
chat template at start (patches/LEDGER.md; bringup/RUNBOOK.md).

The server writes and reads a few strings itself instead of taking them from the template (the audit, Part 3.4):

  server.py    IM_END "<|im_end|>" and "<|endoftext|>" (the stop ids), EFFORT_TURN "<|im_start|>system ... <|im_end|>"
               and EFFORT_TEXT (effort_position "end", #458), REASONING_WRAP_UP "... </think>" (thinking budget, #123),
               IMAGE_PAD / VISION_START "<|vision_end|>" (images), --tail-role-token = the id of "system"
  frontend.py  "<think>" / "</think>" (reasoning split, #537's marks), the tool-call parser: CALL_START "<tool_call>",
               CALL_END "</tool_call>", FUNC_START "<function=", FUNC_END "</function>", "<parameter=", PARAM_END
               "</parameter>" (Qwen's XML tool format; #700's fix assumes it), the web search loop (on that parser)

A model whose tokenizer or template disagrees gets one line per disagreement.  Where a local patch depends on the
string, that patch is turned off (its switch), as the engine's patch guards do: the web search switch and #700's
parser need the template to write Qwen XML tool calls."""
from __future__ import annotations


SPECIALS = {   # the string -> what in the server depends on it
    "<|im_end|>": "the stop ids, the effort turn (server.py IM_END)",
    "<|im_start|>": "the effort turn (server.py EFFORT_TURN)",
    "<|endoftext|>": "the stop ids (server.py)",
    "<think>": "the reasoning split (frontend.py), #537's marks",
    "</think>": "the reasoning split, the thinking budget's wrap-up (server.py REASONING_WRAP_UP)",
    "<tool_call>": "the tool-call parser (frontend.py CALL_START), web search",
    "</tool_call>": "the tool-call parser (frontend.py CALL_END)",
}
VISION = {"<|image_pad|>": "images (server.py IMAGE_PAD)", "<|vision_start|>": "images (server.py VISION_START)",
          "<|vision_end|>": "images"}
TEMPLATE_WRITES = {   # what the template must write for the server's own parsing to match it
    "<|im_start|>": "chat turns", "<|im_end|>": "chat turns", "<think>": "the reasoning block",
    "</think>": "the reasoning block", "<tool_call>": "tool calls", "<function=": "Qwen XML tool calls",
    "<parameter=": "Qwen XML tool arguments",
}


def single_token(tok, s: str) -> bool:
    try:
        return len(tok.encode(s, parse_special=True)) == 1
    except Exception:  # noqa: BLE001 - a stand-in tokenizer
        return False


def check(tok, template_text: str, vision: bool = False, eos_id: int | None = None) -> list[str]:
    """-> the disagreement lines (empty: the model is what the server assumes).  Turns off the patches whose string
    is missing (the caller drops web search when "web search" is in apply()'s answer)."""
    lines = []
    for s, what in {**SPECIALS, **(VISION if vision else {})}.items():
        if not single_token(tok, s):
            lines.append(f"'{s}' is not one special token in this model's vocabulary: {what}")
    for s, what in TEMPLATE_WRITES.items():
        if template_text and s not in template_text:
            lines.append(f"the chat template does not write '{s}' ({what})")
    if eos_id is not None:
        try:
            im_end = tok.encode("<|im_end|>", parse_special=True)
            if len(im_end) == 1 and im_end[0] != eos_id:
                lines.append(f"the model's end-of-turn id {eos_id} is not <|im_end|> ({im_end[0]}): the stop ids are "
                             f"<|im_end|> and <|endoftext|>")
        except Exception:  # noqa: BLE001
            pass
    return lines


def apply(lines: list[str]) -> dict:
    """The patches to turn off for these disagreements."""
    off = {}
    if any("'<function='" in l or "'<tool_call>'" in l or "'<parameter='" in l for l in lines):
        # (round 14: #700's prose-tag rule is upstream's own code since v0.1.40 - it has no switch and is not ours to
        # turn off; only our web search writes Qwen XML calls itself)
        off["web search"] = "off"
    return off
