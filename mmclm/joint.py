"""Joint request layout: every question about one state in one prompt,
all decided by the joint head in one pass.

Prompt (user turn of Gemma's chat template; media first so the prefix cache shares them across requests):
    <media>
    <state text>

    Questions:
    [Q1] <instruction>
    1. <option>
    2. <option>
    [Q2] <instruction>
    1. yes
    2. no

    Answer each question with the number of the correct option only.
A yes/no ("noul") question is rendered as a two-option choice ("yes", "no"); the head returns p(yes).
"""

from __future__ import annotations


INSTRUCTION = "Answer each question with the number of the correct option only."
MAX_OPTIONS = 255






def options_text(q: dict) -> list[str]:
    return ["yes", "no"] if q.get("type") == "noul" else list(q["options"])


def prompt_text(state_text: str, shown: list[tuple[dict, list[int]]]) -> str:
    """The user-turn text for a state and its (shuffled) questions."""
    blocks = []
    for i, (q, perm) in enumerate(shown):
        opts = options_text(q)
        lines = "\n".join(f"{j + 1}. {opts[p]}" for j, p in enumerate(perm))
        head = f"[Q{i + 1}] {q.get('instruction') or ''}".rstrip()
        blocks.append(f"{head}\n{lines}")
    body = "\n".join(blocks)
    state_text = (state_text or "").strip()
    return f"{state_text}\n\nQuestions:\n{body}\n\n{INSTRUCTION}" if state_text else f"Questions:\n{body}\n\n{INSTRUCTION}"








