"""Ask the model what kind of game this is, before it tries to play it.

Goal inference is the hard part of ARC-AGI-3: nothing states the objective, and
every probe costs an action the score squares. Analogy narrows the search --
"this is a piece being rotated into a slot" rules out most of the hypothesis
space in one step -- but analogy recall is visual-motor, and a single still
board carries no motion to recall from.

So this runs off to the side, out of the play conversation: it is shown a
filmstrip and a motion overlay, and it answers in three turns, in this order.

  1. what moved, with no rules and no goal named
  2. the ten things it already knows that move like that
  3. only then, the category, the goal, and three hypotheses that can die

The order is the point. Asked for hypotheses first, the model reasons from the
last panel and skips the motion entirely; asked to recall first, it commits to
a description it then has to stay consistent with.

Every failure here returns None. The layer is an addition, not a dependency:
if the endpoint times out, the JSON is malformed, or PIL is missing, the game
carries on exactly as it would without it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from inference.agent.hypotheses import Hypothesis, parse_framing

log = logging.getLogger(__name__)

ChatFn = Callable[[list[dict[str, Any]], int], str]

TURN1_TOKENS = 700
TURN2_TOKENS = 900
TURN3_TOKENS = 1200


def _motion_prompt(valid_actions: list[str], notes_tail: list[str]) -> str:
    actions = ", ".join(valid_actions) if valid_actions else "unknown"
    known = ""
    if notes_tail:
        joined = "\n".join(f"- {line}" for line in notes_tail)
        known = f"\n\nWhat you already recorded this game:\n{joined}"
    return (
        "You are shown a filmstrip: consecutive frames from one level of a grid puzzle game, "
        "left to right in time order, joined by \">\" arrows. Each panel is labelled with the "
        f"action that produced it. Valid actions in this game: {actions}"
        "\n\nDescribe the motion, and only the motion. For each arrow, say what changed between "
        "those two panels, naming the panels."
        "\n\nSeparate two things that are easy to confuse. Say whether a shape changed ORIENTATION, "
        "and say whether it changed POSITION. Both can happen at once: a piece that swings around "
        "another object turns and moves along an arc, and calling that only a translation loses the "
        "pivot. If nothing changed between two panels, say so plainly -- many actions do nothing, "
        "and that is a finding."
        "\n\nOne thing to name and then set aside. A long thin strip flush against an edge that "
        "advances a little every panel, in the same direction, regardless of which action was "
        "taken, is a step or time counter. Say that you see it, say which edge, and then ignore "
        "it. It is not an object, it is not a target, and it is not evidence about any action."
        "\n\nDo not guess the rules. Do not name the goal. Motion only."
        f"{known}"
    )


TURN2_PROMPT = (
    "Here is a SECOND image of the same transitions: every cell that changed is drawn onto one "
    "board, faint = earlier, solid = later. It is NOT a later frame and NOT a later point in time. "
    "It is a summary of the motion you just described, collapsed onto one board. It is reliable for "
    "things that travelled and unreliable for things that turned in place, which overlap themselves."
    "\n\nNow recall. Given that motion, list the TOP 10 things you already know that move like this: "
    "video games, arcade or puzzle mechanics, physical processes, everyday phenomena, well-known "
    "GIFs. Rank them by how closely the motion matches. For each, give the name and the one sentence "
    "of motion that makes it match."
    "\n\nThe ten must not be ten costumes for one idea -- a hinge, a clock hand, a windmill and a "
    "propeller are one entry, not four. At most three may share a mechanism. At least four must be "
    "actual games or puzzle mechanics you could name the rules of."
    "\n\nYou are recalling, not deducing. Do not analyse the puzzle yet."
)


def _derive_prompt(dead_names: list[str]) -> str:
    refuted = ""
    if dead_names:
        refuted = (
            "\n\nAlready refuted in this game, do not propose again: "
            + ", ".join(dead_names)
        )
    return (
        "From your top 10, work out what kind of problem this is."
        "\n\nEvery `because` must be consistent with the motion you described in your first answer. "
        "If you now think that description was wrong, say so explicitly and say what you think "
        "instead -- but do not quietly contradict it."
        "\n\nReturn a single JSON object and nothing else:"
        "\n{"
        "\n  \"category\": one of physics|pattern|geometry|color|object|spatial|logic|construct|other,"
        "\n  \"why_category\": one sentence naming which of your top 10 drove it,"
        "\n  \"goal_hypothesis\": what would count as winning here, in one sentence. If a target,"
        "\n     template, or silhouette is displayed anywhere on the board, say where it is."
        "\n     Never name the edge counter as the goal or the target: it advances on its own,"
        "\n     so nothing you do can be aimed at it.,"
        "\n  \"hypotheses\": [ exactly 3 objects ]"
        "\n}"
        "\n\nThe 3 hypotheses must sit on DIFFERENT axes: one about what an action does, one about "
        "what the goal is, one about a constraint or failure condition. Three phrasings of one idea "
        "is a failure, because they die together and leave you with nothing."
        "\n\nEach object needs: \"name\", \"because\" (cite a panel or a top-10 entry), \"predicts\" "
        "(something the very next action could confirm), \"killed_if\" (something the very next "
        "action could trigger), \"confidence\" (a number between 0 and 1, not a word)."
        f"{refuted}"
    )


@dataclass
class FramingResult:
    category: str
    goal_hypothesis: str
    hypotheses: list[Hypothesis]
    motion: str
    recall: str
    raw: str


def _user(text: str, image_url: str | None = None) -> dict[str, Any]:
    if not image_url:
        return {"role": "user", "content": text}
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }


def run_framing(
    chat: ChatFn,
    *,
    filmstrip_url: str,
    trail_url: str | None,
    valid_actions: list[str],
    notes_tail: list[str],
    dead_names: list[str],
) -> FramingResult | None:
    """Three turns off to the side. Returns None if anything at all goes wrong."""
    messages: list[dict[str, Any]] = [_user(_motion_prompt(valid_actions, notes_tail), filmstrip_url)]
    try:
        motion = chat(messages, TURN1_TOKENS)
        if not str(motion).strip():
            log.warning("framing: motion turn returned nothing")
            return None
        messages.append({"role": "assistant", "content": motion})

        messages.append(_user(TURN2_PROMPT, trail_url))
        recall = chat(messages, TURN2_TOKENS)
        messages.append({"role": "assistant", "content": recall or ""})

        messages.append(_user(_derive_prompt(dead_names)))
        raw = chat(messages, TURN3_TOKENS)
    except Exception as exc:
        log.warning("framing call failed, continuing without it: %s", exc)
        return None

    category, goal, hypotheses = parse_framing(raw)
    if not hypotheses:
        log.warning("framing produced no falsifiable hypotheses; discarding")
        return None
    return FramingResult(
        category=category,
        goal_hypothesis=goal,
        hypotheses=hypotheses,
        motion=str(motion),
        recall=str(recall or ""),
        raw=str(raw),
    )
