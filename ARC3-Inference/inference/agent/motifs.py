"""A vocabulary of game shapes, and a way to look one up without paying for all of them.

Twenty public games were written up by hand and the recurring structures pulled
out of them: push puzzles, rotations, threading, symmetry completion. The
competition documents the private set as not overlapping with the public one,
so none of those games' solutions are worth carrying. The *names* are, because
naming a shape is what turns "press things and see" into "if this is a push
puzzle, walking into a block is the one probe that settles it".

Two tiers, for a reason that is arithmetic. The window this harness runs in is
32k tokens and the conversation already gets trimmed to fit. The 22 one-line
summaries cost ~440 tokens, which buys the vocabulary. Each detailed entry
costs another ~150, which is cheap once but 2.9k for all of them -- so the
detail is fetched from Python, on the turn it is wanted, and nothing is spent
on the twenty-one motifs a given game is not.

The catalog is generated: code/tools/build_motif_catalog.py in the parent repo.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CATALOG_PATH = Path(__file__).with_name("motif_catalog.json")


@lru_cache(maxsize=1)
def catalog() -> list[dict[str, Any]]:
    """Every motif, or an empty list if the file did not ship.

    Empty is a supported state, not an error: the harness has to run when the
    catalog is missing exactly as it runs when the feature is switched off.
    """
    try:
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("motif catalog unavailable, continuing without it: %s", exc)
        return []
    motifs = data.get("motifs")
    return motifs if isinstance(motifs, list) else []


def slugs() -> list[str]:
    return [str(m.get("slug", "")) for m in catalog() if m.get("slug")]


def detail(slug: str) -> dict[str, Any] | None:
    wanted = str(slug or "").strip().lower().replace("_", "-").replace(" ", "-")
    for motif in catalog():
        if str(motif.get("slug", "")).lower() == wanted:
            return {k: v for k, v in motif.items() if k != "has_detail"}
    return None


def summary_block() -> str:
    """The always-on part of the prompt. Empty string when there is nothing to say."""
    motifs = catalog()
    if not motifs:
        return ""
    lines = "\n".join(
        f"  {m['slug']}: {m['summary']}" for m in motifs if m.get("slug") and m.get("summary")
    )
    if not lines:
        return ""
    return (
        "\n\nGrid games of this kind keep reusing a small number of shapes. These are the ones "
        "seen most often, as a vocabulary for saying what you are looking at:\n"
        f"{lines}\n"
        "- This list is a prompt for hypotheses, not a set of answers. These names came from a "
        "different set of games than the one you are playing, and the games here were built not to "
        "overlap with those. A motif that fits tells you what to probe first; it never tells you "
        "what the rules are, and a game may be none of these or two of them at once.\n"
        "- From Python, `motif(\"sokoban\")` returns what to look for, which probe distinguishes it "
        "from its neighbours, and what winning usually looks like. `motifs()` lists the names. "
        "Fetch one when you have a real candidate -- naming the shape early is worth an action, "
        "and reading all of them is not.\n"
        "- The suggested action mappings are the ones seen most often, never a rule. Action "
        "numbering differs per game by design; treat them as what to try first.\n"
    )
