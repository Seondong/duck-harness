"""Falsifiable framing hypotheses, owned by the harness rather than the model.

The seven-slot note is rewritten by the model every turn, which is why what it
holds decays. A hypothesis is the opposite: the harness writes it once, and the
model may only kill it or confirm it, always with evidence. A frame that cannot
die is not a hypothesis, it is a belief, and a wrong belief held confidently
costs actions -- which the score squares.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

MAX_ALIVE = 3

_CATEGORIES = {
    "physics", "pattern", "geometry", "color", "object",
    "spatial", "logic", "construct", "other",
}


@dataclass
class Hypothesis:
    name: str
    because: str = ""
    predicts: str = ""
    killed_if: str = ""
    confidence: float = 0.0
    status: str = "alive"          # alive | dead
    born_step: int = 0
    died_step: int | None = None
    evidence: list[str] = field(default_factory=list)

    def as_line(self) -> str:
        mark = "alive" if self.status == "alive" else "dead "
        body = self.predicts if self.status == "alive" else (self.evidence[-1] if self.evidence else "")
        return f"[H {mark}] {self.name} -- {body}".strip()

    def as_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "because": self.because,
            "predicts": self.predicts,
            "killed_if": self.killed_if,
            "confidence": self.confidence,
            "status": self.status,
            "evidence": list(self.evidence),
        }


class HypothesisSet:
    """All hypotheses this game has held, alive and dead."""

    def __init__(self) -> None:
        self._items: list[Hypothesis] = []
        self.category: str = ""
        self.goal_hypothesis: str = ""

    def __len__(self) -> int:
        return len(self._items)

    @property
    def alive(self) -> list[Hypothesis]:
        return [h for h in self._items if h.status == "alive"]

    @property
    def dead(self) -> list[Hypothesis]:
        return [h for h in self._items if h.status != "alive"]

    def dead_names(self) -> list[str]:
        return [h.name for h in self.dead]

    def reset(self) -> None:
        self._items = []
        self.category = ""
        self.goal_hypothesis = ""

    def adopt(
        self,
        hypotheses: list[Hypothesis],
        *,
        category: str = "",
        goal_hypothesis: str = "",
        step: int = 0,
    ) -> list[Hypothesis]:
        """Take a fresh batch. Names already refuted are dropped, not revived."""
        refuted = {name.lower() for name in self.dead_names()}
        live = {h.name.lower() for h in self.alive}
        added: list[Hypothesis] = []
        for item in hypotheses:
            key = item.name.lower()
            if not key or key in refuted or key in live:
                continue
            item.born_step = step
            self._items.append(item)
            added.append(item)
            if len(self.alive) >= MAX_ALIVE:
                break
        if category:
            self.category = category
        if goal_hypothesis:
            self.goal_hypothesis = goal_hypothesis
        return added

    def _find(self, name: str) -> Hypothesis | None:
        key = str(name).strip().lower()
        if not key:
            return None
        for item in self._items:
            if item.name.lower() == key:
                return item
        for item in self._items:
            if key in item.name.lower() or item.name.lower() in key:
                return item
        return None

    def kill(self, name: str, evidence: str, *, step: int = 0) -> Hypothesis | None:
        item = self._find(name)
        if item is None or item.status != "alive":
            return None
        item.status = "dead"
        item.died_step = step
        item.evidence.append(str(evidence).strip())
        return item

    def confirm(self, name: str, evidence: str) -> Hypothesis | None:
        item = self._find(name)
        if item is None or item.status != "alive":
            return None
        item.evidence.append(str(evidence).strip())
        return item

    def apply_verdicts(self, verdicts: list[dict[str, Any]], *, step: int = 0) -> list[str]:
        """Returns note lines for whatever actually changed."""
        lines: list[str] = []
        for verdict in verdicts or []:
            if not isinstance(verdict, dict):
                continue
            name = str(verdict.get("name", "")).strip()
            evidence = str(verdict.get("evidence", "")).strip()
            kind = str(verdict.get("verdict", "")).strip().lower()
            if not name or not evidence:
                continue
            if kind == "dead":
                item = self.kill(name, evidence, step=step)
                if item is not None:
                    lines.append(f"[H dead ] {item.name} -- {evidence}")
            elif kind == "alive":
                item = self.confirm(name, evidence)
                if item is not None:
                    lines.append(f"[H holds] {item.name} -- {evidence}")
        return lines

    def payload(self) -> list[dict[str, Any]]:
        return [h.as_payload() for h in self._items if h.status == "alive"]

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        if self.category:
            lines.append(f"Framing category: {self.category}")
        if self.goal_hypothesis:
            lines.append(f"Goal hypothesis: {self.goal_hypothesis}")
        for item in self.alive:
            lines.append(f"  alive: {item.name} -- predicts {item.predicts}; dies if {item.killed_if}")
        for item in self.dead[-3:]:
            last = item.evidence[-1] if item.evidence else ""
            lines.append(f"  refuted: {item.name} -- {last}")
        return lines


def _strip_fences(text: str) -> str:
    body = str(text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", body, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return body


def _coerce_confidence(value: Any) -> float:
    """The model answers `0.85` when asked, and `"High"` when it forgets."""
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    word = str(value or "").strip().lower()
    return {"high": 0.85, "medium": 0.6, "med": 0.6, "low": 0.35}.get(word, 0.5)


def parse_framing(text: str) -> tuple[str, str, list[Hypothesis]]:
    """Best-effort read of the framing turn. Never raises."""
    body = _strip_fences(text)
    payload: Any = None
    try:
        payload = json.loads(body)
    except Exception:
        start, end = body.find("{"), body.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(body[start:end + 1])
            except Exception:
                payload = None
    if not isinstance(payload, dict):
        return "", "", []

    category = str(payload.get("category", "") or "").strip().lower()
    if category not in _CATEGORIES:
        category = ""
    goal = str(payload.get("goal_hypothesis", "") or "").strip()

    parsed: list[Hypothesis] = []
    for raw in payload.get("hypotheses") or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "") or "").strip()
        killed_if = str(raw.get("killed_if", "") or "").strip()
        if not name or not killed_if:
            # Without a kill condition it is not falsifiable, so it is not one
            # of these.
            continue
        parsed.append(
            Hypothesis(
                name=name[:80],
                because=str(raw.get("because", "") or "").strip()[:400],
                predicts=str(raw.get("predicts", "") or "").strip()[:400],
                killed_if=killed_if[:400],
                confidence=_coerce_confidence(raw.get("confidence")),
            )
        )
    return category, goal, parsed[:MAX_ALIVE]
