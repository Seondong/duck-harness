"""The framing layer must add hypotheses when it works and change nothing when it does not."""
from __future__ import annotations

import base64

import pytest

from inference.agent import motion_context
from inference.agent.framing import run_framing
from inference.agent.hypotheses import HypothesisSet, parse_framing
from inference.agent.runtime_state import Frame, HistoryEntry


def _frame(step: int, level: int, mark: int = 0) -> Frame:
    grid = tuple(tuple(mark if (r == step % 8 and c == 3) else 0 for c in range(8)) for r in range(8))
    return Frame(grid=grid, step=step, level=level)


def _history(levels: list[int], marks: list[int] | None = None) -> list[HistoryEntry]:
    marks = marks or [1] * len(levels)
    return [
        HistoryEntry(action=f"ACT{i}", frame=_frame(i, level, marks[i]))
        for i, level in enumerate(levels)
    ]


class TestWindowSelection:
    def test_rejects_a_window_that_crosses_a_level(self):
        history = _history([1, 1, 1, 2, 2, 2])
        assert motion_context.select_window(history, None) is None

    def test_rejects_a_window_where_nothing_moved(self):
        frames = [HistoryEntry(action="UP", frame=Frame(grid=((0, 0), (0, 0)), step=i, level=1))
                  for i in range(6)]
        assert motion_context.select_window(frames, None) is None

    def test_accepts_a_moving_single_level_window(self):
        window = motion_context.select_window(_history([1] * 6), None)
        assert window is not None and len(window) == 6

    def test_refuses_to_frame_a_new_level_from_the_previous_one(self):
        """The bug that shipped: framing fires on entering a level, and the
        only footage in hand at that moment belongs to the level just left.
        A single-level window is not enough -- it has to be *this* level."""
        history = _history([1] * 10 + [2, 2])
        current = _frame(step=12, level=2, mark=1)
        assert motion_context.select_window(history, current) is None

    def test_frames_the_new_level_once_it_has_enough_frames_of_its_own(self):
        history = _history([1] * 6 + [2] * 6)
        current = _frame(step=12, level=2, mark=1)
        window = motion_context.select_window(history, current)
        assert window is not None
        assert {frame.level for frame, _ in window} == {2}

    def test_tolerates_one_dead_transition_but_not_two(self):
        def frames(steps: list[int]) -> list[HistoryEntry]:
            return [HistoryEntry(action="UP", frame=_frame(step, 1, 1)) for step in steps]

        # steps 0..5 all differ; repeating one step repeats its board
        assert motion_context.select_window(frames([0, 1, 1, 2, 3, 4]), None) is not None
        assert motion_context.select_window(frames([0, 1, 1, 2, 2, 3]), None) is None

    def test_prefers_the_window_where_more_actually_happened(self):
        """A HUD tick keeps a still board technically alive; pick the busy window."""
        def quiet(step: int) -> Frame:
            grid = [[0] * 8 for _ in range(8)]
            grid[0][step % 8] = 5          # a one-cell timer, ticking
            return Frame(grid=tuple(tuple(r) for r in grid), step=step, level=1)

        def busy(step: int) -> Frame:
            grid = [[0] * 8 for _ in range(8)]
            grid[0][step % 8] = 5
            for c in range(6):             # a big object sliding
                grid[4][(c + step) % 8] = 9
            return Frame(grid=tuple(tuple(r) for r in grid), step=step, level=1)

        history = [HistoryEntry(action="UP", frame=busy(i)) for i in range(6)]
        history += [HistoryEntry(action="UP", frame=quiet(i)) for i in range(6, 12)]
        window = motion_context.select_window(history, None)
        assert window is not None
        # the busy stretch is the older one, and should still win
        moving = sum(1 for frame, _ in window if any(9 in row for row in frame.grid))
        assert moving >= 4

    def test_never_spans_a_reset(self):
        history = _history([1] * 6)
        history[3] = HistoryEntry(action="RESET", frame=history[3].frame)
        assert motion_context.select_window(history, None) is None

    def test_needs_a_full_window(self):
        assert motion_context.select_window(_history([1, 1, 1]), None) is None


class TestRendering:
    def test_both_images_are_png_data_urls(self):
        window = motion_context.select_window(_history([1] * 6), None)
        for url in (motion_context.filmstrip_data_url(window), motion_context.trail_data_url(window)):
            assert url.startswith("data:image/png;base64,")
            base64.b64decode(url.split(",", 1)[1])

    def test_the_trail_is_drawn_larger_than_one_strip_panel(self):
        # A single 64x64 panel at the strip's upscale lands under the 256x256
        # floor the processor rescales from, so it is drawn at its own scale.
        assert motion_context.trail_upscale() > motion_context.strip_upscale()

    def test_mouse_labels_are_shortened_to_fit_under_a_panel(self):
        assert motion_context.short_action("MOUSE(row=40, col=22)") == "M(40,22)"
        assert motion_context.short_action("LEFT") == "LEFT"


class TestParsing:
    def test_reads_a_fenced_json_block(self):
        text = """```json
        {"category": "spatial", "goal_hypothesis": "fill the slot",
         "hypotheses": [{"name": "rot", "predicts": "p", "killed_if": "k", "confidence": 0.9}]}
        ```"""
        category, goal, hypotheses = parse_framing(text)
        assert (category, goal) == ("spatial", "fill the slot")
        assert hypotheses[0].confidence == pytest.approx(0.9)

    def test_a_worded_confidence_becomes_a_number(self):
        text = '{"hypotheses": [{"name": "a", "killed_if": "k", "confidence": "High"}]}'
        _, _, hypotheses = parse_framing(text)
        assert 0.0 < hypotheses[0].confidence <= 1.0

    def test_drops_a_hypothesis_that_cannot_be_refuted(self):
        text = '{"hypotheses": [{"name": "vibes", "predicts": "good things"}]}'
        assert parse_framing(text)[2] == []

    def test_garbage_yields_nothing_rather_than_raising(self):
        assert parse_framing("the model rambled and never emitted json") == ("", "", [])


class TestHypothesisSet:
    def _seeded(self) -> HypothesisSet:
        hs = HypothesisSet()
        _, _, parsed = parse_framing(
            '{"hypotheses": ['
            '{"name": "gravity", "predicts": "blocks fall", "killed_if": "one floats"},'
            '{"name": "goal", "predicts": "slot fills", "killed_if": "nothing happens"}]}'
        )
        hs.adopt(parsed, category="physics", goal_hypothesis="fill it")
        return hs

    def test_a_kill_moves_it_out_of_alive_and_records_why(self):
        hs = self._seeded()
        lines = hs.apply_verdicts([{"name": "gravity", "verdict": "dead", "evidence": "(12,7) floated"}])
        assert [h.name for h in hs.alive] == ["goal"]
        assert hs.dead[0].evidence == ["(12,7) floated"]
        assert lines and "gravity" in lines[0]

    def test_a_refuted_name_is_not_readopted(self):
        hs = self._seeded()
        hs.apply_verdicts([{"name": "gravity", "verdict": "dead", "evidence": "it floated"}])
        _, _, again = parse_framing(
            '{"hypotheses": [{"name": "gravity", "predicts": "x", "killed_if": "y"}]}'
        )
        assert hs.adopt(again) == []

    def test_a_verdict_without_evidence_is_ignored(self):
        hs = self._seeded()
        hs.apply_verdicts([{"name": "gravity", "verdict": "dead", "evidence": ""}])
        assert [h.name for h in hs.alive] == ["gravity", "goal"]

    def test_confirming_keeps_it_alive_and_adds_evidence(self):
        hs = self._seeded()
        hs.apply_verdicts([{"name": "gravity", "verdict": "alive", "evidence": "3 blocks fell"}])
        alive = {h.name: h for h in hs.alive}
        assert alive["gravity"].evidence == ["3 blocks fell"]


class TestFailsClosed:
    """Whatever breaks, the caller gets None and the game is unaffected."""

    def test_a_dead_endpoint_returns_none(self):
        def chat(messages, budget):
            raise RuntimeError("connection refused")

        assert run_framing(chat, filmstrip_url="data:,", trail_url=None,
                           valid_actions=["UP"], notes_tail=[], dead_names=[]) is None

    def test_an_empty_first_turn_returns_none(self):
        assert run_framing(lambda m, b: "", filmstrip_url="data:,", trail_url=None,
                           valid_actions=[], notes_tail=[], dead_names=[]) is None

    def test_unparseable_json_returns_none(self):
        assert run_framing(lambda m, b: "no json here", filmstrip_url="data:,", trail_url=None,
                           valid_actions=[], notes_tail=[], dead_names=[]) is None

    def test_a_good_run_carries_the_image_and_returns_hypotheses(self):
        seen: list[int] = []

        def chat(messages, budget):
            seen.append(sum(
                1 for m in messages
                if isinstance(m.get("content"), list)
                for part in m["content"] if part.get("type") == "image_url"
            ))
            return ('{"category": "physics", "goal_hypothesis": "g",'
                    ' "hypotheses": [{"name": "a", "predicts": "p", "killed_if": "k",'
                    ' "confidence": 0.8}]}')

        result = run_framing(chat, filmstrip_url="data:image/png;base64,AA",
                             trail_url="data:image/png;base64,BB",
                             valid_actions=["UP"], notes_tail=["saw a thing"], dead_names=["old"])
        assert result is not None
        assert result.category == "physics"
        assert [h.name for h in result.hypotheses] == ["a"]
        # filmstrip on the first turn, trail added on the second, both still
        # there on the third
        assert seen == [1, 2, 2]

    def test_the_refuted_list_reaches_the_model(self):
        captured: list[str] = []

        def chat(messages, budget):
            captured.append(str(messages[-1].get("content")))
            return '{"hypotheses": [{"name": "b", "killed_if": "k"}]}'

        run_framing(chat, filmstrip_url="data:,", trail_url=None, valid_actions=[],
                    notes_tail=[], dead_names=["already-dead"])
        assert any("already-dead" in text for text in captured)
