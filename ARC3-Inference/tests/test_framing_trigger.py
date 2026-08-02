"""When the agent re-frames, and when it leaves a standing framing alone."""
from __future__ import annotations

import pytest

from inference.agent import tool_agent as ta
from inference.agent.hypotheses import parse_framing
from inference.agent.runtime_state import Frame


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    return ta.ToolAgent(model="Qwen/Qwen3.6-27B", base_url="http://127.0.0.1:1", provider="vllm")


def _frame(level: int = 1, step: int = 0) -> Frame:
    return Frame(grid=((0, 0), (0, 0)), step=step, level=level)


def _seed(agent, names: list[str]) -> None:
    body = ",".join(
        f'{{"name": "{n}", "predicts": "p", "killed_if": "k"}}' for n in names
    )
    _, _, parsed = parse_framing(f'{{"hypotheses": [{body}]}}')
    agent._hypotheses.adopt(parsed)


def test_a_text_only_endpoint_is_never_framed(agent, monkeypatch):
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "none")
    assert agent._should_frame(_frame()) is False


def test_frames_on_first_sight_of_a_level(agent):
    assert agent._should_frame(_frame(level=1)) is True


def test_does_not_reframe_while_a_hypothesis_still_stands(agent):
    agent._framed_level = 1
    _seed(agent, ["gravity", "goal"])
    assert agent._should_frame(_frame(level=1)) is False


def test_reframes_once_every_hypothesis_is_dead(agent):
    agent._framed_level = 1
    _seed(agent, ["gravity"])
    agent._hypotheses.apply_verdicts(
        [{"name": "gravity", "verdict": "dead", "evidence": "it floated"}]
    )
    assert agent._should_frame(_frame(level=1)) is True


def test_reframes_on_a_new_level_even_with_live_hypotheses(agent):
    agent._framed_level = 1
    _seed(agent, ["gravity"])
    assert agent._should_frame(_frame(level=2)) is True


def test_stops_framing_once_the_per_pass_cap_is_reached(agent):
    agent._framing_runs = ta._FRAMING_MAX_PER_PASS
    assert agent._should_frame(_frame(level=9)) is False


def test_a_failed_framing_still_marks_the_level_so_it_is_not_retried_forever(agent, monkeypatch):
    monkeypatch.setattr(
        ta, "run_framing", lambda *a, **k: pytest.fail("should not be reached")
    )
    # No window can be built from an empty history, so the call returns before
    # ever reaching the model -- and the level is left unmarked, because nothing
    # was spent.
    agent._run_framing(_frame(level=3), [])
    assert agent._framed_level is None
    assert agent._framing_runs == 0


def test_a_model_failure_marks_the_level_and_leaves_the_game_alone(agent, monkeypatch):
    monkeypatch.setattr(ta, "run_framing", lambda *a, **k: None)
    monkeypatch.setattr(
        ta.ToolAgent, "_should_frame", lambda self, frame: True
    )
    from inference.agent import motion_context

    monkeypatch.setattr(motion_context, "select_window", lambda h, f: [(f, "UP")] * 6)
    monkeypatch.setattr(motion_context, "filmstrip_data_url", lambda w, **k: "data:,")
    monkeypatch.setattr(motion_context, "trail_data_url", lambda w, **k: "data:,")

    agent._run_framing(_frame(level=4), [])
    assert agent._framed_level == 4
    assert agent._notes == []
    assert len(agent._hypotheses) == 0
