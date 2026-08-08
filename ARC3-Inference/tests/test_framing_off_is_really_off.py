"""FRAMING_ENABLED=0 has to remove every trace, not just the side-call.

The 08-04 submission set FRAMING_ENABLED=0 and still carried three prompt lines
telling the model to spend its actions deciding "live hypotheses", plus a
sandbox `hypotheses` list that was permanently empty and kill/confirm functions
nobody could ever usefully call. Those lines were in the 0.86 run and the 0.78
run and in neither of the 1.10/1.16 runs, which is the one thing that separates
the two pairs.

A flag that only half-disables a feature makes every A/B built on it a lie, so
these tests assert the whole surface goes away.
"""
from __future__ import annotations

import importlib

import pytest


def _reloaded(monkeypatch, enabled: str):
    """`_FRAMING_ENABLED` is read at import, so the module has to come back."""
    monkeypatch.setenv("FRAMING_ENABLED", enabled)
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    from inference.agent import tool_agent

    return importlib.reload(tool_agent)


@pytest.fixture(autouse=True)
def _restore_module():
    yield
    from inference.agent import tool_agent

    importlib.reload(tool_agent)


# The base prompt uses the word "hypotheses" in an unrelated sentence about
# segmenting the board, so match the framing block's own wording instead.
FRAMING_MARKERS = ("kill(name, evidence)", "confirm(name, evidence)", "live hypothesis")


def test_the_system_prompt_says_nothing_about_hypotheses_when_framing_is_off(monkeypatch):
    module = _reloaded(monkeypatch, "0")
    prompt = module._build_system_prompt(tool_output_tokens=1024)
    for marker in FRAMING_MARKERS:
        assert marker not in prompt, marker


def test_the_system_prompt_documents_hypotheses_when_framing_is_on(monkeypatch):
    module = _reloaded(monkeypatch, "1")
    prompt = module._build_system_prompt(tool_output_tokens=1024)
    for marker in FRAMING_MARKERS:
        assert marker in prompt, marker


def test_the_notes_guidance_survives_either_way(monkeypatch):
    """The PRO-LONG notes layer is not part of framing and must not move."""
    for flag in ("0", "1"):
        module = _reloaded(monkeypatch, flag)
        prompt = module._build_system_prompt(tool_output_tokens=1024)
        assert "notes" in prompt, flag


def _sandbox(code: str, state: dict):
    from inference.agent import python_tool_sandbox

    return python_tool_sandbox.run_sandboxed_python(
        code=code,
        timeout_seconds=20,
        initial_state=state,
        action_handler=lambda actions: {"error": "no actions in this test"},
    )


def test_the_sandbox_hides_kill_and_confirm_when_the_state_omits_hypotheses():
    result = _sandbox(
        "g = dir()\nprint([n for n in ('hypotheses', 'kill', 'confirm') if n in g])",
        {"notes": []},
    )
    assert "[]" in str(result.get("stdout", "")), result


def test_the_sandbox_offers_them_when_the_state_carries_hypotheses():
    result = _sandbox(
        "kill('gravity', 'block rose at (3,4) on step 7')\nprint(len(hypotheses))",
        {
            "notes": [],
            "hypotheses": [{"name": "gravity", "predicts": "p", "killed_if": "k"}],
        },
    )
    assert "1" in str(result.get("stdout", "")), result
    assert any(v.get("name") == "gravity" for v in result.get("verdicts", [])), result
