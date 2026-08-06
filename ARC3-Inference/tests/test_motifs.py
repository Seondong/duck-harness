"""The motif vocabulary: cheap when on, absent when off, and honest about cost.

Two things have gone wrong on this harness before and both are pinned here. A
feature flag that only half-disabled its feature made an A/B meaningless
(FRAMING_ENABLED), and a layer that was never run inside the real loop shipped
with a bug every unit test passed. So: the flag has to remove the whole
surface, and the sandbox path has to be exercised for real rather than mocked.
"""
from __future__ import annotations

import importlib
import json

import pytest

from inference.agent import motifs


@pytest.fixture(autouse=True)
def _restore_module():
    yield
    from inference.agent import tool_agent

    importlib.reload(tool_agent)


def _reloaded(monkeypatch, enabled: str):
    monkeypatch.setenv("MOTIFS_ENABLED", enabled)
    monkeypatch.setenv("FRAMING_ENABLED", "0")
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    from inference.agent import tool_agent

    return importlib.reload(tool_agent)


class TestCatalog:
    def test_it_shipped(self):
        assert len(motifs.catalog()) == 22

    def test_every_motif_has_a_slug_and_a_one_line_summary(self):
        for entry in motifs.catalog():
            assert entry["slug"], entry
            assert entry["summary"].strip(), entry

    def test_lookup_is_forgiving_about_how_the_name_is_typed(self):
        for spelling in ("sokoban", "Sokoban", "  SOKOBAN  ", "piece_insertion", "piece insertion"):
            assert motifs.detail(spelling) is not None, spelling

    def test_an_unknown_name_returns_nothing_rather_than_guessing(self):
        assert motifs.detail("tetris-but-backwards") is None

    def test_no_public_game_is_named_anywhere_in_it(self):
        """The private set is documented as not overlapping with the public one,
        so per-game attributions are a false cue with no upside."""
        blob = json.dumps(motifs.catalog(), ensure_ascii=False).lower()
        for game in ("tu93", "sk48", "ls20", "vc33", "cd82", "ft09", "ka59", "wa30", "sb26"):
            assert game not in blob, game

    def test_it_is_written_in_the_language_the_rest_of_the_prompt_uses(self):
        blob = json.dumps(motifs.catalog(), ensure_ascii=False)
        hangul = sum(1 for ch in blob if "가" <= ch <= "힣")
        assert hangul == 0, f"{hangul} Korean characters left in the shipped catalog"


class TestPromptCost:
    def test_the_always_on_block_stays_small(self):
        """~440 tokens against a 32k window that already trims the conversation.
        Guarded in characters so the test needs no tokenizer."""
        block = motifs.summary_block()
        assert 0 < len(block) < 3000, len(block)

    def test_the_block_says_the_names_are_not_answers(self):
        block = motifs.summary_block().lower()
        assert "not a set of answers" in block
        assert "never a rule" in block

    def test_the_system_prompt_is_untouched_when_the_flag_is_off(self, monkeypatch):
        off = _reloaded(monkeypatch, "0")._build_system_prompt(tool_output_tokens=1024)
        on = _reloaded(monkeypatch, "1")._build_system_prompt(tool_output_tokens=1024)
        assert "sokoban" not in off.lower()
        assert "sokoban" in on.lower()
        assert len(on) > len(off)


class TestSandbox:
    def _run(self, code: str, state: dict):
        from inference.agent import python_tool_sandbox

        return python_tool_sandbox.run_sandboxed_python(
            code=code,
            timeout_seconds=20,
            initial_state=state,
            action_handler=lambda actions: {"error": "no actions in this test"},
        )

    def test_the_helpers_are_absent_when_no_catalog_is_passed(self):
        result = self._run("print([n for n in ('motif', 'motifs') if n in dir()])", {"notes": []})
        assert "[]" in str(result.get("stdout", "")), result

    def test_a_lookup_returns_the_entry_and_is_recorded(self):
        result = self._run(
            "d = motif('sokoban')\nprint(d['summary'])\nprint(len(d['probes']))",
            {"notes": [], "motif_catalog": motifs.catalog()},
        )
        assert "Push blocks" in str(result.get("stdout", "")), result
        assert result.get("motif_lookups") == ["sokoban"], result

    def test_listing_the_names_costs_no_lookup(self):
        result = self._run(
            "print(len(motifs()))",
            {"notes": [], "motif_catalog": motifs.catalog()},
        )
        assert "22" in str(result.get("stdout", "")), result
        assert result.get("motif_lookups") == [], result

    def test_a_wrong_name_returns_the_available_ones_and_is_still_recorded(self):
        result = self._run(
            "print(motif('sokobon')['error'])",
            {"notes": [], "motif_catalog": motifs.catalog()},
        )
        assert "no motif named" in str(result.get("stdout", "")), result
        # Recorded with a marker: a model guessing at names is worth seeing.
        assert result.get("motif_lookups") == ["sokobon?"], result

    def test_a_summary_only_motif_says_so_instead_of_returning_a_bare_stub(self):
        result = self._run(
            "print(motif('key-door').get('note', ''))",
            {"notes": [], "motif_catalog": motifs.catalog()},
        )
        assert "Summary only" in str(result.get("stdout", "")), result
