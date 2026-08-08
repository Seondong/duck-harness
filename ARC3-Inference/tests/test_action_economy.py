"""The two things the prompt never told the model about its own cost.

RHAE is (H/A)^2 per cleared level with a hard cutoff at 5H, and the paper is
explicit that "internal operations that do not alter the environment, such as
tool calls, reasoning steps, or retries within the model itself, are not
counted as actions". The prompt carried neither fact -- only the advice
"optimize for as few in-game actions as possible", which does not say that the
penalty is quadratic, that overshooting forfeits every later level, or that
thinking is unmetered.

`actions_this_level` is the other half: the score is computed per level, and
`current_frame.step` counts the whole game, so nothing in the sandbox said how
much of the level's budget was already gone.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _restore_module():
    yield
    from inference.agent import tool_agent

    importlib.reload(tool_agent)


def _reloaded(monkeypatch, **env):
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    monkeypatch.setenv("FRAMING_ENABLED", "0")
    monkeypatch.setenv("MOTIFS_ENABLED", "0")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    from inference.agent import tool_agent

    return importlib.reload(tool_agent)


class TestPrompt:
    def test_the_quadratic_penalty_and_the_cutoff_are_both_stated(self, monkeypatch):
        prompt = _reloaded(monkeypatch, ACTION_ECONOMY="1")._build_system_prompt(
            tool_output_tokens=1024
        )
        assert "squared" in prompt
        assert "five times H" in prompt
        assert "25%" in prompt  # the concrete cost of being twice as slow

    def test_it_says_thinking_is_not_metered(self, monkeypatch):
        prompt = _reloaded(monkeypatch, ACTION_ECONOMY="1")._build_system_prompt(
            tool_output_tokens=1024
        )
        assert "NOT actions" in prompt
        assert "cost your score" in prompt

    def test_it_also_says_a_plan_never_executed_scores_nothing(self, monkeypatch):
        """The counterweight. Told only that thinking is free, an agent can
        reason until the wall clock ends the game having taken no actions."""
        prompt = _reloaded(monkeypatch, ACTION_ECONOMY="1")._build_system_prompt(
            tool_output_tokens=1024
        )
        assert "never execute also scores zero" in prompt and "the clock" in prompt

    def test_reset_is_named_as_an_action(self, monkeypatch):
        """sum(actions_per_level) == len(history): RESET is charged like the rest."""
        prompt = _reloaded(monkeypatch, ACTION_ECONOMY="1")._build_system_prompt(
            tool_output_tokens=1024
        )
        assert "RESET and undo are actions" in prompt

    def test_the_flag_removes_all_of_it(self, monkeypatch):
        off = _reloaded(monkeypatch, ACTION_ECONOMY="0")._build_system_prompt(
            tool_output_tokens=1024
        )
        assert "What an action costs" not in off

    def test_the_block_is_small_against_a_32k_window(self, monkeypatch):
        on = _reloaded(monkeypatch, ACTION_ECONOMY="1")._build_system_prompt(tool_output_tokens=1024)
        off = _reloaded(monkeypatch, ACTION_ECONOMY="0")._build_system_prompt(tool_output_tokens=1024)
        assert 0 < len(on) - len(off) < 2500


def _sandbox(code: str, state: dict):
    from inference.agent import python_tool_sandbox

    return python_tool_sandbox.run_sandboxed_python(
        code=code,
        timeout_seconds=20,
        initial_state=state,
        action_handler=lambda actions: {"error": "no actions in this test"},
    )


def _frame(step: int, level: int) -> dict:
    return {"ascii": "", "step": step, "level": level, "shape": [2, 2], "grid": [[0, 0], [0, 0]]}


class TestActionsThisLevel:
    def test_it_counts_only_the_current_level(self):
        state = {
            "current_frame": _frame(7, 2),
            "history": [
                {"action": "", "frame": _frame(0, 1)},
                {"action": "UP", "frame": _frame(1, 1)},
                {"action": "UP", "frame": _frame(2, 1)},
                {"action": "UP", "frame": _frame(3, 2)},
                {"action": "LEFT", "frame": _frame(4, 2)},
            ],
            "valid_actions": ["UP"],
            "notes": [],
            "action_economy": True,
        }
        result = _sandbox("print(actions_this_level)", state)
        assert "2" in str(result.get("stdout", "")), result

    def test_the_seed_entry_with_no_action_is_not_counted(self):
        state = {
            "current_frame": _frame(0, 1),
            "history": [{"action": "", "frame": _frame(0, 1)}],
            "valid_actions": ["UP"],
            "notes": [],
            "action_economy": True,
        }
        result = _sandbox("print(actions_this_level)", state)
        assert str(result.get("stdout", "")).strip() == "0", result


class TestRememberedCode:
    def test_the_helpers_are_absent_unless_the_parent_enables_them(self):
        result = _sandbox(
            "g = dir()\nprint([n for n in ('remember', 'forget', 'remembered') if n in g])",
            {"notes": [], "current_frame": _frame(0, 1), "history": []},
        )
        assert "[]" in str(result.get("stdout", "")), result

    def test_a_stored_function_is_usable_immediately_and_reported_up(self):
        result = _sandbox(
            "remember('dyn', 'def step(x):\\n    return x + 1')\nprint(step(41))",
            {"notes": [], "current_frame": _frame(0, 1), "history": [], "remembered_code": {}},
        )
        assert "42" in str(result.get("stdout", "")), result
        assert "dyn" in result.get("remembered_edits", {}), result

    def test_a_stored_function_survives_into_the_next_call(self):
        """The whole point: a fresh subprocess that still has last turn's model."""
        first = _sandbox(
            "remember('dyn', 'def step(x):\\n    return x * 2')",
            {"notes": [], "current_frame": _frame(0, 1), "history": [], "remembered_code": {}},
        )
        carried = {k: v for k, v in first["remembered_edits"].items() if v is not None}
        second = _sandbox(
            "print(step(21))",
            {"notes": [], "current_frame": _frame(1, 1), "history": [], "remembered_code": carried},
        )
        assert "42" in str(second.get("stdout", "")), second

    def test_forgetting_removes_it(self):
        result = _sandbox(
            "forget('dyn')",
            {
                "notes": [],
                "current_frame": _frame(0, 1),
                "history": [],
                "remembered_code": {"dyn": "def step(x):\n    return x"},
            },
        )
        assert result["remembered_edits"] == {"dyn": None}, result

    def test_a_broken_stored_source_is_reported_and_does_not_kill_the_call(self):
        """A bad world model must not brick every later turn."""
        result = _sandbox(
            "print(len(remembered_errors))\nprint('still running')",
            {
                "notes": [],
                "current_frame": _frame(0, 1),
                "history": [],
                "remembered_code": {"bad": "this is not python ("},
            },
        )
        stdout = str(result.get("stdout", ""))
        assert "1" in stdout and "still running" in stdout, result
        assert not result.get("error"), result


class TestPersistCodePromptGating:
    """The FRAMING_ENABLED lesson: a flag has to remove the documentation too,
    or the prompt advertises a mechanism the runtime does not provide."""

    def test_off_says_nothing_about_remember(self, monkeypatch):
        prompt = _reloaded(monkeypatch, PERSIST_CODE="0")._build_system_prompt(
            tool_output_tokens=1024
        )
        assert "remember(" not in prompt
        assert "Code that outlives the turn" not in prompt

    def test_on_documents_it(self, monkeypatch):
        prompt = _reloaded(monkeypatch, PERSIST_CODE="1")._build_system_prompt(
            tool_output_tokens=1024
        )
        assert "remember(name, source)" in prompt
        assert "predicted next state" in prompt


class TestPersistenceThroughTheAgent:
    """Not the sandbox in isolation -- the parent path that has to carry the
    store between two fresh subprocesses. The framing layer shipped a bug that
    every unit test passed because nothing exercised the seam."""

    def _agent(self, monkeypatch):
        module = _reloaded(monkeypatch, PERSIST_CODE="1")
        return module.ToolAgent(
            model="Qwen/Qwen3.6-27B", base_url="http://127.0.0.1:1", provider="vllm"
        )

    def _state_file(self, tmp_path):
        from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state

        path = tmp_path / "state.json"
        frame = Frame(grid=((0, 0), (0, 0)), step=1, level=1)
        write_runtime_state(
            path,
            current_frame=frame,
            history=[HistoryEntry(action="", frame=frame), HistoryEntry(action="UP", frame=frame)],
        )
        return path

    def test_a_world_model_written_one_turn_is_callable_the_next(self, monkeypatch, tmp_path):
        agent = self._agent(monkeypatch)
        agent._current_valid_actions = ["UP"]
        path = self._state_file(tmp_path)

        agent._run_python_tool(
            path, {"code": "remember('dyn', 'def predict(x):\\n    return x + 1')"}
        )
        assert "dyn" in agent._remembered_code

        second = agent._run_python_tool(path, {"code": "print(predict(41))"})
        assert "42" in second.content, second.content

    def test_forgetting_survives_the_seam_too(self, monkeypatch, tmp_path):
        agent = self._agent(monkeypatch)
        agent._current_valid_actions = ["UP"]
        path = self._state_file(tmp_path)
        agent._run_python_tool(path, {"code": "remember('dyn', 'x = 1')"})
        agent._run_python_tool(path, {"code": "forget('dyn')"})
        assert agent._remembered_code == {}

    def test_actions_this_level_is_visible_through_the_agent(self, monkeypatch, tmp_path):
        agent = self._agent(monkeypatch)
        agent._current_valid_actions = ["UP"]
        path = self._state_file(tmp_path)
        result = agent._run_python_tool(path, {"code": "print(actions_this_level)"})
        assert "1" in result.content, result.content


def test_the_code_store_evicts_oldest_first_when_it_fills(monkeypatch, tmp_path):
    """It is re-executed at the top of every later call, so unbounded growth
    is a per-turn tax for the rest of the game."""
    module = _reloaded(monkeypatch, PERSIST_CODE="1", PERSIST_CODE_MAX_CHARS="120")
    agent = module.ToolAgent(
        model="Qwen/Qwen3.6-27B", base_url="http://127.0.0.1:1", provider="vllm"
    )
    agent._current_valid_actions = ["UP"]
    from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state

    path = tmp_path / "state.json"
    frame = Frame(grid=((0, 0), (0, 0)), step=0, level=1)
    write_runtime_state(path, current_frame=frame, history=[HistoryEntry(action="", frame=frame)])

    for index in range(4):
        agent._run_python_tool(
            path, {"code": f"remember('m{index}', 'x{index} = ' + '1' * 50)"}
        )
    assert sum(len(v) for v in agent._remembered_code.values()) <= 120
    assert "m0" not in agent._remembered_code
    assert "m3" in agent._remembered_code


class TestActionEffects:
    def _state(self):
        def grid(mark):
            g = [[0] * 8 for _ in range(8)]
            g[mark % 8][0] = 1          # the "gameplay" cell that moves
            g[0][7] = mark % 3          # a one-cell HUD tick, always changing
            return g

        def frame(step, mark):
            return {"ascii": "", "step": step, "level": 1, "shape": [8, 8], "grid": grid(mark)}

        return {
            "current_frame": frame(4, 4),
            "history": [
                {"action": "", "frame": frame(0, 0)},
                {"action": "UP", "frame": frame(1, 1)},      # moves the cell + HUD
                {"action": "LEFT", "frame": frame(2, 1)},    # HUD only
                {"action": "LEFT", "frame": frame(3, 1)},    # HUD only
                {"action": "UP", "frame": frame(4, 4)},      # moves again
            ],
            "valid_actions": ["UP", "LEFT"],
            "notes": [],
            "action_economy": True,
        }

    def test_it_separates_a_real_move_from_a_hud_tick(self):
        result = _sandbox(
            "e = action_effects()\nprint(e['UP']['median_cells'], e['LEFT']['median_cells'])",
            self._state(),
        )
        up, left = str(result.get("stdout", "")).split()
        assert int(up) > int(left), result
        assert int(left) <= 1, result

    def test_it_counts_tries_and_changes(self):
        result = _sandbox(
            "e = action_effects()\nprint(e['LEFT']['tried'], e['UP']['tried'])", self._state()
        )
        assert str(result.get("stdout", "")).strip() == "2 2", result

    def test_an_action_never_taken_is_simply_absent(self):
        result = _sandbox("print('DOWN' in action_effects())", self._state())
        assert "False" in str(result.get("stdout", "")), result

    def test_it_is_documented_only_with_the_economy_block(self, monkeypatch):
        off = _reloaded(monkeypatch, ACTION_ECONOMY="0")._build_system_prompt(tool_output_tokens=1024)
        on = _reloaded(monkeypatch, ACTION_ECONOMY="1")._build_system_prompt(tool_output_tokens=1024)
        assert "action_effects()" not in off
        assert "action_effects()" in on


class TestTheFlagLeavesNoRuntimeTrace:
    """A replication run is only a replication if the switched-off feature is
    absent from the runtime too, not just from the prompt."""

    def test_neither_global_exists_when_the_feature_is_not_declared(self):
        result = _sandbox(
            "g = dir()\nprint([n for n in ('actions_this_level', 'action_effects') if n in g])",
            {
                "current_frame": _frame(1, 1),
                "history": [{"action": "UP", "frame": _frame(1, 1)}],
                "valid_actions": ["UP"],
                "notes": [],
            },
        )
        assert "[]" in str(result.get("stdout", "")), result

    def test_both_appear_when_it_is(self):
        result = _sandbox(
            "g = dir()\nprint(sorted(n for n in ('actions_this_level', 'action_effects') if n in g))",
            {
                "current_frame": _frame(1, 1),
                "history": [{"action": "UP", "frame": _frame(1, 1)}],
                "valid_actions": ["UP"],
                "notes": [],
                "action_economy": True,
            },
        )
        assert "action_effects" in str(result.get("stdout", "")), result
        assert "actions_this_level" in str(result.get("stdout", "")), result
