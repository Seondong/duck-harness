"""The time policy, exercised through the real play loop and a real engine.

The framing layer shipped a level-entry bug that every unit test passed,
because nothing ever ran it inside ``_HarnessGameSession.play()``. So this
drives the genuine loop -- real ``arc_agi`` engine, real action execution, real
``should_stop`` polling, real teardown -- and swaps out only the model, which
is the one part that needs a GPU. A stub analyzer that presses buttons is
enough to prove the budget, the stop reason and the ``[level]`` line are wired
to the code path a submission actually takes.

Skipped when the competition environment files are not on this machine.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

ENV_DIR = Path(os.environ.get("ARC3_TEST_ENVIRONMENTS_DIR", "/Users/sundong/Documents/arc-agi-3/environment_files"))

pytestmark = pytest.mark.skipif(
    not ENV_DIR.is_dir(), reason=f"no offline environments at {ENV_DIR}"
)


class _ButtonMasher:
    """Stands in for the model: takes the first valid action, every turn."""

    def __init__(self) -> None:
        self.generated_tokens = 0
        self.turns = 0

    def analyze(self, state_path, action_count, *, valid_actions, step_env, **kwargs):
        from inference.agent.tool_agent import AnalyzerTurnResult

        self.turns += 1
        self.generated_tokens += 10
        # Cycle the actions rather than repeat one: a game that only ever sees
        # ACTION1 tends to sit against a wall, and the point is to move.
        name = valid_actions[self.turns % len(valid_actions)]
        result = step_env({"actions": [{"action": name}]})
        # A refusal here is usually the budget expiring between the loop's
        # should_stop() check and this call, which is the normal way a turn
        # ends. Report it rather than raising, so the note records the stop
        # reason instead of an exception from the stand-in model.
        return AnalyzerTurnResult(step_executed=bool(result.get("executed")))


def _session(game, solver, tmp_path):
    from inference.framework.solver import _HarnessGameSession

    return _HarnessGameSession(
        solver=solver,
        game=game,
        analyzer=_ButtonMasher(),
        game_index=0,
        pass_index=0,
        state_path=tmp_path / "state.json",
        transcript_path=tmp_path / "transcript.txt",
        analysis_html_relpath="analysis.html",
        stop_event=threading.Event(),
        viewer_data_path=tmp_path / "viewer.json",
    )


def _play_one(tmp_path, **solver_kwargs):
    import arc_agi

    import taaf.game
    import taaf.game_api
    from inference.framework.solver import HarnessSolver

    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR)
    )
    game_ids = [info.game_id for info in arcade.available_environments]
    assert game_ids, "offline arcade exposed no environments"

    spec = taaf.game_api.ArcadeSpec(
        operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR)
    )
    game = taaf.game_api.GameAPI(env_name=game_ids[0], arcade_spec=spec)
    game.start_game(taaf.game.RunSession(record_intermediate_states=False))

    solver = HarnessSolver(**solver_kwargs)
    solver.setup()
    session = _session(game, solver, tmp_path)
    session.play()
    return session, game.game_run


def test_the_loop_stops_on_the_base_budget_and_says_so(tmp_path):
    session, run = _play_one(
        tmp_path, max_runtime_s_per_game=3.0, level_extension_s=0.0
    )
    assert session.stop_reason in {"time_limit", "won", "finished"}
    assert run.solver_note is not None
    assert run.solver_note.startswith("stop=")
    assert "elapsed=" in run.solver_note
    # The score is computed even though the clock, not the game, ended it.
    assert run.final_score is not None


def test_the_global_guard_stops_the_loop_and_is_named_in_the_note(tmp_path):
    session, run = _play_one(
        tmp_path,
        max_runtime_s_per_game=600.0,
        level_extension_s=0.0,
        global_runtime_s=2.0,
    )
    assert session.stop_reason in {"global_deadline", "won", "finished"}
    assert "stop=" in (run.solver_note or "")


def test_an_action_limit_still_ends_the_run_cleanly(tmp_path):
    session, run = _play_one(
        tmp_path, max_runtime_s_per_game=600.0, max_actions_per_game=5
    )
    assert session.stop_reason in {"action_limit", "won", "finished"}
    assert run.final_score is not None


class _WonALevel:
    """The engine's state, with ``just_won_level`` forced on."""

    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped

    def __getattr__(self, name):
        if name == "just_won_level":
            return True
        return getattr(self._wrapped, name)


def test_clearing_a_level_extends_the_budget_through_the_real_action_path(tmp_path, capsys):
    """The wiring, not the arithmetic.

    ``budget_s`` is unit-tested above; what this pins down is that a real
    ``_execute_action`` on a real engine reaches ``_record_level_cleared``. The
    framing layer's shipped bug was exactly this kind of gap -- correct pieces,
    never run together.
    """
    import arc_agi

    import taaf.game
    import taaf.game_api
    from inference.framework.solver import HarnessSolver

    spec = taaf.game_api.ArcadeSpec(
        operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR)
    )
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR)
    )
    game_id = [info.game_id for info in arcade.available_environments][0]
    game = taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec)
    game.start_game(taaf.game.RunSession(record_intermediate_states=False))

    solver = HarnessSolver(
        max_runtime_s_per_game=600.0, level_extension_s=900.0, max_runtime_ceiling_s_per_game=3600.0
    )
    solver.setup()
    session = _session(game, solver, tmp_path)
    session.seed_initial_history()

    assert session.budget_s() == 600.0
    assert session.last_level_at is None

    real_execute = game.execute_action
    game.execute_action = lambda *a, **k: _WonALevel(real_execute(*a, **k))

    import arcengine

    session._execute_action(
        arcengine.ActionInput(id=arcengine.GameAction.ACTION1, data={}),
        batch_index=1,
        batch_size=1,
        generated_tokens=0,
    )

    assert session.last_level_at is not None
    assert session.budget_s() > 600.0
    assert "[level]" in capsys.readouterr().out


def test_the_solver_survives_a_pickle_round_trip_with_the_new_fields(tmp_path):
    """The deployed bundle is a pickle; a field that does not survive it is a
    setting that silently reverts on Kaggle."""
    import pickle

    from inference.framework.solver import HarnessSolver

    solver = HarnessSolver(
        max_runtime_s_per_game=7920.0,
        level_extension_s=2700.0,
        max_runtime_ceiling_s_per_game=18000.0,
        global_runtime_s=25200.0,
    )
    restored = pickle.loads(pickle.dumps(solver))
    assert restored.max_runtime_s_per_game == 7920.0
    assert restored.level_extension_s == 2700.0
    assert restored.max_runtime_ceiling_s_per_game == 18000.0
    assert restored.global_runtime_s == 25200.0
    # Stamped at setup(), never carried across the pickle.
    assert restored.global_deadline_monotonic() is None
    restored.setup()
    assert restored.global_deadline_monotonic() is not None
