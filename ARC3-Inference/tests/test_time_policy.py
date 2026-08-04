"""When a game is allowed to keep playing, and when its clock runs out.

The policy under test: every game gets a base budget, and a game that cleared a
level *recently* gets until that clear plus an extension, capped by a ceiling
and by a whole-run guard. The rules that matter are the ones that keep it from
being a free-for-all -- a game that cleared a level long ago must die on the
base budget like it always did, or the extension stops being an addition and
starts being a redistribution of GPU nobody measured.
"""
from __future__ import annotations

import pytest

from inference.framework.solver import HarnessSolver, _HarnessGameSession

BASE = 600.0
EXTENSION = 300.0
CEILING = 2000.0


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr("inference.framework.solver.time.monotonic", fake)
    return fake


def _session(
    clock,
    *,
    base: float | None = BASE,
    extension: float = EXTENSION,
    ceiling: float | None = CEILING,
    global_runtime: float | None = None,
) -> _HarnessGameSession:
    solver = HarnessSolver(
        max_runtime_s_per_game=base,
        level_extension_s=extension,
        max_runtime_ceiling_s_per_game=ceiling,
        global_runtime_s=global_runtime,
    )
    if global_runtime is not None:
        solver.setup()
    session = _HarnessGameSession.__new__(_HarnessGameSession)
    session.solver = solver
    session.started_at = clock.now
    session.last_level_at = None
    session.levels_cleared = 0
    session.stop_reason = ""
    return session


def test_a_game_that_never_clears_anything_gets_exactly_the_base(clock):
    session = _session(clock)
    assert session.budget_s() == BASE
    clock.advance(BASE - 1)
    assert session.runtime_limit_reached() is False
    clock.advance(2)
    assert session.runtime_limit_reached() is True


def test_clearing_a_level_late_buys_time_past_the_base(clock):
    session = _session(clock)
    clock.advance(BASE - 60)
    session.last_level_at = clock.now
    # The clear happened at t=540, so the game now runs to 540 + 300 = 840.
    assert session.budget_s() == pytest.approx(BASE - 60 + EXTENSION)
    clock.advance(120)  # t=660, past the base budget
    assert session.runtime_limit_reached() is False


def test_clearing_a_level_early_then_stalling_still_dies_on_the_base(clock):
    """The whole point of keying off the last clear rather than counting them.

    A game that cleared level 1 at t=30 and then went quiet has no claim on
    anyone else's GPU at t=600.
    """
    session = _session(clock)
    clock.advance(30)
    session.last_level_at = clock.now
    clock.advance(BASE - 30)
    assert session.budget_s() == BASE
    assert session.runtime_limit_reached() is True


def test_the_extension_never_shortens_the_base(clock):
    session = _session(clock)
    session.last_level_at = clock.now  # cleared at t=0; 0 + 300 < base 600
    assert session.budget_s() == BASE


def test_a_game_clearing_levels_forever_still_hits_the_ceiling(clock):
    session = _session(clock)
    for _ in range(20):
        clock.advance(EXTENSION - 30)
        session.last_level_at = clock.now
    assert session.budget_s() == CEILING
    clock.advance(CEILING)
    assert session.runtime_limit_reached() is True


def test_a_ceiling_below_the_base_caps_the_extension_not_the_base(clock):
    """Misconfiguration should cost the extension, never the original budget."""
    session = _session(clock, ceiling=BASE / 2)
    clock.advance(BASE - 60)
    session.last_level_at = clock.now
    assert session.budget_s() == BASE


def test_no_ceiling_means_progress_can_extend_indefinitely(clock):
    session = _session(clock, ceiling=None)
    for _ in range(20):
        clock.advance(EXTENSION - 30)
        session.last_level_at = clock.now
    assert session.budget_s() > CEILING


def test_zero_extension_reproduces_the_old_fixed_budget(clock):
    """The configuration the 1.16 run used has to still be reachable."""
    session = _session(clock, extension=0.0)
    clock.advance(BASE - 60)
    session.last_level_at = clock.now
    assert session.budget_s() == BASE


def test_an_unbounded_solver_is_left_unbounded(clock):
    session = _session(clock, base=None)
    session.last_level_at = clock.now
    assert session.budget_s() is None
    assert session.runtime_limit_reached() is False


def test_the_global_guard_fires_regardless_of_per_game_budget(clock):
    session = _session(clock, global_runtime=100.0)
    assert session.global_limit_reached() is False
    clock.advance(150)
    assert session.global_limit_reached() is True
    # Still inside its own generous per-game budget.
    assert session.runtime_limit_reached() is False


def test_no_global_runtime_means_no_global_deadline(clock):
    session = _session(clock)
    clock.advance(100_000)
    assert session.global_limit_reached() is False


def test_time_remaining_reports_the_tighter_of_the_two_clocks(clock):
    session = _session(clock, global_runtime=100.0)
    remaining = session.timing_payload()["time_remaining_seconds"]
    assert remaining == pytest.approx(100.0)  # global, not the 600s base


def test_time_remaining_follows_the_extension(clock):
    session = _session(clock)
    clock.advance(BASE - 60)
    assert session.timing_payload()["time_remaining_seconds"] == pytest.approx(60.0)
    session.last_level_at = clock.now
    assert session.timing_payload()["time_remaining_seconds"] == pytest.approx(EXTENSION)


def test_stop_reason_distinguishes_a_plain_timeout_from_an_extended_one(clock):
    session = _session(clock)
    clock.advance(BASE + 1)
    session.stop_reason = ""
    assert session._stopping("time_limit") is True

    extended = _session(clock)
    clock.advance(BASE - 60)
    extended.last_level_at = clock.now
    assert extended.budget_s() > BASE


def test_the_first_stop_reason_is_the_one_that_is_kept(clock):
    """should_stop() is polled long after the loop exits; last writer would lie."""
    session = _session(clock)
    session._stopping("time_limit")
    session._stopping("finished")
    assert session.stop_reason == "time_limit"
