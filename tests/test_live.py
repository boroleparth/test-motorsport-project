"""Mid-race simulation (f1sim.live) and live win probability (f1sim.winprob)."""
import json
from dataclasses import replace
from importlib import resources

import numpy as np
import pytest

from f1sim import load_tracks
from f1sim.laps import load_laps
from f1sim.live import Plan, plan_options, race_state, simulate_from, tyre_cost
from f1sim.winprob import features, win_probabilities


@pytest.fixture(scope="module")
def race():
    rows = [r for r in load_laps([2025]) if r["round"] == 10]
    return rows, load_tracks()[rows[0]["track"]]


def test_race_state(race):
    rows, track = race
    st = race_state(rows, 20, track)
    running = [c for c in st.cars if c.running]
    assert st.lap == 20 and st.remaining == st.laps_total - 20 and len(running) >= 15
    assert min(c.gap for c in running) == 0.0
    leader = min(running, key=lambda c: c.gap)
    assert leader.code == next(r["driver"] for r in rows if r["lap"] == 20 and r["position"] == 1)
    assert all(c.tyre_age >= 1 and c.compound for c in running)


def test_simulate_from_end_and_late_lead(race):
    rows, track = race
    last = race_state(rows, max(r["laps_total"] for r in rows), track)
    out = simulate_from(last, track, 20, seed=1)
    assert (out.positions == out.positions[0]).all()           # nothing left to race
    winner = next(r["driver"] for r in rows if r["final_position"] == 1)
    assert out.codes[int(np.argmin(out.positions[0]))] == winner
    st = race_state(rows, max(r["laps_total"] for r in rows) - 3, track)
    P = dict(zip(out.codes, simulate_from(st, track, 400, seed=2).matrix()[:, 0]))
    leader = min((c for c in st.cars if c.running), key=lambda c: c.gap)
    assert P[leader.code] > 0.6 and sum(P.values()) == pytest.approx(1)


def test_plans_respect_two_compound_rule(race):
    rows, track = race
    st = race_state(rows, 10, track)
    car = next(c for c in st.cars if c.running)
    car = replace(car, compound="MEDIUM", used={"MEDIUM"}, tyre_age=10)
    plans = plan_options(car, st, track, 1.0)
    assert plans and all(p.pits for p, _ in plans)              # must stop at least once
    assert all(any(c != 1 for c in p.comps) for p, _ in plans)   # and fit another compound
    done = replace(car, used={"MEDIUM", "HARD"})
    late = replace(st, lap=st.laps_total - 3)
    assert any(not p.pits for p, _ in plan_options(done, late, track, 1.0))   # rule met: staying out is allowed
    forced = simulate_from(st, track, 50, seed=3, force={car.code: Plan((st.lap + 5,), (2,))})
    assert forced.positions.shape == (50, len(st.cars))


def test_tyre_cost_has_no_cliff():
    ages = np.arange(0, 60)
    d = np.diff(tyre_cost(1, ages, 1.0))
    assert np.allclose(d, d[0]) and d[0] > 0


def test_win_probabilities(race):
    rows, track = race
    report = json.loads(resources.files("f1sim").joinpath("data", "winprob.json").read_text())
    assert report["test"]["log_loss"] < 1.0 and report["test"]["favourite_won"] > 0.6
    laps_total = max(r["laps_total"] for r in rows)
    late = win_probabilities(race_state(rows, laps_total - 2, track))
    early = win_probabilities(race_state(rows, 5, track))
    assert sum(late.values()) == pytest.approx(1) and sum(early.values()) == pytest.approx(1)
    assert max(late.values()) > max(early.values())
    X, codes = features(race_state(rows, 30, track))
    assert X.shape == (len(codes), 5) and np.isfinite(X).all()
