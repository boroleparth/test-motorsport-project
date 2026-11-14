"""Replay data (f1sim.replay) and the in-browser pit advisor (web/replay-engine.js)."""
import json
import shutil
import subprocess
from importlib import resources

import numpy as np
import pytest

from f1sim import load_tracks
from f1sim.laps import load_laps
from f1sim.live import plan_options, race_state, simulate_from
from f1sim.replay import CAR_FIELDS, engine_constants, race_file
from f1sim.winprob import by_race

NODE = shutil.which("node")


@pytest.fixture(scope="module")
def race():
    rows = by_race(load_laps([2026]))[2026, 22]          # wet start: the two-compound rule is waived
    track = load_tracks()[rows[0]["track"]]
    meta = {"season": 2026, "round": 22, "name": "test", "winner": "NOR"}
    return rows, track, race_file(rows, track, None, meta)


def test_race_file(race):
    rows, track, data = race
    assert data["car_fields"] == CAR_FIELDS and len(data["laps"]) == data["track"]["laps"]
    for lap in data["laps"]:
        assert lap["cars"][0][1] == 0.0                          # leader first, zero gap
        assert sum(lap["win"].values()) == pytest.approx(1, abs=0.02)
    assert data["laps"][-1]["win"] == {"NOR": 1.0}               # final lap shows the real winner
    assert data["laps"][39]["waived"] is True
    json.dumps(data)


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
def test_js_advisor_matches_python(race, tmp_path):
    rows, track, data = race
    lap = 40
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps({"C": engine_constants(), "race": data, "lap": lap}))
    engine = resources.files("f1sim").joinpath("web", "replay-engine.js")
    script = f"""
const E = require({json.dumps(str(engine))}); const I = require({json.dumps(str(inp))});
const e = I.race.laps[I.lap - 1];
const sim = E.simulateFrom(I.C, I.race, e, {{sims: 6000, seed: 3}});
const cars = E.cars(I.race, e).filter(c => c.running);
const plans = cars.slice(0, 5).map(c => E.planOptions(I.C, I.race, e, c, e.deg || I.race.track.deg).map(p => [p.pits, p.comps]));
const adv = E.advise(I.C, I.race, e, cars[1].code, {{sims: 300}});
console.log(JSON.stringify({{rows: sim.rows, plans, adv: adv.map(a => [a.label, a.result.exp_pos])}}));
"""
    js = json.loads(subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True, timeout=300).stdout)
    st = race_state(rows, lap, track)
    py = {r["driver"]: r for r in simulate_from(st, track, 6000, seed=3).summary()}
    for metric, tol in (("win", 0.03), ("podium", 0.04), ("exp_pos", 0.35)):
        diff = np.array([r[metric] - py[r["code"]][metric] for r in js["rows"]])
        assert np.abs(diff).max() < tol, (metric, diff)
    running = sorted([c for c in st.cars if c.running], key=lambda c: c.gap)[:5]
    for car, js_plans in zip(running, js["plans"]):
        py_plans = [[list(p.pits), list(p.comps)] for p, _ in plan_options(car, st, track, st.deg or track.deg)]
        assert js_plans == py_plans, car.code
    labels = [a[0] for a in js["adv"]]
    assert len(labels) >= 3 and any(l.startswith("Pit now") for l in labels)
    assert [a[1] for a in js["adv"]] == sorted(a[1] for a in js["adv"])   # best option first
