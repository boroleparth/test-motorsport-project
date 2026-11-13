"""The browser engine (f1sim/data/engine.js) must agree with the Python engine.

Runs the JavaScript engine under Node and compares its results with
simulate_race on the same scenarios. Skipped when Node is not installed.
"""
import json
import shutil
import subprocess
from dataclasses import replace
from importlib import resources

import numpy as np
import pytest

from f1sim import Strategy, load_grid, load_tracks, simulate_race
from f1sim.dashboard import export_model
from f1sim.models import Grid

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")
SIMS = 6000

SCENARIOS = [
    ("monaco", {}),
    ("britain", {"wet": True}),
    ("bahrain", {"strategies": {"VER": "S-H@40"}}),
    ("italy", {"pace": {"Williams": -1.5}}),
]


def run_js(model, cases, tmp_path, extra=""):
    (tmp_path / "model.json").write_text(json.dumps(model))
    engine = resources.files("f1sim").joinpath("data", "engine.js")
    script = f"""
const E = require({json.dumps(str(engine))});
const m = require({json.dumps(str(tmp_path / 'model.json'))});
const cases = {json.dumps(cases)};
{extra}
console.log(JSON.stringify(cases.map(([k, o]) => E.simulate(m, k, Object.assign({{sims: {SIMS}, seed: 11}}, o)).rows)));
"""
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True, timeout=300)
    return json.loads(out.stdout)


def test_js_engine_matches_python(tmp_path):
    grid, tracks = load_grid(), load_tracks()
    results = run_js(export_model(grid, tracks), SCENARIOS, tmp_path)
    for (key, opts), js in zip(SCENARIOS, results):
        g = grid
        if "pace" in opts:
            g = Grid({n: replace(t, pace=t.pace + opts["pace"].get(n, 0)) for n, t in grid.teams.items()}, grid.drivers)
        py = {r["driver"]: r for r in simulate_race(
            g, tracks[key], SIMS, seed=11, wet=opts.get("wet"),
            strategies={c: Strategy.parse(s) for c, s in opts.get("strategies", {}).items()}).summary()}
        for metric, tol in (("win", 0.03), ("podium", 0.04), ("dnf", 0.03), ("exp_pos", 0.35)):
            diff = np.array([r[metric] - py[r["code"]][metric] for r in js])
            assert np.abs(diff).max() < tol, (key, metric, diff)
            # no systematic shift: the average error over drivers stays near zero
            assert abs(diff.mean()) < tol / 4, (key, metric, diff.mean())


def test_js_controls(tmp_path):
    grid, tracks = load_grid(), load_tracks()
    model = export_model(grid, tracks)
    cases = [["spain", {"wet": False}], ["spain", {"wet": False, "startSlot": {"BOT": 1}}],
             ["spain", {"wet": False, "pace": {"Cadillac": -3}}]]
    base, slot, fast = run_js(model, cases, tmp_path)
    get = lambda rows, c: next(r for r in rows if r["code"] == c)
    assert get(slot, "BOT")["pole"] == 1.0 and get(base, "BOT")["pole"] < 0.01
    assert get(fast, "BOT")["win"] > get(base, "BOT")["win"] + 0.1
    for rows in (base, slot, fast):
        assert sum(r["win"] for r in rows) == pytest.approx(1)

    check = """
for (const bad of ["M-H", "M-X@10", "S-S@20", "M-H@99", "M-H-S@30,20"]) {
  let ok = false; try { E.parseStrategy(bad, 57); } catch (e) { ok = true; }
  if (!ok) throw new Error("accepted " + bad);
}
const s = E.parseStrategy("s-m-h@15,36", 57);
if (s.comps.join() !== "0,1,2" || s.pits.join() !== "15,36") throw new Error("parse");
"""
    run_js(model, [], tmp_path, extra=check)
