"""The live page's browser pipeline (web/live-feed.js) against the Python one.

Uses a saved copy of OpenF1's raw data for the 2026 Brazilian Grand Prix.
"""
import gzip
import json
import shutil
import subprocess
from importlib import resources
from pathlib import Path

import pytest

from f1sim import load_tracks
from f1sim.laps import build_rows
from f1sim.replay import engine_constants, race_file
from f1sim.winprob import load_weights

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")
RAW = Path(__file__).parent / "data" / "openf1_brazil_2026.json.gz"


def run_js(script: str, payload: dict, tmp_path) -> dict:
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps(payload))
    web = resources.files("f1sim").joinpath("web")
    head = (f"const F = require({json.dumps(str(web.joinpath('live-feed.js')))});"
            f"const E = require({json.dumps(str(web.joinpath('replay-engine.js')))});"
            f"const I = require({json.dumps(str(inp))});")
    out = subprocess.run([NODE, "-e", head + script], capture_output=True, text=True, check=True, timeout=300)
    return json.loads(out.stdout)


@pytest.fixture(scope="module")
def brazil():
    raw = json.loads(gzip.open(RAW, "rt").read())
    raw["session"] = {**raw["session"], "year": 2026, "round": 22, "track": "brazil"}
    track = load_tracks()["brazil"]
    rows = build_rows(raw)
    return raw, track, rows, race_file(rows, track, None, {"season": 2026, "round": 22, "name": "", "winner": ""})


def test_js_pipeline_matches_python(brazil, tmp_path):
    raw, track, rows, py = brazil
    payload = {"raw": {k: raw[k] for k in ("drivers", "laps", "stints", "pit", "race_control", "weather")},
               "C": engine_constants(), "w": load_weights().tolist(), "track": py["track"], "laps": [1, 20, 40, 60]}
    js = run_js("""
const rows = F.buildRows(I.raw), race = {track: I.track, car_fields: F.CAR_FIELDS}, out = {rows: rows.length, laps: {}};
for (const lap of I.laps) { const e = F.raceState(rows, lap, I.track, I.C, null, I.track.laps); out.laps[lap] = {e, win: F.winOdds(race, e, I.w, I.C, E)}; }
console.log(JSON.stringify(out));""", payload, tmp_path)
    assert js["rows"] == len(rows)
    for lap, j in js["laps"].items():
        p = py["laps"][int(lap) - 1]
        for a, b in zip(p["cars"], j["e"]["cars"]):
            assert a[0] == b[0] and a[2:6] == b[2:6] and a[7:9] == b[7:9]           # driver, tyres, stops, laps
            assert abs((a[1] or 0) - (b[1] or 0)) < 0.01 and abs(a[6] - b[6]) < 0.001   # gap and pace (rounding)
        assert (p["status"], p["waived"]) == (j["e"]["status"], j["e"]["waived"])
        assert max(abs(p["win"].get(k, 0) - v) for k, v in j["win"].items()) < 0.002


def test_live_cutoff_keeps_the_field(brazil, tmp_path):
    """Cut the feed when the leader finishes lap 30, as a live feed would be."""
    raw, track, rows, py = brazil
    payload = {"raw": {k: raw[k] for k in ("drivers", "laps", "stints", "pit", "race_control", "weather")},
               "C": engine_constants(), "w": load_weights().tolist(), "track": py["track"], "lap": 30}
    js = run_js("""
const ends = I.raw.laps.filter(l => l.lap_number === I.lap && l.lap_duration).map(l => Date.parse(l.date_start) + l.lap_duration * 1e3);
const cut = Math.min(...ends), raw = {...I.raw, laps: I.raw.laps.filter(l => l.date_start && Date.parse(l.date_start) + (l.lap_duration || 0) * 1e3 <= cut + 1)};
const rows = F.buildRows(raw), race = {track: I.track, car_fields: F.CAR_FIELDS};
const e = F.raceState(rows, I.lap - 1, I.track, I.C, null, I.track.laps, true);
console.log(JSON.stringify({running: e.cars.filter(c => c[1] !== null).length, win: F.winOdds(race, e, I.w, I.C, E)}));""",
                payload, tmp_path)
    full = py["laps"][28]                                    # lap 29 from complete data
    assert js["running"] == sum(1 for c in full["cars"] if c[1] is not None)
    assert max(abs(full["win"].get(k, 0) - v) for k, v in js["win"].items()) < 0.05
