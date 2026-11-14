"""Data for the race replay page: every race's state after each lap.

``race_file`` turns one race's lap table into a compact JSON-ready dict: per
lap, each car's gap, tyres, stops, pace estimate and running position (the
inputs the in-browser pit advisor simulates from, see web/replay-engine.js),
plus live win odds from ``f1sim.winprob``. ``build_replays`` writes one file per
race and an index of races for the page.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .laps import load_laps
from . import live as LIVE
from . import race as RACE
from .live import race_state
from .strategy import OFFSET, WEAR
from .models import Grid, Track
from .winprob import by_race, load_weights, pre_race_grids, win_probabilities

COMPOUND_CODE = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
USED_BIT = {"SOFT": 1, "MEDIUM": 2, "HARD": 4}
# fields of each car row in a lap: see race_file
CAR_FIELDS = ["code", "gap", "compound", "age", "stops", "used", "pace", "pace_laps", "laps_done", "dnf_rate"]


def race_file(rows: list[dict], track: Track, grid: Grid | None, meta: dict, weights=None) -> dict:
    """Replay data for one race. ``meta`` adds season, round and name."""
    laps_total = max(r["laps_total"] for r in rows)
    teams = {}
    for r in rows:
        teams.setdefault(r["driver"], r["team"])
    final = {r["driver"]: r["final_position"] for r in rows if r["final_position"]}
    weights = load_weights() if weights is None else weights
    laps = []
    for lap in range(1, laps_total + 1):
        st = race_state(rows, lap, track, grid)
        cars = []
        for c in sorted(st.cars, key=lambda c: (not c.running, c.gap if c.running else -c.laps_done)):
            cars.append([c.code, round(c.gap, 2) if c.running else None, COMPOUND_CODE.get(c.compound, 1),
                         int(c.tyre_age), int(c.stops), sum(USED_BIT.get(u, 0) for u in c.used),
                         round(c.pace, 3), int(c.pace_laps), int(c.laps_done), round(c.dnf_rate, 4)])
        win = win_probabilities(st, weights, track) if lap < laps_total else {
            c[0]: float(final.get(c[0]) == 1) for c in cars}
        laps.append({"lap": lap, "status": st.status, "wet": st.wet, "waived": st.rule_waived,
                     "deg": round(st.deg, 3) if st.deg else None,
                     "cars": cars, "win": {k: round(v, 4) for k, v in win.items() if v >= 0.0005}})
    return {**meta, "track": {"key": track.key, "name": track.name, "laps": laps_total, "pit_loss": track.pit_loss,
                              "overtake": track.overtake, "sc_prob": track.sc_prob, "deg": track.deg},
            "teams": teams, "final": final, "car_fields": CAR_FIELDS, "laps": laps}


def build_replays(tracks: dict[str, Track], out: str | Path, years: list[int] | None = None,
                  log=lambda msg: None) -> list[dict]:
    """Write one JSON file per race to ``out`` and return the index of races."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    races = by_race(load_laps(years))
    grids = pre_race_grids(races)
    weights = load_weights()
    index = []
    for (season, rnd), rows in sorted(races.items()):
        track = tracks.get(rows[0]["track"])
        if track is None:
            continue
        winner = next((r["driver"] for r in rows if r["final_position"] == 1), None)
        meta = {"season": season, "round": rnd, "name": f"{track.name} {season}", "winner": winner}
        data = race_file(rows, track, grids.get((season, rnd)), meta, weights)
        name = f"{season}-{rnd:02d}.json"
        (out / name).write_text(json.dumps(data, separators=(",", ":")))
        index.append({**meta, "file": name, "laps": data["track"]["laps"],
                      "neutralised": sum(1 for l in data["laps"] if l["status"] != "GREEN")})
        log(f"{name} {track.name}: {len(data['laps'])} laps")
    return index


def lead_changes(data: dict) -> int:
    """How many times the favourite changed during a race (used in the index)."""
    favs = [max(l["win"], key=l["win"].get) for l in data["laps"] if l["win"]]
    return int(np.sum([a != b for a, b in zip(favs, favs[1:])]))


def engine_constants() -> dict:
    """Settings the in-browser pit advisor (web/replay-engine.js) shares with f1sim.live."""
    return {
        **{k: float(getattr(RACE, k)) for k in ("FUEL_PER_LAP", "FOLLOW_GAP", "PASS_SCALE", "SC_LAP_FACTOR",
                                                "SC_PIT_FACTOR", "SC_GAP", "WET_NOISE")},
        **{k: float(getattr(LIVE, k)) for k in ("PACE_SCALE", "PACE_SPREAD")},
        "OFFSET": OFFSET.tolist(), "WEAR": WEAR.tolist(), "POINTS": RACE.POINTS.tolist(),
    }
