"""Public prediction log: forecasts saved before each race, scored once it is run.

For every race weekend two snapshots are written to ``data/predictions/``:

* ``pre_quali``  saved before qualifying, from the ratings alone (the simulator
  simulates qualifying itself);
* ``post_quali`` saved after qualifying and before the race, given the real
  qualifying order and lap times (``simulate_race(start_grid=..., quali_times=...)``).

A snapshot is never written once its session has started and never
overwritten, so the files, and the git history that records when they were
committed, show what was predicted before the result was known. Scores are
computed from the saved probabilities whenever the site is built, so nothing
is rewritten after the race.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np

from .evaluate import score_matrix
from .models import Grid, Track
from .race import simulate_race

LOG = Path(__file__).resolve().parent / "data" / "predictions"
KINDS = ("pre_quali", "post_quali")
LABELS = {"pre_quali": "Before qualifying", "post_quali": "After qualifying"}
PRE_WINDOW = dt.timedelta(days=4)       # save the pre-qualifying forecast at most this early
QUALI_RESULTS_AFTER = dt.timedelta(hours=1, minutes=30)  # qualifying is over by then
N_SIMS = 10000


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _time(text: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(text) if text else None


def path_for(season: str | int, rnd: int, track: str, log: Path = LOG) -> Path:
    return Path(log) / f"{season}-{int(rnd):02d}-{track}.json"


def _dumps(entry: dict) -> str:
    """Indented JSON with each list of plain values (a row of probabilities, the codes) on one line."""
    text = json.dumps(entry, indent=1)
    flat = lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]"
    return re.sub(r"\[([^\[\]{}]*)\]", flat, text) + "\n"


def load_log(log: Path = LOG) -> list[dict]:
    """Every saved race entry, oldest first."""
    log = Path(log)
    if not log.is_dir():
        return []
    entries = [json.loads(p.read_text()) for p in sorted(log.glob("*.json"))]
    return sorted(entries, key=lambda e: (str(e["season"]), e["round"]))


def snapshot(grid: Grid, track: Track, start_grid: list[str] | None = None,
             quali_times: dict[str, float] | None = None, n_sims: int = N_SIMS, seed: int = 0) -> dict:
    """Simulate the race and keep what is needed to score it later."""
    res = simulate_race(grid, track, n_sims, seed=seed, start_grid=start_grid, quali_times=quali_times)
    P = res.position_matrix()
    summary = [{"driver": r["driver"], "win": round(r["win"], 4), "podium": round(r["podium"], 4),
                "points": round(r["points"], 4)} for r in sorted(res.summary(), key=lambda r: -r["win"])]
    snap = {"sims": n_sims, "seed": seed, "codes": grid.codes, "summary": summary,
            "dist": [[round(float(p), 4) for p in row] for row in P]}
    if start_grid:
        snap["start_grid"] = list(start_grid)
    if quali_times:
        snap["quali_times"] = {k: v for k, v in quali_times.items() if v}
    return snap


def save(kind: str, race: dict, grid: Grid, track: Track, season: dict, now: dt.datetime | None = None,
         start_grid: list[str] | None = None, quali_times: dict[str, float] | None = None,
         n_sims: int = N_SIMS, log: Path = LOG, force: bool = False) -> Path:
    """Write one snapshot for ``race`` (a :func:`~f1sim.datasource.fetch_schedule` entry).

    Refuses to save after the session it predicts has started, or over an
    existing snapshot (unless ``force``), so the log stays honest.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    now = now or _now()
    deadline = _time(race["qualifying"] if kind == "pre_quali" else race["race"])
    if deadline and now >= deadline and not force:
        raise ValueError(f"too late for a {kind} forecast of {race['name']}: the session started {deadline:%d %b %H:%M} UTC")
    if any(r["round"] == race["round"] for r in season.get("races", [])):
        raise ValueError(f"{race['name']} already has a result")
    if kind == "post_quali" and not start_grid:
        raise ValueError("an after-qualifying forecast needs the qualifying order")
    path = path_for(season["season"], race["round"], race["track"], log)
    entry = json.loads(path.read_text()) if path.exists() else {
        "season": str(season["season"]), "round": race["round"], "track": race["track"], "name": race["name"],
        "qualifying": race["qualifying"], "race": race["race"]}
    if kind in entry and not force:
        raise ValueError(f"{path.name} already has a {kind} forecast (saved {entry[kind]['saved']})")
    done = [r["round"] for r in season.get("races", [])]
    entry[kind] = {"saved": now.replace(microsecond=0).isoformat(),
                   "ratings": f"fitted from {season['season']} rounds 1-{max(done)}" if done else "pre-season",
                   **snapshot(grid, track, start_grid, quali_times, n_sims,
                              seed=int(season["season"]) * 100 + race["round"])}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dumps(entry))
    return path


def next_race(schedule: list[dict], season: dict, now: dt.datetime | None = None) -> dict | None:
    """The first race of ``schedule`` with no result yet that has not started."""
    now = now or _now()
    done = {r["round"] for r in season.get("races", [])}
    for race in schedule:
        start = _time(race["race"])
        if race["round"] not in done and (start is None or start > now):
            return race
    return None


def auto(grid: Grid, tracks: dict[str, Track], season: dict, schedule: list[dict],
         fetch_quali: Callable[[int], list[tuple[str, float | None]]], now: dt.datetime | None = None,
         n_sims: int = N_SIMS, log: Path = LOG) -> list[str]:
    """Save whichever snapshot is due for the next race. Safe to run as often as you like.

    * within ``PRE_WINDOW`` of qualifying and before it starts: the pre-qualifying forecast;
    * after qualifying (once its results are published) and before the race: the
      post-qualifying forecast.

    Returns a line per action taken or skipped, for the build log.
    """
    now = now or _now()
    race = next_race(schedule, season, now)
    if race is None:
        return ["no race left this season"]
    if race["track"] not in tracks:
        return [f"{race['name']}: circuit {race['track']!r} not in tracks.json, not predicted"]
    track = tracks[race["track"]]
    path = path_for(season["season"], race["round"], race["track"], log)
    entry = json.loads(path.read_text()) if path.exists() else {}
    quali, start = _time(race["qualifying"]), _time(race["race"])
    notes = []

    if "pre_quali" not in entry:
        if quali and quali - PRE_WINDOW <= now < quali:
            save("pre_quali", race, grid, track, season, now, n_sims=n_sims, log=log)
            notes.append(f"{race['name']}: saved the pre-qualifying forecast")
        elif quali and now >= quali:
            notes.append(f"{race['name']}: qualifying has started, too late for a pre-qualifying forecast")
        else:
            notes.append(f"{race['name']}: pre-qualifying forecast not due yet (qualifying {race['qualifying']})")

    if "post_quali" not in entry and quali and now >= quali + QUALI_RESULTS_AFTER and (start is None or now < start):
        rows = fetch_quali(race["round"])
        if len(rows) < 10:
            notes.append(f"{race['name']}: qualifying results not published yet")
        else:
            save("post_quali", race, grid, track, season, now, start_grid=[c for c, _ in rows],
                 quali_times={c: t for c, t in rows if t}, n_sims=n_sims, log=log)
            notes.append(f"{race['name']}: saved the post-qualifying forecast")
    return notes


def track_record(entries: list[dict], seasons: list[dict]) -> dict:
    """Score every logged forecast whose race has a result; list the rest as pending."""
    results = {(str(s["season"]), r["round"]): r for s in seasons for r in s.get("races", [])}
    rows, scored = [], {k: [] for k in KINDS}
    for e in entries:
        race = results.get((str(e["season"]), e["round"]))
        row = {"season": str(e["season"]), "round": e["round"], "name": e["name"], "track": e["track"],
               "race_start": e.get("race"), "winner": race["results"][0] if race else None,
               "file": path_for(e["season"], e["round"], e["track"]).name}
        for kind in KINDS:
            snap = e.get(kind)
            if not snap:
                continue
            fav = snap["summary"][0]
            out = {"saved": snap["saved"], "favourite": fav["driver"], "favourite_win": fav["win"],
                   "top": snap["summary"][:3]}
            if kind == "post_quali":
                out["pole"] = snap["start_grid"][0]
            if race:
                s = score_matrix(snap["codes"], np.array(snap["dist"]), race["results"], eps=0.5 / snap["sims"])
                out.update({k: s[k] for k in ("winner_prob", "rps", "favourite_won", "podium_hits", "log_loss")})
                if kind == "post_quali":
                    out["pole_won"] = int(out["pole"] == race["results"][0])
                scored[kind].append(out)
            row[kind] = out
        rows.append(row)
    summary = {}
    for kind, outs in scored.items():
        if outs:
            summary[kind] = {"label": LABELS[kind], "n": len(outs),
                             **{k: float(np.mean([o[k] for o in outs]))
                                for k in ("winner_prob", "rps", "favourite_won", "podium_hits", "log_loss")}}
            if kind == "post_quali":
                summary[kind]["pole_won"] = float(np.mean([o["pole_won"] for o in outs]))
    return {"races": rows, "summary": summary}
