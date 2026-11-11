"""Build a self-contained HTML dashboard from simulation results."""
from __future__ import annotations

import datetime as dt
import json
from importlib import resources

import numpy as np

from .evaluate import compare_strategies
from .fit import walk_forward
from .models import Grid, Track
from .race import simulate_race
from .season import simulate_season
from .strategy import optimal_strategies


def _r(x, digits: int = 4) -> float:
    return round(float(x), digits)


def build_data(grid: Grid, tracks: dict[str, Track], season: dict | None = None,
               n_sims: int = 2000, seed: int | None = 0, accuracy: bool = True) -> dict:
    """Run every simulation the dashboard shows and return it as plain data."""
    seeds = np.random.SeedSequence(seed)
    codes = grid.codes
    calendar = [k for k in (season or {}).get("calendar", []) if k in tracks]
    done = len((season or {}).get("races", []))
    remaining = calendar[done:] if season else []
    featured = remaining[0] if remaining else next(iter(tracks))

    out_tracks = []
    for key, track in tracks.items():
        (child,) = seeds.spawn(1)
        res = simulate_race(grid, track, n_sims, seed=child)
        by_code = {r["driver"]: r for r in res.summary()}
        rows = [{"d": i, "win": _r(by_code[c]["win"]), "podium": _r(by_code[c]["podium"]),
                 "points": _r(by_code[c]["points"]), "pole": _r(by_code[c]["pole"]), "dnf": _r(by_code[c]["dnf"]),
                 "exp_pos": _r(by_code[c]["exp_pos"], 2), "exp_pts": _r(by_code[c]["exp_points"], 2)}
                for i, c in enumerate(codes)]
        out_tracks.append({
            "key": key, "name": track.name, "laps": track.laps, "sc_prob": track.sc_prob,
            "rain_prob": track.rain_prob, "strategy": str(optimal_strategies(track, top=1)[0]),
            "remaining": key in remaining, "rows": rows,
            "dist": [[_r(p, 3) for p in row] for row in res.position_matrix()],
        })

    data = {
        "label": (season or {}).get("season", ""), "generated": dt.date.today().isoformat(), "sims": n_sims,
        "rounds_done": done, "rounds_total": len((season or {}).get("calendar", [])), "next": featured,
        "drivers": [{"code": d.code, "name": d.name, "team": d.team, "pace": _r(d.pace, 3),
                     "quali_gap": _r(d.quali_pace + grid.teams[d.team].pace, 3), "error_rate": _r(d.error_rate, 3)}
                    for d in grid.drivers],
        "teams": [{"name": t.name, "pace": _r(t.pace, 3), "reliability": _r(t.reliability, 3)}
                  for t in sorted(grid.teams.values(), key=lambda t: t.pace)],
        "tracks": out_tracks, "season": None, "strategy": None, "accuracy": None,
    }
    # one-lap gap is shown relative to the quickest qualifier
    best_q = min(d["quali_gap"] for d in data["drivers"])
    for d in data["drivers"]:
        d["quali_gap"] = _r(d["quali_gap"] - best_q, 3)

    if remaining:
        standings = {c: p for c, p in season["standings"].items() if c in codes}
        (child,) = seeds.spawn(1)
        proj = simulate_season(grid, [tracks[k] for k in remaining], n_sims, seed=child, standings=standings)
        team_now = {t: sum(standings.get(d.code, 0) for d in grid.drivers if d.team == t) for t in grid.teams}

        def rows(table, current):
            return [{"name": r["name"], "current": _r(current.get(r["name"], 0), 1), "mean": _r(r["mean_points"], 1),
                     "p05": _r(r["p05"], 1), "p95": _r(r["p95"], 1), "title": _r(r["title"])} for r in table]

        data["season"] = {"races": remaining, "race_names": [tracks[k].name for k in remaining], "sims": n_sims,
                               "drivers": rows(proj.drivers(), standings),
                               "constructors": rows(proj.constructors(), team_now)}

    strat_sims = max(200, n_sims // 5)
    data["strategy"] = {
        "track": featured, "track_name": tracks[featured].name, "sims": strat_sims,
        "drivers": {c: [{**r, "clean_air": _r(r["clean_air"], 1), "exp_pos": _r(r["exp_pos"], 2),
                         "exp_points": _r(r["exp_points"], 2), "win": _r(r["win"]), "podium": _r(r["podium"])}
                        for r in compare_strategies(grid, tracks[featured], c, n_sims=strat_sims, seed=seed)]
                    for c in codes},
    }

    if accuracy and season and done > 6:
        (child,) = seeds.spawn(1)
        wf = walk_forward(season, tracks, grid, start=5, n_sims=min(n_sims, 1500), seed=child)
        names = {r["track"]: r["name"] for r in season["races"]}
        data["accuracy"] = {
            "mean": {k: _r(v) for k, v in wf["mean"].items()},
            "races": [{"track": s["track"], "name": names.get(s["track"], s["track"]), "winner": s["winner"],
                       "winner_prob": _r(s["winner_prob"]), "podium_hits": s["podium_hits"],
                       "favourite": s["favourite"], "favourite_won": s["favourite"] == s["winner"]}
                      for s in wf["races"]],
        }
    return data


def render(data: dict, standalone: bool = True) -> str:
    """Fill the HTML template. ``standalone=False`` returns a fragment without <html>/<body>."""
    template = resources.files("f1sim").joinpath("data", "dashboard.html").read_text()
    page = template.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    if not standalone:
        return page
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n'
            + page + "\n</body>\n</html>\n")


def build_dashboard(grid: Grid, tracks: dict[str, Track], season: dict | None = None, n_sims: int = 2000,
                    seed: int | None = 0, standalone: bool = True, accuracy: bool = True) -> str:
    return render(build_data(grid, tracks, season, n_sims, seed, accuracy), standalone)
