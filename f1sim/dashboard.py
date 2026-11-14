"""Build the website: a home page plus one page per area (forecast, lab, accuracy)."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from importlib import resources

import numpy as np

from .evaluate import compare_strategies
from .benchmark import LABELS, METHODS, PAIRS, benchmark, compare, summarise
from .models import Grid, Track
from .race import simulate_race
from .season import simulate_season
from .strategy import optimal_strategies


def _r(x, digits: int = 4) -> float:
    return round(float(x), digits)


def build_data(grid: Grid, tracks: dict[str, Track], season: dict | None = None,
               n_sims: int = 2000, seed: int | None = 0, accuracy: bool = True,
               history: list[dict] | None = None, holdout: tuple[str, ...] = ("2025", "2026")) -> dict:
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
        "model": export_model(grid, tracks),
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
        seasons = history or [season]
        bench = benchmark(seasons, tracks, grid, start=5, n_sims=min(n_sims, 1000), seed=child)
        years = [s["season"] for s in seasons]
        held = [y for y in years if y in holdout] if len(years) > 1 else years
        means = summarise(bench["races"], held)
        data["accuracy"] = {
            "n_races": means["n_races"], "seasons": held, "n_all": bench["n_races"], "all_seasons": years,
            "tuned_on": [y for y in years if y not in held],
            "methods": [{"label": LABELS[m], "model": m.startswith("model"),
                         **{k: _r(v) for k, v in means[m].items()}} for m in METHODS],
            "tests": [{"a": LABELS[a], "b": LABELS[b], **{k: _r(v, 5) for k, v in compare(bench["races"], a, b).items()
                                                         if k in ("diff", "lo", "hi", "p_a_lower")}} for a, b in PAIRS],
            "calibration": {e: [{"lo": r["lo"], "hi": r["hi"], "n": r["n"], "predicted": _r(r["predicted"]),
                                 "observed": _r(r["observed"])} for r in rows]
                            for e, rows in bench["calibration"].items()},
            "races": [{"name": r["name"], "winner": r["winner"], "favourite": r["favourite"],
                       "winner_prob": _r(r["model:winner_prob"]), "after_prob": _r(r["model+grid:winner_prob"]),
                       "favourite_won": bool(r["model:favourite_won"])}
                      for r in bench["races"] if r["season"] == season["season"]],
        }
    return data


PAGES = [  # (folder, menu label, title, description)
    ("", "Home", "F1 Race Simulator",
     "Formula 1 race and championship predictions from a lap-by-lap simulator fitted to real results."),
    ("forecast", "Forecast", "Race Forecast", "Win, podium and points odds for every circuit, the championship and pit strategies."),
    ("lab", "What-if lab", "What-if Lab", "Rerun an F1 race in your browser with rain, pace, strategy or grid changes."),
    ("replay", "Race replay", "Race Replay", "Step through any Grand Prix since 2023 with lap-by-lap win odds and a pit-stop advisor."),
    ("live", "Live", "Live Race", "Live Formula 1 win odds and pit-stop advice during a Grand Prix."),
    ("accuracy", "Accuracy", "Prediction Accuracy", "How the simulator's predictions compare with real results and simple baselines."),
]
REPO = "https://github.com/boroleparth/test-motorsport-project"
FONTS = ("https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700"
         "&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap")


def _page_data(data: dict, page: str) -> dict:
    """Only the data each page needs, so every page stays small."""
    pick = lambda *keys: {k: data[k] for k in keys}
    if page == "forecast":
        return pick("label", "rounds_done", "rounds_total", "sims", "next", "drivers", "tracks", "season", "strategy")
    if page == "lab":
        return {**pick("next", "drivers", "model"),
                "tracks": [{k: t[k] for k in ("key", "name", "remaining")} for t in data["tracks"]]}
    if page == "accuracy":
        return pick("label", "drivers", "teams", "accuracy")
    if page == "replay":
        from .replay import engine_constants
        return {"races": data.get("replays", []), "C": engine_constants()}
    if page == "live":
        from .laps import CIRCUITS
        from .models import load_tracks
        from .replay import engine_constants
        from .winprob import load_weights
        teams = {t["name"]: t for t in data["teams"]}
        prior = {d["code"]: {"pace": teams[d["team"]]["pace"] + d["pace"],
                             "dnf": 1 - (1 - teams[d["team"]]["reliability"]) * (1 - d["error_rate"])}
                 for d in data["drivers"] if d["team"] in teams}
        tracks = {t.key: {"name": t.name, "laps": t.laps, "pit_loss": t.pit_loss, "overtake": t.overtake,
                          "sc_prob": t.sc_prob, "deg": t.deg} for t in load_tracks().values()}
        return {"C": engine_constants(), "weights": load_weights().tolist(), "prior": prior,
                "tracks": tracks, "circuits": CIRCUITS}
    # home: a summary of the other pages
    home = pick("label", "rounds_done", "rounds_total")
    track = next((t for t in data["tracks"] if t["key"] == data["next"]), None)
    if track:
        top = sorted(track["rows"], key=lambda r: -r["win"])[:3]
        home["next"] = {"name": track["name"], "top": [{"code": data["drivers"][r["d"]]["code"],
                        "name": data["drivers"][r["d"]]["name"], "win": r["win"]} for r in top]}
    if data.get("season"):
        lead = data["season"]["drivers"][0]
        home["title"] = {"code": lead["name"], "p": lead["title"]}
    acc = data.get("accuracy")
    if acc:
        test = next((t for t in acc.get("tests", []) if t["a"] == "Simulator, after qualifying"), None)
        if test:
            home["acc"] = {"diff": test["diff"], "lo": test["lo"], "hi": test["hi"], "races": acc["n_all"],
                           "first": acc["all_seasons"][0]}
    return {"home": home}


def render_page(data: dict, page: str) -> str:
    """One self-contained HTML page of the site (styles and scripts inlined)."""
    web = resources.files("f1sim").joinpath("web")
    name = page or "home"
    root = "../" if page else "./"
    _, _, title, description = next(p for p in PAGES if p[0] == page)
    items = []
    for folder, label, _, _ in PAGES:
        current = ' aria-current="page"' if folder == page else ""
        items.append(f'<li><a href="{root}{folder + "/" if folder else ""}"{current}>{label}</a></li>')
    items.append(f'<li class="ext"><a href="{REPO}">Code on GitHub ↗</a></li>')
    engine = ""
    if page == "lab":
        engine = "<script>\n" + resources.files("f1sim").joinpath("data", "engine.js").read_text() + "\n</script>\n"
    if page == "replay":
        engine = "<script>\n" + web.joinpath("replay-engine.js").read_text() + "\n</script>\n"
    if page == "live":
        engine = "".join("<script>\n" + web.joinpath(f).read_text() + "\n</script>\n"
                         for f in ("replay-engine.js", "live-feed.js"))
    blob = json.dumps(_page_data(data, page), separators=(",", ":")).replace("</", "<\\/")
    full_title = title if not page else f"{title} · F1 Race Simulator"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{full_title}</title>
<meta name="description" content="{description}">
<link rel="stylesheet" href="{FONTS}">
<style>
{web.joinpath("style.css").read_text()}
</style>
</head>
<body>
<nav class="topbar" aria-label="Site"><div class="in">
<a class="brand" href="{root}">F1 Race <span>Sim</span></a>
<ul class="menu">{"".join(items)}</ul>
</div></nav>
<div class="wrap">
{web.joinpath(name + ".html").read_text()}
<footer>Generated {data["generated"]} by f1sim from real results (<a href="https://api.jolpi.ca/ergast/f1/">Jolpica F1 API</a>,
<a href="https://openf1.org">OpenF1</a>). Predictions come from a lap-by-lap simulation; see the
<a href="{REPO}">code and methods on GitHub</a>.</footer>
</div>
<div id="tip" hidden></div>
{engine}<script>
const D = {blob};
{web.joinpath("common.js").read_text()}
{web.joinpath("replay.js").read_text() if page == "live" else ""}
{web.joinpath(name + ".js").read_text()}
</script>
</body>
</html>
"""


def build_site(data: dict, out: str | Path, replays: bool = True) -> list[Path]:
    """Write every page to ``out`` (index.html plus one folder per page).

    With ``replays``, also writes one data file per race to ``replay/data/``.
    """
    out = Path(out)
    written = []
    if replays:
        from .models import load_tracks
        from .replay import build_replays
        data = {**data, "replays": build_replays(load_tracks(), out / "replay" / "data")}
    for folder, *_ in PAGES:
        path = out / folder / "index.html" if folder else out / "index.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_page(data, folder))
        written.append(path)
    (out / ".nojekyll").write_text("")
    return written


def build_dashboard(grid: Grid, tracks: dict[str, Track], out: str | Path, season: dict | None = None,
                    n_sims: int = 2000, seed: int | None = 0, accuracy: bool = True,
                    history: list[dict] | None = None, replays: bool = True) -> list[Path]:
    """Run every simulation and write the multi-page site to ``out``."""
    return build_site(build_data(grid, tracks, season, n_sims, seed, accuracy, history), out, replays)


def export_model(grid: Grid, tracks: dict[str, Track]) -> dict:
    """Everything the JavaScript engine (data/engine.js) needs, from the Python side."""
    from . import race as R
    from . import strategy as T

    team_names = list(grid.teams)
    model = {
        "constants": {k: float(getattr(R, k)) for k in (
            "FUEL_PER_LAP", "FOLLOW_GAP", "PASS_SCALE", "QUALI_SIGMA", "FORM_SIGMA", "GRID_SPACING",
            "START_SIGMA", "SC_LAP_FACTOR", "SC_PIT_FACTOR", "SC_GAP", "WET_LAP_FACTOR", "WET_NOISE", "WET_ERROR")},
        "tyres": {"offset": T.OFFSET.tolist(), "wear": T.WEAR.tolist(), "life": T.LIFE.tolist(), "cliff": T.CLIFF},
        "teams": [{"name": t.name, "pace": t.pace, "reliability": t.reliability,
                   "pit_mean": t.pit_mean, "pit_sigma": t.pit_sigma} for t in grid.teams.values()],
        "drivers": [{"code": d.code, "team": team_names.index(d.team), "pace": d.pace, "quali": d.quali_pace,
                     "consistency": d.consistency, "wet": d.wet, "error_rate": d.error_rate} for d in grid.drivers],
        "tracks": {},
    }
    model["constants"]["POINTS"] = R.POINTS.tolist()
    for key, track in tracks.items():
        options, weights = R._default_options(track)
        model["tracks"][key] = {
            "name": track.name, "laps": track.laps, "base_lap": track.base_lap, "pit_loss": track.pit_loss,
            "overtake": track.overtake, "sc_prob": track.sc_prob, "rain_prob": track.rain_prob, "deg": track.deg,
            "options": [{"comps": [T.COMPOUNDS.index(c) for c in s.compounds], "pits": list(s.pit_laps),
                         "label": str(s), "weight": float(w)} for s, w in zip(options, weights)],
            "candidates": [str(s) for s in optimal_strategies(track, top=6)],
        }
    return model
