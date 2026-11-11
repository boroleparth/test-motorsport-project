"""Benchmark the whole pipeline against simple baselines over many seasons.

Every race is predicted from earlier rounds of the same season only
(walk-forward), four ways:

* ``model``        simulator, before qualifying (it simulates qualifying itself)
* ``model+grid``   simulator, given the real starting grid and qualifying times
* ``standings``    baseline: drivers finish in current championship order
* ``grid``         baseline: drivers finish where they start

Baselines only give an order, so to score them with the same probability
metrics each rank is given the finishing-position distribution that rank
actually produced across the benchmark races. That uses the answers, which
favours the baselines.
"""
from __future__ import annotations

import numpy as np

from .evaluate import score_matrix
from .fit import fit_grid
from .models import Grid, Track
from .race import POINTS, simulate_race

METHODS = ("model", "model+grid", "standings", "grid")
LABELS = {"model": "Simulator, before qualifying", "model+grid": "Simulator, after qualifying",
          "standings": "Baseline: championship order", "grid": "Baseline: starting grid order"}
CAL_EVENTS = {"win": 1, "podium": 3, "top10": 10}
CAL_BINS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.0001)


def _points_before(season: dict, rnd: int) -> dict[str, float]:
    pts: dict[str, float] = {}
    for race in season["races"]:
        if race["round"] >= rnd:
            break
        for pos, code in enumerate(race["results"][: len(POINTS)]):
            pts[code] = pts.get(code, 0) + float(POINTS[pos])
    return pts


def _start_order(race: dict) -> list[str]:
    grid = {c: e["grid"] for c, e in race["entries"].items()}
    return sorted(grid, key=lambda c: (grid[c] <= 0, grid[c]))  # 0 = pit-lane start


def benchmark(seasons: list[dict], tracks: dict[str, Track], base: Grid | None = None, start: int = 5,
              n_sims: int = 1000, seed: int | None = 0, **fit_kwargs) -> dict:
    """Walk-forward test over several seasons. See the module docstring.

    Returns ``{"methods": {name: mean metrics}, "races": [...], "calibration": {...}, "n_races"}``.
    """
    seeds = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    cases = []          # one per race: codes, actual order, model matrices, baseline ranks
    for season in seasons:
        for race in season["races"]:
            if race["round"] <= start or race["track"] not in tracks:
                continue
            grid = fit_grid(season, base, through_round=race["round"] - 1, **fit_kwargs)
            codes, track = grid.codes, tracks[race["track"]]
            a, b = seeds.spawn(2)
            start_order = [c for c in _start_order(race) if c in codes]
            pts = _points_before(season, race["round"])
            cases.append({
                "season": season["season"], "round": race["round"], "track": race["track"],
                "name": race["name"], "codes": codes, "actual": race["results"],
                "model": simulate_race(grid, track, n_sims, seed=a).position_matrix(),
                "model+grid": simulate_race(
                    grid, track, n_sims, seed=b, start_grid=start_order,
                    quali_times={k: e["quali"] for k, e in race["entries"].items()}).position_matrix(),
                "standings": sorted(codes, key=lambda c: -pts.get(c, 0)),
                "grid": start_order + [c for c in codes if c not in start_order],
            })
    if not cases:
        raise ValueError("no races to benchmark")

    # Finishing-position distribution of each baseline rank, pooled over all races.
    size = max(len(c["codes"]) for c in cases)
    for method in ("standings", "grid"):
        counts = np.full((size, size), 0.5)
        for c in cases:
            finish = {code: i for i, code in enumerate([x for x in c["actual"] if x in c["codes"]])}
            for rank, code in enumerate(c[method]):
                counts[rank, finish.get(code, len(c["codes"]) - 1)] += 1
        for c in cases:
            n = len(c["codes"])
            K = counts[:n, :n] / counts[:n, :n].sum(1, keepdims=True)
            P = np.zeros((n, n))
            for rank, code in enumerate(c[method]):
                P[c["codes"].index(code)] = K[rank]
            c[method] = P

    eps = 0.5 / n_sims
    scores = {m: [score_matrix(c["codes"], c[m], c["actual"], eps) for c in cases] for m in METHODS}
    numeric = [k for k, v in scores["model"][0].items() if isinstance(v, (int, float))]
    means = {m: {k: float(np.mean([s[k] for s in scores[m]])) for k in numeric} for m in METHODS}

    # Calibration: when the simulator says p, how often does it happen?
    calibration = {}
    for event, top in CAL_EVENTS.items():
        pred, hit = [], []
        for c in cases:
            finish = {code: i for i, code in enumerate([x for x in c["actual"] if x in c["codes"]])}
            for i, code in enumerate(c["codes"]):
                pred.append(c["model"][i, :top].sum())
                hit.append(finish.get(code, 99) < top)
        pred, hit = np.array(pred), np.array(hit, dtype=float)
        rows = []
        for lo, hi in zip(CAL_BINS[:-1], CAL_BINS[1:]):
            mask = (pred >= lo) & (pred < hi)
            if mask.sum():
                rows.append({"lo": lo, "hi": min(hi, 1.0), "n": int(mask.sum()),
                             "predicted": float(pred[mask].mean()), "observed": float(hit[mask].mean())})
        calibration[event] = rows

    races = [{"season": c["season"], "round": c["round"], "track": c["track"], "name": c["name"],
              **{f"{m}:{k}": scores[m][i][k] for m in METHODS for k in numeric},
              "winner": scores["model"][i]["winner"], "favourite": scores["model"][i]["favourite"]}
             for i, c in enumerate(cases)]
    return {"methods": means, "races": races, "calibration": calibration, "n_races": len(cases),
            "seasons": [s["season"] for s in seasons]}


def summarise(races: list[dict], seasons: list[str] | None = None) -> dict:
    """Mean of every metric per method, optionally for a subset of seasons."""
    rows = [r for r in races if seasons is None or r["season"] in seasons]
    keys = sorted({k.split(":", 1)[1] for k in rows[0] if ":" in k}) if rows else []
    return {m: {k: float(np.mean([r[f"{m}:{k}"] for r in rows])) for k in keys} for m in METHODS} | {"n_races": len(rows)}


PAIRS = (("model", "standings"), ("model+grid", "grid"))


def compare(races: list[dict], a: str, b: str, metric: str = "rps", seasons: list[str] | None = None,
            n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired bootstrap of ``metric(a) - metric(b)`` over races.

    Races are resampled with replacement, keeping each race's two scores
    together, so the interval reflects race-to-race luck. Returns the mean
    difference, its 95% interval, and the share of resamples in which ``a``
    had the lower value (``p_a_lower``).
    """
    rows = [r for r in races if seasons is None or r["season"] in seasons]
    diff = np.array([r[f"{a}:{metric}"] - r[f"{b}:{metric}"] for r in rows], dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"a": a, "b": b, "metric": metric, "n_races": len(rows), "diff": float(diff.mean()),
            "lo": float(lo), "hi": float(hi), "p_a_lower": float((means < 0).mean())}
