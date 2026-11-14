"""Live win probability from the state of a race.

A conditional logit: each running car gets a score from six features of the
race state, and the win probabilities are the softmax of the scores across the
field. Features (``features``):

1. gap to the leader per lap remaining, s
2. log of running position
3. log of running position x share of the race remaining
4. gap per lap remaining x share of the race remaining
5. current pace relative to the field median, s/lap (from ``f1sim.live``)
6. pit-stop time still owed compared with the car owing least, per lap remaining

Tested on every 5th lap of the 2025 and 2026 races after training on 2023-2024
(``train_and_save`` writes the result to data/winprob.json): log loss 0.79 and
the favourite won 72% of the time, against 0.95 and 68% for running the
simulator (``f1sim.live.simulate_from``) from the same states, and 1.98 and 71%
for "the leader wins". Adding the simulator's
odds as a sixth feature did not improve it. The simulator is still what the pit
advisor uses, because only it can answer "what if this car pits now?".
"""
from __future__ import annotations

import json
from collections import defaultdict
from importlib import resources
from pathlib import Path

import numpy as np

from .live import RaceState, plan_options, race_state
from .models import Grid, Track

WEIGHTS = Path(__file__).resolve().parent / "data" / "winprob.json"


def features(state: RaceState, track: Track | None = None) -> tuple[np.ndarray, list[str]]:
    """Feature matrix for the running cars, in running order.

    With ``track``, adds a sixth feature: the time each car still owes in pit
    stops compared with the car owing least (stops left in its best remaining
    plan x the circuit's pit loss), per lap remaining. Without it, a car that has
    not pitted yet looks like a leader during the pit cycle.
    """
    run = sorted([c for c in state.cars if c.running], key=lambda c: c.gap)
    rem, frac = state.remaining, state.remaining / state.laps_total
    med = float(np.median([c.pace for c in run])) if run else 0.0
    rows = [[c.gap / (rem + 1), np.log(p), np.log(p) * frac, c.gap / (rem + 1) * frac, c.pace - med]
            for p, c in enumerate(run, 1)]
    if track is not None and run:
        deg = state.deg or track.deg
        owed = np.array([len(plan_options(c, state, track, deg, top=1)[0][0].pits) for c in run], float)
        owed = (owed - owed.min()) * track.pit_loss / (rem + 1)
        rows = [r + [o] for r, o in zip(rows, owed)]
    return np.array(rows), [c.code for c in run]


def _softmax(s: np.ndarray) -> np.ndarray:
    e = np.exp(s - s.max())
    return e / e.sum()


def fit(samples: list[tuple[np.ndarray, int]], lam: float = 1.0, iters: int = 200) -> np.ndarray:
    """Fit weights by damped Newton steps with a ridge penalty ``lam``.

    ``samples`` are (feature matrix, index of the eventual winner) pairs.
    """
    k = samples[0][0].shape[1]
    w = np.zeros(k)

    def nll(w):
        out = lam / 2 * w @ w
        for X, y in samples:
            s = X @ w
            m = s.max()
            out -= s[y] - m - np.log(np.exp(s - m).sum())
        return out

    cur = nll(w)
    for _ in range(iters):
        g, H = -lam * w, -lam * np.eye(k)
        for X, y in samples:
            p = _softmax(X @ w)
            g += X[y] - p @ X
            H -= (X.T * p) @ X - np.outer(p @ X, p @ X)
        step, t = np.linalg.solve(H, g), 1.0
        while t > 1e-6:
            new = nll(w - t * step)
            if new <= cur:
                break
            t /= 2
        w, done, cur = w - t * step, abs(cur - new) < 1e-10, new
        if done:
            break
    return w


def samples_from(races: dict[tuple, list[dict]], tracks: dict[str, Track], grids: dict[tuple, Grid] | None = None,
                 every: int = 5) -> list[tuple[np.ndarray, int]]:
    """(features, winner index) at every ``every`` laps of each race in ``races``."""
    out = []
    for key, rows in sorted(races.items()):
        track = tracks.get(rows[0]["track"])
        winner = next((r["driver"] for r in rows if r["final_position"] == 1), None)
        if track is None or winner is None:
            continue
        laps_total = max(r["laps_total"] for r in rows)
        for lap in range(every, laps_total, every):
            X, codes = features(race_state(rows, lap, track, (grids or {}).get(key)), track)
            if winner in codes:
                out.append((X, codes.index(winner)))
    return out


def by_race(rows: list[dict]) -> dict[tuple, list[dict]]:
    races = defaultdict(list)
    for r in rows:
        races[r["season"], r["round"]].append(r)
    return races


def pre_race_grids(races: dict[tuple, list[dict]]) -> dict[tuple, Grid]:
    """Ratings each race would have had beforehand: fitted on earlier rounds of its season.

    Used as the prior for drivers' pace in the opening laps.
    """
    from .datasource import bundled_seasons, load_season
    from .fit import fit_grid
    from .models import load_grid

    base, seasons, out = load_grid(), {}, {}
    for season, rnd in races:
        if rnd <= 1 or str(season) not in bundled_seasons():
            continue
        if season not in seasons:
            seasons[season] = load_season(season)
        try:
            out[season, rnd] = fit_grid(seasons[season], base, through_round=rnd - 1)
        except ValueError:
            pass
    return out


def evaluate(w: np.ndarray, samples: list[tuple[np.ndarray, int]]) -> dict:
    """Log loss, Brier score and how often the favourite won."""
    ll, br, fav = [], [], []
    for X, y in samples:
        p = _softmax(X @ w)
        ll.append(-np.log(max(p[y], 1e-3)))
        br.append(float(((p - np.eye(len(p))[y]) ** 2).sum()))
        fav.append(int(np.argmax(p) == y))
    return {"log_loss": float(np.mean(ll)), "brier": float(np.mean(br)), "favourite_won": float(np.mean(fav)),
            "states": len(samples)}


def load_weights() -> np.ndarray:
    return np.array(json.loads(resources.files("f1sim").joinpath("data", "winprob.json").read_text())["weights"])


def win_probabilities(state: RaceState, w: np.ndarray | None = None, track: Track | None = None) -> dict[str, float]:
    """Each running car's chance to win from ``state`` (pass ``track`` when the weights use pit stops owed)."""
    w = load_weights() if w is None else w
    if len(w) == 6 and track is None:
        raise ValueError("these weights use pit stops owed: pass the race's track")
    X, codes = features(state, track if len(w) == 6 else None)
    if not codes:
        return {}
    p = _softmax(X @ w)
    return dict(zip(codes, p.tolist()))


def train_and_save(laps: list[dict], tracks: dict[str, Track], test_years=(2025, 2026), path: Path = WEIGHTS) -> dict:
    """Report a held-out test, then refit on every season and save the weights."""
    races = by_race(laps)
    train = {k: v for k, v in races.items() if k[0] not in test_years}
    test = {k: v for k, v in races.items() if k[0] in test_years}
    grids = pre_race_grids(races)
    s_train, s_test = samples_from(train, tracks, grids), samples_from(test, tracks, grids)
    w_test = fit(s_train)
    report = {"trained_on": sorted({k[0] for k in train}), "tested_on": sorted({k[0] for k in test}),
              "test": evaluate(w_test, s_test)}
    w = fit(s_train + s_test)
    report["weights"] = w.tolist()
    report["fitted_on"] = sorted({k[0] for k in races})
    path.write_text(json.dumps(report, indent=1))
    return report
