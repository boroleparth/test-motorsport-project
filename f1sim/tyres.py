"""Tyre wear, fuel effect and pit-stop loss measured from real laps.

Uses the lap table from ``f1sim.laps``. Within one race, a clean lap's time is
modelled as

    lap_time = driver base pace + fuel * lap + offset[c] + wear[c] * age + curve[c] * age^2

for compound c and tyre age. Fuel burn-off (and the track getting faster) and
tyre age can be told apart because tyre age resets at each stop while the lap
number keeps rising. "Clean" means green flag, dry, not lap 1, not an in or out
lap, and more than ``CLEAN_AIR`` seconds behind the car ahead.

Per-race estimates are pooled into:

* compound offsets and wear rates for the whole field (``global``),
* a wear multiplier per circuit (``tracks[key]["deg"]``),
* the fuel effect per lap,
* the time lost to a pit stop at each circuit (``tracks[key]["pit_loss"]``).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

DRY = ("SOFT", "MEDIUM", "HARD")
CLEAN_AIR = 1.5      # s behind the car ahead
MIN_LAPS = 150       # clean laps needed to fit a race
MAX_AGE = 45
MIN_COMPOUND_LAPS = 60  # a compound needs this many clean laps in a race to be measured


def clean_laps(rows: list[dict]) -> list[dict]:
    """Laps fit for measuring pace: see module docstring."""
    out = []
    for r in rows:
        if (r["lap_time"] and r["lap"] > 1 and r["status"] == "GREEN" and not r["rain"] and not r["pit_in"]
                and not r["pit_out"] and r["compound"] in DRY and r["tyre_age"] is not None
                and 1 <= r["tyre_age"] <= MAX_AGE and r["gap_ahead"] is not None
                and (r["position"] == 1 or r["gap_ahead"] > CLEAN_AIR)):
            out.append(r)
    return out


def _by_race(rows):
    races = defaultdict(list)
    for r in rows:
        races[r["season"], r["round"]].append(r)
    return races


def fit_race(laps: list[dict]) -> dict | None:
    """Fit one race. Returns fuel slope and per-compound offset, wear and curvature.

    Offsets are relative to the medium (or, if no mediums were raced, the hard).
    """
    count = defaultdict(int)
    for r in laps:
        count[r["compound"]] += 1
    laps = [r for r in laps if count[r["compound"]] >= MIN_COMPOUND_LAPS]
    if len(laps) < MIN_LAPS:
        return None
    comps = sorted({r["compound"] for r in laps}, key=DRY.index)
    if len(comps) < 2:
        return None
    base = "MEDIUM" if "MEDIUM" in comps else comps[-1]
    drivers = sorted({r["driver"] for r in laps})
    # columns: driver dummies | lap | per compound: [dummy (not base)] age age^2
    cols = len(drivers) + 1 + sum(3 if c != base else 2 for c in comps)
    X = np.zeros((len(laps), cols))
    y = np.array([r["lap_time"] for r in laps])
    di = {d: i for i, d in enumerate(drivers)}
    j0 = len(drivers) + 1
    slots = {}
    for c in comps:
        slots[c] = j0
        j0 += 3 if c != base else 2
    for k, r in enumerate(laps):
        X[k, di[r["driver"]]] = 1
        X[k, len(drivers)] = r["lap"]
        j, a = slots[r["compound"]], r["tyre_age"]
        if r["compound"] != base:
            X[k, j] = 1
            j += 1
        X[k, j], X[k, j + 1] = a, a * a
    keep = np.ones(len(y), bool)
    for _ in range(3):  # drop outliers (traffic, mistakes, damage) and refit
        beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        res = y - X @ beta
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) * 1.4826
        keep = np.abs(res) < 3 * max(mad, 0.15)
    out = {"fuel": float(beta[len(drivers)]), "n": int(keep.sum()), "base": base, "compounds": {},
           "resid_sd": float(np.std(res[keep]))}
    for c in comps:
        j = slots[c]
        off = 0.0
        if c != base:
            off, j = float(beta[j]), j + 1
        ages = [r["tyre_age"] for r, k in zip(laps, keep) if k and r["compound"] == c]
        out["compounds"][c] = {"offset": off, "wear": float(beta[j]), "curve": float(beta[j + 1]),
                               "laps": len(ages), "max_age": max(ages) if ages else 0}
    return out


def _wavg(pairs):
    pairs = [(v, w) for v, w in pairs if w > 0 and np.isfinite(v)]
    if not pairs:
        return None
    v, w = np.array(pairs).T
    return float(np.sum(v * w) / np.sum(w))


def mean_wear(c: dict, horizon: int = 20) -> float:
    """Average slope of the fitted curve over the first ``horizon`` laps of a stint, s/lap."""
    h = min(horizon, max(c["max_age"], 5))
    return c["wear"] + c["curve"] * h


def pit_losses(rows: list[dict]) -> dict[tuple, list[float]]:
    """Time lost per green-flag pit stop: in-lap + out-lap minus two normal laps."""
    by_driver = defaultdict(dict)
    for r in rows:
        by_driver[r["season"], r["round"], r["driver"]][r["lap"]] = r
    out = defaultdict(list)
    for (season, rnd, _), laps in by_driver.items():
        for n, r in laps.items():
            if not r["pit_in"] or n + 1 not in laps or n < 3:
                continue
            a, b = r, laps[n + 1]
            if a["status"] != "GREEN" or b["status"] != "GREEN" or a["rain"] or not a["lap_time"] or not b["lap_time"]:
                continue
            normal = [laps[k]["lap_time"] for k in range(n - 6, n + 8) if k in laps and k not in (n, n + 1)
                      and laps[k]["lap_time"] and laps[k]["status"] == "GREEN" and not laps[k]["pit_in"]
                      and not laps[k]["pit_out"]]
            if len(normal) >= 4:
                loss = a["lap_time"] + b["lap_time"] - 2 * float(np.median(normal))
                if 8 < loss < 60:
                    out[r["track"]].append(loss)
    return out


def fit(rows: list[dict]) -> dict:
    """Pool per-race fits into field-wide and per-circuit parameters."""
    races = {k: fit_race(v) for k, v in _by_race(clean_laps(rows)).items()}
    races = {k: v for k, v in races.items() if v}
    track_of = {(r["season"], r["round"]): r["track"] for r in rows}

    # Field-wide compound offsets relative to the medium, and wear rates, weighted by laps.
    offset, wear = {}, {}
    for c in DRY:
        wear[c] = _wavg([(mean_wear(f["compounds"][c]), f["compounds"][c]["laps"])
                         for f in races.values() if c in f["compounds"]])
    off_s = [(f["compounds"]["SOFT"]["offset"], f["compounds"]["SOFT"]["laps"])
             for f in races.values() if f["base"] == "MEDIUM" and "SOFT" in f["compounds"]]
    off_h = [(f["compounds"]["HARD"]["offset"], f["compounds"]["HARD"]["laps"])
             for f in races.values() if f["base"] == "MEDIUM" and "HARD" in f["compounds"]]
    offset = {"SOFT": _wavg(off_s), "MEDIUM": 0.0, "HARD": _wavg(off_h)}

    # Per circuit: wear multiplier = observed wear / field-wide wear, averaged over compounds.
    tracks = defaultdict(lambda: {"ratios": [], "races": 0})
    for k, f in races.items():
        t = tracks[track_of[k]]
        t["races"] += 1
        for c, v in f["compounds"].items():
            if wear.get(c) and v["laps"] >= 40:
                t["ratios"].append((mean_wear(v) / wear[c], v["laps"]))
    losses = pit_losses(rows)
    out_tracks = {}
    for key in set(tracks) | set(losses):
        t = tracks.get(key, {"ratios": [], "races": 0})
        deg = _wavg(t["ratios"])
        out_tracks[key] = {
            "deg": round(float(np.clip(deg, 0.3, 2.5)), 3) if deg is not None else None,
            "pit_loss": round(float(np.median(losses[key])), 2) if len(losses.get(key, [])) >= 5 else None,
            "races": t["races"], "stops": len(losses.get(key, [])),
        }
    return {
        "races": len(races),
        "fuel": _wavg([(f["fuel"], f["n"]) for f in races.values()]),
        "global": {c: {"offset": offset[c], "wear": wear[c]} for c in DRY},
        "tracks": out_tracks,
    }


def stint_error(rows: list[dict], offset: dict, wear: dict, fuel: float, deg: dict) -> float:
    """How well a tyre model predicts lap-time shape within clean stints (RMSE, s).

    Each stint is centred on its own mean, so only the shape (fuel + wear) is
    tested, not the driver's absolute pace.
    """
    stints = defaultdict(list)
    for r in clean_laps(rows):
        stints[r["season"], r["round"], r["driver"], r["stint"]].append(r)
    errs = []
    for laps in stints.values():
        if len(laps) < 8:
            continue
        actual = np.array([r["lap_time"] for r in laps])
        k = deg.get(laps[0]["track"]) or 1.0
        pred = np.array([offset[r["compound"]] + wear[r["compound"]] * k * r["tyre_age"] + fuel * r["lap"] for r in laps])
        e = (actual - actual.mean()) - (pred - pred.mean())
        e = e[np.abs(e) < 3]  # ignore traffic and incidents
        errs.extend(e.tolist())
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def real_strategies(rows: list[dict], top: int = 10) -> dict[str, dict]:
    """What teams actually did, per circuit: stop counts and first-stop timing.

    Uses dry races and drivers classified in the top ``top``. Returns, per track,
    the median number of stops and the median first stop as a fraction of race
    distance.
    """
    wet = {(r["season"], r["round"]) for r in rows if r["rain"]}
    stops, first, final = defaultdict(int), {}, {}
    for r in rows:
        k = (r["season"], r["round"], r["track"], r["driver"])
        if r["pit_in"] and 1 < r["lap"] < r["laps_total"]:
            stops[k] += 1
            first.setdefault(k, r["lap"] / r["laps_total"])
        if r["final_position"]:
            final[k] = r["final_position"]
    out = defaultdict(lambda: {"stops": [], "first": []})
    for k, pos in final.items():
        if pos <= top and (k[0], k[1]) not in wet and stops[k] >= 1:
            out[k[2]]["stops"].append(stops[k])
            out[k[2]]["first"].append(first[k])
    return {t: {"stops": float(np.median(v["stops"])), "first": float(np.median(v["first"])), "n": len(v["stops"])}
            for t, v in out.items() if len(v["stops"]) >= 8}


def calibrate_tracks(tracks: dict, target: dict[str, dict], pit_loss: dict[str, float] | None = None,
                     grid=tuple(np.round(np.arange(0.3, 2.51, 0.05), 2))) -> dict[str, float]:
    """Wear multiplier per circuit that makes the simulator's fastest plan match real strategies.

    For each circuit with a target, tries every value in ``grid`` and keeps the
    one whose optimal strategy has the real number of stops and the closest
    first-stop timing. ``pit_loss`` overrides the circuit's pit-lane loss first.
    """
    from dataclasses import replace

    from .strategy import optimal_strategies

    out = {}
    for key, goal in target.items():
        if key not in tracks:
            continue
        base = tracks[key]
        if pit_loss and pit_loss.get(key):
            base = replace(base, pit_loss=pit_loss[key])
        best, best_cost = None, None
        for deg in grid:
            plan = optimal_strategies(replace(base, deg=float(deg)), top=1)[0]
            n, f = len(plan.pit_laps), plan.pit_laps[0] / base.laps
            cost = 3 * abs(n - goal["stops"]) + abs(f - goal["first"]) + 0.02 * abs(deg - 1.0)
            if best_cost is None or cost < best_cost:
                best, best_cost = float(deg), cost
        out[key] = best
    return out


def safety_car_rates(rows: list[dict], prior: float = 20.0) -> tuple[float, dict[str, float]]:
    """Chance of at least one full safety car or red flag per race, per circuit.

    Each circuit's own rate is shrunk towards the field-wide rate with a prior
    worth ``prior`` races: with only a few races per circuit, the field rate
    predicts better (on 2026, a prior of 20 races scored a Brier of 0.257 against
    0.360 for raw per-circuit rates and 0.283 for the old estimates).
    Returns (field rate, {track: rate}).
    """
    lead, laps, track = {}, {}, {}
    for r in rows:
        k = (r["season"], r["round"])
        if r["position"] == 1:
            lead[k + (r["lap"],)] = r["status"]
        laps[k], track[k] = r["laps_total"], r["track"]
    had = defaultdict(list)
    for k, n in laps.items():
        had[track[k]].append(any(lead.get(k + (i,)) in ("SC", "RED") for i in range(1, n + 1)))
    field = float(np.mean([x for v in had.values() for x in v]))
    return field, {t: float((sum(v) + prior * field) / (len(v) + prior)) for t, v in had.items()}
