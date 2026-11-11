"""Fit team and driver ratings from real results (see ``datasource.fetch_season``).

Pace is measured, not guessed:

* one-lap pace from each driver's best qualifying lap, as a fraction of the
  fastest lap of that session;
* race pace from the finishing gap to the winner per lap (lapped cars are
  placed by laps lost).

Each is split into ``team + driver`` by robust (median-like) ridge regression, with recent
races weighted more heavily because cars develop through a season. Retirement
rates are shrunk towards the field average so one unlucky race doesn't dominate.
"""
from __future__ import annotations

import numpy as np

from .models import Driver, Grid, Team

REF_LAP = 88.0        # seconds; converts fractional deficits to s/lap
MAX_DEFICIT = 0.04    # deficits beyond 4% are treated as outliers (crash, rain, no lap)
TEAM_NAMES = {"RB F1 Team": "Racing Bulls"}


def team_name(api_name: str) -> str:
    return TEAM_NAMES.get(api_name, api_name.replace(" F1 Team", ""))


def _observations(races: list[dict]):
    """Yield (round, driver, team, quali_deficit | None, race_deficit | None)."""
    for race in races:
        entries = race["entries"]
        q_best = min((e["quali"] for e in entries.values() if e["quali"]), default=None)
        winner = entries[race["results"][0]]
        for code, e in entries.items():
            q = e["quali"] / q_best - 1 if e["quali"] and q_best else None
            r = None
            if e["finished"] and winner["time"]:
                if e["time"]:
                    r = (e["time"] - winner["time"]) / winner["time"]
                elif e["laps"] < race["laps"]:
                    r = (race["laps"] - e["laps"] + 0.5) / race["laps"]
            yield race["round"], code, team_name(e["team"]), q, r


def _decompose(obs, teams, drivers, weights, ridge=3.0):
    """Solve deficit = team + driver by robust weighted ridge regression.

    Returns (team_effect, driver_effect) with the best driver in each team at 0
    and the best team at 0.
    """
    ti = {t: i for i, t in enumerate(teams)}
    di = {d: len(teams) + i for i, d in enumerate(drivers)}
    A = np.zeros((len(obs), len(teams) + len(drivers)))
    y = np.zeros(len(obs))
    w = np.zeros(len(obs))
    for k, (rnd, code, team, value) in enumerate(obs):
        A[k, ti[team]] = A[k, di[code]] = 1.0
        y[k], w[k] = min(value, MAX_DEFICIT), weights[rnd]
    penalty = np.zeros_like(A[0])
    penalty[len(teams):] = ridge * w.mean()       # shrink driver effects only
    # Iteratively reweighted least squares approximates a median fit, so one
    # bad afternoon (damage, a penalty, a wrong tyre call) doesn't move a rating.
    robust = np.ones(len(obs))
    for _ in range(25):
        sw = np.sqrt(w * robust)[:, None]
        lhs = (A * sw).T @ (A * sw) + np.diag(penalty + 1e-9)
        beta = np.linalg.solve(lhs, (A * sw).T @ (y * sw[:, 0]))
        robust = 1.0 / np.maximum(np.abs(y - A @ beta), 1e-3)
        robust /= robust.mean()
    return dict(zip(teams, beta[: len(teams)])), dict(zip(drivers, beta[len(teams):]))


def fit_grid(season: dict, base: Grid | None = None, through_round: int | None = None,
             half_life: float = 4.0, race_weight: float = 0.3,
             results_weight: float = 0.4) -> Grid:
    """Build a Grid from a season file.

    ``through_round`` uses only races up to that round (for out-of-sample tests).
    ``half_life`` is the number of rounds after which a race counts half as much.
    ``race_weight`` blends race-gap pace (1.0) with qualifying pace (0.0).
    ``results_weight`` pulls each driver's race pace towards the order implied by
    their average finishing position (0 = measured gaps only, 1 = finishing order only).
    ``base`` supplies values that results cannot identify (lap-time consistency,
    pit-stop times); defaults are used for anyone not in it.
    """
    races = [r for r in season["races"] if through_round is None or r["round"] <= through_round]
    if not races:
        raise ValueError("no races to fit")
    last = races[-1]
    weights = {r["round"]: 0.5 ** ((last["round"] - r["round"]) / half_life) for r in races}

    # Line-up: whoever entered the most recent race, at their team in that race.
    lineup = {code: team_name(e["team"]) for code, e in last["entries"].items()}
    teams = sorted(set(lineup.values()))

    rows = list(_observations(races))
    all_teams = sorted({t for _, _, t, _, _ in rows})
    all_drivers = sorted({d for _, d, _, _, _ in rows})
    effects = []
    for idx in (3, 4):  # qualifying, race
        obs = [(r[0], r[1], r[2], r[idx]) for r in rows if r[idx] is not None]
        effects.append(_decompose(obs, all_teams, all_drivers, weights))
    (q_team, q_drv), (r_team, r_drv) = effects

    def normalise(team_eff, drv_eff):
        team_eff = {t: team_eff[t] for t in teams}
        drv_eff = {d: drv_eff[d] for d in lineup}
        for t in teams:  # move the within-team baseline onto the team
            shift = min(drv_eff[d] for d in lineup if lineup[d] == t)
            team_eff[t] += shift
            for d in lineup:
                if lineup[d] == t:
                    drv_eff[d] -= shift
        best = min(team_eff.values())
        return ({t: (v - best) * REF_LAP for t, v in team_eff.items()},
                {d: v * REF_LAP for d, v in drv_eff.items()})

    q_team, q_drv = normalise(q_team, q_drv)
    r_team, r_drv = normalise(r_team, r_drv)
    blend = lambda q, r: (1 - race_weight) * q + race_weight * r

    # Retirements, shrunk towards the field rate with a prior worth 10 starts.
    starts, dnfs = {}, {}
    for race in races:
        for code, e in race["entries"].items():
            if e["status"] == "Did not start":
                continue
            for k in (code, team_name(e["team"])):
                starts[k] = starts.get(k, 0) + weights[race["round"]]
                dnfs[k] = dnfs.get(k, 0) + weights[race["round"]] * (not e["finished"])
    field = sum(dnfs[d] for d in all_drivers if d in dnfs) / max(sum(starts[d] for d in all_drivers if d in starts), 1e-9)
    rate = lambda k: (dnfs.get(k, 0) + 10 * field) / (starts.get(k, 0) + 10)

    base_teams = base.teams if base else {}
    base_drivers = {d.code: d for d in base.drivers} if base else {}
    names = {**{c: d.name for c, d in base_drivers.items()}, **season.get("drivers", {})}

    # Race pace per driver (car + driver), from measured gaps.
    total = {c: blend(q_team[lineup[c]], r_team[lineup[c]]) + blend(q_drv[c], r_drv[c]) for c in lineup}
    if results_weight > 0:
        # Finishing positions say who beats whom more reliably than time gaps do,
        # but not by how much. Keep the measured spread of pace and re-deal it in
        # order of average finishing position, then blend the two.
        num, den = {}, {}
        for race in races:
            finishers = [c for c in race["results"] if race["entries"][c]["finished"]]
            for pos, code in enumerate(finishers):
                num[code] = num.get(code, 0) + weights[race["round"]] * pos
                den[code] = den.get(code, 0) + weights[race["round"]]
        by_gap = sorted(lineup, key=lambda c: total[c])
        avg = {c: num[c] / den[c] if den.get(c) else float(by_gap.index(c)) for c in lineup}
        ladder = sorted(total.values())
        dealt = {c: ladder[i] for i, c in enumerate(sorted(lineup, key=lambda c: avg[c]))}
        total = {c: (1 - results_weight) * total[c] + results_weight * dealt[c] for c in lineup}
    team_pace = {t: min(total[c] for c in lineup if lineup[c] == t) for t in teams}
    best = min(team_pace.values())
    team_pace = {t: v - best for t, v in team_pace.items()}
    total = {c: v - best for c, v in total.items()}

    out_teams = {}
    for t in sorted(teams, key=team_pace.get):
        b = base_teams.get(t, Team(t))
        out_teams[t] = Team(t, pace=round(float(team_pace[t]), 3),
                            reliability=round(float(0.6 * rate(t)), 3), pit_mean=b.pit_mean, pit_sigma=b.pit_sigma)
    # The simulator adds team.pace to both sessions, so the driver's qualifying
    # number carries the difference between the car's one-lap and race pace.
    out_drivers = []
    for code in sorted(lineup, key=lambda c: (list(out_teams).index(lineup[c]), total[c])):
        t = lineup[code]
        b = base_drivers.get(code)
        out_drivers.append(Driver(
            code=code, name=names.get(code, code), team=t,
            pace=round(float(total[code] - team_pace[t]), 3),
            quali=round(float(q_drv[code] + q_team[t] - team_pace[t]), 3),
            consistency=b.consistency if b else 0.35, wet=0.0,
            error_rate=round(float(0.4 * rate(code)), 3),
        ))
    return Grid(out_teams, out_drivers)


def grid_to_json(grid: Grid, note: str = "") -> dict:
    return {
        "_note": note,
        "teams": [{k: (float(v) if isinstance(v, float) else v) for k, v in vars(t).items()} for t in grid.teams.values()],
        "drivers": [{k: (float(v) if isinstance(v, float) else v) for k, v in vars(d).items()} for d in grid.drivers],
    }


def walk_forward(season: dict, tracks: dict, base: Grid | None = None, start: int = 5,
                 n_sims: int = 2000, seed: int | None = 0, **fit_kwargs) -> dict:
    """Honest test of the whole pipeline: predict each race using only earlier races.

    For every round after ``start``, ratings are fitted on the rounds before it
    and the prediction is scored against what actually happened.
    """
    from .evaluate import score_race
    from .race import simulate_race

    scores = []
    seeds = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    for race in season["races"]:
        if race["round"] <= start or race["track"] not in tracks:
            continue
        grid = fit_grid(season, base, through_round=race["round"] - 1, **fit_kwargs)
        (child,) = seeds.spawn(1)
        result = simulate_race(grid, tracks[race["track"]], n_sims, seed=child)
        scores.append({"track": race["track"], **score_race(result, race["results"])})
    if not scores:
        raise ValueError("no races to test after the starting round")
    metrics = [k for k, v in scores[0].items() if isinstance(v, (int, float))]
    return {"races": scores, "mean": {k: float(np.mean([s[k] for s in scores])) for k in metrics}}
