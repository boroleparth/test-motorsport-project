"""Score simulator predictions against real results, and compare strategies."""
from __future__ import annotations

import numpy as np

from .models import Grid, Track
from .race import RaceResult, simulate_race
from .strategy import Strategy, optimal_strategies


def _ranks(x: np.ndarray) -> np.ndarray:
    return x.argsort().argsort().astype(float)


def score_matrix(codes: list[str], P: np.ndarray, actual: list[str], eps: float = 1e-3) -> dict:
    """Score a position-probability matrix ``P[driver, position]`` against a finishing order.

    ``actual`` lists driver codes from first to last. Drivers missing from it are
    treated as classified last; codes not in ``codes`` are ignored.

    Metrics (lower is better unless noted):
      favourite     the driver rated most likely to win
      winner_prob   probability given to the real winner (higher is better)
      log_loss      -ln(winner_prob)
      brier         multi-class Brier score of the win probabilities
      rps           mean ranked probability score over all drivers' positions
      spearman      rank correlation of expected vs actual position (higher is better)
      podium_hits   how many of the predicted top 3 finished on the podium (higher is better)
      favourite_won 1 if the favourite won, else 0 (higher is better)
    """
    n = len(codes)
    order = [c.upper() for c in actual if c.upper() in codes]
    if not order:
        raise ValueError("no drivers in the actual result match the grid")
    order += [c for c in codes if c not in order]
    actual_pos = np.array([order.index(c) for c in codes])  # 0-based

    win = P[:, 0]
    winner = codes.index(order[0])
    onehot = np.zeros(n)
    onehot[winner] = 1.0

    cdf = np.cumsum(P, axis=1)
    step = (np.arange(n)[None, :] >= actual_pos[:, None]).astype(float)
    rps = float(((cdf - step) ** 2).sum(1).mean() / (n - 1))

    exp_pos = (P * np.arange(n)).sum(1)
    spearman = float(np.corrcoef(_ranks(exp_pos), actual_pos)[0, 1])
    predicted_top3 = set(np.argsort(exp_pos)[:3])
    actual_top3 = {codes.index(c) for c in order[:3]}
    favourite = codes[int(win.argmax())]

    return {
        "winner": order[0],
        "favourite": favourite,
        "winner_prob": float(win[winner]),
        "log_loss": float(-np.log(max(win[winner], eps))),
        "brier": float(((win - onehot) ** 2).sum()),
        "rps": rps,
        "spearman": spearman,
        "podium_hits": len(predicted_top3 & actual_top3),
        "favourite_won": int(favourite == order[0]),
    }


def score_race(result: RaceResult, actual: list[str]) -> dict:
    """Score a simulated race against the real finishing order (see :func:`score_matrix`)."""
    # floor so a winner never seen in the simulations still gives a finite loss
    return score_matrix(result.grid.codes, result.position_matrix(), actual, eps=0.5 / result.n_sims)


def backtest(grid: Grid, tracks: dict[str, Track], races: list[dict], n_sims: int = 2000, seed: int | None = None) -> dict:
    """Score predictions for many past races.

    ``races`` is a list of ``{"track": <key>, "results": [codes...], "wet": bool (optional)}``.
    Returns per-race scores plus the mean of every numeric metric.
    """
    seeds = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    scores = []
    for race in races:
        (child,) = seeds.spawn(1)
        result = simulate_race(grid, tracks[race["track"]], n_sims, seed=child, wet=race.get("wet"))
        scores.append({"track": race["track"], **score_race(result, race["results"])})
    metrics = [k for k, v in scores[0].items() if isinstance(v, (int, float))]
    return {"races": scores, "mean": {k: float(np.mean([s[k] for s in scores])) for k in metrics}}


def compare_strategies(
    grid: Grid,
    track: Track,
    driver: str,
    candidates: list[Strategy] | None = None,
    n_sims: int = 2000,
    seed: int | None = 0,
    wet: bool | None = False,
) -> list[dict]:
    """Race each candidate strategy for one driver against the same field.

    Every candidate reuses the same seed (common random numbers) so differences
    come from the strategy rather than sampling noise. Sorted best-first.
    """
    candidates = candidates or optimal_strategies(track, top=5)
    i = grid.index(driver)
    rows = []
    for strat in candidates:
        res = simulate_race(grid, track, n_sims, seed=seed, strategies={driver: strat}, wet=wet)
        pos = res.positions[:, i]
        rows.append(
            {
                "strategy": str(strat),
                "clean_air": strat.clean_air_time(track),
                "exp_pos": float(pos.mean()),
                "exp_points": float(res.points[:, i].mean()),
                "win": float((pos == 1).mean()),
                "podium": float((pos <= 3).mean()),
            }
        )
    return sorted(rows, key=lambda r: (-r["exp_points"], r["exp_pos"]))
