"""Lap-by-lap Monte Carlo race engine, vectorised across simulations."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .models import Grid, Track
from .strategy import COMPOUNDS, Strategy, optimal_strategies, tyre_penalty

POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1])

MAX_STOPS = 3
NO_PIT = 10**6
FUEL_PER_LAP = 0.03      # s of lap time per lap of fuel on board
FOLLOW_GAP = 0.25        # minimum gap behind a car you failed to pass, s
PASS_SCALE = 0.25        # softness of the pass-probability curve, s
QUALI_SIGMA = 0.22       # one-lap noise, s
GRID_SPACING = 0.3       # time gap per grid slot at the start, s
SC_LAP_FACTOR = 1.4      # safety-car lap time relative to base lap
SC_PIT_FACTOR = 0.5      # fraction of normal pit loss when stopping under SC
SC_GAP = 0.5             # gap per position once the field is bunched, s
WET_LAP_FACTOR = 0.10    # wet laps are this much slower
WET_NOISE = 1.8          # lap-time noise multiplier in the wet
WET_ERROR = 2.5          # driver-error multiplier in the wet


@dataclass
class RaceResult:
    """Outcome of ``n_sims`` simulated races. Arrays are (n_sims, n_drivers)."""

    grid: Grid
    track: Track
    positions: np.ndarray   # classified finishing position, 1-based
    grid_positions: np.ndarray  # starting position, 1-based
    dnf: np.ndarray         # bool
    wet: np.ndarray         # (n_sims,) bool

    @property
    def n_sims(self) -> int:
        return self.positions.shape[0]

    @property
    def points(self) -> np.ndarray:
        table = np.zeros(self.positions.shape[1] + 1)
        k = min(len(POINTS), self.positions.shape[1])
        table[1 : k + 1] = POINTS[:k]
        return table[self.positions]

    def position_matrix(self) -> np.ndarray:
        """P[d, p] = probability driver d is classified in position p+1."""
        n = self.positions.shape[1]
        out = np.zeros((n, n))
        for d in range(n):
            out[d] = np.bincount(self.positions[:, d] - 1, minlength=n) / self.n_sims
        return out

    def summary(self) -> list[dict]:
        """Per-driver probabilities, sorted by expected points."""
        pos, pts = self.positions, self.points
        rows = [
            {
                "driver": d.code,
                "team": d.team,
                "win": float((pos[:, i] == 1).mean()),
                "podium": float((pos[:, i] <= 3).mean()),
                "points": float((pos[:, i] <= 10).mean()),
                "pole": float((self.grid_positions[:, i] == 1).mean()),
                "dnf": float(self.dnf[:, i].mean()),
                "exp_pos": float(pos[:, i].mean()),
                "exp_points": float(pts[:, i].mean()),
            }
            for i, d in enumerate(self.grid.drivers)
        ]
        return sorted(rows, key=lambda r: (-r["exp_points"], r["exp_pos"]))


@lru_cache(maxsize=None)
def _default_options(track: Track) -> tuple[tuple[Strategy, ...], tuple[float, ...]]:
    options = optimal_strategies(track, top=3)
    times = np.array([s.clean_air_time(track) for s in options])
    w = np.exp(-(times - times.min()) / 1.5)
    return tuple(options), tuple(w / w.sum())


def _pad(strategy: Strategy, laps: int) -> tuple[list[int], list[int]]:
    if len(strategy.pit_laps) > MAX_STOPS:
        raise ValueError(f"at most {MAX_STOPS} stops supported")
    if strategy.pit_laps and strategy.pit_laps[-1] >= laps:
        raise ValueError(f"strategy {strategy} does not fit a {laps}-lap race")
    pits = list(strategy.pit_laps) + [NO_PIT] * (MAX_STOPS + 1 - len(strategy.pit_laps))
    comps = [COMPOUNDS.index(c) for c in strategy.compounds]
    return pits, comps + [comps[-1]] * (MAX_STOPS + 1 - len(comps))


def _assign_strategies(rng, grid, track, S, overrides):
    options, weights = _default_options(track)
    padded = [_pad(s, track.laps) for s in options]
    opt_pits = np.array([p for p, _ in padded])
    opt_comps = np.array([c for _, c in padded])
    N = len(grid.drivers)
    choice = rng.choice(len(options), size=(S, N), p=weights)
    pits, comps = opt_pits[choice], opt_comps[choice]
    jitter = rng.integers(-2, 3, size=pits.shape)
    pits = np.where(pits < NO_PIT, np.clip(pits + jitter, 1, track.laps - 1), NO_PIT)
    for k in range(1, MAX_STOPS + 1):  # keep stops strictly increasing after jitter
        pits[..., k] = np.where(pits[..., k] < NO_PIT, np.maximum(pits[..., k], pits[..., k - 1] + 1), NO_PIT)
    for code, strat in (overrides or {}).items():
        i = grid.index(code)
        p, c = _pad(strat, track.laps)
        pits[:, i, :], comps[:, i, :] = p, c
    return pits, comps


def _safety_cars(rng, track, S, wet):
    """Return (sc_active, sc_start) boolean arrays of shape (S, laps)."""
    L = track.laps
    lam = -np.log(max(1.0 - track.sc_prob, 1e-9)) * np.where(wet, 1.5, 1.0)
    count = rng.poisson(lam)
    laps = np.arange(1, L + 1)[None, :]
    active = np.zeros((S, L), dtype=bool)
    for k in range(3):
        start = rng.integers(2, max(L - 3, 3), size=S)[:, None]
        dur = rng.integers(3, 6, size=S)[:, None]
        active |= (count[:, None] > k) & (laps >= start) & (laps < start + dur)
    start_mask = active.copy()
    start_mask[:, 1:] &= ~active[:, :-1]
    return active, start_mask


def simulate_race(
    grid: Grid,
    track: Track,
    n_sims: int = 2000,
    seed: int | None = None,
    strategies: dict[str, Strategy] | None = None,
    wet: bool | None = None,
) -> RaceResult:
    """Simulate qualifying and the race ``n_sims`` times.

    ``strategies`` pins specific drivers (by code) to a fixed strategy; everyone
    else samples from the track's near-optimal strategies with pit-lap jitter.
    ``wet`` forces wet/dry; by default each simulation draws from ``track.rain_prob``.
    """
    rng = np.random.default_rng(seed)
    drivers = grid.drivers
    teams = [grid.teams[d.team] for d in drivers]
    S, N, L = n_sims, len(drivers), track.laps
    rows = np.arange(S)

    car = np.array([t.pace for t in teams])
    drv = np.array([d.pace for d in drivers])
    sigma = np.array([d.consistency for d in drivers])
    wet_skill = np.array([d.wet for d in drivers])
    pit_delta = np.array([t.pit_mean - 2.5 for t in teams])
    pit_sigma = np.array([t.pit_sigma for t in teams])

    is_wet = np.full(S, bool(wet)) if wet is not None else rng.random(S) < track.rain_prob
    w = is_wet[:, None]

    # --- qualifying -------------------------------------------------------
    q = car + np.array([d.quali_pace for d in drivers]) + w * wet_skill
    q = q + rng.normal(0, QUALI_SIGMA, (S, N)) * np.where(w, 2.0, 1.0)
    grid_pos = q.argsort(1).argsort(1)

    # --- retirements ------------------------------------------------------
    p_mech = np.array([t.reliability for t in teams])
    p_err = np.minimum(np.array([d.error_rate for d in drivers]) * np.where(w, WET_ERROR, 1.0), 0.9)
    dnf = rng.random((S, N)) < 1 - (1 - p_mech) * (1 - p_err)
    dnf_lap = np.where(dnf, rng.integers(1, L + 1, (S, N)), L + 1)  # retires during this lap

    sc_active, sc_start = _safety_cars(rng, track, S, is_wet)
    pits, comps = _assign_strategies(rng, grid, track, S, strategies)

    stint = np.zeros((S, N), dtype=int)
    age = np.zeros((S, N))
    t = grid_pos * GRID_SPACING

    for lap in range(1, L + 1):
        active = lap < dnf_lap
        sc = sc_active[:, lap - 1]
        scc = sc[:, None]
        comp = np.take_along_axis(comps, stint[..., None], 2)[..., 0]
        next_pit = np.take_along_axis(pits, stint[..., None], 2)[..., 0]

        lt = (
            track.base_lap + car + drv
            + tyre_penalty(comp, age, track.deg)
            + FUEL_PER_LAP * (L - lap)
            + rng.normal(0, 1, (S, N)) * sigma * np.where(w, WET_NOISE, 1.0)
            + w * (WET_LAP_FACTOR * track.base_lap + wet_skill)
        )
        if lap == 1:  # standing start and first-lap scramble
            lt += 4.0 + rng.normal(0, 0.8, (S, N))
        lt = np.where(scc, track.base_lap * SC_LAP_FACTOR, lt)

        opportunistic = sc_start[:, lap - 1][:, None] & (next_pit > lap) & (next_pit - lap <= 6)
        pit = active & (lap < L) & ((next_pit == lap) | opportunistic)
        loss = track.pit_loss * np.where(scc, SC_PIT_FACTOR, 1.0) + pit_delta + rng.exponential(1.0, (S, N)) * pit_sigma
        new = np.where(active, t + lt + pit * loss, np.inf)

        if lap > 1:
            # Cars are processed in start-of-lap order; a car catching the one
            # ahead either completes the pass or is held FOLLOW_GAP behind it.
            order = np.argsort(t, axis=1)
            u = rng.random((S, N))
            for p in range(1, N):
                a, b = order[:, p - 1], order[:, p]
                ta, tb = new[rows, a], new[rows, b]
                fight = np.isfinite(ta) & np.isfinite(tb) & ~sc & ~pit[rows, a] & ~pit[rows, b] & (tb < ta + FOLLOW_GAP)
                gain = np.where(fight, ta, 0.0) + FOLLOW_GAP - np.where(fight, tb, 0.0)
                p_pass = 1.0 / (1.0 + np.exp(-(gain - track.overtake) / PASS_SCALE))
                held = fight & (u[:, p] >= p_pass)
                new[rows, b] = np.where(held, ta + FOLLOW_GAP, tb)

        if sc.any():  # bunch the field behind the safety car, preserving order
            rank = new.argsort(1).argsort(1)
            target = new.min(1, keepdims=True) + rank * SC_GAP
            new = np.where(scc & active, np.minimum(new, target), new)

        stint = stint + pit
        age = np.where(pit, 0.0, age + 1.0)
        t = new

    # Classification: finishers by time, then retirements by distance covered.
    key = np.where(dnf, 1e7 - (dnf_lap - 1) * 1e3 + rng.random((S, N)), t)
    positions = key.argsort(1).argsort(1) + 1
    return RaceResult(grid, track, positions, grid_pos + 1, dnf, is_wet)
