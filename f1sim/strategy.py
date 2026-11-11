"""Tyre model and pit-strategy search."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .models import Track

COMPOUNDS = ("S", "M", "H")
# Per-compound: pace offset vs a fresh soft (s), linear wear (s/lap), usable life (laps).
OFFSET = np.array([0.0, 0.45, 0.85])
WEAR = np.array([0.130, 0.065, 0.040])
LIFE = np.array([14.0, 26.0, 38.0])
CLIFF = 0.03  # s per (lap beyond life)^2


def tyre_penalty(compound, age, deg: float = 1.0):
    """Lap-time penalty (s) for a compound index at a given tyre age. Vectorised."""
    compound = np.asarray(compound)
    age = np.asarray(age, dtype=float)
    over = np.maximum(age - LIFE[compound] / deg, 0.0)
    return OFFSET[compound] + WEAR[compound] * deg * age + CLIFF * over**2


@dataclass(frozen=True)
class Strategy:
    compounds: tuple[str, ...]   # one per stint, e.g. ("M", "H")
    pit_laps: tuple[int, ...]    # lap at the end of which each stop is made

    def __post_init__(self) -> None:
        if len(self.compounds) != len(self.pit_laps) + 1:
            raise ValueError("need exactly one more compound than pit stops")
        if any(c not in COMPOUNDS for c in self.compounds):
            raise ValueError(f"compounds must be among {COMPOUNDS}")
        if list(self.pit_laps) != sorted(set(self.pit_laps)) or any(p < 1 for p in self.pit_laps):
            raise ValueError("pit laps must be positive and strictly increasing")

    @classmethod
    def parse(cls, text: str) -> "Strategy":
        """Parse 'M-H@22' or 'S-M-M@15,36'."""
        tyres, _, stops = text.partition("@")
        laps = tuple(int(x) for x in stops.split(",")) if stops else ()
        return cls(tuple(tyres.upper().split("-")), laps)

    def __str__(self) -> str:
        s = "-".join(self.compounds)
        return f"{s}@{','.join(map(str, self.pit_laps))}" if self.pit_laps else s

    def clean_air_time(self, track: Track) -> float:
        """Tyre + pit-stop time cost over a race with no traffic (s, relative)."""
        total, start = 0.0, 0
        for comp, end in zip(self.compounds, (*self.pit_laps, track.laps)):
            if end <= start:
                raise ValueError(f"strategy {self} does not fit a {track.laps}-lap race")
            total += float(tyre_penalty(COMPOUNDS.index(comp), np.arange(end - start), track.deg).sum())
            start = end
        return total + track.pit_loss * len(self.pit_laps)


def optimal_strategies(track: Track, top: int = 3, max_stops: int = 2) -> list[Strategy]:
    """Best dry strategies by clean-air time, one per distinct compound sequence.

    Enforces the two-compound rule. Returns up to ``top`` strategies, fastest first.
    """
    n = track.laps
    # cost[c, k] = total penalty of running k laps on compound c
    cost = np.zeros((3, n + 1))
    for c in range(3):
        cost[c, 1:] = np.cumsum(tyre_penalty(c, np.arange(n), track.deg))

    best: dict[tuple[int, ...], tuple[float, tuple[int, ...]]] = {}

    def consider(seq, pits, time):
        if seq not in best or time < best[seq][0]:
            best[seq] = (time, pits)

    for seq in product(range(3), repeat=2):
        if seq[0] == seq[1]:
            continue
        for p in range(3, n - 2):
            consider(seq, (p,), cost[seq[0], p] + cost[seq[1], n - p] + track.pit_loss)
    if max_stops >= 2:
        for seq in product(range(3), repeat=3):
            if len(set(seq)) < 2:
                continue
            for p1 in range(3, n - 5):
                for p2 in range(p1 + 3, n - 2):
                    t = cost[seq[0], p1] + cost[seq[1], p2 - p1] + cost[seq[2], n - p2] + 2 * track.pit_loss
                    consider(seq, (p1, p2), t)

    ranked = sorted(best.items(), key=lambda kv: kv[1][0])[:top]
    return [Strategy(tuple(COMPOUNDS[c] for c in seq), pits) for seq, (_, pits) in ranked]
