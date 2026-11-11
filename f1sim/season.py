"""Championship simulation over a calendar of races."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .models import Grid, Track
from .race import simulate_race


@dataclass
class SeasonResult:
    grid: Grid
    driver_points: np.ndarray       # (n_sims, n_drivers)
    team_names: list[str]
    team_points: np.ndarray         # (n_sims, n_teams)

    @staticmethod
    def _table(names, pts) -> list[dict]:
        champ = np.bincount(pts.argmax(1), minlength=len(names)) / pts.shape[0]
        rows = [
            {
                "name": n,
                "title": float(champ[i]),
                "mean_points": float(pts[:, i].mean()),
                "p05": float(np.percentile(pts[:, i], 5)),
                "p95": float(np.percentile(pts[:, i], 95)),
            }
            for i, n in enumerate(names)
        ]
        return sorted(rows, key=lambda r: -r["mean_points"])

    def drivers(self) -> list[dict]:
        return self._table(self.grid.codes, self.driver_points)

    def constructors(self) -> list[dict]:
        return self._table(self.team_names, self.team_points)


def simulate_season(
    grid: Grid,
    tracks: Iterable[Track],
    n_sims: int = 1000,
    seed: int | None = None,
    standings: dict[str, float] | None = None,
) -> SeasonResult:
    """Simulate every race in ``tracks`` and total the points.

    ``standings`` gives points already scored (driver code -> points), so the
    remainder of a season can be simulated from the current championship state.
    """
    seeds = np.random.SeedSequence(seed)
    pts = np.zeros((n_sims, len(grid.drivers)))
    for code, p in (standings or {}).items():
        pts[:, grid.index(code)] += p
    for track in tracks:
        (child,) = seeds.spawn(1)
        pts += simulate_race(grid, track, n_sims, seed=child).points

    team_names = list(grid.teams)
    team_pts = np.zeros((n_sims, len(team_names)))
    for i, d in enumerate(grid.drivers):
        team_pts[:, team_names.index(d.team)] += pts[:, i]
    return SeasonResult(grid, pts, team_names, team_pts)
