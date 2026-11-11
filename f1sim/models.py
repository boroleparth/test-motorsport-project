"""Data model: teams, drivers, tracks, and JSON loaders."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class Team:
    name: str
    pace: float = 0.0          # car deficit to the fastest car, s/lap (lower is faster)
    reliability: float = 0.05  # probability of a mechanical DNF per race
    pit_mean: float = 2.5      # typical stationary time, s
    pit_sigma: float = 0.5     # scale of slow-stop tail, s


@dataclass(frozen=True)
class Driver:
    code: str
    name: str
    team: str
    pace: float = 0.2          # race-pace deficit to the best driver, s/lap
    quali: float | None = None  # one-lap deficit, s (defaults to pace)
    consistency: float = 0.35  # lap-time standard deviation, s
    wet: float = 0.0           # extra s/lap in the wet (negative = wet specialist)
    error_rate: float = 0.04   # probability of a race-ending driver error per race

    @property
    def quali_pace(self) -> float:
        return self.pace if self.quali is None else self.quali


@dataclass(frozen=True)
class Track:
    key: str
    name: str
    laps: int
    base_lap: float            # reference dry lap for the fastest car on fresh softs, s
    pit_loss: float            # total time lost for a normal green-flag stop, s
    overtake: float = 0.8      # pace advantage (s/lap) needed for a 50% pass chance per lap
    sc_prob: float = 0.5       # probability of at least one safety car
    rain_prob: float = 0.1     # probability of a wet race
    deg: float = 1.0           # tyre degradation multiplier


@dataclass
class Grid:
    teams: dict[str, Team]
    drivers: list[Driver] = field(default_factory=list)

    def __post_init__(self) -> None:
        codes = [d.code for d in self.drivers]
        if len(set(codes)) != len(codes):
            raise ValueError("duplicate driver codes in grid")
        for d in self.drivers:
            if d.team not in self.teams:
                raise ValueError(f"driver {d.code} references unknown team {d.team!r}")

    @property
    def codes(self) -> list[str]:
        return [d.code for d in self.drivers]

    def index(self, code: str) -> int:
        try:
            return self.codes.index(code.upper())
        except ValueError:
            raise KeyError(f"unknown driver {code!r}; known: {', '.join(self.codes)}") from None


def _read(path: str | Path | None, default_name: str) -> dict:
    if path is None:
        text = resources.files("f1sim").joinpath("data", default_name).read_text()
    else:
        text = Path(path).read_text()
    return json.loads(text)


def load_grid(path: str | Path | None = None) -> Grid:
    """Load teams and drivers from JSON (bundled illustrative grid if no path)."""
    raw = _read(path, "grid.json")
    teams = {t["name"]: Team(**t) for t in raw["teams"]}
    return Grid(teams=teams, drivers=[Driver(**d) for d in raw["drivers"]])


def load_tracks(path: str | Path | None = None) -> dict[str, Track]:
    """Load the calendar as an ordered {key: Track} dict."""
    raw = _read(path, "tracks.json")
    return {t["key"]: Track(**t) for t in raw["tracks"]}
