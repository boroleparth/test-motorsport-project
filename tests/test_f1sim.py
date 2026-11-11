import json
from dataclasses import replace

import numpy as np
import pytest

from f1sim import (Strategy, backtest, compare_strategies, load_grid, load_tracks,
                   optimal_strategies, score_race, simulate_race, simulate_season)
from f1sim.cli import main
from f1sim.models import Driver, Grid, Team
from f1sim.strategy import tyre_penalty


@pytest.fixture(scope="module")
def grid():
    return load_grid()


@pytest.fixture(scope="module")
def tracks():
    return load_tracks()


def test_bundled_data_loads(grid, tracks):
    assert len(grid.drivers) == 22 and len(grid.teams) == 11
    assert len(tracks) == 24
    assert all(sum(d.team == t for d in grid.drivers) == 2 for t in grid.teams)


def test_grid_validation():
    with pytest.raises(ValueError):
        Grid({"A": Team("A")}, [Driver("X", "x", "B")])
    with pytest.raises(ValueError):
        Grid({"A": Team("A")}, [Driver("X", "x", "A"), Driver("X", "y", "A")])


def test_positions_are_permutations(grid, tracks):
    res = simulate_race(grid, tracks["bahrain"], 300, seed=1)
    n = len(grid.drivers)
    assert (np.sort(res.positions, axis=1) == np.arange(1, n + 1)).all()
    assert (np.sort(res.grid_positions, axis=1) == np.arange(1, n + 1)).all()
    assert np.allclose(res.position_matrix().sum(0), 1) and np.allclose(res.position_matrix().sum(1), 1)
    assert res.points.sum(1) == pytest.approx(101)


def test_seed_reproducible(grid, tracks):
    a = simulate_race(grid, tracks["monaco"], 100, seed=7)
    b = simulate_race(grid, tracks["monaco"], 100, seed=7)
    c = simulate_race(grid, tracks["monaco"], 100, seed=8)
    assert (a.positions == b.positions).all()
    assert (a.positions != c.positions).any()


def test_retirements_classified_behind_finishers(grid, tracks):
    res = simulate_race(grid, tracks["italy"], 300, seed=2)
    finishers = (~res.dnf).sum(1, keepdims=True)
    assert (res.positions[res.dnf] > np.broadcast_to(finishers, res.dnf.shape)[res.dnf]).all()
    assert 0.02 < res.dnf.mean() < 0.25


def test_faster_car_wins_more():
    teams = {"Fast": Team("Fast", pace=0.0, reliability=0.0), "Slow": Team("Slow", pace=1.0, reliability=0.0)}
    drivers = [Driver(f"F{i}", "f", "Fast", pace=0.1, error_rate=0.0) for i in range(2)]
    drivers += [Driver(f"S{i}", "s", "Slow", pace=0.1, error_rate=0.0) for i in range(2)]
    res = simulate_race(Grid(teams, drivers), load_tracks()["bahrain"], 300, seed=3, wet=False)
    wins = (res.positions == 1).mean(0)
    assert wins[:2].sum() > 0.95
    assert not res.dnf.any()


def test_harder_overtaking_protects_pole(grid, tracks):
    def pole_to_win(overtake):
        r = simulate_race(grid, replace(tracks["bahrain"], overtake=overtake), 1500, seed=4, wet=False)
        return ((r.grid_positions == 1) & (r.positions == 1)).sum() / r.n_sims
    assert pole_to_win(3.0) > pole_to_win(0.3) + 0.05


def test_wet_flag(grid, tracks):
    assert simulate_race(grid, tracks["bahrain"], 50, seed=1, wet=True).wet.all()
    assert not simulate_race(grid, tracks["belgium"], 50, seed=1, wet=False).wet.any()


def test_tyre_model():
    assert tyre_penalty(0, 0) < tyre_penalty(1, 0) < tyre_penalty(2, 0)   # soft fastest when new
    assert tyre_penalty(0, 30) > tyre_penalty(2, 30)                      # hard fastest when old
    assert tyre_penalty(1, 10, deg=1.5) > tyre_penalty(1, 10, deg=1.0)


def test_strategy_parsing_and_validation():
    s = Strategy.parse("s-m-m@15,36")
    assert s == Strategy(("S", "M", "M"), (15, 36)) and str(s) == "S-M-M@15,36"
    for bad in ("M-H", "M-X@10", "M-H-S@30,20"):
        with pytest.raises(ValueError):
            Strategy.parse(bad)


def test_optimal_strategies(tracks):
    for track in tracks.values():
        opts = optimal_strategies(track, top=3)
        times = [s.clean_air_time(track) for s in opts]
        assert times == sorted(times)
        assert all(len(set(s.compounds)) >= 2 and s.pit_laps[-1] < track.laps for s in opts)
    # more degradation on the same circuit means more stops
    stops = lambda deg: len(optimal_strategies(replace(tracks["bahrain"], deg=deg), top=1)[0].pit_laps)
    assert stops(0.7) == 1 and stops(2.0) == 2


def test_bad_strategy_is_punished(grid, tracks):
    rows = compare_strategies(grid, tracks["bahrain"], "VER",
                              [Strategy.parse("M-H@24"), Strategy.parse("S-H@50")], n_sims=400)
    assert rows[0]["strategy"] == "M-H@24"
    assert rows[0]["exp_points"] > rows[1]["exp_points"] + 3
    with pytest.raises(ValueError):
        simulate_race(grid, tracks["bahrain"], 10, strategies={"VER": Strategy.parse("M-H@80")})


def test_season(grid, tracks):
    cal = [tracks[k] for k in ("bahrain", "monaco", "italy")]
    res = simulate_season(grid, cal, 100, seed=5)
    assert res.driver_points.sum(1) == pytest.approx(303)
    assert res.team_points.sum(1) == pytest.approx(303)
    assert sum(r["title"] for r in res.drivers()) == pytest.approx(1)
    lead = simulate_season(grid, cal, 100, seed=5, standings={"BOT": 500})
    assert lead.drivers()[0]["name"] == "BOT" and lead.drivers()[0]["title"] == 1.0


def test_scoring(grid, tracks):
    res = simulate_race(grid, tracks["bahrain"], 500, seed=6, wet=False)
    likely = [r["driver"] for r in sorted(res.summary(), key=lambda r: r["exp_pos"])]
    good, bad = score_race(res, likely), score_race(res, likely[::-1])
    assert good["spearman"] == pytest.approx(1) and bad["spearman"] == pytest.approx(-1)
    assert good["rps"] < bad["rps"] and good["brier"] < bad["brier"] and good["log_loss"] < bad["log_loss"]
    assert good["podium_hits"] == 3 and np.isfinite(bad["log_loss"])
    partial = score_race(res, likely[:5] + ["ZZZ"])   # unknown code ignored, rest classified last
    assert partial["winner"] == likely[0]
    with pytest.raises(ValueError):
        score_race(res, ["ZZZ"])


def test_backtest_and_cli(grid, tracks, tmp_path, capsys):
    races = [{"track": "bahrain", "results": ["NOR", "VER", "PIA"], "wet": False},
             {"track": "monaco", "results": ["LEC", "NOR", "PIA"]}]
    out = backtest(grid, tracks, races, n_sims=200, seed=1)
    assert len(out["races"]) == 2 and 0 <= out["mean"]["rps"] <= 1
    f = tmp_path / "results.json"
    f.write_text(json.dumps(races))
    for argv in (["tracks"], ["race", "bahrain", "--sims", "50", "--seed", "1", "--strategy", "VER=M-H@24"],
                 ["season", "--sims", "20", "--start", "qatar"], ["strategy", "monaco", "LEC", "--sims", "50"],
                 ["backtest", str(f), "--sims", "50"]):
        main(argv)
        assert capsys.readouterr().out.strip()
    with pytest.raises(SystemExit):
        main(["race", "nowhere"])
