import json
from dataclasses import replace

import numpy as np
import pytest

from f1sim import (Strategy, backtest, fit_grid, load_season, walk_forward, compare_strategies, load_grid, load_tracks,
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
    assert len(tracks) == 33
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


def test_harder_overtaking_traps_a_fast_car(tracks):
    # One driver is quickest in the race but qualifies at the back.
    teams = {"T": Team("T", reliability=0.0)}
    drivers = [Driver("ACE", "a", "T", pace=0.0, quali=2.0, error_rate=0.0)]
    drivers += [Driver(f"D{i}", "d", "T", pace=0.6, quali=0.0, error_rate=0.0) for i in range(9)]
    def ace_wins(overtake):
        track = replace(tracks["spain"], overtake=overtake, sc_prob=0.0)
        r = simulate_race(Grid(teams, drivers), track, 500, seed=4, wet=False)
        return (r.positions[:, 0] == 1).mean()
    assert ace_wins(0.3) > ace_wins(3.0) + 0.3


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
        assert all(a <= b + 1e-6 for a, b in zip(times, times[1:]))
        assert all(len(set(s.compounds)) >= 2 and s.pit_laps[-1] < track.laps for s in opts)
    # more degradation on the same circuit means more stops
    stops = lambda deg: len(optimal_strategies(replace(tracks["bahrain"], deg=deg), top=1)[0].pit_laps)
    assert stops(0.7) == 1 and stops(2.0) == 2


def test_bad_strategy_is_punished(grid, tracks):
    best = optimal_strategies(tracks["bahrain"], top=1)[0]
    rows = compare_strategies(grid, tracks["bahrain"], "VER", [best, Strategy.parse("S-H@50")], n_sims=400)
    assert rows[0]["strategy"] == str(best)
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
             {"track": "monaco", "results": ["LEC", "NOR", "PIA"], "round": 2}]
    out = backtest(grid, tracks, races, n_sims=200, seed=1)
    assert len(out["races"]) == 2 and 0 <= out["mean"]["rps"] <= 1
    f = tmp_path / "results.json"
    f.write_text(json.dumps(races))
    for argv in (["tracks"], ["race", "bahrain", "--sims", "50", "--seed", "1", "--strategy", "VER=M-H@24"],
                 ["season", "--sims", "20", "--start", "qatar"], ["season", "--sims", "20"],
                 ["season", "--sims", "5", "--full"], ["fit", "--through", "8", "--out", str(tmp_path / "g.json")],
                 ["race", "monaco", "--sims", "20", "--grid", str(tmp_path / "g.json")],
                 ["backtest", "--walk-forward", "--start-round", "18", "--sims", "50"],
                 ["benchmark", "--seasons", "2026", "--start-round", "18", "--sims", "50", "--holdout", "2026"],
                 ["race", "monaco", "--sims", "20", "--start-grid", "VER,NOR"], ["strategy", "monaco", "LEC", "--sims", "50"],
                 ["backtest", str(f), "--sims", "50"]):
        main(argv)
        assert capsys.readouterr().out.strip()
    with pytest.raises(SystemExit):
        main(["race", "nowhere"])


def test_fit_from_real_results(grid):
    season = load_season()
    fitted = fit_grid(season, grid)
    last = season["races"][-1]
    assert set(fitted.codes) == set(last["entries"]) and len(fitted.teams) == 11
    assert min(t.pace for t in fitted.teams.values()) < 0.2
    assert all(0 < t.reliability < 0.5 for t in fitted.teams.values())
    for team in fitted.teams:  # the better driver in each team is the baseline
        assert min(d.pace for d in fitted.drivers if d.team == team) < 0.2
    # the championship leader's team should be rated near the front
    leader = max(season["standings"], key=season["standings"].get)
    leader_team = next(d.team for d in fitted.drivers if d.code == leader)
    assert list(fitted.teams).index(leader_team) < 3
    # fitting on fewer rounds only uses those rounds
    early = fit_grid(season, grid, through_round=3)
    assert set(early.codes) == set(season["races"][2]["entries"])
    with pytest.raises(ValueError):
        fit_grid(season, grid, through_round=0)


def test_fitted_ratings_beat_uninformed(grid, tracks):
    season = load_season()
    flat = Grid({t.name: Team(t.name) for t in grid.teams.values()},
                [Driver(d.code, d.name, d.team) for d in grid.drivers])
    races = [r for r in season["races"] if r["round"] > 14]
    naive = backtest(flat, tracks, races, n_sims=300, seed=1)["mean"]
    wf = walk_forward(season, tracks, grid, start=14, n_sims=300, seed=1)["mean"]
    assert wf["rps"] < naive["rps"] - 0.03 and wf["spearman"] > 0.4


def test_dashboard(grid, tracks, tmp_path, capsys):
    from f1sim.dashboard import build_data, build_site
    data = build_data(grid, tracks, load_season(), n_sims=200, seed=1, accuracy=False)
    assert data["next"] in tracks and len(data["tracks"]) == len(tracks)
    for t in data["tracks"]:
        assert sum(r["win"] for r in t["rows"]) == pytest.approx(1, abs=0.01)
        assert len(t["dist"]) == len(grid.drivers)
    assert sum(r["title"] for r in data["season"]["drivers"]) == pytest.approx(1, abs=0.01)
    assert set(data["strategy"]["drivers"]) == set(grid.codes)
    full = build_data(grid, {k: tracks[k] for k in list(tracks)[:23]}, load_season(), n_sims=100, seed=1,
                      history=[load_season()])
    assert len(full["accuracy"]["methods"]) == 4 and full["accuracy"]["races"]
    json.dumps(full)                                   # everything is plain JSON
    pages = build_site(data, tmp_path / "site", replays=False)
    names = {p.relative_to(tmp_path / "site").as_posix() for p in pages}
    assert names == {"index.html", "forecast/index.html", "lab/index.html", "replay/index.html", "live/index.html",
                     "accuracy/index.html"}
    for p in pages:
        html = p.read_text()
        assert html.startswith("<!doctype html>") and 'aria-current="page"' in html and "Code on GitHub" in html
    assert "F1Sim" in (tmp_path / "site/lab/index.html").read_text()            # engine only on the lab page
    assert "F1Sim" not in (tmp_path / "site/forecast/index.html").read_text()
    main(["dashboard", "--sims", "200", "--quick", "--out", str(tmp_path / "cli")])
    assert (tmp_path / "cli/index.html").exists() and capsys.readouterr().out
    assert (tmp_path / "cli/replay/index.html").exists()


def test_start_grid_and_qualifying_update(grid, tracks):
    order = grid.codes[::-1]                      # reverse of the rated order
    res = simulate_race(grid, tracks["monaco"], 50, seed=1, start_grid=order)
    assert (res.grid_positions == np.arange(len(order), 0, -1)).all()
    partial = simulate_race(grid, tracks["monaco"], 50, seed=1, start_grid=order[:3])
    assert (partial.grid_positions[:, [grid.index(c) for c in order[:3]]] == [1, 2, 3]).all()
    # a backmarker who qualifies on pole by a mile is expected to race better than rated
    slow = grid.codes[-1]
    times = {c: 90.0 + i * 0.1 for i, c in enumerate(c for c in grid.codes if c != slow)}
    times[slow] = 89.0
    plain = simulate_race(grid, tracks["spain"], 400, seed=2, wet=False)
    told = simulate_race(grid, tracks["spain"], 400, seed=2, wet=False, quali_times=times)
    i = grid.index(slow)
    assert told.positions[:, i].mean() < plain.positions[:, i].mean() - 0.5


def test_benchmark(grid, tracks):
    from f1sim.benchmark import METHODS, benchmark, summarise
    out = benchmark([load_season(2025), load_season(2026)], tracks, grid, start=17, n_sims=200, seed=1)
    assert out["n_races"] == len(out["races"]) > 5 and set(out["methods"]) == set(METHODS)
    for m in METHODS:
        assert 0 < out["methods"][m]["rps"] < 0.25 and 0 <= out["methods"][m]["favourite_won"] <= 1
    assert summarise(out["races"], ["2026"])["n_races"] < out["n_races"]
    from f1sim.benchmark import compare
    c = compare(out["races"], "model+grid", "grid", n_boot=500)
    assert c["lo"] <= c["diff"] <= c["hi"] and 0 <= c["p_a_lower"] <= 1 and c["n_races"] == out["n_races"]
    for rows in out["calibration"].values():
        assert sum(r["n"] for r in rows) >= 20 * out["n_races"]
        assert all(0 <= r["observed"] <= 1 for r in rows)
