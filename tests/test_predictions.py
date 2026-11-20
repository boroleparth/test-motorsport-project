"""The public prediction log: when forecasts are saved, that they are never rewritten, and scoring."""
import datetime as dt
import json

import pytest

from f1sim import load_grid, load_season, load_tracks
from f1sim import predictions as P

UTC = dt.timezone.utc
RACE = {"round": 21, "track": "las_vegas", "name": "Las Vegas Grand Prix",
        "qualifying": "2026-11-21T04:00:00+00:00", "race": "2026-11-22T04:00:00+00:00"}
QUALI = [("RUS", 92.1), ("VER", 92.2), ("NOR", 92.3), ("LEC", 92.35), ("PIA", 92.4), ("HAM", 92.5),
         ("ANT", 92.6), ("ALO", 92.7), ("SAI", 92.8), ("GAS", 92.9), ("HUL", None)]


@pytest.fixture(scope="module")
def setup():
    season = load_season(2026)
    season = {**season, "races": [r for r in season["races"] if r["round"] <= 20]}
    return load_grid(), load_tracks(), season


def at(text):
    return dt.datetime.fromisoformat(text).replace(tzinfo=UTC)


def run(setup, now, log, quali=QUALI):
    grid, tracks, season = setup
    return P.auto(grid, tracks, season, [RACE], lambda rnd: quali, now=at(now), n_sims=300, log=log)


def test_auto_saves_each_forecast_once_in_its_window(setup, tmp_path):
    path = P.path_for("2026", 21, "las_vegas", tmp_path)
    assert "not due yet" in run(setup, "2026-11-15T12:00", tmp_path)[0] and not path.exists()
    assert "saved the pre-qualifying" in run(setup, "2026-11-19T12:00", tmp_path)[0]
    first = path.read_text()
    assert run(setup, "2026-11-20T12:00", tmp_path) == [] and path.read_text() == first   # never rewritten
    # qualifying under way: too early for the post-qualifying forecast
    assert run(setup, "2026-11-21T04:30", tmp_path) == []
    assert "not published" in run(setup, "2026-11-21T06:00", tmp_path, quali=[])[0]
    assert "saved the post-qualifying" in run(setup, "2026-11-21T06:00", tmp_path)[0]
    entry = json.loads(path.read_text())
    assert entry["pre_quali"] == json.loads(first)["pre_quali"]
    post = entry["post_quali"]
    assert post["start_grid"][0] == "RUS" and "HUL" not in post["quali_times"]
    assert post["saved"].startswith("2026-11-21T06:00") and post["ratings"] == "fitted from 2026 rounds 1-20"
    for snap in (entry["pre_quali"], post):
        assert len(snap["dist"]) == len(snap["codes"])
        assert all(abs(sum(row) - 1) < 0.01 for row in snap["dist"])
    # the pole-sitter is much likelier to win once the grid is known
    win = lambda s, c: next(r["win"] for r in s["summary"] if r["driver"] == c)
    assert win(post, "RUS") > win(entry["pre_quali"], "RUS")


def test_too_late_and_no_overwrite(setup, tmp_path):
    grid, tracks, season = setup
    track = tracks["las_vegas"]
    # missed the pre-qualifying window: say so, save nothing
    assert "too late" in run(setup, "2026-11-21T05:00", tmp_path)[0]
    with pytest.raises(ValueError, match="too late"):
        P.save("pre_quali", RACE, grid, track, season, at("2026-11-21T04:00"), n_sims=100, log=tmp_path)
    with pytest.raises(ValueError, match="too late"):
        P.save("post_quali", RACE, grid, track, season, at("2026-11-22T04:01"), start_grid=["VER"], n_sims=100, log=tmp_path)
    with pytest.raises(ValueError, match="qualifying order"):
        P.save("post_quali", RACE, grid, track, season, at("2026-11-21T06:00"), n_sims=100, log=tmp_path)
    P.save("pre_quali", RACE, grid, track, season, at("2026-11-20T00:00"), n_sims=100, log=tmp_path)
    with pytest.raises(ValueError, match="already has"):
        P.save("pre_quali", RACE, grid, track, season, at("2026-11-20T01:00"), n_sims=100, log=tmp_path)
    brazil = {**RACE, "round": 20, "track": "brazil", "qualifying": None, "race": None}
    with pytest.raises(ValueError, match="already has a result"):
        P.save("pre_quali", brazil, grid, tracks["brazil"], load_season(2026), n_sims=100, log=tmp_path)
    assert run(setup, "2026-11-22T05:00", tmp_path) == ["no race left this season"]


def test_track_record_scores_against_results(setup, tmp_path):
    grid, tracks, season = setup
    run(setup, "2026-11-19T12:00", tmp_path)
    run(setup, "2026-11-21T06:00", tmp_path)
    entries = P.load_log(tmp_path)
    pending = P.track_record(entries, [season])
    assert pending["summary"] == {} and pending["races"][0]["winner"] is None
    assert "winner_prob" not in pending["races"][0]["pre_quali"]
    result = {"round": 21, "track": "las_vegas", "name": RACE["name"], "results": [c for c, _ in QUALI]}
    done = P.track_record(entries, [{**season, "races": season["races"] + [result]}])
    row = done["races"][0]
    assert row["winner"] == "RUS" and row["file"] == "2026-21-las_vegas.json"
    assert row["post_quali"]["pole_won"] == 1 and done["summary"]["post_quali"]["n"] == 1
    for kind in P.KINDS:
        assert 0 < row[kind]["winner_prob"] < 1 and 0 < row[kind]["rps"] < 0.25
    assert row["post_quali"]["winner_prob"] > row["pre_quali"]["winner_prob"]
    json.dumps(done)


def test_bundled_log_is_valid():
    """Every committed forecast was saved before its session and can be scored."""
    for e in P.load_log():
        assert P.path_for(e["season"], e["round"], e["track"]).exists()
        for kind, deadline in (("pre_quali", e["qualifying"]), ("post_quali", e["race"])):
            if kind in e:
                assert dt.datetime.fromisoformat(e[kind]["saved"]) < dt.datetime.fromisoformat(deadline)
                assert len(e[kind]["dist"]) == len(e[kind]["codes"])


def test_cli_record_and_page_data(setup, tmp_path, capsys):
    from f1sim.cli import main
    from f1sim.dashboard import _page_data
    run(setup, "2026-11-19T12:00", tmp_path)
    main(["predict", "record", "--log", str(tmp_path)])
    assert "Las Vegas Grand Prix" in capsys.readouterr().out
    rec = P.track_record(P.load_log(tmp_path), [setup[2]])
    page = _page_data({"label": "", "drivers": [], "teams": [], "accuracy": None, "track_record": rec}, "accuracy")
    assert page["track_record"] == rec and page["log_url"].endswith("/f1sim/data/predictions/")
