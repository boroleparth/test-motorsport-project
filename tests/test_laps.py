"""Lap data (f1sim.laps) and the fits built on it (f1sim.tyres)."""
import numpy as np
import pytest

from f1sim import load_tracks
from f1sim.laps import build_rows, load_laps, _neutralised
from f1sim.tyres import calibrate_tracks, clean_laps, fit_race, pit_losses, real_strategies


@pytest.fixture(scope="module")
def season():
    return load_laps([2025])


def test_bundled_lap_tables(season):
    assert {r["season"] for r in season} == {2025}
    races = {r["round"] for r in season}
    assert len(races) >= 20 and len(season) > 20000
    for rnd in list(races)[:5]:
        race = [r for r in season if r["round"] == rnd]
        laps_total = race[0]["laps_total"]
        winner = {r["driver"] for r in race if r["final_position"] == 1}
        assert len(winner) == 1
        # the winner leads at the flag, and positions at each lap are 1..n
        last = [r for r in race if r["lap"] == laps_total]
        assert min(last, key=lambda r: r["end_time"])["driver"] in winner
        lap10 = sorted(r["position"] for r in race if r["lap"] == 10)
        assert lap10 == list(range(1, len(lap10) + 1))
    assert sum(r["pit_in"] for r in season) > 400
    assert {r["status"] for r in season} <= {"GREEN", "SC", "VSC", "RED"}


def test_build_rows_from_raw():
    t = lambda s: f"2025-01-01T12:{s // 60:02d}:{s % 60:02d}+00:00"
    laps, stints, drivers = [], [], []
    for num, code, pace in ((1, "AAA", 90), (2, "BBB", 91)):
        drivers.append({"driver_number": num, "name_acronym": code, "team_name": "T"})
        for n in range(1, 6):
            laps.append({"driver_number": num, "lap_number": n, "date_start": t((n - 1) * pace),
                         "lap_duration": pace, "is_pit_out_lap": n == 4 and num == 2})
        stints.append({"driver_number": num, "stint_number": 1, "lap_start": 1, "lap_end": 3 if num == 2 else 5,
                       "compound": "MEDIUM", "tyre_age_at_start": 0})
    stints.append({"driver_number": 2, "stint_number": 2, "lap_start": 4, "lap_end": 5, "compound": "HARD",
                   "tyre_age_at_start": 0})
    raw = {"session": {"year": 2025, "round": 1, "track": "x"}, "drivers": drivers, "laps": laps,
           "stints": stints, "pit": [], "weather": [],
           "race_control": [{"date": t(100), "lap_number": 2, "category": "SafetyCar", "flag": None,
                             "message": "SAFETY CAR DEPLOYED"},
                            {"date": t(200), "lap_number": 3, "category": "SafetyCar", "flag": None,
                             "message": "SAFETY CAR IN THIS LAP"}]}
    rows = {(r["driver"], r["lap"]): r for r in build_rows(raw)}
    assert rows["AAA", 5]["final_position"] == 1 and rows["BBB", 5]["final_position"] == 2
    assert rows["BBB", 2]["gap_ahead"] == pytest.approx(2.0)
    assert rows["BBB", 3]["pit_in"] == 1 and rows["BBB", 4]["pit_out"] == 1  # pit inferred from the out lap
    assert rows["BBB", 4]["compound"] == "HARD" and rows["BBB", 4]["tyre_age"] == 1
    assert [rows["AAA", n]["status"] for n in range(1, 6)] == ["GREEN", "SC", "SC", "GREEN", "GREEN"]


def test_neutralised_handles_vsc_and_red():
    msgs = [{"date": "1", "lap_number": 5, "message": "VIRTUAL SAFETY CAR DEPLOYED", "flag": None},
            {"date": "2", "lap_number": 6, "message": "VIRTUAL SAFETY CAR ENDING", "flag": None},
            {"date": "3", "lap_number": 9, "message": "RED FLAG", "flag": "RED"},
            {"date": "4", "lap_number": 10, "message": "TRACK CLEAR", "flag": "GREEN"}]
    s = _neutralised(msgs, 20)
    assert s == {5: "VSC", 6: "VSC", 9: "RED", 10: "RED"}


def test_fits_on_real_laps(season):
    clean = clean_laps(season)
    assert 0.3 * len(season) < len(clean) < len(season)
    race = [r for r in clean if r["round"] == 2]
    f = fit_race(race)
    assert f is not None and -0.2 < f["fuel"] < 0.05 and f["resid_sd"] < 1.0
    losses = pit_losses(season)
    allv = [v for vals in losses.values() for v in vals]
    assert len(allv) > 200 and 15 < float(np.median(allv)) < 30
    target = real_strategies(season)
    assert len(target) >= 15 and all(1 <= g["stops"] <= 6 for g in target.values())
    key = "italy" if "italy" in target else next(iter(target))
    deg = calibrate_tracks(load_tracks(), {key: target[key]})
    assert 0.3 <= deg[key] <= 2.5
