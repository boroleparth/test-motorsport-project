"""Update tracks.json from real lap data: pit-stop loss and tyre-wear multiplier.

* Pit-stop loss: median measured loss per circuit over the most recent seasons
  that have at least 8 green-flag stops there (see f1sim.tyres.pit_losses).
* Safety-car probability ("sc_prob"): the field-wide rate of full safety cars
  and red flags, nudged towards each circuit's own history (see
  f1sim.tyres.safety_car_rates); circuits with no lap data get the field rate.
* Wear multiplier ("deg"): tuned so the simulator's fastest strategy matches what
  teams actually did (stop count and first-stop timing). The current rules era
  (2026 onward) is used where it has data, earlier seasons otherwise.

Run after `f1sim`'s lap data has been refreshed (f1sim/data/laps/).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from f1sim import load_tracks
from f1sim.laps import load_laps
from f1sim.tyres import calibrate_tracks, pit_losses, real_strategies, safety_car_rates

ERA_START = 2026
TRACKS = Path(__file__).resolve().parent.parent / "f1sim" / "data" / "tracks.json"


def main() -> None:
    rows = load_laps()
    seasons = sorted({r["season"] for r in rows})
    recent = [r for r in rows if r["season"] >= ERA_START]
    older = [r for r in rows if r["season"] < ERA_START]

    # pit loss: newest seasons first, until 8 stops are collected
    loss = {}
    by_season = {s: pit_losses([r for r in rows if r["season"] == s]) for s in seasons}
    for key in {k for v in by_season.values() for k in v}:
        sample = []
        for s in reversed(seasons):
            sample += by_season[s].get(key, [])
            if len(sample) >= 8:
                break
        if len(sample) >= 8:
            loss[key] = round(float(np.median(sample)), 1)

    target = {**real_strategies(older), **real_strategies(recent)}
    tracks = load_tracks()
    deg = calibrate_tracks(tracks, target, loss)

    field_sc, sc = safety_car_rates(rows)
    raw = json.loads(TRACKS.read_text())
    changed = 0
    for t in raw["tracks"]:
        k = t["key"]
        if k in loss and loss[k] != t["pit_loss"]:
            t["pit_loss"], changed = loss[k], changed + 1
        p_sc = round(sc.get(k, field_sc), 2)
        if p_sc != t["sc_prob"]:
            t["sc_prob"], changed = p_sc, changed + 1
        if k in deg and deg[k] != t["deg"]:
            t["deg"], changed = deg[k], changed + 1
    raw["_note"] = ("Circuit parameters. pit_loss (s) is measured from real pit stops and deg (tyre-wear multiplier) "
                    "is tuned so the fastest strategy matches what teams ran, and sc_prob is the measured rate of full safety "
                    "cars and red flags (shrunk towards the field rate), all by scripts/calibrate_tracks.py from "
                    f"OpenF1 lap data {seasons[0]}-{seasons[-1]}; other values are estimates. base_lap/pit_loss in "
                    "seconds; overtake = pace advantage (s/lap) needed for a 50% chance to pass per lap; "
                    "sc_prob/rain_prob are per-race probabilities.")
    TRACKS.write_text('{\n  "_note": %s,\n  "tracks": [\n%s\n  ]\n}\n' % (
        json.dumps(raw["_note"]), ",\n".join("    " + json.dumps(t) for t in raw["tracks"])))
    print(f"Updated {changed} values: pit loss at {len(loss)} circuits, wear at {len(deg)} circuits, "
          f"safety-car odds everywhere (field rate {field_sc:.2f})")


if __name__ == "__main__":
    main()
