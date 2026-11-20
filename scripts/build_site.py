"""Refresh the data and build the dashboard website into ``site/``.

Used by the GitHub Pages workflow. Steps:
1. download the current season's results (kept only if a race has been run);
2. refit the ratings from the latest season;
3. with ``--predict``, save any forecast due for the next race to the public
   prediction log (``f1sim/data/predictions/``; the workflow commits it);
4. write the website to ``site/``.

If the download fails, the bundled data is used, so the site still builds.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from f1sim import fit_grid, load_grid, load_season, load_tracks
from f1sim.dashboard import build_dashboard
from f1sim.datasource import bundled_seasons, fetch_season
from f1sim.fit import grid_to_json

DATA = Path(__file__).resolve().parent.parent / "f1sim" / "data"


def refresh(year: int) -> bool:
    """Download ``year`` into the bundled data. Returns True if the data changed."""
    try:
        season = fetch_season(year)
    except Exception as exc:  # network trouble must not break the site build
        print(f"Could not download {year} ({exc}); using bundled data")
        return False
    if not season["races"]:
        print(f"No {year} races run yet; using bundled data")
        return False
    path = DATA / f"results_{year}.json"
    old = path.read_text() if path.exists() else ""
    new = json.dumps(season, separators=(",", ":"))
    if new == old:
        print(f"{year} results unchanged ({len(season['races'])} races)")
        return False
    path.write_text(new)
    print(f"Saved {len(season['races'])} races of {year}")
    return True


def predict() -> None:
    """Save the pre- or post-qualifying forecast for the next race if one is due."""
    from f1sim import predictions
    from f1sim.datasource import fetch_qualifying, fetch_schedule

    season = load_season()
    try:
        schedule = fetch_schedule(season["season"])
        notes = predictions.auto(load_grid(), load_tracks(), season, schedule,
                                 lambda rnd: fetch_qualifying(season["season"], rnd))
    except Exception as exc:  # a missed forecast must not stop the site build
        notes = [f"Could not save a forecast ({exc})"]
    for line in notes:
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="site")
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--no-fetch", action="store_true", help="use the bundled data as it is")
    parser.add_argument("--predict", action="store_true",
                        help="save whichever race-weekend forecast is due to the prediction log")
    args = parser.parse_args()

    if not args.no_fetch and refresh(dt.date.today().year):
        season = load_season()
        rounds = season["races"][-1]["round"]
        note = (f"Fitted by scripts/build_site.py from {season['season']} rounds 1-{rounds} (api.jolpi.ca). "
                "consistency, wet and pit-stop values are assumptions, not measurements.")
        (DATA / "grid.json").write_text(json.dumps(grid_to_json(fit_grid(season, load_grid()), note), indent=1))

    if args.predict:
        predict()

    history = [load_season(y) for y in bundled_seasons()]
    pages = build_dashboard(load_grid(), load_tracks(), args.out, load_season(), n_sims=args.sims, seed=0,
                            history=history)
    for page in pages:
        print(f"Wrote {page} ({page.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
