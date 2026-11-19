# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

f1sim: a Monte Carlo Formula 1 race and championship simulator (Python + NumPy) with a static
website on GitHub Pages. `HANDOFF.md` has the current state, open items and next steps; `README.md`
has the methods and benchmark results.

## Commands

```bash
pip install -e ".[dev]"
pytest                                   # whole suite
pytest tests/test_f1sim.py::test_season  # one test
pytest -k js                             # JS-vs-Python checks (skipped if `node` is missing)
f1sim dashboard --out site --quick       # build all pages locally; drops the benchmark and replay data
f1sim dashboard --out site               # full build, ~6 min
f1sim benchmark                          # pre-race benchmark vs baselines, with bootstrap intervals
python scripts/build_site.py --out site  # what CI runs: fetch current season, refit grid.json, build
f1sim predict                            # save the race-weekend forecast that is due (`record` scores the log)
python scripts/calibrate_tracks.py       # rewrite tracks.json pit loss / sc_prob / deg from lap data
```

No linter or formatter is configured. Every CLI command takes `--sims`, `--seed`, `--grid`, `--tracks`.

## Architecture

Two separate prediction paths:

- **Pre-race** — `race.simulate_race` simulates qualifying and the race lap by lap, vectorised as
  `(n_sims, n_drivers)` arrays. Inputs are a `Grid` (teams + drivers with pace in s/lap deficits) and
  a `Track` (`models.py`). `start_grid` / `quali_times` condition on real qualifying. `season.py`
  chains races into a championship. Ratings in `data/grid.json` are fitted from results by `fit.py`
  from `data/results_YYYY.json` (Jolpica/Ergast, via `datasource.py`).
- **Mid-race** — `laps.py` turns OpenF1 data into a lap table (`data/laps/*.csv.gz`, 2023 onward);
  `live.race_state` builds the state at any lap and `live.simulate_from` runs the rest of the race
  (tyres wear steadily mid-race, no cliff — deliberately different from the pre-race tyre model).
  Live **win odds** come from `winprob.py`, a 6-feature conditional logit (weights in
  `data/winprob.json`), not from the simulator; the simulator is used for what-ifs and the pit advisor.

`predictions.py` keeps the public prediction log (`data/predictions/`): forecasts saved before and after
qualifying, refused once their session has started or if one already exists, and scored at build time by
`track_record` for the Accuracy page. Never edit or regenerate saved forecasts.

`tyres.py` derives circuit parameters (pit loss, wear multiplier, safety-car rate) from lap data;
`scripts/calibrate_tracks.py` writes them into `data/tracks.json`.

### Website and the JavaScript ports

GitHub Pages serves static files only, so browser features run JS ports of Python engines:

| JS (browser) | Ports | Checked by |
|---|---|---|
| `f1sim/data/engine.js` (what-if lab) | `race.py` | `tests/test_engine_js.py` |
| `f1sim/web/replay-engine.js` (pit advisor) | `live.py` | `tests/test_replay.py` |
| `f1sim/web/live-feed.js` (live page) | `laps.build_rows`, `live.race_state`, `winprob` | `tests/test_live_page.py` |

**Changing engine logic or constants in Python means changing the matching JS port too**; the tests
run each port under Node and compare against Python. Constants and parameters reach the JS through
exported data (`dashboard.export_model`, `replay.engine_constants`) rather than being duplicated.

`dashboard.build_data` runs all simulations into one dict; `_page_data` slices out only what each page
needs; `render_page` inlines `web/style.css`, `common.js`, `<page>.html`, `<page>.js` and the data
(`const D = …`) into one self-contained `index.html` per page (`PAGES` lists them). Replay races are
written as separate `replay/data/<season>-<round>.json` files by `replay.py`.

CI (`.github/workflows/`): tests on every push (Python 3.11 and 3.13); `pages.yml` runs
`scripts/build_site.py --predict` on push to `main`, Mondays and every 3 h Thursday–Sunday; it may refetch
results and refit `grid.json` (not committed) and commits only new files in `f1sim/data/predictions/` to `main`.

## Conventions

- **Out-of-sample evidence decides model changes.** Walk-forward (each race predicted from earlier
  rounds only), settings tuned on older seasons and reported on 2025–2026, paired-bootstrap intervals
  (`benchmark.py`). Changes that don't win on held-out data are dropped — several have been.
- Prefer measured values over guesses; values that are still assumptions (compound offsets, base
  wear, consistency, pit spread, overtaking, rain odds) are flagged as such in docs.
- Driver codes are three-letter uppercase (`VER`); track keys are the `tracks.json` keys
  (`las_vegas`, `abu_dhabi`); Jolpica circuit IDs map to them in `datasource.CIRCUIT_KEYS`.
- Strategies are written `COMPOUNDS@PIT_LAPS`, e.g. `S-M-M@15,36`; at most 3 stops (`race.MAX_STOPS`).
- Network: lap data comes from OpenF1 (`api.openf1.org`); results from `api.jolpi.ca`. Builds fall
  back to bundled data if downloads fail.
- One `main` branch; branches only for work in progress. Repo: `boroleparth/test-motorsport-project`.
