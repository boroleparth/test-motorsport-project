# Handoff: f1sim

State as of 15 November 2026. Everything below is on `main` in
`boroleparth/test-motorsport-project`, all 36 tests pass (41 on the work branch with the prediction log), and the site is live at
https://boroleparth.github.io/test-motorsport-project/.

## What it is

A Formula 1 race and championship simulator with a public website. Races are
simulated lap by lap thousands of times using car and driver pace fitted from
real results; every prediction is tested against what actually happened.

| Page | What it does |
|---|---|
| Home | Next race at a glance, links to everything |
| Forecast | Win/podium/points odds per circuit, championship projection, strategy comparison |
| What-if lab | Rerun a race in the browser with rain, pace, strategy or grid changes |
| Race replay | Any Grand Prix since 2023, lap by lap: win-odds chart, running order, pit advisor |
| Live | Follows a race from OpenF1 every 20 s; between races, replays the last race as if live |
| Accuracy | Benchmark against baselines, calibration, fitted car and driver pace |

## Key results (all out of sample)

- **Pre-race, 207 races 2014–2026:** after qualifying the simulator beats
  "finish where you start" (ranked probability score better by 0.0048, 95%
  interval 0.0021–0.0075; favourite won 62% vs 54% for pole). Before
  qualifying it is level with championship order.
- **Live win odds, every 5th lap of 2025–2026 (trained on 2023–2024):** learned
  model log loss 0.75, favourite won 74%; simulator 0.95 / 68%; "leader wins"
  1.98 / 71%. Both models are well calibrated.
- **Pit advisor:** on lap 40 of Brazil 2026 its top pick for Verstappen was the
  stop Red Bull actually made.

## Key architectural decisions

1. **Monte Carlo simulator, not a neural network.** About 20 races a season is
   too little to learn outcomes end to end; the simulator answers what-if
   questions a statistical model cannot. Learned models are used where data is
   plentiful (lap-level) and only if they win on held-out seasons.
2. **Simulator for what-ifs, learned model for win odds.** The 6-feature
   conditional logit (`winprob.py`) beat the simulator on live odds, and
   combining them did not help, so each is used where it is best.
3. **Strict out-of-sample testing.** Walk-forward (each race predicted from
   earlier rounds only), settings tuned on older seasons, reported on newer
   ones, paired-bootstrap intervals. Several changes were dropped because they
   failed this (direct lap-time tyre-wear fit; per-circuit safety-car history).
4. **Data over guesses, calibrated to revealed behaviour.** Pit loss is
   measured from real stops; tyre wear per circuit is tuned so the simulator's
   best strategy matches what teams ran; safety-car odds are the field rate
   shrunk towards each circuit. 2026's new rules broke the link with earlier
   tyre data, so the current era is used where it has data.
5. **Mid-race tyres wear steadily, with no cliff.** Real in-stint lap times are
   nearly flat; the pre-race cliff gave cars on long stints a phantom speed-up.
6. **Static site, computation in the browser.** GitHub Pages serves files only,
   so the lab, pit advisor and live page run JavaScript ports of the Python
   engines (`engine.js`, `replay-engine.js`, `live-feed.js`), fed parameters
   exported from Python. Tests run each port under Node against Python.
7. **One `main` branch, one site with linked pages.** Branches only for work in
   progress. Each page is a self-contained HTML file embedding only its data;
   replay races load one JSON file at a time.
8. **Live data differs from history:** when the leader finishes a lap the field
   is still on it, so the live page shows the last lap the whole lead-lap field
   has completed.

## Files

| Path | Role |
|---|---|
| `f1sim/race.py`, `strategy.py`, `models.py`, `season.py` | Pre-race engine, tyre model and strategy search, data model, championship |
| `f1sim/fit.py`, `datasource.py` | Ratings fitted from results; Jolpica (Ergast) downloader |
| `f1sim/evaluate.py`, `benchmark.py` | Scoring, walk-forward benchmark vs baselines, calibration, bootstrap |
| `f1sim/laps.py` | OpenF1 lap-by-lap download and lap table (2023 onward) |
| `f1sim/tyres.py` | Pit loss, tyre-wear calibration, safety-car rates from lap data |
| `f1sim/live.py` | Race state from any lap; simulate the rest of the race; remaining plans |
| `f1sim/winprob.py` | Live win-probability model (weights in `data/winprob.json`) |
| `f1sim/replay.py` | Per-race data for the replay page |
| `f1sim/dashboard.py`, `cli.py` | Runs the simulations and builds the site; command-line tool |
| `f1sim/web/` | Page templates and scripts; `engine.js` (in `data/`), `replay-engine.js`, `live-feed.js` are the browser ports |
| `f1sim/data/` | Results 2014–2026, lap tables 2023–2026, fitted grid, circuits, win-model weights |
| `scripts/build_site.py`, `calibrate_tracks.py` | Site build used by CI; circuit calibration from lap data |
| `.github/workflows/` | Tests on every push; site rebuild on push to `main` and every Monday |
| `tests/` | 36 tests, including JS-vs-Python checks and a saved OpenF1 race |

## How to run

```bash
pip install -e ".[dev]"
pytest
f1sim dashboard --out site --quick   # all pages locally; full build ~6 min without --quick
f1sim benchmark                      # pre-race benchmark with intervals
```

## Open items and caveats

- **Live feed needs a paid OpenF1 account.** Checked during Las Vegas FP1 (20 Nov 2026):
  while any F1 session runs, OpenF1 answers 401 to every unauthenticated request, past
  sessions included, with no CORS header, so the browser sees only a failed request. The live
  page now says so; it works between sessions (the replay of the last race) and the Race replay
  page is unaffected. Real-time odds would need an OpenF1 key, which a static site cannot hide;
  that would need a small server or proxy.
- **Known limitation:** win odds can swing briefly during pit cycles in unusual
  races (wet races, very low tyre wear), when it is unclear who still has to stop.
- **Network:** F1's own timing archive now needs an F1 account, so lap data
  comes from OpenF1. The cloud environment `boroleparth` allows `api.jolpi.ca`,
  `api.openf1.org` and `livetiming.formula1.com`.
- **Stale branch:** `claude/determined-rubin-sj41az` in `boroleparth/boroleparth`
  holds an early copy that was never pushed; the project moved to this repo.
- Assumed, not measured: tyre compound offsets and base wear, lap-time
  consistency, pit-stop spread, overtaking difficulty, rain odds. Sprint races,
  virtual safety cars (in the simulator), grid penalties and red-flag restarts
  are not modelled.

## Next steps

1. **Race-weekend prediction log: built, waiting on two things.** `f1sim/predictions.py`,
   `f1sim predict`, `scripts/build_site.py --predict` and a "Track record" section on the
   Accuracy page (see README, "Prediction log"). The Las Vegas pre-qualifying forecast was
   saved on 19 Nov (Antonelli 24.2%, Verstappen 22.0%, Norris 15.3%). Still to do:
   - **Push it before Las Vegas qualifying (21 Nov 04:00 UTC)**, or the git history no
     longer shows the forecast was made beforehand.
   - Automation is in `pages.yml`: every 3 hours Thursday to Sunday the site build runs
     `scripts/build_site.py --predict` and commits any new forecast in
     `f1sim/data/predictions/` to `main` (only the build job can write to the repo). It takes
     effect once merged to `main`. Check the first run on the Las Vegas weekend.
2. **Repository polish:** README with screenshots, version tags, link from the
   profile repo.
3. **Off-season models:** overtaking learned from lap data; virtual safety cars
   in the simulators; a model of when teams pit, so rivals react in the advisor.
4. **2027 features:** pre-season mode with a user-set pecking order; fantasy helper.
