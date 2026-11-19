# f1sim — Formula 1 race & season simulator

[![Tests](https://github.com/boroleparth/test-motorsport-project/actions/workflows/tests.yml/badge.svg)](https://github.com/boroleparth/test-motorsport-project/actions/workflows/tests.yml)

**Live dashboard: https://boroleparth.github.io/test-motorsport-project/**

A Monte Carlo simulator for **predicting** F1 races and championships and for
**testing** ideas (pit strategies, car upgrades, "what if it rains?") against a
consistent model. Pure Python + NumPy; 2,000 full race simulations take well under a second.

The bundled ratings are **fitted from real 2026 results** (rounds 1–20, from the
public Jolpica/Ergast API), and the whole pipeline is benchmarked against
simple baselines on races it has not seen.

**Benchmark: 207 races, 2014–2026**, each predicted from earlier rounds of its
season only (`f1sim benchmark`).

| Method | Favourite won | Podium drivers found | Order match | Probability score (0 = perfect) |
|---|---|---|---|---|
| Simulator, before qualifying | 45% | 1.80 of 3 | 0.62 | 0.125 |
| Baseline: championship order | 45% | 1.81 of 3 | 0.61 | 0.127 |
| Simulator, after qualifying | 62% | 1.94 of 3 | 0.66 | 0.118 |
| Baseline: starting grid order | 54% | 2.03 of 3 | 0.63 | 0.122 |

- **After qualifying the simulator beats grid order**: ranked probability score
  better by 0.0048, 95% paired-bootstrap interval 0.0021 to 0.0075, and its
  favourite wins 62% of races against 54% for the pole-sitter.
- **Before qualifying it is level with championship order**: better by 0.0013,
  interval −0.0003 to 0.0029, which includes zero.
- On the 34 races of 2025–2026, which were never used to choose a setting, the
  pattern is the same (0.125 vs 0.130 before qualifying, 0.114 vs 0.117 after).
- The baselines are scored generously (see `f1sim/benchmark.py`), and settings
  were chosen on subsets of these seasons, so treat the margins as indicative.

What is *not* measured: lap-time consistency, pit-stop times, wet-weather skill
and the circuit parameters in `tracks.json` are assumptions. This is a
statistical simulator, not a neural network — see "How the model works".

## Install

```bash
git clone https://github.com/boroleparth/test-motorsport-project
cd test-motorsport-project
pip install -e ".[dev]"
```

## Use

```bash
f1sim dashboard --out site                     # build the website into site/
f1sim tracks                                   # circuits and their fastest strategy
f1sim race monaco --seed 1                     # win / podium / points probabilities
f1sim race britain --wet                       # force a wet race (--dry to force dry)
f1sim race bahrain --strategy VER=S-M-M@15,36  # pin one driver's strategy
f1sim strategy bahrain VER                     # compare the 5 fastest strategies for a driver
f1sim strategy bahrain VER M-H@24 H-M@33       # ...or your own candidates
f1sim season                                   # finish the real season from current standings
f1sim season --full --sims 1000                # whole calendar from zero points
f1sim fetch 2026 --out season.json             # download real results (needs network)
f1sim fit --season-file season.json --out grid.json   # fit ratings from them
f1sim race qatar --grid grid.json              # predict with your fitted ratings
f1sim benchmark                                # simulator vs baselines over 2014–2026, with intervals and calibration
f1sim race qatar --start-grid ANT,VER,RUS      # predict once the real grid is known
f1sim backtest --walk-forward                  # predict each race of a season from earlier rounds
f1sim backtest results.json                    # score fixed ratings against a results file
```

Strategies are written `COMPOUNDS@PIT_LAPS`, e.g. `M-H@22` (mediums, pit at the
end of lap 22, hards to the flag). Every command takes `--sims`, `--seed`,
`--grid my_grid.json` and `--tracks my_tracks.json`.

The same things from Python:

```python
from f1sim import load_grid, load_tracks, simulate_race, simulate_season, score_race

grid, tracks = load_grid(), load_tracks()
result = simulate_race(grid, tracks["italy"], n_sims=5000, seed=42)
for row in result.summary()[:5]:
    print(row["driver"], f"{row['win']:.1%}")

result.position_matrix()      # P[driver, finishing position]
season = simulate_season(grid, tracks.values(), n_sims=1000, seed=42)
season.drivers()[:3]          # title probability, expected points, 5–95% range
```

## Website

**https://boroleparth.github.io/test-motorsport-project/** is built from this
repository on every push to `main` and every Monday. It has one page per area,
joined by a shared menu:

| Page | What it shows |
|---|---|
| [Home](https://boroleparth.github.io/test-motorsport-project/) | The next race at a glance and links to everything |
| [Forecast](https://boroleparth.github.io/test-motorsport-project/forecast/) | Win, podium and points odds for any circuit, the championship projection, pit strategies compared |
| [What-if lab](https://boroleparth.github.io/test-motorsport-project/lab/) | Rerun a race in your browser with rain, pace, strategy or grid changes |
| [Race replay](https://boroleparth.github.io/test-motorsport-project/replay/) | Step through any Grand Prix since 2023 with lap-by-lap win odds and a pit advisor |
| [Live](https://boroleparth.github.io/test-motorsport-project/live/) | Win odds and pit advice during a race; between races, the last race replayed as if live |
| [Accuracy](https://boroleparth.github.io/test-motorsport-project/accuracy/) | The benchmark against baselines, calibration, and fitted car and driver pace |

`f1sim dashboard
--out site` builds the same pages locally (about six minutes; `--quick` skips
the accuracy benchmark and the replay data). Each page is a single self-contained HTML file, so it
also works opened straight from disk. The page templates, shared styles and
scripts are in `f1sim/web/`.

### Prediction log

Every race weekend the simulator's forecast is saved twice, before
qualifying and after it (given the real qualifying order and lap times), to
`f1sim/data/predictions/<season>-<round>-<circuit>.json`. A forecast is never
saved once its session has started and never overwritten, so the git history
shows what was predicted before the result was known. The Accuracy page's
"Track record" scores each logged forecast once the race is run.

```bash
f1sim predict          # save whichever forecast is due for the next race (safe to run any time)
f1sim predict pre      # save the before-qualifying forecast now
f1sim predict post     # save the after-qualifying forecast (add --start-grid if penalties change the grid)
f1sim predict record   # score the log against real results
```

On race weekends the site workflow does this by itself: every 3 hours from
Thursday to Sunday it runs `scripts/build_site.py --predict` and commits any new
forecast to the repository.

### What-if lab

The dashboard includes a what-if lab that runs the simulator in the visitor's
browser: pick a race, force rain, change a car's pace, set a driver's pit
strategy or starting position, and compare the result with the forecast. Each
run simulates the race with and without the changes using the same random
draws, so the differences come from the changes, not from sampling noise.

The browser engine is `f1sim/data/engine.js`, a port of `f1sim/race.py` that
reads every parameter from the Python side (`export_model` in
`f1sim/dashboard.py`). `tests/test_engine_js.py` runs it under Node and checks
that it agrees with the Python engine.

## Fitting and testing against reality

`f1sim fetch` saves a season (results, qualifying times, standings) as JSON.
`f1sim fit` turns it into ratings:

- **one-lap pace** from each driver's best qualifying lap relative to the fastest;
- **race pace** from the finishing gap to the winner per lap;
- each split into team + driver by a robust (median-like) regression, with
  recent races weighted more because cars develop;
- race pace is then pulled part-way towards the order implied by average
  finishing position, which says who beats whom more reliably than time gaps;
- **retirement rates** per team and driver, shrunk towards the field average.

`f1sim benchmark` runs that test over every bundled season (2014–2026) against
two baselines, championship order and starting-grid order, and prints a
calibration table (when the simulator says 30%, how often does it happen),
with paired-bootstrap intervals on the difference from each baseline.
Given the real grid and qualifying times (`start_grid`, `quali_times` in
`simulate_race`), the simulator also updates race pace with what qualifying
revealed.

`f1sim backtest --walk-forward` does the same for one season: before each race it
refits using earlier rounds only, predicts, and scores against the real result.
It reports the probability given to the real winner, log loss, Brier score,
ranked probability score (RPS, 0 = perfect) and rank correlation of predicted
vs actual order. Change the model, re-run, keep what improves the mean.

`f1sim backtest file.json` also accepts a hand-written list:

```json
[{"track": "monaco", "results": ["LEC", "PIA", "NOR"], "wet": false}]
```

## Lap-by-lap data

`f1sim/data/laps/` holds every lap of every Grand Prix since 2023 (about
100,000 driver-laps from 90 races), downloaded from the public
[OpenF1](https://openf1.org) API by `f1sim/laps.py`: lap time, running
position, gaps, tyre compound and age, pit stops, safety-car status and rain.
`load_laps()` reads it back.

`f1sim/tyres.py` and `scripts/calibrate_tracks.py` use it to set two circuit
parameters from data instead of estimates:

- **Pit-stop loss**, measured from in-laps and out-laps. On 2026 races, values
  measured from 2023–2025 are off by 2.1 s on average, against 2.7 s for the
  previous estimates.
- **Tyre-wear multiplier**, tuned so the simulator's fastest strategy matches
  what teams actually ran (number of stops and when the first stop comes).
  Tuned on 2023–2025, this gets the stop count right at 22 of 24 circuits
  (previously 17), but it does not carry over to 2026's new cars (12 of 18
  either way), so the current rules era is used wherever it has data.

- **Safety-car odds**: about half of races since 2023 had a full safety car
  or red flag. With three races per circuit, which circuits get them is close
  to unpredictable: on 2026 the field-wide rate predicted better than either
  each circuit's history or the old estimates, so every circuit gets the field
  rate nudged slightly towards its own history (43–57%). Virtual safety cars
  (46% of races) are not simulated yet.

Measuring tyre wear directly from lap times was tried and dropped: within a
stint, real lap times are nearly flat (typically 0.00–0.04 s a lap slower as
tyres age, because fuel burn-off almost cancels wear), and the direct fit
predicted 2026 stints worse than assuming no change at all. The race-prediction
benchmark is unchanged by the new circuit values.

## Mid-race prediction

`f1sim/live.py` picks up a race from any lap's real state: gaps, tyre compound
and age, stops made, which compounds each car has used (the two-compound
rule), safety-car status and retirements. Each car's current pace comes from
its own recent clean laps, blended with its pre-race rating early on, and tyre
wear from the race's own completed stints. `simulate_from` runs the remaining
laps many times, with each car following one of its best remaining plans;
`plan_options` lists those plans, which the pit advisor will compare.

`f1sim/winprob.py` gives live win odds from six features of the race state
(gap per lap remaining, running position, share of the race left, current
pace, and pit-stop time still owed). Tested on every 5th lap of the 2025 and 2026 races after training on
2023–2024:

| Method | Log loss (lower is better) | Favourite won |
|---|---|---|
| Learned model (`winprob`) | 0.75 | 74% |
| Simulator from the same states | 0.95 | 68% |
| "The current leader wins" | 1.98 | 71% |

Both the learned model and the simulator are well calibrated (when the
simulator says 59% it happens 58% of the time). Adding the simulator's odds to
the learned model did not improve it, so the learned model gives the win odds
and the simulator answers what-if questions such as "what if he pits now?".

One bug found along the way: the pre-race tyre model's sudden "cliff" for old
tyres gave cars on long stints a phantom speed-up once they pitted, because
their real lap times showed no cliff. Mid-race, tyres wear steadily instead.

### Race replay and pit advisor

The replay page loads one race at a time (`replay/data/<season>-<round>.json`,
written by `f1sim/replay.py` when the site is built): the race state after every
lap, and live win odds from `winprob`. The pit advisor runs in the browser
(`f1sim/web/replay-engine.js`, a port of `f1sim/live.py`): from the chosen lap it
reruns the rest of the race 400 times for each option (the driver's best
remaining plans, and pitting now for each dry compound) and ranks them by
expected finishing position. `tests/test_replay.py` checks that it agrees with
the Python simulator. Known limitation: in races where the rules or the tyres
make it unclear who still has to stop (wet races, very low tyre wear), the odds
can swing briefly during the pit cycle.

### Live page

During a Grand Prix the live page polls OpenF1 every 20 seconds from the
visitor's browser (OpenF1 allows cross-origin requests) and runs the whole
pipeline there: `f1sim/web/live-feed.js` ports the lap table
(`laps.build_rows`), the race state (`live.race_state`) and the win model
(`winprob`), and the replay page's chart and pit advisor display the result.
`tests/test_live_page.py` checks the port against Python on saved OpenF1 data,
including a feed cut off mid-race. Live data differs from history in one way
that needed handling: when the leader completes a lap the rest of the field is
still on it, so the page shows the last lap the whole lead-lap field has
completed.

Whether OpenF1 shares data during a session for free could not be tested before
a race weekend; if it does not, the page says so and shows the race once the
data is published. Between races, "watch the last race as if live" feeds the
latest race through the same pipeline in time order.

## How the model works

Each simulation runs qualifying, then the race lap by lap for every car:

```
lap time = circuit base lap + car pace + driver pace
         + tyre penalty (compound offset + wear × age + cliff past tyre life)
         + fuel load + random variation (driver consistency)
```

- **Strategy** — for each circuit the 1- and 2-stop strategies are searched for
  the fastest clean-air race (two-compound rule enforced); each car samples one
  of the best three with pit-lap jitter unless you pin it.
- **Overtaking** — a car that catches the one ahead passes with a probability
  that depends on its pace advantage versus the circuit's `overtake` difficulty;
  otherwise it is held 0.25 s behind. This is what makes grid position matter at Monaco.
- **Safety cars** — random periods that neutralise lap times, bunch the field
  and halve pit loss; cars near their stop window pit opportunistically.
- **Weekend form** — each team's car pace is shifted by a random amount per
  weekend (0.4 s/lap spread, chosen on 2014–2021), because cars suit
  some circuits better than others.
- **Retirements** — mechanical (team `reliability`) and driver error (`error_rate`).
- **Rain** — per-circuit probability; wet races add lap-time variance, more
  driver errors, and each driver's `wet` skill offset (0 for everyone in the
  fitted grid, since results alone can't measure it).

### Known simplifications

No neural network or machine-learned lap model; no sprint races (real sprint
points already scored are carried in the standings), grid penalties, red flags, virtual safety car, team orders,
intermediate/wet tyre choices, or changing weather mid-race; tyre parameters
are generic rather than per-event compound allocations; undercut/overcut only
emerge from tyre age, not from out-lap warm-up.

## Layout

```
f1sim/models.py     Team / Driver / Track / Grid dataclasses and JSON loaders
f1sim/strategy.py   tyre model, Strategy, optimal-strategy search
f1sim/race.py       lap-by-lap race engine (vectorised over simulations)
f1sim/season.py     championship simulation
f1sim/evaluate.py   prediction scoring, backtesting, strategy comparison
f1sim/datasource.py download a season of real results
f1sim/fit.py        fit ratings from results; walk-forward testing
f1sim/benchmark.py  multi-season benchmark against baselines; calibration
f1sim/laps.py       lap-by-lap race data from OpenF1 (2023 onward)
f1sim/tyres.py      pit-stop loss, tyre wear and strategy calibration from lap data
f1sim/live.py       simulate the rest of a race from any lap's real state
f1sim/winprob.py    live win probability; weights in f1sim/data/winprob.json
f1sim/replay.py     per-race data for the replay page
f1sim/web/          page templates, styles and scripts; replay-engine.js and live-feed.js run the
                    pit advisor and the live pipeline in the browser
f1sim/dashboard.py  runs the simulations and builds the website
f1sim/data/engine.js the race engine in JavaScript, for the what-if lab
f1sim/cli.py        command-line interface
f1sim/data/         fitted 2026 grid, 2014–2026 results, circuit parameters
tests/              pytest suite (run `pytest`)
scripts/            build_site.py: refresh data and build the website;
                    calibrate_tracks.py: set circuit values from lap data
.github/workflows/  tests on every push; dashboard publishing to GitHub Pages
```
