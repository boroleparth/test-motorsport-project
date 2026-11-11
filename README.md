# f1sim — Formula 1 race & season simulator

A Monte Carlo simulator for **predicting** F1 races and championships and for
**testing** ideas (pit strategies, car upgrades, "what if it rains?") against a
consistent model. Pure Python + NumPy; 2,000 full race simulations take well under a second.

The bundled ratings are **fitted from real 2026 results** (rounds 1–20, from the
public Jolpica/Ergast API), and the whole pipeline is benchmarked against
simple baselines on races it has not seen.

**Benchmark: 34 races from 2025–2026**, each predicted from earlier rounds of
its season only. Model settings were tuned on 2022–2024, so these are held out.

| Method | Favourite won | Podium drivers found | Order match | Probability score (0 = perfect) |
|---|---|---|---|---|
| Simulator, before qualifying | 24% | 1.71 of 3 | 0.61 | 0.126 |
| Baseline: championship order | 26% | 1.79 of 3 | 0.59 | 0.129 |
| Simulator, after qualifying | 65% | 2.12 of 3 | 0.67 | 0.113 |
| Baseline: starting grid order | 65% | 2.03 of 3 | 0.67 | 0.116 |

Read honestly: the simulator is slightly ahead of the matching baseline on
ranked probability score in both settings, and level on picking the winner.
With 34 races those margins are small. The baselines are strong, and they are
scored generously (see `f1sim/benchmark.py`). Reproduce with `f1sim benchmark`.

What is *not* measured: lap-time consistency, pit-stop times, wet-weather skill
and the circuit parameters in `tracks.json` are assumptions. This is a
statistical simulator, not a neural network — see "How the model works".

## Install

```bash
cd f1-simulator
pip install -e ".[dev]"
```

## Use

```bash
f1sim dashboard --out dashboard.html           # everything below as one HTML page
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
f1sim benchmark                                # simulator vs baselines over 2022–2026, plus calibration
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

## Dashboard

`f1sim dashboard` runs every simulation and writes one self-contained HTML file
you can open in a browser: race odds and finishing-position heatmap for any
circuit, championship projection from the real standings, a pit-strategy
comparison for each driver at the next race, the walk-forward accuracy record,
and the fitted car and driver pace. It takes about a minute and a half at the
default 2,000 simulations per circuit; `--quick` skips the multi-season
accuracy section.

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

`f1sim benchmark` runs that test over every bundled season (2022–2026) against
two baselines, championship order and starting-grid order, and prints a
calibration table (when the simulator says 30%, how often does it happen).
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
  weekend (0.3 s/lap spread, tuned on the walk-forward test), because cars suit
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
f1sim/dashboard.py  builds the HTML dashboard (template in f1sim/data/dashboard.html)
f1sim/cli.py        command-line interface
f1sim/data/         fitted 2026 grid, 2022–2026 results, circuit parameters
tests/              pytest suite (run `pytest`)
```
