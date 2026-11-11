# f1sim — Formula 1 race & season simulator

A Monte Carlo simulator for **predicting** F1 races and championships and for
**testing** ideas (pit strategies, car upgrades, "what if it rains?") against a
consistent model. Pure Python + NumPy; 2,000 full race simulations take well under a second.

> **The bundled ratings are illustrative placeholders, not fitted to real data.**
> The driver/team/circuit numbers in `f1sim/data/*.json` are rough hand-set
> estimates so the tool runs out of the box. Predictions are only as good as
> those inputs — edit them, then use `f1sim backtest` to check them against real results.

## Install

```bash
cd f1-simulator
pip install -e ".[dev]"
```

## Use

```bash
f1sim tracks                                   # circuits and their fastest strategy
f1sim race monaco --seed 1                     # win / podium / points probabilities
f1sim race britain --wet                       # force a wet race (--dry to force dry)
f1sim race bahrain --strategy VER=S-M-M@15,36  # pin one driver's strategy
f1sim strategy bahrain VER                     # compare the 5 fastest strategies for a driver
f1sim strategy bahrain VER M-H@24 H-M@33       # ...or your own candidates
f1sim season --sims 1000                       # championship probabilities
f1sim season --start qatar --standings pts.json  # finish a season from current points
f1sim backtest results.json                    # score predictions against real results
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

## Testing predictions against reality

Put real results in a JSON file (finishing order as driver codes) and run `f1sim backtest`:

```json
[
  {"track": "bahrain", "results": ["VER", "NOR", "LEC", "..."], "wet": false},
  {"track": "monaco",  "results": ["LEC", "PIA", "NOR", "..."]}
]
```

For each race it reports the probability the model gave the real winner, log
loss, Brier score, ranked probability score (RPS, 0 = perfect) and the rank
correlation between predicted and actual order. Tune the ratings, re-run, and
keep changes that improve the mean scores.

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
- **Retirements** — mechanical (team `reliability`) and driver error (`error_rate`).
- **Rain** — per-circuit probability; wet races add lap-time variance, more
  driver errors, and each driver's `wet` skill offset.

### Known simplifications

No sprint races, grid penalties, red flags, virtual safety car, team orders,
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
f1sim/cli.py        command-line interface
f1sim/data/         illustrative grid and calendar
tests/              pytest suite (run `pytest`)
```
