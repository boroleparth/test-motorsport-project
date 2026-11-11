"""Command-line interface: ``f1sim <command>`` or ``python -m f1sim <command>``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .datasource import fetch_season, load_season
from .evaluate import backtest, compare_strategies
from .fit import fit_grid, grid_to_json, walk_forward
from .models import load_grid, load_tracks
from .race import simulate_race
from .season import simulate_season
from .strategy import Strategy, optimal_strategies


def _table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    """columns: (key, header, format) where format is 'pct', 'f1', 'f2' or 's'."""
    fmt = {"pct": lambda v: f"{100 * v:5.1f}%", "f1": lambda v: f"{v:.1f}", "f2": lambda v: f"{v:.2f}", "f3": lambda v: f"{v:.3f}", "s": str}
    cells = [[fmt[f](r[k]) for k, _, f in columns] for r in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, (_, h, _) in enumerate(columns)]
    line = lambda vals: "  ".join(
        v.ljust(w) if columns[i][2] == "s" else v.rjust(w) for i, (v, w) in enumerate(zip(vals, widths))
    )
    return "\n".join([line([h for _, h, _ in columns]), line(["-" * w for w in widths]), *map(line, cells)])


def _track(tracks, key):
    if key not in tracks:
        sys.exit(f"unknown track {key!r}; run `f1sim tracks` to list them")
    return tracks[key]


def _wet(args):
    return True if args.wet else False if args.dry else None


def cmd_tracks(args, grid, tracks):
    rows = [{**vars(t), "best": str(optimal_strategies(t, top=1)[0])} for t in tracks.values()]
    print(_table(rows, [("key", "key", "s"), ("name", "circuit", "s"), ("laps", "laps", "s"),
                        ("overtake", "overtake", "f1"), ("sc_prob", "SC", "pct"),
                        ("rain_prob", "rain", "pct"), ("deg", "deg", "f2"), ("best", "fastest strategy", "s")]))


def cmd_race(args, grid, tracks):
    track = _track(tracks, args.track)
    pinned = {}
    for item in args.strategy or []:
        code, _, text = item.partition("=")
        pinned[code.upper()] = Strategy.parse(text)
    res = simulate_race(grid, track, args.sims, args.seed, strategies=pinned, wet=_wet(args))
    print(f"{track.name}: {track.laps} laps, {args.sims} simulations, {100 * res.wet.mean():.0f}% wet\n")
    print(_table(res.summary(), [("driver", "drv", "s"), ("team", "team", "s"), ("win", "win", "pct"),
                                 ("podium", "podium", "pct"), ("points", "top10", "pct"), ("pole", "pole", "pct"),
                                 ("dnf", "DNF", "pct"), ("exp_pos", "E[pos]", "f1"), ("exp_points", "E[pts]", "f1")]))


def cmd_season(args, grid, tracks):
    standings = None
    if args.full:
        keys = [k for k in load_season(args.season_file)["calendar"] if k in tracks]
    elif args.start or args.standings:
        keys = list(tracks)
        if args.start:
            _track(tracks, args.start)
            keys = keys[keys.index(args.start):]
        standings = json.loads(Path(args.standings).read_text()) if args.standings else None
    else:  # finish the real season from the current standings
        season = load_season(args.season_file)
        keys = [k for k in season["calendar"][len(season["races"]):] if k in tracks]
        standings = {c: p for c, p in season["standings"].items() if c in grid.codes}
        print(f"After round {len(season['races'])} of {len(season['calendar'])}, real points carried over.")
    if not keys:
        sys.exit("no races left to simulate; use --full to simulate the whole calendar")
    res = simulate_season(grid, [tracks[k] for k in keys], args.sims, args.seed, standings)
    cols = [("name", "name", "s"), ("title", "title", "pct"), ("mean_points", "E[pts]", "f1"),
            ("p05", "5%", "f1"), ("p95", "95%", "f1")]
    print(f"{len(keys)} races ({', '.join(keys) if len(keys) <= 6 else keys[0] + ' ... ' + keys[-1]}), "
          f"{args.sims} simulated seasons\n\nDrivers' championship")
    print(_table(res.drivers(), cols))
    print("\nConstructors' championship")
    print(_table(res.constructors(), cols))


def cmd_strategy(args, grid, tracks):
    track = _track(tracks, args.track)
    cands = [Strategy.parse(s) for s in args.candidates] if args.candidates else None
    rows = compare_strategies(grid, track, args.driver.upper(), cands, args.sims,
                              0 if args.seed is None else args.seed, wet=bool(args.wet))
    print(f"{args.driver.upper()} at {track.name} ({args.sims} simulations per strategy)\n")
    print(_table(rows, [("strategy", "strategy", "s"), ("clean_air", "clean-air s", "f1"), ("exp_pos", "E[pos]", "f2"),
                        ("exp_points", "E[pts]", "f2"), ("win", "win", "pct"), ("podium", "podium", "pct")]))


def cmd_backtest(args, grid, tracks):
    data = load_season(args.results)
    season = data if isinstance(data, dict) else {"races": data}
    if args.walk_forward:
        out = walk_forward(season, tracks, grid, start=args.start_round, n_sims=args.sims, seed=args.seed)
        print(f"Walk-forward: each race predicted from ratings fitted on earlier rounds only.\n")
    else:
        races = [r for r in season["races"] if r["track"] in tracks]
        out = backtest(grid, tracks, races, args.sims, args.seed)
    cols = [("track", "track", "s"), ("winner", "winner", "s"), ("winner_prob", "P(winner)", "pct"),
            ("log_loss", "logloss", "f2"), ("brier", "brier", "f2"), ("rps", "RPS", "f3"),
            ("spearman", "spearman", "f2"), ("podium_hits", "podium hits", "s")]
    mean = {"track": "MEAN", "winner": "", **out["mean"]}
    mean["podium_hits"] = f"{mean['podium_hits']:.2f}"
    print(_table(out["races"] + [mean], cols))


def cmd_fetch(args, grid, tracks):
    season = fetch_season(args.year)
    Path(args.out).write_text(json.dumps(season, separators=(",", ":")))
    print(f"Saved {len(season['races'])} of {len(season['calendar'])} races to {args.out}")


def cmd_fit(args, grid, tracks):
    season = load_season(args.season_file)
    fitted = fit_grid(season, grid, through_round=args.through)
    rounds = [r["round"] for r in season["races"] if args.through is None or r["round"] <= args.through]
    print(f"Fitted from {season.get('season', '?')} rounds {rounds[0]}-{rounds[-1]}\n")
    print(_table([vars(t) for t in fitted.teams.values()],
                 [("name", "team", "s"), ("pace", "pace s/lap", "f3"), ("reliability", "mech DNF", "pct")]))
    print()
    print(_table([vars(d) for d in fitted.drivers],
                 [("code", "drv", "s"), ("team", "team", "s"), ("pace", "race +s", "f3"),
                  ("quali", "quali +s", "f3"), ("error_rate", "error DNF", "pct")]))
    if args.out:
        note = f"Fitted by `f1sim fit` from {season.get('season', '?')} rounds {rounds[0]}-{rounds[-1]}."
        Path(args.out).write_text(json.dumps(grid_to_json(fitted, note), indent=1))
        print(f"\nSaved to {args.out}; use it with --grid {args.out}")


def main(argv: list[str] | None = None) -> None:
    def common(sims: int = 2000) -> argparse.ArgumentParser:
        c = argparse.ArgumentParser(add_help=False)
        c.add_argument("--sims", type=int, default=sims, help="number of Monte Carlo simulations")
        c.add_argument("--seed", type=int, default=None, help="random seed for reproducible output")
        c.add_argument("--grid", help="path to a grid JSON file (default: bundled illustrative grid)")
        c.add_argument("--tracks", dest="tracks_file", help="path to a tracks JSON file")
        return c

    weather = argparse.ArgumentParser(add_help=False)
    g = weather.add_mutually_exclusive_group()
    g.add_argument("--wet", action="store_true", help="force a wet race")
    g.add_argument("--dry", action="store_true", help="force a dry race")

    parser = argparse.ArgumentParser(prog="f1sim", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("tracks", parents=[common()], help="list circuits").set_defaults(func=cmd_tracks)

    p = sub.add_parser("race", parents=[common(), weather], help="predict a single race")
    p.add_argument("track")
    p.add_argument("--strategy", action="append", metavar="DRV=M-H@22", help="pin a driver's strategy (repeatable)")
    p.set_defaults(func=cmd_race)

    p = sub.add_parser("season", parents=[common(500)], help="simulate the championship (default: rest of the real season)")
    p.add_argument("--full", action="store_true", help="simulate the whole calendar from zero points")
    p.add_argument("--start", help="first race to simulate (track key)")
    p.add_argument("--standings", help="JSON file of {driver code: points already scored}")
    p.add_argument("--season-file", help="season file from `f1sim fetch` (default: bundled 2026)")
    p.set_defaults(func=cmd_season)

    p = sub.add_parser("strategy", parents=[common(), weather], help="compare pit strategies for one driver")
    p.add_argument("track")
    p.add_argument("driver")
    p.add_argument("candidates", nargs="*", metavar="M-H@22", help="strategies to test (default: 5 fastest)")
    p.set_defaults(func=cmd_strategy)

    p = sub.add_parser("backtest", parents=[common()], help="score predictions against real results")
    p.add_argument("results", nargs="?", help="season file from `f1sim fetch`, or a JSON list of "
                   '{"track": key, "results": [codes...]} (default: bundled 2026)')
    p.add_argument("--walk-forward", action="store_true",
                   help="refit ratings before each race using earlier rounds only (honest out-of-sample test)")
    p.add_argument("--start-round", type=int, default=5, help="first rounds used only for fitting (walk-forward)")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("fetch", parents=[common()], help="download a season of real results")
    p.add_argument("year")
    p.add_argument("--out", default="season.json")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("fit", parents=[common()], help="fit team and driver ratings from real results")
    p.add_argument("--season-file", help="season file from `f1sim fetch` (default: bundled 2026)")
    p.add_argument("--through", type=int, help="use only races up to this round")
    p.add_argument("--out", help="write the fitted grid to this JSON file")
    p.set_defaults(func=cmd_fit)

    args = parser.parse_args(argv)
    args.func(args, load_grid(args.grid), load_tracks(args.tracks_file))


if __name__ == "__main__":
    main()
