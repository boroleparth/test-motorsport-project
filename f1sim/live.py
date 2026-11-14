"""Simulate the rest of a race from any lap's real state.

``race_state(rows, lap, track)`` reads the lap table (``f1sim.laps``) of one
race and returns its state at the end of ``lap``: gaps, tyres, stops, safety-car
status, retirements, and each driver's current pace estimated from their own
recent laps. ``simulate_from(state, track)`` runs the remaining laps many times
and returns finishing-position probabilities. ``replay(rows, track)`` does this
for every lap of a race, which is what the replay page and the live mode show.

The lap-by-lap rules (overtaking, safety cars, pit-lane loss, retirements) are
the same as the pre-race engine in ``f1sim.race``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import race as R
from .models import Grid, Track
from .strategy import COMPOUNDS, OFFSET, WEAR

DRY = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
PRIOR_LAPS = 5.0         # pre-race rating counts as this many observed laps
RECENT = 10              # laps used to estimate current pace
CLEAN_GAP = 1.0          # s behind the car ahead for a lap to count as clean
DNF_RATE = 0.06          # per-race retirement chance when no rating is known
PACE_SPREAD = 0.15       # uncertainty in each driver's estimated pace, s/lap
PACE_SCALE = 1.5         # share of the estimated pace differences carried into the rest of the race


@dataclass
class Car:
    code: str
    team: str
    gap: float                 # s behind the leader at the end of the lap; inf if retired
    laps_done: int
    compound: str              # SOFT/MEDIUM/HARD, or INTERMEDIATE/WET
    tyre_age: int
    stops: int
    used: set = field(default_factory=set)   # dry compounds used so far
    pace: float = 0.0          # s/lap relative to the field's typical pace (lower is faster)
    pace_laps: int = 0         # clean laps behind the estimate
    dnf_rate: float = DNF_RATE

    @property
    def running(self) -> bool:
        return np.isfinite(self.gap)


@dataclass
class RaceState:
    lap: int
    laps_total: int
    cars: list[Car]
    status: str = "GREEN"      # leader's status on this lap
    wet: bool = False
    deg: float | None = None   # tyre-wear multiplier estimated from this race
    rule_waived: bool = False  # wet-weather tyres used: the two-dry-compound rule no longer applies

    @property
    def remaining(self) -> int:
        return self.laps_total - self.lap


# ---------------------------------------------------------------- state -----

def _prior_pace(grid: Grid | None, codes: list[str]) -> dict[str, float]:
    if grid is None:
        return {}
    by = {d.code: grid.teams[d.team].pace + d.pace for d in grid.drivers}
    known = [by[c] for c in codes if c in by]
    if not known:
        return {}
    mid = float(np.median(known))
    return {c: by[c] - mid for c in codes if c in by}


def _race_deg(rows: list[dict], lap: int, track: Track) -> float | None:
    """Wear multiplier implied by this race's own completed clean stints so far."""
    stints: dict[tuple, list] = {}
    for r in rows:
        if (r["lap"] <= lap and r["lap"] > 1 and r["lap_time"] and r["status"] == "GREEN" and not r["rain"]
                and not r["pit_in"] and not r["pit_out"] and r["compound"] in DRY
                and (r["position"] == 1 or (r["gap_ahead"] or 0) > CLEAN_GAP)):
            stints.setdefault((r["driver"], r["stint"], r["compound"]), []).append(r)
    slopes, comps = [], []
    for (_, _, comp), laps in stints.items():
        if len(laps) >= 8:
            x = np.array([r["lap"] for r in laps], float)
            y = np.array([r["lap_time"] for r in laps], float)
            b = np.polyfit(x, y, 1)[0]
            if abs(b) < 0.4:
                slopes.append(b)
                comps.append(DRY[comp])
    if len(slopes) < 3:
        return None
    # model's in-stint slope is wear * deg minus the fuel gain; solve for deg
    wear = float(np.mean([WEAR[c] for c in comps]))
    implied = (float(np.median(slopes)) + R.FUEL_PER_LAP) / wear
    weight = len(slopes) / (len(slopes) + 6.0)          # shrink towards the circuit value
    return float(np.clip(weight * implied + (1 - weight) * track.deg, 0.3, 2.5))


def race_state(rows: list[dict], lap: int, track: Track, grid: Grid | None = None) -> RaceState:
    """State of one race at the end of ``lap`` (rows = that race's lap table)."""
    laps_total = max(r["laps_total"] for r in rows)
    by_driver: dict[str, dict[int, dict]] = {}
    for r in rows:
        by_driver.setdefault(r["driver"], {})[r["lap"]] = r
    codes = sorted(by_driver)
    prior = _prior_pace(grid, codes)

    # field reference lap time per lap, from clean laps
    ref: dict[int, float] = {}
    for n in range(2, lap + 1):
        times = [d[n]["lap_time"] for d in by_driver.values() if n in d and d[n]["lap_time"]
                 and d[n]["status"] == "GREEN" and not d[n]["pit_in"] and not d[n]["pit_out"]]
        if len(times) >= 5:
            ref[n] = float(np.median(times))

    leader_status = next((d[lap]["status"] for d in by_driver.values() if lap in d and d[lap]["position"] == 1),
                         "GREEN")
    wet = any(d[lap]["rain"] or d[lap]["compound"] in ("INTERMEDIATE", "WET")
              for d in by_driver.values() if lap in d)
    waived = any(r["compound"] in ("INTERMEDIATE", "WET") for r in rows if r["lap"] <= lap)
    cars = []
    for code in codes:
        laps = by_driver[code]
        done = [n for n in laps if n <= lap]
        last = laps[max(done)] if done else None
        if last is None:
            continue
        # running if the driver completed this lap, or is the slowest car still lapped but moving
        latest_lap = max(laps)
        running = lap in laps or (latest_lap > lap and max(done) >= lap - 3)
        retired = latest_lap <= lap and (last["final_position"] is None) and lap < laps_total
        gap = last["gap_leader"] if (running and not retired) else np.inf
        if running and lap not in laps:   # a lapped car: add the laps it is down
            gap = last["gap_leader"] + (lap - max(done)) * (ref.get(lap) or 90.0)
        resid = []
        for n in sorted(done)[::-1]:
            r = laps[n]
            if (n in ref and r["lap_time"] and r["status"] == "GREEN" and not r["pit_in"] and not r["pit_out"]
                    and not r["rain"] and (r["position"] == 1 or (r["gap_ahead"] or 0) > CLEAN_GAP)):
                resid.append(r["lap_time"] - ref[n])
            if len(resid) >= RECENT:
                break
        resid = [x for x in resid if abs(x) < 4]
        p0 = prior.get(code, 0.0)
        pace = (sum(resid) + PRIOR_LAPS * p0) / (len(resid) + PRIOR_LAPS)
        used = {laps[n]["compound"] for n in done if laps[n]["compound"] in DRY}
        cars.append(Car(code=code, team=last["team"], gap=float(gap), laps_done=max(done),
                        compound=last["compound"] or "MEDIUM", tyre_age=last["tyre_age"] or 1,
                        stops=sum(laps[n]["pit_in"] for n in done if n < laps_total), used=used,
                        pace=pace, pace_laps=len(resid)))
    if grid is not None:
        rel = {d.code: 1 - (1 - grid.teams[d.team].reliability) * (1 - d.error_rate) for d in grid.drivers}
        for c in cars:
            c.dnf_rate = rel.get(c.code, DNF_RATE)
    return RaceState(lap=lap, laps_total=laps_total, cars=cars, status=leader_status, wet=wet,
                     deg=_race_deg(rows, lap, track), rule_waived=waived)


# ------------------------------------------------------------- strategy -----

def tyre_cost(compound, age, deg: float):
    """Mid-race tyre penalty, s: compound offset plus steady wear, with no cliff.

    Real lap times stay nearly flat through a stint (see f1sim.tyres), and the
    pace estimate already reflects each car's current tyres, so the pre-race
    model's sudden cliff for old tyres would hand cars on long stints a phantom
    speed-up once they pit.
    """
    compound = np.asarray(compound)
    return OFFSET[compound] + WEAR[compound] * deg * np.asarray(age, dtype=float)


@dataclass(frozen=True)
class Plan:
    """Remaining stops: laps (absolute, end of lap) and the compound fitted at each."""
    pits: tuple[int, ...] = ()
    comps: tuple[int, ...] = ()

    def __str__(self) -> str:
        if not self.pits:
            return "stay out"
        return ", ".join(f"pit lap {p} for {COMPOUNDS[c]}" for p, c in zip(self.pits, self.comps))


def plan_options(car: Car, state: RaceState, track: Track, deg: float, top: int = 3) -> list[tuple[Plan, float]]:
    """Best remaining plans for one car by clean-air time cost (lowest first)."""
    left = state.remaining
    if left <= 1 or car.compound not in DRY:
        return [(Plan(), 0.0)]
    c0, a0 = DRY[car.compound], car.tyre_age
    ages = np.arange(0, a0 + left + 2)
    pen = np.stack([tyre_cost(c, ages, deg) for c in range(3)])            # pen[c, age]
    cum = np.concatenate([np.zeros((3, 1)), np.cumsum(pen[:, 1:], axis=1)], axis=1)  # cum[c, k] = sum age 1..k
    cur = lambda k: cum[c0, a0 + k] - cum[c0, a0]                          # k more laps on current set
    fresh = lambda c, k: cum[c, k]                                         # k laps on a new set
    loss = track.pit_loss
    need = len(car.used | {car.compound}) < 2 and not (state.wet or state.rule_waived)  # two-compound rule unmet
    best: dict[tuple, tuple[float, Plan]] = {}

    def add(key, cost, plan):
        if key not in best or cost < best[key][0]:
            best[key] = (cost, plan)

    if not need:
        add(("stay",), cur(left), Plan())
    for k in range(0 if state.status != "GREEN" else 1, left):
        for c in range(3):
            if need and c == c0:
                continue
            add((1, c), cur(k) + loss + fresh(c, left - k), Plan((state.lap + max(k, 1),), (c,)))
    if left > 8:
        for k1 in range(1, left - 5):
            base1 = cur(k1) + 2 * loss
            for c1 in range(3):
                for c2 in range(3):
                    if need and c0 == c1 == c2:
                        continue
                    ks = np.arange(k1 + 3, left - 2)
                    if not len(ks):
                        continue
                    tot = base1 + fresh(c1, ks - k1) + fresh(c2, left - ks)
                    j = int(np.argmin(tot))
                    add((2, c1, c2), float(tot[j]), Plan((state.lap + k1, state.lap + int(ks[j])), (c1, c2)))
    ranked = sorted(best.values(), key=lambda v: v[0])[:top]
    return [(p, float(c)) for c, p in ranked] or [(Plan(), 0.0)]


# ------------------------------------------------------------ simulation ----

@dataclass
class Outcome:
    codes: list[str]
    positions: np.ndarray       # (sims, cars), 1-based

    def matrix(self) -> np.ndarray:
        n = self.positions.shape[1]
        return np.stack([np.bincount(self.positions[:, i] - 1, minlength=n) for i in range(n)]) / len(self.positions)

    def summary(self) -> list[dict]:
        P = self.matrix()
        pts = np.zeros(len(self.codes))
        pts[: min(10, len(pts))] = R.POINTS[: min(10, len(pts))]
        rows = [{"driver": c, "win": float(P[i, 0]), "podium": float(P[i, :3].sum()),
                 "exp_pos": float(P[i] @ np.arange(1, len(self.codes) + 1)),
                 "exp_points": float(P[i] @ pts)} for i, c in enumerate(self.codes)]
        return sorted(rows, key=lambda r: r["exp_pos"])


def simulate_from(state: RaceState, track: Track, n_sims: int = 1000, seed: int | None = None,
                  force: dict[str, Plan] | None = None) -> Outcome:
    """Run the rest of the race ``n_sims`` times from ``state``.

    ``force`` pins drivers (by code) to a remaining plan, for the pit advisor.
    """
    rng = np.random.default_rng(seed)
    cars = state.cars
    S, N, n0, L = n_sims, len(cars), state.lap, state.laps_total
    rows = np.arange(S)
    deg = state.deg or track.deg
    running0 = np.array([c.running for c in cars])
    gap = np.array([c.gap if c.running else 0.0 for c in cars])
    codes = [c.code for c in cars]

    if state.remaining <= 0:
        key = np.where(running0, gap, 1e7 - np.array([c.laps_done for c in cars]) * 1e3)
        pos = key.argsort().argsort() + 1
        return Outcome(codes, np.tile(pos, (S, 1)))

    # pace: estimate plus uncertainty that shrinks with laps observed
    pace_sd = PACE_SPREAD + 0.4 / np.sqrt(1 + np.array([c.pace_laps for c in cars]))
    pace = PACE_SCALE * np.array([c.pace for c in cars]) + rng.normal(0, 1, (S, N)) * pace_sd
    sigma = 0.35 * (R.WET_NOISE if state.wet else 1.0)

    # remaining plans: each car samples one of its best options; forced cars take theirs
    pits = np.full((S, N, 3), R.NO_PIT)
    comps = np.zeros((S, N, 4), int)
    comp_now = np.array([DRY.get(c.compound, 1) for c in cars])
    age0 = np.array([c.tyre_age for c in cars], float)
    for i, car in enumerate(cars):
        comps[:, i, :] = comp_now[i]
        if not car.running:
            continue
        if force and car.code in force:
            options, w = [force[car.code]], np.array([1.0])
        else:
            opts = plan_options(car, state, track, deg)
            options = [p for p, _ in opts]
            costs = np.array([c for _, c in opts])
            w = np.exp(-(costs - costs.min()) / 1.5)
            w /= w.sum()
        pick = rng.choice(len(options), size=S, p=w)
        for j, plan in enumerate(options):
            m = pick == j
            for k, (p, c) in enumerate(zip(plan.pits, plan.comps)):
                jitter = rng.integers(-2, 3, size=m.sum()) if (p - n0 > 3 and not (force and car.code in force)) else 0
                pits[m, i, k] = np.clip(p + jitter, n0 + 1 if state.status == "GREEN" else n0, L - 1)
                comps[m, i, k + 1] = c
        for k in (1, 2):
            pits[:, i, k] = np.where(pits[:, i, k] < R.NO_PIT, np.maximum(pits[:, i, k], pits[:, i, k - 1] + 1), R.NO_PIT)

    # safety cars over the remaining laps
    lam = -np.log(max(1 - track.sc_prob, 1e-9)) * (1.5 if state.wet else 1.0)
    hazard = lam / L
    sc = np.zeros((S, L + 2), bool)
    if state.status in ("SC", "RED"):
        end = n0 + rng.integers(1, 4, size=S)
        for l in range(n0 + 1, min(n0 + 4, L + 2)):
            sc[:, l] |= l <= end
    for l in range(n0 + 1, L + 1):
        start = (rng.random(S) < hazard) & ~sc[:, l - 1]
        dur = rng.integers(3, 6, size=S)
        for d in range(5):
            idx = l + d
            if idx <= L:
                sc[:, idx] |= start & (d < dur)
    sc_start = sc.copy()
    sc_start[:, 1:] &= ~sc[:, :-1]

    # retirements over the remaining laps
    per_lap = 1 - (1 - np.array([c.dnf_rate for c in cars])) ** (1 / L)
    p_out = 1 - (1 - per_lap) ** state.remaining
    out = rng.random((S, N)) < p_out
    dnf_lap = np.where(out, rng.integers(n0 + 1, L + 1, (S, N)), L + 1)
    dnf_lap = np.where(running0, dnf_lap, n0)   # already retired

    stint = np.zeros((S, N), int)
    age = np.tile(age0, (S, 1))
    # the pace estimate already reflects the tyres each car is on now, so future
    # laps add only the change in tyre penalty from that state
    base_pen = tyre_cost(comp_now, age0, deg)
    t = np.tile(gap, (S, 1))
    base = 90.0
    for lap in range(n0 + 1, L + 1):
        active = lap < dnf_lap
        sc_now = sc[:, lap]
        scc = sc_now[:, None]
        comp = np.take_along_axis(comps, stint[..., None], 2)[..., 0]
        nxt = np.take_along_axis(pits, np.minimum(stint, 2)[..., None], 2)[..., 0]
        nxt = np.where(stint >= 3, R.NO_PIT, nxt)
        lt = (base + pace + tyre_cost(comp, age, deg) - base_pen
              - R.FUEL_PER_LAP * (lap - n0) + rng.normal(0, 1, (S, N)) * sigma)
        lt = np.where(scc, base * R.SC_LAP_FACTOR, lt)
        opportunistic = sc_start[:, lap][:, None] & (nxt > lap) & (nxt - lap <= 6)
        pit = active & (lap < L) & ((nxt == lap) | opportunistic)
        loss = track.pit_loss * np.where(scc, R.SC_PIT_FACTOR, 1.0) + rng.exponential(0.5, (S, N))
        new = np.where(active, t + lt + pit * loss, np.inf)
        order = np.argsort(t, axis=1)
        u = rng.random((S, N))
        for p in range(1, N):
            a, b = order[:, p - 1], order[:, p]
            ta, tb = new[rows, a], new[rows, b]
            fight = (np.isfinite(ta) & np.isfinite(tb) & ~sc_now & ~pit[rows, a] & ~pit[rows, b]
                     & (tb < ta + R.FOLLOW_GAP))
            gain = np.where(fight, ta, 0.0) + R.FOLLOW_GAP - np.where(fight, tb, 0.0)
            p_pass = 1.0 / (1.0 + np.exp(-(gain - track.overtake) / R.PASS_SCALE))
            held = fight & (u[:, p] >= p_pass)
            new[rows, b] = np.where(held, ta + R.FOLLOW_GAP, tb)
        if sc_now.any():
            rank = new.argsort(1).argsort(1)
            target = new.min(1, keepdims=True) + rank * R.SC_GAP
            new = np.where(scc & active, np.minimum(new, target), new)
        stint = stint + pit
        age = np.where(pit, 0.0, age + 1.0)
        t = new

    done = np.array([c.laps_done for c in cars])
    key = np.where(dnf_lap <= L, 1e7 - np.where(running0, dnf_lap - 1, done) * 1e3 + rng.random((S, N)), t)
    return Outcome(codes, key.argsort(1).argsort(1) + 1)


def replay(rows: list[dict], track: Track, grid: Grid | None = None, n_sims: int = 500,
           seed: int = 0, every: int = 1) -> list[dict]:
    """Win and podium probabilities after every ``every`` laps of a race."""
    laps_total = max(r["laps_total"] for r in rows)
    out = []
    for lap in range(1, laps_total, every):
        state = race_state(rows, lap, track, grid)
        res = simulate_from(state, track, n_sims, seed=seed + lap)
        P = res.matrix()
        out.append({"lap": lap, "status": state.status,
                    "drivers": {c: {"win": float(P[i, 0]), "podium": float(P[i, :3].sum())}
                                for i, c in enumerate(res.codes)}})
    return out
