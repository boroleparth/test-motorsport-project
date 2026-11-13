"""Lap-by-lap race data from the OpenF1 API (api.openf1.org), 2023 onward.

``download_laps`` fetches every completed Grand Prix of the given seasons and
writes one gzipped CSV per season to ``f1sim/data/laps/``, with one row per
driver per lap. Raw API responses are cached on disk so a rerun only fetches
races it has not seen.

Columns of the lap table:

==============  ===============================================================
season, round   season year and round number among that season's Grands Prix
track           key into tracks.json
driver, team    three-letter code and team name
lap             lap number (1-based)
lap_time        lap time, s (blank when the timing feed has none)
end_time        time from the start of the race to the end of this lap, s
position        running position at the end of the lap, by end_time
gap_leader      gap to the leader at the end of the lap, s
gap_ahead       gap to the car one position ahead, s
compound        SOFT, MEDIUM, HARD, INTERMEDIATE or WET
tyre_age        laps on this set at the end of the lap, including laps from
                before the race (used sets)
stint           stint number, 1-based
pit_in          1 if the driver entered the pit lane at the end of this lap (from pit
                records, or from the next lap being an out lap)
pit_out         1 if this lap started in the pit lane
pit_lane        time in the pit lane, s, on pit_in laps (blank when not recorded)
status          track status on the leader's lap: GREEN, SC, VSC or RED
rain            1 if rainfall was recorded during the lap
laps_total      laps completed by the winner
final_position  classified position by distance and time; blank if the driver
                did not complete 90% of the race
==============  ===============================================================
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.openf1.org/v1"
ENDPOINTS = ("drivers", "laps", "stints", "pit", "race_control", "weather")
DATA = Path(__file__).resolve().parent / "data" / "laps"
COLUMNS = ["season", "round", "track", "driver", "team", "lap", "lap_time", "end_time", "position",
           "gap_leader", "gap_ahead", "compound", "tyre_age", "stint", "pit_in", "pit_out", "pit_lane",
           "status", "rain", "laps_total", "final_position"]

# OpenF1 circuit_short_name -> key in tracks.json
CIRCUITS = {
    "Sakhir": "bahrain", "Jeddah": "saudi", "Melbourne": "australia", "Suzuka": "japan", "Shanghai": "china",
    "Miami": "miami", "Imola": "imola", "Monte Carlo": "monaco", "Montreal": "canada", "Catalunya": "spain",
    "Spielberg": "austria", "Silverstone": "britain", "Hungaroring": "hungary", "Spa-Francorchamps": "belgium",
    "Zandvoort": "netherlands", "Monza": "italy", "Madring": "madrid", "Baku": "azerbaijan",
    "Singapore": "singapore", "Austin": "usa", "Mexico City": "mexico", "Interlagos": "brazil",
    "Las Vegas": "las_vegas", "Lusail": "qatar", "Yas Marina Circuit": "abu_dhabi", "Kuala Lumpur": "sepang",
}


def _get(path: str, pause: float = 0.4, tries: int = 6):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(f"{API}/{path}", headers={"User-Agent": "f1sim"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.load(resp)
            time.sleep(pause)
            return data
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return []
            if exc.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
        time.sleep(2 ** attempt)
    return []


def _ts(text: str | None) -> float | None:
    return dt.datetime.fromisoformat(text).timestamp() if text else None


def race_sessions(year: int) -> list[dict]:
    """Completed Grand Prix race sessions of a season, in date order, with round numbers."""
    now = dt.datetime.now(dt.timezone.utc)
    sessions = sorted(_get(f"sessions?year={year}&session_name=Race"), key=lambda s: s["date_start"])
    out = []
    for rnd, s in enumerate(sessions, 1):
        if s.get("date_end") and dt.datetime.fromisoformat(s["date_end"]) < now:
            out.append({**s, "round": rnd, "track": CIRCUITS.get(s["circuit_short_name"], s["circuit_short_name"])})
    return out


def fetch_race(session: dict, cache: Path) -> dict:
    """Raw OpenF1 data for one race, cached as one JSON file per session."""
    path = cache / f"{session['session_key']}.json"
    if path.exists():
        return json.loads(path.read_text())
    raw = {"session": session}
    for ep in ENDPOINTS:
        raw[ep] = _get(f"{ep}?session_key={session['session_key']}")
    if raw["laps"]:  # only cache races that have data
        cache.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw, separators=(",", ":")))
    return raw


def _neutralised(messages: list[dict], laps_total: int) -> dict[int, str]:
    """Leader lap -> SC, VSC or RED, from race-control messages."""
    status: dict[int, str] = {}
    open_since: dict[str, int] = {}
    for m in sorted(messages, key=lambda m: m.get("date") or ""):
        lap, text, flag = m.get("lap_number"), (m.get("message") or "").upper(), (m.get("flag") or "").upper()
        if lap is None:
            continue
        kind = None
        if "VIRTUAL SAFETY CAR" in text or text.startswith("VSC"):
            kind = "VSC"
        elif "SAFETY CAR" in text:
            kind = "SC"
        if flag == "RED":
            open_since["RED"] = lap
        elif "RED" in open_since and (flag == "GREEN" or "RESUME" in text or "RESTART" in text):
            for k in range(open_since.pop("RED"), lap + 1):
                status[k] = "RED"
        if kind and "DEPLOYED" in text:
            open_since.setdefault(kind, lap)
        elif kind and ("ENDING" in text or "IN THIS LAP" in text) and kind in open_since:
            for k in range(open_since.pop(kind), lap + 1):
                status.setdefault(k, kind)
    for kind, start in open_since.items():  # never closed: assume to the end
        for k in range(start, laps_total + 1):
            status.setdefault(k, kind)
    return status


def build_rows(raw: dict) -> list[dict]:
    """Turn one race's raw OpenF1 data into lap-table rows (see module docstring)."""
    s = raw["session"]
    drivers = {d["driver_number"]: d for d in raw["drivers"]}
    laps: dict[int, dict[int, dict]] = {}
    for lap in raw["laps"]:
        laps.setdefault(lap["driver_number"], {})[lap["lap_number"]] = lap

    # End-of-lap timestamps: the next lap's start, else start + duration.
    end: dict[int, dict[int, float]] = {}
    for num, by_lap in laps.items():
        for n, lap in by_lap.items():
            nxt = by_lap.get(n + 1)
            t = _ts(nxt["date_start"]) if nxt and nxt.get("date_start") else None
            if t is None and lap.get("date_start") and lap.get("lap_duration"):
                t = _ts(lap["date_start"]) + lap["lap_duration"]
            if t is not None:
                end.setdefault(num, {})[n] = t
    if not end:
        return []
    starts = [_ts(by_lap[1]["date_start"]) for by_lap in laps.values() if 1 in by_lap and by_lap[1].get("date_start")]
    t0 = min(starts) if starts else min(min(v.values()) for v in end.values()) - 100
    laps_total = max(max(v) for v in end.values())

    # Running order at the end of each lap.
    pos, gap_leader, gap_ahead = {}, {}, {}
    for n in range(1, laps_total + 1):
        done = sorted((t[n], num) for num, t in end.items() if n in t)
        for p, (t, num) in enumerate(done, 1):
            pos[num, n] = p
            gap_leader[num, n] = t - done[0][0]
            gap_ahead[num, n] = t - done[p - 2][0] if p > 1 else 0.0

    # Classification: most laps, then earliest finish.
    last = {num: max(t) for num, t in end.items()}
    finish = sorted(end, key=lambda num: (-last[num], end[num][last[num]]))
    final = {num: i + 1 for i, num in enumerate(finish) if last[num] >= 0.9 * laps_total}

    stints = {}
    for st in raw["stints"]:
        for n in range(st.get("lap_start") or 1, (st.get("lap_end") or laps_total) + 1):
            stints[st["driver_number"], n] = st
    pits = {(p["driver_number"], p["lap_number"]): p for p in raw["pit"] if p.get("lap_number")}
    status = _neutralised(raw["race_control"], laps_total)
    rain_times = [_ts(w["date"]) for w in raw["weather"] if w.get("rainfall")]

    rows = []
    for num, by_end in end.items():
        d = drivers.get(num, {})
        for n in sorted(by_end):
            lap = laps[num].get(n, {})
            st = stints.get((num, n))
            t_start = _ts(lap.get("date_start")) if lap.get("date_start") else by_end[n] - (lap.get("lap_duration") or 90)
            pit = pits.get((num, n))
            # Some races have no pit records; the next lap starting in the pit lane is the same signal.
            pitted = pit is not None or bool(laps[num].get(n + 1, {}).get("is_pit_out_lap"))
            rows.append({
                "season": s["year"], "round": s["round"], "track": s["track"],
                "driver": d.get("name_acronym") or str(num), "team": d.get("team_name") or "",
                "lap": n, "lap_time": lap.get("lap_duration"), "end_time": round(by_end[n] - t0, 3),
                "position": pos[num, n], "gap_leader": round(gap_leader[num, n], 3),
                "gap_ahead": round(gap_ahead[num, n], 3),
                "compound": (st or {}).get("compound") or "",
                "tyre_age": ((st.get("tyre_age_at_start") or 0) + n - st["lap_start"] + 1) if st else None,
                "stint": (st or {}).get("stint_number"),
                "pit_in": int(pitted), "pit_out": int(bool(lap.get("is_pit_out_lap"))),
                "pit_lane": pit.get("lane_duration") or pit.get("pit_duration") if pit else None,
                "status": status.get(n, "GREEN"),
                "rain": int(any(t_start <= t <= by_end[n] for t in rain_times)),
                "laps_total": laps_total, "final_position": final.get(num),
            })
    return rows


def write_season(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["round"], r["lap"], r["position"]))
    with gzip.open(path, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if r[k] is None else r[k] for k in COLUMNS})


def download_laps(years: list[int], cache: Path, out: Path = DATA, log=print) -> dict[int, int]:
    """Download and write the lap table for each season. Returns races written per season."""
    written = {}
    for year in years:
        rows, n = [], 0
        for s in race_sessions(year):
            raw = fetch_race(s, cache)
            race_rows = build_rows(raw)
            if race_rows:
                rows += race_rows
                n += 1
                log(f"{year} round {s['round']:2d} {s['track']:12s} {len(race_rows):5d} laps")
            else:
                log(f"{year} round {s['round']:2d} {s['track']:12s} no lap data")
        if rows:
            write_season(rows, out / f"laps_{year}.csv.gz")
        written[year] = n
    return written


def load_laps(years: list[int] | None = None, folder: Path = DATA) -> list[dict]:
    """Read the lap table(s) back as dicts with numbers converted."""
    paths = sorted(folder.glob("laps_*.csv.gz"))
    if years is not None:
        paths = [p for p in paths if int(p.name[5:9]) in years]
    ints = {"season", "round", "lap", "position", "tyre_age", "stint", "pit_in", "pit_out", "rain",
            "laps_total", "final_position"}
    floats = {"lap_time", "end_time", "gap_leader", "gap_ahead", "pit_lane"}
    out = []
    for p in paths:
        with gzip.open(p, "rt", newline="") as f:
            for r in csv.DictReader(f):
                for k in ints:
                    r[k] = int(float(r[k])) if r[k] != "" else None
                for k in floats:
                    r[k] = float(r[k]) if r[k] != "" else None
                out.append(r)
    return out
