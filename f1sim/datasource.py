"""Download real results from the Jolpica (Ergast-compatible) F1 API.

The season file produced by :func:`fetch_season` is plain JSON, so it can be
committed, edited by hand, and used offline by ``f1sim fit`` and ``f1sim backtest``.
"""
from __future__ import annotations

import json
import time
import urllib.request
from importlib import resources
from pathlib import Path

API = "https://api.jolpi.ca/ergast/f1"

# Jolpica circuitId -> key in tracks.json
CIRCUIT_KEYS = {
    "albert_park": "australia", "shanghai": "china", "suzuka": "japan", "bahrain": "bahrain",
    "jeddah": "saudi", "miami": "miami", "villeneuve": "canada", "monaco": "monaco",
    "catalunya": "spain", "red_bull_ring": "austria", "silverstone": "britain", "spa": "belgium",
    "hungaroring": "hungary", "zandvoort": "netherlands", "monza": "italy", "madring": "madrid",
    "baku": "azerbaijan", "marina_bay": "singapore", "americas": "usa", "rodriguez": "mexico",
    "interlagos": "brazil", "vegas": "las_vegas", "losail": "qatar", "yas_marina": "abu_dhabi",
}


def _get(path: str, pause: float = 0.3) -> dict:
    """Fetch every page of an endpoint and return the merged MRData."""
    merged, offset = None, 0
    while True:
        req = urllib.request.Request(f"{API}/{path}.json?limit=100&offset={offset}",
                                     headers={"User-Agent": "f1sim"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)["MRData"]
        table = next(v for k, v in data.items() if k.endswith("Table"))
        list_key = next(k for k, v in table.items() if isinstance(v, list))
        if merged is None:
            merged = table
        else:  # a race can be split across pages: merge by round
            by_round = {r.get("round"): r for r in merged[list_key]}
            for item in table[list_key]:
                prev = by_round.get(item.get("round"))
                inner = next((k for k, v in item.items() if isinstance(v, list)), None)
                if prev is not None and inner:
                    prev[inner].extend(item[inner])
                else:
                    merged[list_key].append(item)
        offset += int(data["limit"])
        if offset >= int(data["total"]):
            return merged
        time.sleep(pause)


def _lap_seconds(text: str | None) -> float | None:
    if not text:
        return None
    try:
        mins, _, secs = text.rpartition(":")
        return (int(mins) if mins else 0) * 60 + float(secs)
    except ValueError:
        return None


def _code(driver: dict) -> str:
    return driver.get("code") or driver["driverId"][:3].upper()


def fetch_season(year: int | str) -> dict:
    """Download a season: calendar, race results, qualifying times and standings.

    Returns ``{"season", "calendar": [track keys], "drivers": {code: name}, "races": [...],
    "standings": {code: points}}``.
    Each race has ``track``, ``name``, ``round``, ``laps``, ``results`` (codes in
    finishing order) and ``entries`` (per-driver team, grid, laps, finished flag,
    race time in seconds if on the lead lap, best qualifying lap in seconds).
    """
    def key(race):
        cid = race["Circuit"]["circuitId"]
        return CIRCUIT_KEYS.get(cid, cid)

    calendar = [key(r) for r in _get(f"{year}/races")["Races"]]
    quali = {}
    for race in _get(f"{year}/qualifying")["Races"]:
        for q in race["QualifyingResults"]:
            laps = [t for t in (_lap_seconds(q.get(s)) for s in ("Q1", "Q2", "Q3")) if t]
            if laps:
                quali[(race["round"], _code(q["Driver"]))] = min(laps)

    races, names = [], {}
    for race in _get(f"{year}/results")["Races"]:
        rows = sorted(race["Results"], key=lambda r: int(r["position"]))
        winner_laps = int(rows[0]["laps"])
        entries = {}
        for r in rows:
            code = _code(r["Driver"])
            names[code] = f'{r["Driver"]["givenName"]} {r["Driver"]["familyName"]}'
            millis = r.get("Time", {}).get("millis")
            entries[code] = {
                "team": r["Constructor"]["name"],
                "grid": int(r["grid"]),
                "laps": int(r["laps"]),
                "finished": r["positionText"].isdigit(),
                "status": r["status"],
                "time": int(millis) / 1000 if millis else None,
                "quali": quali.get((race["round"], code)),
            }
        races.append({"round": int(race["round"]), "track": key(race), "name": race["raceName"],
                      "laps": winner_laps, "results": [_code(r["Driver"]) for r in rows], "entries": entries})

    lists = _get(f"{year}/driverstandings")["StandingsLists"]
    standings = {_code(s["Driver"]): float(s["points"]) for s in lists[-1]["DriverStandings"]} if lists else {}
    return {"season": str(year), "source": "api.jolpi.ca", "calendar": calendar, "drivers": names,
            "races": sorted(races, key=lambda r: r["round"]), "standings": standings}


def load_season(path: str | Path | None = None) -> dict:
    """Load a season file (the bundled 2026 season if no path)."""
    if path is None:
        return json.loads(resources.files("f1sim").joinpath("data", "results_2026.json").read_text())
    return json.loads(Path(path).read_text())
