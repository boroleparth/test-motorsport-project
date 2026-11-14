/* Live race pipeline for the browser: raw OpenF1 data -> lap rows -> race state -> win odds.
 *
 * Ports of f1sim/laps.py (build_rows), f1sim/live.py (race_state) and
 * f1sim/winprob.py (features), producing lap entries in the same format as the
 * replay data so the replay engine (replay-engine.js) can run the pit advisor.
 * tests/test_live_page.py checks the port against the Python pipeline.
 *
 * Works in a browser (window.LiveFeed) and in Node (require).
 */
(function (root) {
  "use strict";

  const API = "https://api.openf1.org/v1";
  const ENDPOINTS = ["drivers", "laps", "stints", "pit", "race_control", "weather"];
  const DRY = { SOFT: 0, MEDIUM: 1, HARD: 2 };
  const CODE = { SOFT: 0, MEDIUM: 1, HARD: 2, INTERMEDIATE: 3, WET: 4 };
  const BIT = { SOFT: 1, MEDIUM: 2, HARD: 4 };
  const CAR_FIELDS = ["code", "gap", "compound", "age", "stops", "used", "pace", "pace_laps", "laps_done", "dnf_rate"];
  const PRIOR_LAPS = 5, RECENT = 10, CLEAN_GAP = 1.0, DNF_RATE = 0.06;
  const ts = s => (s ? Date.parse(s) / 1000 : null);
  const median = a => { const s = a.slice().sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

  /** Fetch every endpoint for a session. Returns {status, raw} where status is "ok", "empty" or an HTTP code. */
  async function fetchSession(sessionKey, fetchFn) {
    const f = fetchFn || fetch, raw = {};
    for (const ep of ENDPOINTS) {
      const res = await f(`${API}/${ep}?session_key=${sessionKey}`);
      if (!res.ok) return { status: res.status, raw: null };
      raw[ep] = await res.json();
    }
    return { status: raw.laps.length ? "ok" : "empty", raw };
  }

  function neutralised(messages, lapsTotal) {
    const status = {}, open = {};
    const sorted = messages.slice().sort((a, b) => (a.date || "").localeCompare(b.date || ""));
    for (const m of sorted) {
      const lap = m.lap_number, text = (m.message || "").toUpperCase(), flag = (m.flag || "").toUpperCase();
      if (lap == null) continue;
      let kind = null;
      if (text.includes("VIRTUAL SAFETY CAR") || text.startsWith("VSC")) kind = "VSC";
      else if (text.includes("SAFETY CAR")) kind = "SC";
      if (flag === "RED") open.RED = lap;
      else if ("RED" in open && (flag === "GREEN" || text.includes("RESUME") || text.includes("RESTART"))) {
        for (let k = open.RED; k <= lap; k++) status[k] = "RED";
        delete open.RED;
      }
      if (kind && text.includes("DEPLOYED")) { if (!(kind in open)) open[kind] = lap; }
      else if (kind && (text.includes("ENDING") || text.includes("IN THIS LAP")) && kind in open) {
        for (let k = open[kind]; k <= lap; k++) if (!(k in status)) status[k] = kind;
        delete open[kind];
      }
    }
    for (const [kind, start] of Object.entries(open))
      for (let k = start; k <= lapsTotal; k++) if (!(k in status)) status[k] = kind;
    return status;
  }

  /** Port of f1sim.laps.build_rows (without season/round/track). */
  function buildRows(raw) {
    const drivers = {}; raw.drivers.forEach(d => { drivers[d.driver_number] = d; });
    const laps = {};
    raw.laps.forEach(l => { (laps[l.driver_number] = laps[l.driver_number] || {})[l.lap_number] = l; });
    const end = {};
    for (const [num, byLap] of Object.entries(laps))
      for (const [n, lap] of Object.entries(byLap)) {
        const nxt = byLap[+n + 1];
        let t = nxt && nxt.date_start ? ts(nxt.date_start) : null;
        if (t === null && lap.date_start && lap.lap_duration) t = ts(lap.date_start) + lap.lap_duration;
        if (t !== null) (end[num] = end[num] || {})[n] = t;
      }
    const nums = Object.keys(end);
    if (!nums.length) return [];
    const starts = Object.values(laps).filter(b => b[1] && b[1].date_start).map(b => ts(b[1].date_start));
    const t0 = starts.length ? Math.min(...starts) : Math.min(...nums.map(n => Math.min(...Object.values(end[n])))) - 100;
    const lapsTotal = Math.max(...nums.map(n => Math.max(...Object.keys(end[n]).map(Number))));
    const pos = {}, gapL = {}, gapA = {};
    for (let n = 1; n <= lapsTotal; n++) {
      const done = nums.filter(num => end[num][n] !== undefined).map(num => [end[num][n], num]).sort((a, b) => a[0] - b[0] || (a[1] < b[1] ? -1 : 1));
      done.forEach(([t, num], p) => {
        const k = num + ":" + n;
        pos[k] = p + 1; gapL[k] = t - done[0][0]; gapA[k] = p ? t - done[p - 1][0] : 0;
      });
    }
    const last = {}; nums.forEach(n => { last[n] = Math.max(...Object.keys(end[n]).map(Number)); });
    const finish = nums.slice().sort((a, b) => last[b] - last[a] || end[a][last[a]] - end[b][last[b]]);
    const final = {}; finish.forEach((n, i) => { if (last[n] >= 0.9 * lapsTotal) final[n] = i + 1; });
    const stints = {};
    raw.stints.forEach(st => { for (let n = st.lap_start || 1; n <= (st.lap_end || lapsTotal); n++) stints[st.driver_number + ":" + n] = st; });
    const pits = {}; raw.pit.forEach(p => { if (p.lap_number) pits[p.driver_number + ":" + p.lap_number] = p; });
    const status = neutralised(raw.race_control, lapsTotal);
    const rain = raw.weather.filter(w => w.rainfall).map(w => ts(w.date));
    const rows = [];
    for (const num of nums) {
      const d = drivers[num] || {};
      for (const n of Object.keys(end[num]).map(Number).sort((a, b) => a - b)) {
        const lap = laps[num][n] || {}, st = stints[num + ":" + n], k = num + ":" + n;
        const tStart = lap.date_start ? ts(lap.date_start) : end[num][n] - (lap.lap_duration || 90);
        const pit = pits[k], nextLap = laps[num][n + 1];
        rows.push({
          driver: d.name_acronym || String(num), team: d.team_name || "", lap: n, lap_time: lap.lap_duration ?? null,
          end_time: end[num][n] - t0, position: pos[k], gap_leader: gapL[k], gap_ahead: gapA[k],
          compound: st ? st.compound || "" : "",
          tyre_age: st ? (st.tyre_age_at_start || 0) + n - st.lap_start + 1 : null,
          stint: st ? st.stint_number : null,
          pit_in: pit || (nextLap && nextLap.is_pit_out_lap) ? 1 : 0, pit_out: lap.is_pit_out_lap ? 1 : 0,
          status: status[n] || "GREEN", rain: rain.some(t => tStart <= t && t <= end[num][n]) ? 1 : 0,
          laps_total: lapsTotal, final_position: final[num] ?? null,
        });
      }
    }
    return rows;
  }

  function raceDeg(rows, lap, track, C) {
    const stints = {};
    for (const r of rows)
      if (r.lap <= lap && r.lap > 1 && r.lap_time && r.status === "GREEN" && !r.rain && !r.pit_in && !r.pit_out
          && r.compound in DRY && (r.position === 1 || (r.gap_ahead || 0) > CLEAN_GAP))
        (stints[r.driver + "|" + r.stint + "|" + r.compound] = stints[r.driver + "|" + r.stint + "|" + r.compound] || []).push(r);
    const slopes = [], comps = [];
    for (const [key, laps] of Object.entries(stints)) {
      if (laps.length < 8) continue;
      const x = laps.map(r => r.lap), y = laps.map(r => r.lap_time);
      const mx = x.reduce((a, b) => a + b) / x.length, my = y.reduce((a, b) => a + b) / y.length;
      let sxy = 0, sxx = 0;
      x.forEach((v, i) => { sxy += (v - mx) * (y[i] - my); sxx += (v - mx) ** 2; });
      const b = sxy / sxx;
      if (Math.abs(b) < 0.4) { slopes.push(b); comps.push(DRY[key.split("|")[2]]); }
    }
    if (slopes.length < 3) return null;
    const wear = comps.reduce((a, c) => a + C.WEAR[c], 0) / comps.length;
    const implied = (median(slopes) + C.FUEL_PER_LAP) / wear, w = slopes.length / (slopes.length + 6);
    return Math.min(Math.max(w * implied + (1 - w) * track.deg, 0.3), 2.5);
  }

  /**
   * Port of f1sim.live.race_state, returned as a replay-format lap entry.
   * prior: {code: {pace, dnf}} pre-race ratings; lapsTotal: scheduled race length;
   * live: data stops at the present, so cars still on track have no later laps.
   */
  function raceState(rows, lap, track, C, prior, lapsTotal, live) {
    const L = lapsTotal || Math.max(...rows.map(r => r.laps_total));
    const by = {};
    rows.forEach(r => { (by[r.driver] = by[r.driver] || {})[r.lap] = r; });
    const codes = Object.keys(by).sort();
    const known = codes.filter(c => prior && prior[c]).map(c => prior[c].pace);
    const mid = known.length ? median(known) : 0;
    const ref = {};
    for (let n = 2; n <= lap; n++) {
      const t = Object.values(by).filter(d => d[n] && d[n].lap_time && d[n].status === "GREEN" && !d[n].pit_in && !d[n].pit_out).map(d => d[n].lap_time);
      if (t.length >= 5) ref[n] = median(t);
    }
    const leader = Object.values(by).find(d => d[lap] && d[lap].position === 1);
    const status = leader ? leader[lap].status : "GREEN";
    const wet = Object.values(by).some(d => d[lap] && (d[lap].rain || d[lap].compound === "INTERMEDIATE" || d[lap].compound === "WET"));
    const waived = rows.some(r => r.lap <= lap && (r.compound === "INTERMEDIATE" || r.compound === "WET"));
    const cars = [];
    for (const code of codes) {
      const laps = by[code], done = Object.keys(laps).map(Number).filter(n => n <= lap);
      if (!done.length) continue;
      const lastLap = Math.max(...done), last = laps[lastLap], latest = Math.max(...Object.keys(laps).map(Number));
      // During a race the feed is cut off mid-lap and lapped cars are always behind, so a car
      // counts as running if it completed a lap recently (complete historical data uses later laps).
      const running = lap in laps || ((live || latest > lap) && lastLap >= lap - 3);
      const retired = latest <= lap && last.final_position == null && lap < L;
      let gap = running && !retired ? last.gap_leader : null;
      if (gap !== null && !(lap in laps)) gap = last.gap_leader + (lap - lastLap) * (ref[lap] || 90);
      const resid = [];
      for (const n of done.slice().sort((a, b) => b - a)) {
        const r = laps[n];
        if (n in ref && r.lap_time && r.status === "GREEN" && !r.pit_in && !r.pit_out && !r.rain
            && (r.position === 1 || (r.gap_ahead || 0) > CLEAN_GAP)) resid.push(r.lap_time - ref[n]);
        if (resid.length >= RECENT) break;
      }
      const kept = resid.filter(x => Math.abs(x) < 4);
      const p0 = prior && prior[code] ? prior[code].pace - mid : 0;
      const pace = (kept.reduce((a, b) => a + b, 0) + PRIOR_LAPS * p0) / (kept.length + PRIOR_LAPS);
      let used = 0; done.forEach(n => { used |= BIT[laps[n].compound] || 0; });
      const stops = done.filter(n => n < L && laps[n].pit_in).length;
      cars.push([code, gap === null ? null : gap, CODE[last.compound] ?? 1, last.tyre_age || 1, stops, used,
                 pace, kept.length, lastLap, prior && prior[code] ? prior[code].dnf : DNF_RATE]);
    }
    cars.sort((a, b) => (a[1] === null) - (b[1] === null) || (a[1] === null ? b[8] - a[8] : a[1] - b[1]));
    return { lap, status, wet, waived, deg: raceDeg(rows, lap, track, C), cars };
  }

  /** Port of f1sim.winprob.features + softmax: {code: chance to win}. */
  function winOdds(race, entry, weights, C, Replay) {
    const L = race.track.laps, rem = L - entry.lap, frac = rem / L;
    const run = Replay.cars(race, entry).filter(c => c.running).sort((a, b) => a.gap - b.gap);
    if (!run.length) return {};
    const med = median(run.map(c => c.pace));
    const deg = entry.deg || race.track.deg;
    const owed = run.map(c => Replay.planOptions(C, race, entry, c, deg, 1)[0].pits.length);
    const lo = Math.min(...owed);
    const scores = run.map((c, i) => {
      const p = i + 1, g = c.gap / (rem + 1);
      const x = [g, Math.log(p), Math.log(p) * frac, g * frac, c.pace - med, (owed[i] - lo) * race.track.pit_loss / (rem + 1)];
      return x.reduce((s, v, k) => s + v * weights[k], 0);
    });
    const m = Math.max(...scores), e = scores.map(s => Math.exp(s - m)), tot = e.reduce((a, b) => a + b, 0);
    const out = {}; run.forEach((c, i) => { out[c.code] = e[i] / tot; });
    return out;
  }

  const api = { fetchSession, buildRows, raceState, winOdds, neutralised, CAR_FIELDS, API };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LiveFeed = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
