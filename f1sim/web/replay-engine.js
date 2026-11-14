/* Mid-race simulator for the replay page's pit advisor.
 *
 * A port of f1sim/live.py (plan_options and simulate_from), run one simulation
 * at a time. It reads a race state exported by f1sim/replay.py and the shared
 * settings from engine_constants(), so it follows the same rules as the Python
 * version; tests/test_replay.py checks that the two agree.
 *
 * Works in a browser (window.Replay) and in Node (require).
 */
(function (root) {
  "use strict";

  function rng(seed) {
    let a = seed >>> 0, spare = null;
    const u = () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const normal = () => {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let x, y, r;
      do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r >= 1 || r === 0);
      const f = Math.sqrt(-2 * Math.log(r) / r);
      spare = y * f;
      return x * f;
    };
    const int = (lo, hi) => lo + Math.floor(u() * (hi - lo)); // [lo, hi)
    const exponential = () => -Math.log(1 - u());
    return { u, normal, int, exponential };
  }

  const NO_PIT = 1e6, BASE = 90, DRY = 3;
  const COMPOUNDS = ["Soft", "Medium", "Hard", "Intermediate", "Wet"];

  /** Car rows of a lap as objects (see CAR_FIELDS in f1sim/replay.py). */
  function cars(race, lapEntry) {
    const f = race.car_fields;
    return lapEntry.cars.map(row => {
      const o = {};
      f.forEach((k, i) => { o[k] = row[i]; });
      o.running = o.gap !== null;
      return o;
    });
  }

  function tyreCost(C, comp, age, deg) {
    const c = Math.min(comp, 2);
    return C.OFFSET[c] + C.WEAR[c] * deg * age;
  }

  /** Best remaining plans for one car, lowest clean-air cost first: [{pits, comps, cost}]. */
  function planOptions(C, race, lapEntry, car, deg, top = 3) {
    const L = race.track.laps, n0 = lapEntry.lap, left = L - n0;
    if (left <= 1 || car.compound >= DRY) return [{ pits: [], comps: [], cost: 0 }];
    const c0 = car.compound, a0 = car.age, maxAge = a0 + left + 2;
    const cum = [0, 1, 2].map(c => {
      const out = new Float64Array(maxAge + 1);
      for (let a = 1; a <= maxAge; a++) out[a] = out[a - 1] + tyreCost(C, c, a, deg);
      return out;
    });
    const cur = k => cum[c0][a0 + k] - cum[c0][a0];
    const fresh = (c, k) => cum[c][k];
    const loss = race.track.pit_loss;
    const used = car.used | (1 << c0);
    const need = (used & (used - 1)) === 0 && !(lapEntry.wet || lapEntry.waived);
    const best = new Map();
    const add = (key, cost, plan) => { if (!best.has(key) || cost < best.get(key).cost) best.set(key, { ...plan, cost }); };
    if (!need) add("stay", cur(left), { pits: [], comps: [] });
    for (let k = lapEntry.status !== "GREEN" ? 0 : 1; k < left; k++)
      for (let c = 0; c < 3; c++) {
        if (need && c === c0) continue;
        add("1-" + c, cur(k) + loss + fresh(c, left - k), { pits: [n0 + Math.max(k, 1)], comps: [c] });
      }
    if (left > 8)
      for (let k1 = 1; k1 < left - 5; k1++) {
        const base1 = cur(k1) + 2 * loss;
        for (let c1 = 0; c1 < 3; c1++) for (let c2 = 0; c2 < 3; c2++) {
          if (need && c0 === c1 && c1 === c2) continue;
          let bestK = -1, bestT = Infinity;
          for (let k2 = k1 + 3; k2 < left - 2; k2++) {
            const t = base1 + fresh(c1, k2 - k1) + fresh(c2, left - k2);
            if (t < bestT) { bestT = t; bestK = k2; }
          }
          if (bestK >= 0) add(`2-${c1}-${c2}`, bestT, { pits: [n0 + k1, n0 + bestK], comps: [c1, c2] });
        }
      }
    const ranked = [...best.values()].sort((a, b) => a.cost - b.cost).slice(0, top);
    return ranked.length ? ranked : [{ pits: [], comps: [], cost: 0 }];
  }

  /** Describe a plan in words. */
  function describe(plan) {
    if (!plan.pits.length) return "Stay out";
    return plan.pits.map((p, i) => `Pit lap ${p} for ${COMPOUNDS[plan.comps[i]].toLowerCase()}`).join(", then ");
  }

  /**
   * Simulate the rest of the race from one lap. opts: {sims, seed, force: {code: {pits, comps}}}.
   * Returns {codes, P (position matrix), rows: [{code, win, podium, exp_pos, exp_pts}]}.
   */
  function simulateFrom(C, race, lapEntry, opts) {
    opts = opts || {};
    const S = opts.sims || 500, R = rng(opts.seed == null ? 1 : opts.seed);
    const tr = race.track, L = tr.laps, n0 = lapEntry.lap;
    const list = cars(race, lapEntry), N = list.length;
    const deg = lapEntry.deg || tr.deg;
    const codes = list.map(c => c.code);
    const counts = list.map(() => new Float64Array(N));

    if (L - n0 <= 0) {
      const key = list.map(c => c.running ? c.gap : 1e7 - c.laps_done * 1e3);
      const order = [...key.keys()].sort((a, b) => key[a] - key[b]);
      order.forEach((i, p) => { counts[i][p] = S; });
    } else {
      // plan choices per car
      const choices = list.map(car => {
        if (!car.running) return null;
        const forced = opts.force && opts.force[car.code];
        if (forced) return { options: [forced], w: [1], forced: true };
        const options = planOptions(C, race, lapEntry, car, deg);
        const lo = Math.min(...options.map(o => o.cost));
        const w = options.map(o => Math.exp(-(o.cost - lo) / 1.5));
        const tot = w.reduce((a, b) => a + b, 0);
        return { options, w: w.map(x => x / tot), forced: false };
      });
      const sigma = 0.35 * (lapEntry.wet ? C.WET_NOISE : 1);
      const lam = -Math.log(Math.max(1 - tr.sc_prob, 1e-9)) * (lapEntry.wet ? 1.5 : 1), hazard = lam / L;
      const pace = new Float64Array(N), t = new Float64Array(N), nw = new Float64Array(N), age = new Float64Array(N);
      const basePen = new Float64Array(N), dnfLap = new Int32Array(N), stint = new Int32Array(N), isPit = new Uint8Array(N);
      const pits = list.map(() => new Float64Array(3)), comps = list.map(() => new Int32Array(4));
      const sc = new Uint8Array(L + 2), scStart = new Uint8Array(L + 2);
      const idx = [...Array(N).keys()];

      for (let s = 0; s < S; s++) {
        for (let i = 0; i < N; i++) {
          const c = list[i];
          const sd = C.PACE_SPREAD + 0.4 / Math.sqrt(1 + c.pace_laps);
          pace[i] = C.PACE_SCALE * c.pace + R.normal() * sd;
          t[i] = c.running ? c.gap : 0;
          age[i] = c.age; stint[i] = 0;
          basePen[i] = tyreCost(C, c.compound >= DRY ? 1 : c.compound, c.age, deg);
          comps[i].fill(c.compound >= DRY ? 1 : c.compound);
          pits[i].fill(NO_PIT);
          const ch = choices[i];
          if (ch) {
            let r = R.u(), j = 0;
            while (j < ch.w.length - 1 && r >= ch.w[j]) { r -= ch.w[j]; j++; }
            const plan = ch.options[j];
            for (let k = 0; k < plan.pits.length; k++) {
              const p = plan.pits[k];
              const jit = p - n0 > 3 && !ch.forced ? R.int(-2, 3) : 0;
              pits[i][k] = Math.min(Math.max(p + jit, lapEntry.status === "GREEN" ? n0 + 1 : n0), L - 1);
              comps[i][k + 1] = plan.comps[k];
            }
            for (let k = 1; k < 3; k++) if (pits[i][k] < NO_PIT) pits[i][k] = Math.max(pits[i][k], pits[i][k - 1] + 1);
          }
        }
        // safety cars
        sc.fill(0);
        if (lapEntry.status === "SC" || lapEntry.status === "RED") {
          const end = n0 + R.int(1, 4);
          for (let l = n0 + 1; l < Math.min(n0 + 4, L + 2); l++) if (l <= end) sc[l] = 1;
        }
        for (let l = n0 + 1; l <= L; l++) {
          const start = R.u() < hazard && !sc[l - 1], dur = R.int(3, 6);
          if (start) for (let d = 0; d < 5 && l + d <= L; d++) if (d < dur) sc[l + d] = 1;
        }
        for (let l = 1; l <= L + 1; l++) scStart[l] = sc[l] && !sc[l - 1] ? 1 : 0;
        // retirements
        for (let i = 0; i < N; i++) {
          const c = list[i];
          if (!c.running) { dnfLap[i] = n0; continue; }
          const perLap = 1 - Math.pow(1 - c.dnf_rate, 1 / L), pOut = 1 - Math.pow(1 - perLap, L - n0);
          dnfLap[i] = R.u() < pOut ? R.int(n0 + 1, L + 1) : L + 1;
        }
        for (let lap = n0 + 1; lap <= L; lap++) {
          const scNow = sc[lap] === 1;
          for (let i = 0; i < N; i++) {
            const active = lap < dnfLap[i];
            const comp = comps[i][Math.min(stint[i], 3)];
            const nxt = stint[i] >= 3 ? NO_PIT : pits[i][stint[i]];
            let lt = BASE + pace[i] + tyreCost(C, comp, age[i], deg) - basePen[i] - C.FUEL_PER_LAP * (lap - n0) + R.normal() * sigma;
            if (scNow) lt = BASE * C.SC_LAP_FACTOR;
            const opp = scStart[lap] && nxt > lap && nxt - lap <= 6;
            isPit[i] = active && lap < L && (nxt === lap || opp) ? 1 : 0;
            const loss = tr.pit_loss * (scNow ? C.SC_PIT_FACTOR : 1) + R.exponential() * 0.5;
            nw[i] = active ? t[i] + lt + isPit[i] * loss : Infinity;
          }
          if (!scNow) {
            const ord = idx.slice().sort((a, b) => t[a] - t[b]);
            for (let p = 1; p < N; p++) {
              const a = ord[p - 1], b = ord[p], ta = nw[a], tb = nw[b];
              const u = R.u();
              if (!isFinite(ta) || !isFinite(tb) || isPit[a] || isPit[b] || !(tb < ta + C.FOLLOW_GAP)) continue;
              const pPass = 1 / (1 + Math.exp(-(ta + C.FOLLOW_GAP - tb - tr.overtake) / C.PASS_SCALE));
              if (u >= pPass) nw[b] = ta + C.FOLLOW_GAP;
            }
          } else {
            const ord = idx.slice().sort((a, b) => nw[a] - nw[b]);
            const lead = nw[ord[0]];
            ord.forEach((i, r) => { if (lap < dnfLap[i]) nw[i] = Math.min(nw[i], lead + r * C.SC_GAP); });
          }
          for (let i = 0; i < N; i++) {
            if (isPit[i]) { stint[i]++; age[i] = 0; } else age[i] += 1;
            t[i] = nw[i];
          }
        }
        const key = list.map((c, i) => dnfLap[i] <= L
          ? 1e7 - (c.running ? dnfLap[i] - 1 : c.laps_done) * 1e3 + R.u() : t[i]);
        idx.slice().sort((a, b) => key[a] - key[b]).forEach((i, p) => { counts[i][p]++; });
      }
    }
    const P = counts.map(r => Array.from(r, c => c / S));
    const rows = codes.map((code, i) => {
      const r = P[i];
      return { code, win: r[0], podium: r[0] + (r[1] || 0) + (r[2] || 0),
               exp_pos: r.reduce((x, p, k) => x + p * (k + 1), 0),
               exp_pts: r.reduce((x, p, k) => x + p * (C.POINTS[k] || 0), 0) };
    });
    return { codes, P, rows };
  }

  /**
   * Pit advisor: compare options for one driver at one lap.
   * Options: the car's best plans, plus pitting now for each dry compound.
   */
  function advise(C, race, lapEntry, code, opts) {
    opts = opts || {};
    const list = cars(race, lapEntry), car = list.find(c => c.code === code);
    if (!car || !car.running) return [];
    const deg = lapEntry.deg || race.track.deg, L = race.track.laps, n0 = lapEntry.lap;
    const seen = new Set(), options = [];
    const push = (plan, label) => {
      const key = plan.pits.join(",") + "|" + plan.comps.join(",");
      if (seen.has(key)) return;
      seen.add(key); options.push({ plan, label: label || describe(plan) });
    };
    for (const p of planOptions(C, race, lapEntry, car, deg, 4)) push({ pits: p.pits, comps: p.comps });
    if (L - n0 > 2 && car.compound < DRY) {
      const used = car.used | (1 << car.compound);
      for (let c = 0; c < 3; c++) {
        // pitting now must still leave the two-compound rule satisfiable
        const after = used | (1 << c), ok = (after & (after - 1)) !== 0 || lapEntry.wet || lapEntry.waived;
        if (ok) push({ pits: [n0 + 1], comps: [c] }, `Pit now for ${COMPOUNDS[c].toLowerCase()}`);
      }
    }
    return options.map(o => {
      const res = simulateFrom(C, race, lapEntry, { sims: opts.sims || 400, seed: opts.seed || 7, force: { [code]: o.plan } });
      return { ...o, result: res.rows.find(r => r.code === code) };
    }).sort((a, b) => a.result.exp_pos - b.result.exp_pos);
  }

  const api = { simulateFrom, planOptions, advise, describe, cars, COMPOUNDS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Replay = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
