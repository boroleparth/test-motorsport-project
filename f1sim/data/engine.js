/* f1sim race engine in JavaScript, for the dashboard's what-if lab.
 *
 * A line-for-line port of f1sim/race.py, run one simulation at a time instead
 * of vectorised. Every constant and parameter comes from the `model` object
 * that f1sim/dashboard.py exports from the Python side, so the two engines
 * share one source of truth. tests/test_engine_js.py checks that they agree.
 *
 * Works in a browser (window.F1Sim) and in Node (require).
 */
(function (root) {
  "use strict";

  // --- random numbers (seeded, so a scenario and its baseline share draws) ---
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
    const poisson = lam => { const L = Math.exp(-lam); let k = 0, p = 1; do { k++; p *= u(); } while (p > L); return k - 1; };
    const exponential = () => -Math.log(1 - u());
    return { u, normal, int, poisson, exponential };
  }

  const COMPOUNDS = ["S", "M", "H"];

  function tyrePenalty(model, c, age, deg) {
    const T = model.tyres;
    const over = Math.max(age - T.life[c] / deg, 0);
    return T.offset[c] + T.wear[c] * deg * age + T.cliff * over * over;
  }

  /** Parse "M-H@24" or "S-M-M@15,36" into {comps:[...], pits:[...]}, or throw. */
  function parseStrategy(text, laps) {
    const [tyres, stops] = String(text).trim().toUpperCase().split("@");
    const comps = tyres.split("-").map(c => COMPOUNDS.indexOf(c));
    const pits = stops ? stops.split(",").map(x => parseInt(x, 10)) : [];
    if (comps.some(c => c < 0)) throw new Error("Tyres must be S, M or H, joined by dashes, like M-H@24");
    if (comps.length !== pits.length + 1) throw new Error("Give one more tyre than pit stops, like M-H@24 or S-M-H@15,36");
    if (pits.some(p => !(p >= 1))) throw new Error("Pit laps must be whole numbers, like M-H@24");
    for (let i = 1; i < pits.length; i++) if (pits[i] <= pits[i - 1]) throw new Error("Pit laps must go up, like S-M-H@15,36");
    if (pits.length > 3) throw new Error("At most three stops");
    if (laps && pits.length && pits[pits.length - 1] >= laps) throw new Error(`The last stop must be before lap ${laps}`);
    if (new Set(comps).size < 2) throw new Error("Use at least two different tyres; the rules require it in a dry race");
    return { comps, pits };
  }

  /**
   * Simulate a race `sims` times.
   * opts: {sims, seed, wet: null|true|false, pace: {team: +s/lap}, strategies: {code: "M-H@24"},
   *        startSlot: {code: position 1..N}}
   * Returns per-driver probabilities and the position matrix P[d][p].
   */
  function simulate(model, trackKey, opts) {
    opts = opts || {};
    const C = model.constants, tr = model.tracks[trackKey];
    if (!tr) throw new Error("Unknown race " + trackKey);
    const drivers = model.drivers, teams = model.teams;
    const N = drivers.length, L = tr.laps, S = opts.sims || 1000;
    const R = rng(opts.seed == null ? 1 : opts.seed);
    const teamOf = drivers.map(d => d.team);
    const extra = opts.pace || {};
    const basePace = teams.map(t => t.pace + (extra[t.name] || 0));

    const pinned = {};
    for (const [code, text] of Object.entries(opts.strategies || {})) {
      const i = drivers.findIndex(d => d.code === code);
      if (i >= 0 && text) pinned[i] = parseStrategy(text, L);
    }
    const slots = {};
    for (const [code, pos] of Object.entries(opts.startSlot || {})) {
      const i = drivers.findIndex(d => d.code === code);
      if (i >= 0 && pos >= 1 && pos <= N) slots[i] = pos - 1;
    }

    const counts = drivers.map(() => new Float64Array(N));
    const pole = new Float64Array(N), dnfCount = new Float64Array(N);
    let wetRaces = 0;

    const car = new Float64Array(N), q = new Float64Array(N), t = new Float64Array(N), nw = new Float64Array(N);
    const dnfLap = new Int32Array(N), stint = new Int32Array(N), age = new Float64Array(N);
    const pits = drivers.map(() => new Int32Array(4)), comps = drivers.map(() => new Int32Array(4));
    const isPit = new Uint8Array(N), gridPos = new Int32Array(N);
    const scActive = new Uint8Array(L + 2), scStart = new Uint8Array(L + 2);
    const opt = tr.options, optW = opt.map(o => o.weight);
    const NO_PIT = 1e6;

    for (let s = 0; s < S; s++) {
      // weekend form: one draw per team
      const form = teams.map(() => R.normal() * C.FORM_SIGMA);
      for (let i = 0; i < N; i++) car[i] = basePace[teamOf[i]] + form[teamOf[i]];
      const wet = opts.wet == null ? R.u() < tr.rain_prob : !!opts.wet;
      if (wet) wetRaces++;

      // qualifying
      for (let i = 0; i < N; i++) {
        const d = drivers[i];
        q[i] = car[i] + d.quali + (wet ? d.wet : 0) + R.normal() * C.QUALI_SIGMA * (wet ? 2 : 1);
      }
      let order = Array.from({ length: N }, (_, i) => i).sort((a, b) => q[a] - q[b]);
      const fixed = Object.keys(slots).map(Number).sort((a, b) => slots[a] - slots[b]);
      if (fixed.length) {
        order = order.filter(i => !(i in slots));
        for (const i of fixed) order.splice(Math.min(slots[i], order.length), 0, i);
      }
      order.forEach((i, p) => { gridPos[i] = p; });
      pole[order[0]]++;

      // retirements
      for (let i = 0; i < N; i++) {
        const pm = teams[teamOf[i]].reliability;
        const pe = Math.min(drivers[i].error_rate * (wet ? C.WET_ERROR : 1), 0.9);
        const out = R.u() < 1 - (1 - pm) * (1 - pe);
        dnfLap[i] = out ? R.int(1, L + 1) : L + 1;
      }

      // safety cars
      scActive.fill(0); scStart.fill(0);
      const lam = -Math.log(Math.max(1 - tr.sc_prob, 1e-9)) * (wet ? 1.5 : 1);
      const nsc = R.poisson(lam);
      for (let k = 0; k < 3; k++) {
        const st = R.int(2, Math.max(L - 3, 3)), du = R.int(3, 6);
        if (nsc > k) for (let l = st; l < st + du && l <= L; l++) scActive[l] = 1;
      }
      for (let l = 1; l <= L; l++) scStart[l] = scActive[l] && !scActive[l - 1] ? 1 : 0;

      // strategies
      for (let i = 0; i < N; i++) {
        let comp, pit;
        if (pinned[i]) {
          comp = pinned[i].comps; pit = pinned[i].pits;
          for (let k = 0; k < 4; k++) { pits[i][k] = k < pit.length ? pit[k] : NO_PIT; comps[i][k] = comp[Math.min(k, comp.length - 1)]; }
        } else {
          let r = R.u(), j = 0;
          while (j < optW.length - 1 && r >= optW[j]) { r -= optW[j]; j++; }
          comp = opt[j].comps; pit = opt[j].pits;
          for (let k = 0; k < 4; k++) {
            let p = k < pit.length ? Math.min(Math.max(pit[k] + R.int(-2, 3), 1), L - 1) : NO_PIT;
            if (k > 0 && p < NO_PIT) p = Math.max(p, pits[i][k - 1] + 1);
            pits[i][k] = p; comps[i][k] = comp[Math.min(k, comp.length - 1)];
          }
        }
      }

      stint.fill(0); age.fill(0);
      for (let i = 0; i < N; i++) t[i] = gridPos[i] * C.GRID_SPACING;

      for (let lap = 1; lap <= L; lap++) {
        const sc = scActive[lap] === 1;
        for (let i = 0; i < N; i++) {
          const d = drivers[i], active = lap < dnfLap[i];
          const comp = comps[i][stint[i]], next = pits[i][stint[i]];
          let lt = tr.base_lap + car[i] + d.pace + tyrePenalty(model, comp, age[i], tr.deg) + C.FUEL_PER_LAP * (L - lap)
            + R.normal() * d.consistency * (wet ? C.WET_NOISE : 1) + (wet ? C.WET_LAP_FACTOR * tr.base_lap + d.wet : 0);
          if (lap === 1) lt += 4 + R.normal() * C.START_SIGMA;
          if (sc) lt = tr.base_lap * C.SC_LAP_FACTOR;
          const opportunistic = scStart[lap] && next > lap && next - lap <= 6;
          isPit[i] = active && lap < L && (next === lap || opportunistic) ? 1 : 0;
          const loss = tr.pit_loss * (sc ? C.SC_PIT_FACTOR : 1) + (teams[teamOf[i]].pit_mean - 2.5) + R.exponential() * teams[teamOf[i]].pit_sigma;
          nw[i] = active ? t[i] + lt + isPit[i] * loss : Infinity;
        }
        if (lap > 1 && !sc) {
          const ord = Array.from({ length: N }, (_, i) => i).sort((a, b) => t[a] - t[b]);
          for (let p = 1; p < N; p++) {
            const a = ord[p - 1], b = ord[p], ta = nw[a], tb = nw[b];
            if (!isFinite(ta) || !isFinite(tb) || isPit[a] || isPit[b] || !(tb < ta + C.FOLLOW_GAP)) continue;
            const gain = ta + C.FOLLOW_GAP - tb;
            const pPass = 1 / (1 + Math.exp(-(gain - tr.overtake) / C.PASS_SCALE));
            if (R.u() >= pPass) nw[b] = ta + C.FOLLOW_GAP;
          }
        }
        if (sc) {
          const ord = Array.from({ length: N }, (_, i) => i).sort((a, b) => nw[a] - nw[b]);
          const lead = nw[ord[0]];
          ord.forEach((i, r) => { if (lap < dnfLap[i]) nw[i] = Math.min(nw[i], lead + r * C.SC_GAP); });
        }
        for (let i = 0; i < N; i++) {
          if (isPit[i]) { stint[i]++; age[i] = 0; } else age[i] += 1;
          t[i] = nw[i];
        }
      }

      const key = Array.from({ length: N }, (_, i) => dnfLap[i] <= L ? 1e7 - (dnfLap[i] - 1) * 1e3 + R.u() : t[i]);
      Array.from({ length: N }, (_, i) => i).sort((a, b) => key[a] - key[b]).forEach((i, p) => { counts[i][p]++; });
      for (let i = 0; i < N; i++) if (dnfLap[i] <= L) dnfCount[i]++;
    }

    const POINTS = C.POINTS;
    const P = counts.map(row => Array.from(row, c => c / S));
    const rows = drivers.map((d, i) => {
      const r = P[i];
      const sum = (a, b) => r.slice(a, b).reduce((x, y) => x + y, 0);
      return {
        code: d.code, team: teams[teamOf[i]].name, win: r[0], podium: sum(0, 3), points: sum(0, 10),
        pole: pole[i] / S, dnf: dnfCount[i] / S,
        exp_pos: r.reduce((x, p, k) => x + p * (k + 1), 0),
        exp_pts: r.reduce((x, p, k) => x + p * (POINTS[k] || 0), 0),
      };
    });
    return { sims: S, wet: wetRaces / S, rows, P };
  }

  const api = { simulate, parseStrategy, tyrePenalty, COMPOUNDS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.F1Sim = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
