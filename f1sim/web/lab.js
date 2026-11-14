// ---------------- what-if lab ----------------
const M = D.model;
const LAB = { weather: "wet", pace: {}, strategies: {}, slots: {} };
const teamName = i => M.teams[i].name;
const fmtS = v => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(2) + " s";

function labInit() {
  if (!M) { $("lab").hidden = true; return; }
  const sel = $("lab-race");
  const group = (label, items) => items.length ? `<optgroup label="${label}">${items.map(t => `<option value="${t.key}">${esc(t.name)}</option>`).join("")}</optgroup>` : "";
  sel.innerHTML = group("Still to run", D.tracks.filter(t => t.remaining)) + group("Other circuits", D.tracks.filter(t => !t.remaining));
  sel.value = D.next;
  $("lab-team").innerHTML = M.teams.map(t => `<option>${esc(t.name)}</option>`).join("");
  const dOpts = D.drivers.map(d => `<option value="${d.code}">${d.code} · ${esc(d.name)}</option>`).join("");
  $("lab-sdrv").innerHTML = dOpts; $("lab-gdrv").innerHTML = dOpts;
  $("lab-gpos").innerHTML = `<option value="">Qualifies normally</option>` + D.drivers.map((_, i) => `<option value="${i + 1}">Starts P${i + 1}</option>`).join("");
  labStratOptions();
  for (const w of ["auto", "dry", "wet"]) $("w-" + w).addEventListener("click", () => { LAB.weather = w; labChips(); });
  $("lab-team").addEventListener("change", () => { const v = LAB.pace[$("lab-team").value] || 0; $("lab-pace").value = v; $("lab-paceval").textContent = fmtS(v); });
  $("lab-pace").addEventListener("input", () => {
    const v = +$("lab-pace").value, t = $("lab-team").value;
    if (Math.abs(v) < 1e-9) delete LAB.pace[t]; else LAB.pace[t] = v;
    $("lab-paceval").textContent = fmtS(v); labChips();
  });
  $("lab-race").addEventListener("change", () => { LAB.strategies = {}; labStratOptions(); labChips(); labRun(); });
  $("lab-sdrv").addEventListener("change", labStratOptions);
  $("lab-strat").addEventListener("change", () => {
    const v = $("lab-strat").value, c = $("lab-sdrv").value;
    $("lab-customrow").hidden = v !== "custom"; $("lab-err").hidden = true;
    if (v === "custom") { $("lab-custom").focus(); return; }
    if (v) LAB.strategies[c] = v; else delete LAB.strategies[c];
    labChips();
  });
  const addCustom = () => {
    const c = $("lab-sdrv").value, text = $("lab-custom").value.trim();
    try { F1Sim.parseStrategy(text, M.tracks[$("lab-race").value].laps); }
    catch (e) { $("lab-err").textContent = e.message; $("lab-err").hidden = false; return; }
    $("lab-err").hidden = true; LAB.strategies[c] = text.toUpperCase(); labChips();
  };
  $("lab-customadd").addEventListener("click", addCustom);
  $("lab-custom").addEventListener("keydown", e => { if (e.key === "Enter") addCustom(); });
  $("lab-gdrv").addEventListener("change", () => { $("lab-gpos").value = LAB.slots[$("lab-gdrv").value] || ""; });
  $("lab-gpos").addEventListener("change", () => {
    const c = $("lab-gdrv").value, v = $("lab-gpos").value;
    for (const [k, p] of Object.entries(LAB.slots)) if (k !== c && String(p) === v) delete LAB.slots[k];
    if (v) LAB.slots[c] = +v; else delete LAB.slots[c];
    labChips();
  });
  $("lab-run").addEventListener("click", labRun);
  $("lab-reset").addEventListener("click", () => {
    LAB.weather = "auto"; LAB.pace = {}; LAB.strategies = {}; LAB.slots = {};
    $("lab-pace").value = 0; $("lab-paceval").textContent = fmtS(0); $("lab-gpos").value = ""; labStratOptions(); labChips();
  });
  labChips(); labRun();
}

function labStratOptions() {
  const tr = M.tracks[$("lab-race").value], c = $("lab-sdrv").value, cur = LAB.strategies[c];
  const opts = [`<option value="">Team's usual choice</option>`].concat(tr.candidates.map(s => `<option value="${s}">${esc(plan(s))}</option>`));
  if (cur && !tr.candidates.includes(cur)) opts.push(`<option value="${esc(cur)}">${esc(plan(cur))}</option>`);
  opts.push(`<option value="custom">Your own…</option>`);
  $("lab-strat").innerHTML = opts.join("");
  $("lab-strat").value = cur || "";
  $("lab-customrow").hidden = true; $("lab-err").hidden = true;
}

function labChips() {
  for (const w of ["auto", "dry", "wet"]) $("w-" + w).setAttribute("aria-pressed", LAB.weather === w);
  const chips = [];
  if (LAB.weather !== "auto") chips.push([`${LAB.weather === "wet" ? "Wet" : "Dry"} race`, () => { LAB.weather = "auto"; }]);
  for (const [t, v] of Object.entries(LAB.pace)) chips.push([`${t} ${fmtS(v)}/lap`, () => { delete LAB.pace[t]; if ($("lab-team").value === t) { $("lab-pace").value = 0; $("lab-paceval").textContent = fmtS(0); } }]);
  for (const [c, s] of Object.entries(LAB.strategies)) chips.push([`${c}: ${plan(s)}`, () => { delete LAB.strategies[c]; labStratOptions(); }]);
  for (const [c, p] of Object.entries(LAB.slots)) chips.push([`${c} starts P${p}`, () => { delete LAB.slots[c]; if ($("lab-gdrv").value === c) $("lab-gpos").value = ""; }]);
  $("lab-chips").innerHTML = chips.length ? chips.map(([t], i) => `<span class="chip">${esc(t)}<button data-i="${i}" aria-label="Remove ${esc(t)}">×</button></span>`).join("")
    : `<span class="hint">No changes yet: the run will match the forecast.</span>`;
  $("lab-chips").querySelectorAll("button").forEach(b => b.addEventListener("click", () => { chips[+b.dataset.i][1](); labChips(); }));
}

function labRun() {
  const key = $("lab-race").value, sims = +$("lab-sims").value, btn = $("lab-run");
  btn.disabled = true; btn.textContent = "Running…";
  $("lab-status").textContent = `Simulating ${M.tracks[key].name} ${(2 * sims).toLocaleString("en")} times…`;
  setTimeout(() => {
    const t0 = performance.now(), seed = 20261113;
    let base, alt;
    try {
      base = F1Sim.simulate(M, key, { sims, seed });
      alt = F1Sim.simulate(M, key, { sims, seed, wet: LAB.weather === "auto" ? null : LAB.weather === "wet",
        pace: LAB.pace, strategies: LAB.strategies, startSlot: LAB.slots });
    } catch (e) {
      $("lab-status").textContent = "Could not run: " + e.message; btn.disabled = false; btn.textContent = "Run simulation"; return;
    }
    labShow(key, base, alt, sims, (performance.now() - t0) / 1000);
    btn.disabled = false; btn.textContent = "Run simulation";
  }, 30);
}

function labShow(key, base, alt, sims, secs) {
  const B = Object.fromEntries(base.rows.map(r => [r.code, r]));
  const rows = alt.rows.slice().sort((a, b) => b.exp_pts - a.exp_pts || a.exp_pos - b.exp_pos);
  const name = c => D.drivers.find(d => d.code === c).name;
  const changed = Object.keys(LAB.strategies).length + Object.keys(LAB.slots).length + Object.keys(LAB.pace).length + (LAB.weather !== "auto");
  // the driver whose win chance moved most
  const mover = rows.slice().sort((a, b) => Math.abs(b.win - B[b.code].win) - Math.abs(a.win - B[a.code].win))[0];
  const fav = rows.slice().sort((a, b) => b.win - a.win)[0], bfav = base.rows.slice().sort((a, b) => b.win - a.win)[0];
  const tr = M.tracks[key].name;
  $("lab-headline").textContent = !changed
    ? `${name(fav.code)} is favourite at ${tr}, winning ${pct(fav.win)} of simulated races.`
    : Math.abs(mover.win - B[mover.code].win) < 0.005
      ? `Your changes barely move the win odds at ${tr}; ${name(fav.code)} stays favourite at ${pct(fav.win)}.`
      : `${name(mover.code)}'s chance to win at ${tr} goes from ${pct(B[mover.code].win)} to ${pct(mover.win)}` +
        (fav.code !== bfav.code ? `, and ${name(fav.code)} becomes favourite.` : `.`);
  $("lab-status").textContent = `${sims.toLocaleString("en")} simulations with and without your changes, ${secs.toFixed(1)} s in your browser. ` +
    `Changes in win chance smaller than about ${Math.max(1, Math.round(100 / Math.sqrt(sims)))} percentage points are within run-to-run noise.`;
  const d = (v, b, f, digits) => {
    const x = v - b, cls = Math.abs(x) < (digits ? 0.05 : 0.0005) ? "flat" : x > 0 ? "up" : "down";
    return cls === "flat" ? "" : `<span class="delta ${cls}">${x > 0 ? "+" : "−"}${f(Math.abs(x))}</span>`;
  };
  const pp = v => (v * 100).toFixed(1);
  $("lab-table").innerHTML = `<tr><th>Driver</th><th class="n">Win</th><th class="n">Podium</th><th class="n">Exp. points</th><th class="n">Forecast pts</th></tr>` +
    rows.map(r => {
      const b = B[r.code], moved = Math.abs(r.exp_pts - b.exp_pts) >= 0.5;
      return `<tr class="${moved ? "moved" : ""}"><td class="code">${r.code} <small style="color:var(--ink-3)">${esc(r.team)}</small></td>
        <td class="n">${pct(r.win)}${d(r.win, b.win, pp)}</td><td class="n">${pct(r.podium)}${d(r.podium, b.podium, pp)}</td>
        <td class="n">${r.exp_pts.toFixed(1)}${d(r.exp_pts, b.exp_pts, v => v.toFixed(1), 1)}</td><td class="n base">${b.exp_pts.toFixed(1)}</td></tr>`;
    }).join("");
}


labInit();
