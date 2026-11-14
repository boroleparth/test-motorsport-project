// ---------------- race replay ----------------
const RP = { race: null, lap: 1, timer: null, top: [], cache: {}, W: 760 };
const C = D.C;
const SERIES = ["--c1", "--c2", "--c3", "--c4", "--c5"];

function rpInit() {
  if (!D.races.length) { $("rp-err").textContent = "No race data in this build."; $("rp-err").hidden = false; return; }
  const seasons = [...new Set(D.races.map(r => r.season))].sort((a, b) => b - a);
  $("rp-season").innerHTML = seasons.map(s => `<option>${s}</option>`).join("");
  let want = null;
  try { want = localStorage.getItem("f1sim-replay"); } catch (e) {}
  const start = D.races.find(r => r.file === want) || D.races.filter(r => r.season === seasons[0]).slice(-1)[0];
  $("rp-season").value = start.season;
  rpRaces(start.file);
  $("rp-season").addEventListener("change", () => rpRaces());
  $("rp-race").addEventListener("change", () => rpLoad($("rp-race").value));
}

// Controls shared by the replay and live pages: slider, play, chart clicks, keys, pit advisor.
function rpBind() {
  $("rp-lap").addEventListener("input", () => { rpStop(); rpSetLap(+$("rp-lap").value); });
  $("rp-play").addEventListener("click", () => RP.timer ? rpStop() : rpPlay());
  $("rp-advise").addEventListener("click", rpAdvise);
  $("rp-driver").addEventListener("change", () => { $("rp-adv").innerHTML = ""; rpActual(); $("rp-adv-note").textContent = adviseHint(); });
  $("rp-chart").addEventListener("click", e => {
    const box = $("rp-chart").getBoundingClientRect(), x = (e.clientX - box.left) / box.width * RP.W;
    const L = RP.race.track.laps, lap = Math.round((x - 44) / (RP.W - 44 - 70) * (L - 1)) + 1;
    rpStop(); rpSetLap(Math.max(1, Math.min(L, lap)));
  });
  document.addEventListener("keydown", e => {
    if (!RP.race || e.target.closest("input, select, textarea")) return;
    if (e.key === "ArrowRight") { rpStop(); rpSetLap(Math.min(RP.lap + 1, RP.race.laps[RP.race.laps.length - 1].lap)); }
    if (e.key === "ArrowLeft") { rpStop(); rpSetLap(Math.max(RP.lap - 1, 1)); }
  });
}

function rpRaces(file) {
  const s = +$("rp-season").value, list = D.races.filter(r => r.season === s);
  $("rp-race").innerHTML = list.map(r => `<option value="${r.file}">Round ${r.round} · ${esc(r.name.replace(" " + s, ""))}${r.winner ? " · won by " + r.winner : ""}</option>`).join("");
  $("rp-race").value = file && list.some(r => r.file === file) ? file : list[list.length - 1].file;
  rpLoad($("rp-race").value);
}

async function rpLoad(file) {
  rpStop();
  $("rp-err").hidden = true;
  try {
    if (!RP.cache[file]) {
      const res = await fetch("data/" + file);
      if (!res.ok) throw new Error(res.status);
      RP.cache[file] = await res.json();
    }
  } catch (e) {
    $("rp-err").textContent = location.protocol === "file:"
      ? "Race data loads from the website. Open this page at its web address (or serve the folder), not as a file."
      : "Could not load this race's data. Try reloading the page.";
    $("rp-err").hidden = false; $("rp-main").hidden = true; return;
  }
  try { localStorage.setItem("f1sim-replay", file); } catch (e) {}
  rpShow(RP.cache[file]);
  rpSetLap(Math.min(Math.round(RP.race.track.laps * 0.4), RP.race.track.laps));
}

// Show a race (from a replay file, or built live). Draws the chart; call rpSetLap after.
function rpShow(race) {
  RP.race = race;
  const r = RP.race, L = r.track.laps;
  // drivers worth drawing: the five with the highest win chance at any point
  const peak = {};
  r.laps.forEach(l => Object.entries(l.win).forEach(([c, p]) => { peak[c] = Math.max(peak[c] || 0, p); }));
  // live races: rank by the latest odds so the current contenders are drawn
  if (r.live && r.laps.length) { const now = r.laps[r.laps.length - 1].win; Object.keys(peak).forEach(c => { peak[c] = (now[c] || 0) + peak[c] * 1e-3; }); }
  RP.top = Object.keys(peak).sort((a, b) => peak[b] - peak[a]).slice(0, 5);
  $("rp-lap").max = r.laps[r.laps.length - 1].lap;
  const neutral = r.laps.filter(l => l.status !== "GREEN").length;
  $("rp-sub").textContent = `${r.name}, ${L} laps${r.winner ? `, won by ${r.winner}` : ""}. ` +
    `${neutral ? `${neutral} laps under safety car, virtual safety car or red flag (shaded). ` : ""}Odds from the live model after each lap` +
    (r.live ? "." : "; click the chart or use the arrow keys to move.");
  $("rp-main").hidden = false;
  rpChart();
}

const tyre = (c, age) => `<span class="tyre t${c}">${"SMHIW"[c]}</span><span class="age">${age}</span>`;

function rpSetLap(lap) {
  const r = RP.race, L = r.track.laps, e = r.laps.find(x => x.lap === lap) || r.laps[r.laps.length - 1];
  lap = e.lap;
  RP.lap = lap;
  $("rp-lap").value = lap;
  $("rp-lapval").textContent = `Lap ${lap} / ${L}`;
  const st = { SC: "Safety car", VSC: "Virtual safety car", RED: "Red flag" }[e.status];
  $("rp-status").hidden = !st; $("rp-status").textContent = st || "";
  const cur = $("rp-cursor");
  if (cur) { const x = rpX(lap); cur.setAttribute("x1", x); cur.setAttribute("x2", x); }
  RP.top.forEach((c, i) => { const v = $("rp-val-" + i); if (v) v.textContent = pct(e.win[c] || 0); });

  const list = Replay.cars(r, e);
  $("rp-order-h").textContent = `Running order after lap ${lap}`;
  $("rp-order").innerHTML = `<tr><th class="n">Pos</th><th>Driver</th><th class="n">Gap</th><th>Tyre</th><th class="n">Stops</th><th class="n">Win</th></tr>` +
    list.map((c, i) => `<tr data-code="${c.code}" class="${c.code === $("rp-driver").value ? "sel" : ""}" tabindex="0">
      <td class="n">${c.running ? i + 1 : "—"}</td><td class="code">${c.code} <small>${esc(r.teams[c.code] || "")}</small></td>
      <td class="n">${!c.running ? "out" : i === 0 ? "leader" : c.gap >= 1000 ? "lapped" : "+" + c.gap.toFixed(1)}</td>
      <td>${c.running ? tyre(c.compound, c.age) : ""}</td><td class="n">${c.stops}</td><td class="n">${c.running ? pct(e.win[c.code] || 0) : ""}</td></tr>`).join("");
  $("rp-order").querySelectorAll("tr[data-code]").forEach(tr => {
    const pick = () => { if (tr.children[0].textContent === "—") return; $("rp-driver").value = tr.dataset.code; $("rp-driver").dispatchEvent(new Event("change")); rpSetLap(RP.lap); };
    tr.addEventListener("click", pick);
    tr.addEventListener("keydown", ev => { if (ev.key === "Enter") pick(); });
  });

  const keep = $("rp-driver").value;
  const running = list.filter(c => c.running);
  $("rp-driver").innerHTML = running.map(c => `<option value="${c.code}">${c.code} · P${list.indexOf(c) + 1}</option>`).join("");
  $("rp-driver").value = running.some(c => c.code === keep) ? keep : (running[0] || {}).code;
  $("rp-adv").innerHTML = "";
  $("rp-adv-note").textContent = adviseHint();
  rpActual();
}

function adviseHint() {
  const r = RP.race, left = r.track.laps - RP.lap;
  return left <= 1 ? "The race is over; pick an earlier lap." :
    `Reruns the last ${left} laps 400 times for each option, with ${$("rp-driver").value}'s pace and tyre wear estimated from the race so far and rivals following their own best plans.`;
}

// what the driver actually did after this lap
function rpActual() {
  const r = RP.race, code = $("rp-driver").value;
  if (!code || r.live) { $("rp-actual").textContent = ""; return; }
  const stops = [];
  let prev = null;
  for (const e of r.laps.slice(RP.lap - 1)) {
    const c = Replay.cars(r, e).find(x => x.code === code);
    if (!c) break;
    if (prev && c.stops > prev.stops) stops.push(`lap ${e.lap - 1} for ${Replay.COMPOUNDS[c.compound].toLowerCase()}`);
    prev = c;
  }
  const fin = r.final[code];
  $("rp-actual").textContent = `What actually happened: ${stops.length ? "pitted on " + stops.join(", then ") : "no more stops"}, ` +
    (fin ? `finished P${fin}.` : "did not finish.");
}

function rpAdvise() {
  const r = RP.race, e = r.laps.find(x => x.lap === RP.lap), code = $("rp-driver").value, btn = $("rp-advise");
  if (r.track.laps - RP.lap <= 1) return;
  btn.disabled = true; btn.textContent = "Simulating…";
  setTimeout(() => {
    const t0 = performance.now();
    const res = Replay.advise(C, r, e, code, { sims: 400, seed: 11 });
    const best = res[0];
    $("rp-adv").innerHTML = `<tr><th>Option</th><th class="n">Expected finish</th><th class="n">Win</th><th class="n">Podium</th></tr>` +
      res.map(o => `<tr class="${o === best ? "best" : ""}"><td>${esc(o.label)}${o === best ? " · best" : ""}</td>
        <td class="n">P${o.result.exp_pos.toFixed(1)}</td><td class="n">${pct(o.result.win)}</td><td class="n">${pct(o.result.podium)}</td></tr>`).join("");
    $("rp-adv-note").textContent = `${res.length} options, 400 simulations each, ${((performance.now() - t0) / 1000).toFixed(1)} s in your browser. Differences under about half a place are within simulation noise.`;
    btn.disabled = false; btn.textContent = "Compare options";
  }, 30);
}

function rpPlay() {
  const L = RP.race.laps[RP.race.laps.length - 1].lap;
  if (RP.lap >= L) rpSetLap(1);
  $("rp-play").textContent = "❚❚ Pause"; $("rp-play").setAttribute("aria-label", "Pause");
  RP.timer = setInterval(() => { RP.lap >= L ? rpStop() : rpSetLap(RP.lap + 1); }, 350);
}
function rpStop() {
  if (RP.timer) clearInterval(RP.timer);
  RP.timer = null;
  $("rp-play").textContent = "▶ Play"; $("rp-play").setAttribute("aria-label", "Play");
}

const rpX = lap => 44 + (lap - 1) / Math.max(RP.race.track.laps - 1, 1) * (RP.W - 44 - 70);
function rpChart() {
  // draw at the chart's real width (no wider than 760) so text stays readable on phones
  RP.W = Math.round(Math.max(340, Math.min(760, $("rp-chart").parentElement.clientWidth - 34)));
  const r = RP.race, y0 = 14, h = RP.W < 500 ? 200 : 250, Y = v => y0 + h - v * h;
  $("rp-chart").setAttribute("viewBox", `0 0 ${RP.W} ${y0 + h + 28}`);
  let g = "";
  // neutralised laps
  r.laps.forEach(l => {
    if (l.status !== "GREEN") g += `<rect x="${rpX(l.lap) - 3}" y="${y0}" width="${Math.max(6, rpX(2) - rpX(1))}" height="${h}" fill="var(--band)"><title>Lap ${l.lap}: ${l.status === "SC" ? "safety car" : l.status === "VSC" ? "virtual safety car" : "red flag"}</title></rect>`;
  });
  for (const v of [0, 0.25, 0.5, 0.75, 1]) g += `<line x1="44" x2="${RP.W - 70}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line-soft)"/><text x="36" y="${Y(v) + 4}" text-anchor="end" font-size="11" fill="var(--ink-3)">${v * 100}%</text>`;
  const L = r.track.laps, step = RP.W < 500 ? (L > 60 ? 20 : 10) : (L > 60 ? 10 : 5);
  for (let lap = step; lap <= L; lap += step) g += `<text x="${rpX(lap)}" y="${y0 + h + 18}" text-anchor="middle" font-size="11" fill="var(--ink-3)">${lap}</text>`;
  g += `<text x="${rpX(1)}" y="${y0 + h + 18}" text-anchor="middle" font-size="11" fill="var(--ink-3)">1</text>`;
  // series, drawn with the eventual winner last so it sits on top
  const order = RP.top.map((c, i) => [c, i]).sort((a, b) => (a[0] === r.winner) - (b[0] === r.winner));
  for (const [c, i] of order) {
    const pts = r.laps.map(l => `${rpX(l.lap).toFixed(1)},${Y(l.win[c] || 0).toFixed(1)}`).join(" ");
    g += `<polyline points="${pts}" fill="none" stroke="var(${SERIES[i]})" stroke-width="${c === r.winner ? 2.5 : 1.75}" stroke-linejoin="round"><title>${c}</title></polyline>`;
  }
  // end labels, spread so they do not overlap
  const lastEntry = r.laps[r.laps.length - 1];
  const ends = RP.top.map((c, i) => ({ c, i, y: Y(lastEntry.win[c] || 0) })).sort((a, b) => a.y - b.y);
  for (let k = 1; k < ends.length; k++) ends[k].y = Math.max(ends[k].y, ends[k - 1].y + 13);
  const over = ends.length ? ends[ends.length - 1].y - (y0 + h) : 0;          // keep the stack inside the frame
  if (over > 0) ends.forEach(e => { e.y -= over; });
  const lx = r.live ? rpX(lastEntry.lap) + 6 : RP.W - 64;
  ends.forEach(e => { g += `<text x="${lx}" y="${e.y + 4}" font-size="12" font-weight="600" fill="var(${SERIES[e.i]})">${e.c}</text>`; });
  g += `<line id="rp-cursor" x1="${rpX(RP.lap)}" x2="${rpX(RP.lap)}" y1="${y0}" y2="${y0 + h}" stroke="var(--ink)" stroke-width="1.5" stroke-dasharray="3 3"/>`;
  $("rp-chart").innerHTML = g;
  $("rp-legend").innerHTML = RP.top.map((c, i) => `<span><i style="background:var(${SERIES[i]})"></i>${c}${c === r.winner ? " (winner)" : ""} <b class="num" id="rp-val-${i}"></b></span>`).join("");
}

let rpResize = null;
// shared with the live page, which builds races itself instead of loading files
addEventListener("resize", () => { clearTimeout(rpResize); rpResize = setTimeout(() => { if (RP.race) { rpChart(); rpSetLap(RP.lap); } }, 150); });

rpBind();
if (D.races) rpInit();
