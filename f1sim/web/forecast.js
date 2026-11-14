function renderRace() {
  const t = D.tracks.find(x => x.key === $("race").value);
  const rows = t.rows.slice().sort((a, b) => b.win - a.win);
  const fav = rows[0], pole = t.rows.slice().sort((a, b) => b.pole - a.pole)[0];
  $("facts").innerHTML = [
    [drv(fav.d).code + " " + pct(fav.win), "Favourite to win"],
    [drv(pole.d).code + " " + pct(pole.pole), "Favourite for pole"],
    [plan(t.strategy), "Fastest dry strategy"],
    [pct(t.sc_prob), "Safety car chance"],
    [pct(t.rain_prob), "Rain chance"],
    [t.laps + " laps", t.name],
  ].map(([b, s]) => `<div class="fact"><b>${esc(b)}</b><small>${esc(s)}</small></div>`).join("");

  const top = rows.slice(0, 10), max = Math.max(top[0].win, 0.05);
  bars($("winbars"), top.map(r => ({ label: drv(r.d).code, sub: drv(r.d).team, value: r.win, text: pct(r.win),
    tip: `${drv(r.d).name}\nWin ${pct(r.win)} · Podium ${pct(r.podium)}\nPole ${pct(r.pole)} · Retires ${pct(r.dnf)}\nExpected points ${r.exp_pts.toFixed(1)}` })), max);

  const order = t.rows.slice().sort((a, b) => a.exp_pos - b.exp_pos), n = D.drivers.length;
  const cls = p => p < 0.005 ? 0 : p < 0.02 ? 1 : p < 0.05 ? 2 : p < 0.10 ? 3 : p < 0.20 ? 4 : p < 0.35 ? 5 : 6;
  let h = `<tr><th class="d"></th>${Array.from({ length: n }, (_, i) => `<th>${i + 1}</th>`).join("")}</tr>`;
  for (const r of order) {
    h += `<tr><th class="d">${drv(r.d).code}</th>` + t.dist[r.d].map((p, i) =>
      `<td class="c${cls(p)}" data-tip="${drv(r.d).code} finishes P${i + 1}: ${pct(p)}"></td>`).join("") + "</tr>";
  }
  $("heat").innerHTML = h;
  $("ramp").innerHTML = "Less likely " + [0, 1, 2, 3, 4, 5, 6].map(i => `<i class="c${i}"></i>`).join("") + " More likely (darkest step: 35% or more; retirements are placed last)";

  $("racetable").innerHTML = `<tr><th>Driver</th><th>Team</th><th class="n">Win</th><th class="n">Podium</th><th class="n">Top 10</th><th class="n">Pole</th><th class="n">Retires</th><th class="n">Avg finish</th><th class="n">Exp. points</th></tr>` +
    t.rows.slice().sort((a, b) => b.exp_pts - a.exp_pts || a.exp_pos - b.exp_pos).map(r => `<tr><td class="code">${drv(r.d).code}</td><td>${esc(drv(r.d).team)}</td>
      <td class="n">${pct(r.win)}</td><td class="n">${pct(r.podium)}</td><td class="n">${pct(r.points)}</td><td class="n">${pct(r.pole)}</td>
      <td class="n">${pct(r.dnf)}</td><td class="n">${r.exp_pos.toFixed(1)}</td><td class="n">${r.exp_pts.toFixed(1)}</td></tr>`).join("");
  try { localStorage.setItem("f1sim-race", t.key); } catch (e) {}
}

function renderChamp(which) {
  const S = D.season;
  if (!S) { $("champ").hidden = true; return; }
  const rows = S[which].filter(r => r.p95 > 0).slice(0, which === "drivers" ? 12 : 11);
  const max = Math.max(...rows.map(r => r.p95)) * 1.02;
  const x = v => (v / max * 100).toFixed(2) + "%";
  bars($("champbars"), rows.map(r => ({ label: r.name, text: r.mean.toFixed(0) + " pts",
    tip: `${r.name}\nPoints today ${r.current.toFixed(0)}\nExpected final ${r.mean.toFixed(0)} (${r.p05.toFixed(0)}–${r.p95.toFixed(0)})\nWins the title in ${pct(r.title)} of seasons`,
    inner: `<span class="range" style="left:${x(r.p05)};width:${x(Math.max(r.p95 - r.p05, 0))}"></span><span class="tick" style="left:${x(r.current)}"></span><span class="dot" style="left:${x(r.mean)}"></span>` })), max);
  const lead = S[which][0];
  $("champlead").textContent = `${lead.name} takes the ${which === "drivers" ? "drivers'" : "constructors'"} title in ${pct(lead.title)} of simulated seasons.`;
  $("seg-d").setAttribute("aria-pressed", which === "drivers");
  $("seg-c").setAttribute("aria-pressed", which === "constructors");
}

function renderStrat() {
  const S = D.strategy;
  if (!S) { $("strat").hidden = true; return; }
  const rows = S.drivers[$("sdriver").value] || [];
  $("strattable").innerHTML = `<tr><th>Strategy</th><th class="n">Clean-air cost</th><th class="n">Avg finish</th><th class="n">Exp. points</th><th class="n">Win</th><th class="n">Podium</th></tr>` +
    rows.map((r, i) => `<tr class="${i ? "" : "best"}"><td>${esc(plan(r.strategy))}${i ? "" : " · best"}</td><td class="n">${r.clean_air.toFixed(1)} s</td>
      <td class="n">${r.exp_pos.toFixed(2)}</td><td class="n">${r.exp_points.toFixed(2)}</td><td class="n">${pct(r.win)}</td><td class="n">${pct(r.podium)}</td></tr>`).join("");
}


function initForecast() {
  const done = D.rounds_done, total = D.rounds_total;
  $("sub").textContent = `${D.label} season after round ${done} of ${total}. ${D.sims.toLocaleString("en")} simulated races per circuit, on car and driver pace fitted from the real results.`;
  const sel = $("race");
  const group = (label, items) => items.length ? `<optgroup label="${label}">${items.map(t => `<option value="${t.key}">${esc(t.name)}</option>`).join("")}</optgroup>` : "";
  sel.innerHTML = group("Still to run", D.tracks.filter(t => t.remaining)) + group("Other circuits", D.tracks.filter(t => !t.remaining));
  let saved = null; try { saved = localStorage.getItem("f1sim-race"); } catch (e) {}
  sel.value = D.tracks.some(t => t.key === saved) ? saved : D.next;

  if (D.season) $("champsub").textContent = `Real points so far plus ${D.season.races.length} simulated race${D.season.races.length === 1 ? "" : "s"} (${D.season.race_names.join(", ")}), ${D.season.sims.toLocaleString("en")} times.`;
  if (D.strategy) {
    $("stratsub").textContent = `Pit strategies compared for one driver at ${D.strategy.track_name}.`;
    $("sdriver").innerHTML = D.drivers.filter(d => D.strategy.drivers[d.code]).map(d => `<option value="${d.code}">${d.code} · ${esc(d.name)}</option>`).join("");
  }

}

initForecast();
$("race").addEventListener("change", renderRace);
$("sdriver").addEventListener("change", renderStrat);
$("seg-d").addEventListener("click", () => renderChamp("drivers"));
$("seg-c").addEventListener("click", () => renderChamp("constructors"));
renderRace(); renderChamp("drivers"); renderStrat();
