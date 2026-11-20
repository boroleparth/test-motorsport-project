function renderCal(event) {
  const A = D.accuracy, rows = A.calibration[event], L = 44, T = 12, W = 300, H = 244;
  const x = v => L + v * W, y = v => T + H - v * H;
  let g = "";
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    g += `<line x1="${x(0)}" x2="${x(1)}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line-soft)"/>` +
      `<text x="${L - 8}" y="${y(v) + 4}" text-anchor="end" fill="var(--ink-3)" font-size="11">${v * 100}%</text>` +
      `<text x="${x(v)}" y="${T + H + 16}" text-anchor="middle" fill="var(--ink-3)" font-size="11">${v * 100}%</text>`;
  }
  g += `<line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="var(--ink-3)" stroke-dasharray="4 4"/>` +
    `<text x="${x(0.8)}" y="${y(0.68)}" text-anchor="start" fill="var(--ink-3)" font-size="11">perfect</text>` +
    `<polyline fill="none" stroke="var(--mark)" stroke-width="2" points="${rows.map(r => x(r.predicted) + "," + y(r.observed)).join(" ")}"/>`;
  for (const r of rows) g += `<circle cx="${x(r.predicted)}" cy="${y(r.observed)}" r="5" fill="var(--mark)" stroke="var(--panel)" stroke-width="2"/>` +
    `<circle cx="${x(r.predicted)}" cy="${y(r.observed)}" r="14" fill="transparent" data-tip="Predicted ${pct(r.predicted)} on average\nHappened ${pct(r.observed)} of the time\n${r.n} driver-races"/>`;
  g += `<text x="${x(0.5)}" y="${T + H + 34}" text-anchor="middle" fill="var(--ink-2)" font-size="12">Chance the simulator gave</text>` +
    `<text transform="translate(11 ${y(0.5)}) rotate(-90)" text-anchor="middle" fill="var(--ink-2)" font-size="12">How often it happened</text>`;
  $("calchart").innerHTML = g;
  const label = { win: "a win", podium: "a podium", top10: "a top-10 finish" }[event];
  $("calnote").textContent = `Every driver in ${A.n_all} races (${A.all_seasons[0]}–${A.all_seasons[A.all_seasons.length - 1]}), grouped by the chance the simulator gave them of ${label} before qualifying. Points on the dashed line are perfectly calibrated; hover for counts.`;
  for (const e of ["win", "podium", "top10"]) $("cal-" + e).setAttribute("aria-pressed", e === event);
}
for (const e of ["win", "podium", "top10"]) $("cal-" + e).addEventListener("click", () => renderCal(e));

function initRecord() {
  const R = D.track_record;
  if (!R || !R.races.length) return;
  $("rec").hidden = false;
  const when = t => new Date(t).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZoneName: "short" });
  const cell = (o, winner) => {
    if (!o) return `<td>–</td><td class="n">–</td>`;
    const tip = `Saved ${when(o.saved)}\n` + o.top.map(t => `${t.driver} ${pct(t.win)}`).join("\n");
    const fav = winner ? (o.favourite_won ? `<b>${o.favourite}</b> ✓` : o.favourite) : o.favourite;
    return `<td data-tip="${esc(tip)}"><span class="code">${fav}</span> ${pct(o.favourite_win)}</td>` +
      `<td class="n">${o.winner_prob === undefined ? "–" : pct(o.winner_prob)}</td>`;
  };
  $("rectable").innerHTML = `<tr><th>Race</th><th>Winner</th><th>Favourite before qualifying</th><th class="n">Chance</th>` +
    `<th>Favourite after qualifying</th><th class="n">Chance</th><th>Pole</th></tr>` +
    R.races.slice().reverse().map(r => `<tr><td><a href="${D.log_url}${r.file}">${esc(r.name)}</a> ${r.season}</td>` +
      (r.winner ? `<td class="code">${r.winner}</td>` : `<td class="note">Race ${when(r.race_start)}</td>`) +
      cell(r.pre_quali, r.winner) + cell(r.post_quali, r.winner) +
      `<td class="code">${r.post_quali ? r.post_quali.pole + (r.winner && r.post_quali.pole_won ? " ✓" : "") : "–"}</td></tr>`).join("");
  const S = R.summary, parts = [];
  for (const k of ["pre_quali", "post_quali"]) if (S[k]) parts.push(`${S[k].label.toLowerCase()}, the favourite won ${Math.round(S[k].favourite_won * S[k].n)} of ${S[k].n} and the winner was given ${pct(S[k].winner_prob)} on average`);
  if (S.post_quali) parts.push(`the pole-sitter won ${Math.round(S.post_quali.pole_won * S.post_quali.n)} of ${S.post_quali.n}`);
  const n = Math.max(0, ...Object.values(S).map(s => s.n));
  $("recsum").textContent = n ? `Scored so far (${n} race${n > 1 ? "s" : ""}): ` + parts.join("; ") + ". Too few races to judge yet; the benchmark below covers 207." :
    "No logged race has been run yet. The first forecasts are scored after the race.";
}

function initAccuracy() {
  const A = D.accuracy;
  if (!A) $("acc").hidden = true; else {
    $("accsub").textContent = `${A.n_races} races from ${A.seasons.join(" and ")}, each predicted using only the earlier rounds of its season, then compared with the real result.`;
    const best = (k, hi) => (hi ? Math.max : Math.min)(...A.methods.map(m => m[k]));
    const cell = (m, k, hi, f) => `<td class="n">${m[k] === best(k, hi) ? "<b>" + f(m[k]) + "</b>" : f(m[k])}</td>`;
    $("acctable").innerHTML = `<tr><th>Method</th><th class="n">Favourite won</th><th class="n">Chance given to winner</th><th class="n">Podium drivers found</th><th class="n">Order match</th><th class="n">Probability score</th></tr>` +
      A.methods.map(m => `<tr class="${m.model ? "best" : ""}"><td>${esc(m.label)}</td>${cell(m, "favourite_won", 1, pct)}${cell(m, "winner_prob", 1, pct)}
        ${cell(m, "podium_hits", 1, v => v.toFixed(2) + " of 3")}${cell(m, "spearman", 1, v => v.toFixed(2))}${cell(m, "rps", 0, v => v.toFixed(3))}</tr>`).join("");
    $("accnote").textContent = `Bold is the best in each column. Order match: 1 is a perfect finishing order. Probability score (ranked probability score): 0 is perfect. Compare the simulator before qualifying with championship order, and after qualifying with grid order.` +
      (A.tuned_on.length ? ` No model setting was chosen using these seasons.` : "") +
      (A.tests || []).map(t => ` Over all ${A.n_all} races (${A.all_seasons[0]}–${A.all_seasons[A.all_seasons.length - 1]}), ${t.a.toLowerCase()} against ${t.b.toLowerCase().replace("baseline: ", "")}: probability score ${t.diff < 0 ? "better" : "worse"} by ${Math.abs(t.diff).toFixed(4)} (95% interval ${t.lo.toFixed(4)} to ${t.hi.toFixed(4)}).`).join("");
    const top = Math.max(...A.races.map(r => r.winner_prob), 0.1), base = 1 / D.drivers.length;
    $("acccols").innerHTML = A.races.map(r => `<div class="col${r.favourite_won ? " top" : ""}" data-tip="${esc(r.name)}\nWinner ${r.winner}: given ${pct(r.winner_prob)}\nAfter qualifying: ${pct(r.after_prob)}\nSimulator's favourite: ${r.favourite}">
      <span>${Math.round(r.winner_prob * 100)}</span><i style="height:${r.winner_prob / top * 82}%"></i></div>`).join("") +
      `<div class="baseline" style="bottom:${base / top * 82}%"><em>blind guess ${pct(base)}</em></div>`;
    $("acclabels").innerHTML = A.races.map(r => `<span>${r.winner}</span>`).join("");
    renderCal("podium");
  }

  const tmax = Math.max(...D.teams.map(t => t.pace), 0.5);
  bars($("teambars"), D.teams.map(t => ({ label: t.name, value: t.pace, text: "+" + t.pace.toFixed(2) + " s",
    tip: `${t.name}\n+${t.pace.toFixed(3)} s per lap\nMechanical retirement ${pct(t.reliability)} per race` })), tmax);
  $("drivertable").innerHTML = `<tr><th>Driver</th><th>Team</th><th class="n">Race</th><th class="n">One lap</th><th class="n">Error exit</th></tr>` +
    D.drivers.map(d => `<tr><td class="code">${d.code}</td><td>${esc(d.team)}</td><td class="n">+${d.pace.toFixed(2)}</td><td class="n">+${d.quali_gap.toFixed(2)}</td><td class="n">${pct(d.error_rate)}</td></tr>`).join("");
}

initRecord();
initAccuracy();
