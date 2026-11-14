function initHome() {
  const H = D.home;
  $("next-sub").textContent = `${H.label} season after round ${H.rounds_done} of ${H.rounds_total}.`;
  const facts = H.next ? H.next.top.map((r, i) => [`${r.code} ${pct(r.win)}`, i ? `${i + 1}. ${r.name}` : `Favourite at ${H.next.name}`]) : [];
  if (H.title) facts.push([`${H.title.code} ${pct(H.title.p)}`, "Chance of the drivers' title"]);
  $("next-facts").innerHTML = facts.map(([b, s]) => `<div class="fact"><b>${esc(b)}</b><small>${esc(s)}</small></div>`).join("");
  const cards = [
    ["forecast/", "Forecast", H.next ? `${H.next.top[0].code} ${pct(H.next.top[0].win)}` : "", `Win, podium and points odds for any circuit, the championship projection, and pit strategies compared.`, "See the forecast"],
    ["lab/", "What-if lab", "Your scenario", "Force rain, change a car's pace, set a pit strategy or grid slot, and rerun the race in your browser.", "Open the lab"],
    ["accuracy/", "Accuracy", !H.acc ? "" : H.acc.hi < 0 ? "Beats the grid" : "Level with the grid",
      !H.acc ? "How the predictions compare with real results."
        : `Once qualifying is known, its odds ${H.acc.hi < 0 ? "beat" : "match"} "everyone finishes where they start" over ${H.acc.races} races since ${H.acc.first}, each predicted before it was run${H.acc.hi < 0 ? ", by a margin outside the error bars" : ""}.`,
      "Check the record"],
    ["replay/", "Race replay", "Every race since 2023", "Step through any Grand Prix lap by lap with live win odds, and ask the pit advisor what each driver should have done.", "Replay a race"],
    ["live/", "Live", "During each race", "Win odds and pit advice that update as laps complete, straight from the timing feed. Between races, watch the last one replayed as if live.", "Go live"],
  ];
  $("cards").innerHTML = cards.map(([href, t, big, text, go]) => href
    ? `<a class="card" href="${href}"><h3>${esc(t)}</h3><span class="big">${esc(big)}</span><p>${esc(text)}</p><span class="go">${esc(go)} →</span></a>`
    : `<div class="card soon"><h3>${esc(t)}</h3><span class="big">${esc(big)}</span><p>${esc(text)}</p><span class="go">${esc(go)}</span></div>`).join("");
}

initHome();
