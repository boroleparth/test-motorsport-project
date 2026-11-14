const $ = id => document.getElementById(id);
const pct = v => (v * 100 < 0.05 && v > 0 ? "<0.1" : (v * 100).toFixed(1)) + "%";
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const TYRE = { S: "Soft", M: "Medium", H: "Hard" };
const plan = s => { const [t, p] = s.split("@"); return t.split("-").map(c => TYRE[c]).join(" → ") + (p ? ", pit lap " + p.split(",").join(" and ") : ""); };
const drv = i => D.drivers[i];

function bars(el, rows, max) {
  el.innerHTML = rows.map(r => `<div class="row" data-tip="${esc(r.tip || "")}">
    <span class="lab">${esc(r.label)}${r.sub ? `<small>${esc(r.sub)}</small>` : ""}</span>
    <span class="track">${r.inner || `<span class="bar" style="width:${Math.max(0, r.value / max * 100)}%"></span>`}</span>
    <span class="val">${esc(r.text)}</span></div>`).join("");
}


const tip = $("tip");
document.addEventListener("pointermove", e => {
  const el = e.target.closest && e.target.closest("[data-tip]");
  if (!el || !el.dataset.tip) { tip.hidden = true; return; }
  tip.textContent = el.dataset.tip; tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.max(8, Math.min(e.clientX + 14, innerWidth - w - 8)) + "px";
  tip.style.top = (e.clientY + 18 + h > innerHeight ? e.clientY - h - 10 : e.clientY + 18) + "px";
});
document.addEventListener("pointerleave", () => { tip.hidden = true; });
