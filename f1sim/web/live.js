// ---------------- live race ----------------
// Reuses the replay page's chart, running order and pit advisor (replay.js).
const LV = { mode: null, raw: null, track: null, entries: [], timer: null, session: null, demo: null };
const POLL_MS = 20000;
// OpenF1 answers 401 to every unauthenticated request, past sessions included, while any F1
// session (practice, qualifying or race) is running, and that answer has no CORS header, so
// the browser only sees a failed request. Checked live during Las Vegas FP1, 20 Nov 2026.
const CLOSED = "OpenF1, the free timing service this page uses, closes all its data to non-paying users " +
  "while any Formula 1 session is running (practice, qualifying or the race) and opens it again when the session ends. " +
  "So live odds cannot be shown during a session; afterwards this page can replay it, and the Race replay page has every race once its data is published.";
const fmtTime = iso => new Date(iso).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

async function getJSON(path) {
  const res = await fetch(`${LiveFeed.API}/${path}`);
  if (!res.ok) throw Object.assign(new Error("HTTP " + res.status), { status: res.status });
  return res.json();
}

function status(head, note, live) {
  $("lv-headline").textContent = head;
  $("lv-note").textContent = note || "";
  $("lv-dot").className = "lv-dot" + (live ? " on" : "");
}

async function lvInit() {
  $("lv-demo").addEventListener("click", lvDemo);
  $("lv-stopdemo").addEventListener("click", lvStopDemo);
  let latest, races = [];
  try {
    latest = (await getJSON("sessions?session_key=latest"))[0];
    const year = new Date().getUTCFullYear();
    races = await getJSON(`sessions?year=${year}&session_name=Race`);
  } catch (e) {
    if (navigator.onLine === false) status("You appear to be offline.", "Reconnect and reload the page.");
    else status("Live timing is closed right now.", CLOSED + " If no session is on, OpenF1 may be down; try again later.");
    return;
  }
  const now = Date.now();
  const live = races.find(s => Date.parse(s.date_start) - 15 * 60e3 < now && now < Date.parse(s.date_end) + 45 * 60e3);
  const next = races.filter(s => Date.parse(s.date_start) > now).sort((a, b) => a.date_start.localeCompare(b.date_start))[0];
  LV.last = races.filter(s => Date.parse(s.date_end) < now).sort((a, b) => b.date_start.localeCompare(a.date_start))[0] || latest;
  if (live) return lvLive(live);
  const until = next ? Date.parse(next.date_start) - now : null;
  status(next ? `Next race: ${next.location}, ${fmtTime(next.date_start)}` : "No more races scheduled this season.",
    next ? `Starts in ${Math.floor(until / 864e5)} days ${Math.floor(until / 36e5) % 24} hours (your local time). Come back during the race, or watch the last race (${LV.last.location}) replayed as if it were live, using exactly the same live pipeline.` : "");
  $("lv-actions").hidden = !LV.last;
}

function trackFor(session) {
  const key = D.circuits[session.circuit_short_name];
  const t = D.tracks[key];
  return t ? { key, ...t } : null;
}

// Build lap entries for every completed lap from raw data, reusing ones already built.
function lvUpdate(raw, session, lapsTotal) {
  const rows = LiveFeed.buildRows(raw);
  const race = { name: `${session.location} ${new Date(session.date_start).getUTCFullYear()}`, live: true, winner: null,
                 track: LV.track, car_fields: LiveFeed.CAR_FIELDS, teams: {}, final: {}, laps: LV.entries };
  rows.forEach(r => { race.teams[r.driver] = r.team; });
  if (!rows.length) return null;
  // the last lap every car on the lead lap has completed (the leader is one lap ahead of the rest)
  const leaderLap = Math.max(...rows.filter(r => r.position === 1).map(r => r.lap));
  const finished = leaderLap >= lapsTotal && rows.filter(r => r.lap === leaderLap).length >= 3;
  const upTo = finished ? leaderLap : leaderLap - 1;
  for (let lap = (LV.entries.length ? LV.entries[LV.entries.length - 1].lap + 1 : 1); lap <= upTo; lap++) {
    const e = LiveFeed.raceState(rows, lap, LV.track, D.C, D.prior, lapsTotal, true);
    e.win = LiveFeed.winOdds(race, e, D.weights, D.C, Replay);
    LV.entries.push(e);
  }
  return race;
}

function lvRender(race, note) {
  if (!race || !race.laps.length) return;
  const wasAt = RP.race ? RP.lap : null, latest = race.laps[race.laps.length - 1].lap;
  rpShow(race);
  // follow the newest lap unless the reader has scrolled back
  rpSetLap(wasAt === null || wasAt >= latest - 1 ? latest : wasAt);
  const e = race.laps[race.laps.length - 1], fav = Object.entries(e.win).sort((a, b) => b[1] - a[1])[0];
  status(`${race.name}: lap ${latest} of ${race.track.laps}${fav ? ` · ${fav[0]} ${pct(fav[1])} to win` : ""}`, note, true);
}

async function lvLive(session) {
  LV.track = trackFor(session);
  if (!LV.track) { status(`${session.location} is live, but this circuit is not in the simulator yet.`); return; }
  LV.entries = [];
  const tick = async () => {
    try {
      const { status: st, raw } = await LiveFeed.fetchSession(session.session_key);
      if (st !== "ok") {
        status(`${session.location}: race day`, st === "empty"
          ? "Waiting for the first completed lap. The page checks every 20 seconds."
          : `${CLOSED} (OpenF1 answered ${st}.) The page keeps checking every 20 seconds.`, true);
        return;
      }
      LV.entries = LV.entries.slice(0, Math.max(0, LV.entries.length - 3));   // timing data can be corrected for a few laps
      lvRender(lvUpdate(raw, session, LV.track.laps), `Updated ${new Date().toLocaleTimeString()}; refreshes every 20 seconds.`);
    } catch (e) {
      status(`${session.location}: race day`, navigator.onLine === false ? "You appear to be offline; retrying in 20 seconds."
        : `${CLOSED} The page keeps checking every 20 seconds.`, true);
    }
  };
  await tick();
  LV.timer = setInterval(tick, POLL_MS);
}

// Demo: feed the last race through the live pipeline, cutting the data off after each lap.
async function lvDemo() {
  const s = LV.last;
  LV.track = trackFor(s);
  if (!LV.track) { status(`${s.location} is not in the simulator yet.`); return; }
  $("lv-demo").disabled = true; status(`Loading ${s.location}…`);
  let full;
  try { full = (await LiveFeed.fetchSession(s.session_key)).raw; } catch (e) { full = null; }
  $("lv-demo").disabled = false;
  if (!full || !full.laps.length) {
    status("Could not load the last race's data.", "OpenF1 closes its data to non-paying users while any Formula 1 session is running; if one is on, try again when it ends. Otherwise try again in a minute.");
    return;
  }
  const ends = {};      // leader's lap end times, to cut the feed
  full.laps.forEach(l => { if (l.date_start && l.lap_duration) { const t = Date.parse(l.date_start) + l.lap_duration * 1e3; ends[l.lap_number] = Math.min(ends[l.lap_number] || Infinity, t); } });
  const before = (iso, t) => !iso || Date.parse(iso) <= t;
  LV.entries = []; LV.demo = { lap: 0, full, ends, session: s };
  $("lv-stopdemo").hidden = false; $("lv-demo").hidden = true;
  const step = () => {
    const d = LV.demo; if (!d) return;
    d.lap++;
    const cut = d.ends[d.lap];
    if (cut === undefined) {     // chequered flag: the complete data includes the final lap
      lvStopDemo();
      lvRender(lvUpdate(d.full, s, LV.track.laps));
      status(`${s.location} replay finished.`, "That is the whole race, shown exactly as the live page would have shown it.");
      return;
    }
    const raw = {
      drivers: d.full.drivers,
      laps: d.full.laps.filter(l => l.date_start && Date.parse(l.date_start) + (l.lap_duration || 0) * 1e3 <= cut + 1),
      stints: d.full.stints.filter(x => (x.lap_start || 1) <= d.lap).map(x => ({ ...x, lap_end: Math.min(x.lap_end || d.lap, d.lap) })),
      pit: d.full.pit.filter(x => before(x.date, cut)),
      race_control: d.full.race_control.filter(x => before(x.date, cut)),
      weather: d.full.weather.filter(x => before(x.date, cut)),
    };
    LV.entries = LV.entries.filter(e => e.lap < d.lap);
    lvRender(lvUpdate(raw, s, LV.track.laps), `Replay of ${s.location}, fed lap by lap through the live pipeline.`);
    LV.timer = setTimeout(step, +$("lv-speed").value);
  };
  step();
}

function lvStopDemo() {
  clearTimeout(LV.timer); LV.demo = null;
  $("lv-stopdemo").hidden = true; $("lv-demo").hidden = false;
}

lvInit();
