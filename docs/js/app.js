/* App controller: state (scenario, time, tracked agent), animation loop, storm cues, URL state.
 * The map is full width until an agent is tracked; the panel exists only while tracking.
 * URL parameters: ?run=storm|base  &t=<tick>  &agent=<id> (track it)  &story=1  &kiosk=1 */
const App = (() => {
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const S = { run: null, R: null, tick: 90, playing: false, speed: 12, sel: null, selDay: 1, file: null, kiosk: false, follow: true };
  let last = 0, panelAt = 0, urlAt = 0, followAt = 0, lastDay = 0;
  const $ = id => document.getElementById(id);

  async function boot() {
    const q = new URLSearchParams(location.search);
    S.kiosk = q.get('kiosk') === '1';
    document.body.classList.toggle('kiosk', S.kiosk);
    const progress = msg => { $('loadingText').textContent = msg; };
    try {
      await Data.loadCore(progress);
    } catch (err) {
      progress(`Could not load the data (${err.message}). If you opened this file directly, serve the folder instead: python3 -m http.server`);
      return;
    }
    MapView.init(pick, () => { if (S.sel != null && !Story.active) setFollow(false); });
    Timeline.init($('tlCanvas'), scrub);
    Panel.init({ close: stopTracking, day: d => showDay(d, true),
                 jump: (t, trip) => { setTick(t); if (trip != null) MapView.focusTrips(S.R, [trip], { maxZoom: 15 }); } });
    Story.init(api);
    wireUi();
    Data.onChunk = (R, d) => {                          // a day's routes arrived: redraw with real streets
      if (R !== S.R) return;
      if (S.sel != null && d <= S.selDay) MapView.showRoute(S.R, S.sel, S.selDay);
      MapView.draw();
    };
    if (q.has('t')) S.tick = Math.max(0, Math.min(Data.TICKS - 1, parseFloat(q.get('t')) || 0));
    await setRun(q.get('run') === 'base' ? 'base' : Data.D.meta.default_run, progress);
    const agent = parseInt(q.get('agent'), 10);
    if (Data.D.index.has(agent) && !S.kiosk) await track(Data.D.index.get(agent), { focus: !q.has('t') });
    $('loading').hidden = true;
    window.__readyAt = Math.round(performance.now());  // ms from navigation to an interactive map (for load tests)
    if (q.get('story') === '1' && !S.kiosk) Story.start(Data.D.meta.featured[0]);
    else if (S.kiosk || !reduceMotion) play();
    requestAnimationFrame(loop);
    setTimeout(async () => {                           // the other scenario, for the ghost line + About stats
      await Data.loadRun(S.run === 'storm' ? 'base' : 'storm');
      refreshTimeline();
      aboutStats();
    }, 1200);
  }

  function wireUi() {
    $('runToggle').addEventListener('click', e => { const b = e.target.closest('[data-run]'); if (b) setRun(b.dataset.run); });
    $('playBtn').addEventListener('click', () => (S.playing ? pause() : play()));
    $('speedSel').addEventListener('change', e => { S.speed = parseFloat(e.target.value); });
    $('aboutBtn').addEventListener('click', () => { aboutStats(); $('about').hidden = false; });
    $('aboutClose').addEventListener('click', () => { $('about').hidden = true; });
    $('about').addEventListener('click', e => { if (e.target.id === 'about') $('about').hidden = true; });
    $('aboutNotes').textContent = `${Data.D.meta.notes.routes} ${Data.D.meta.notes.gaps}`;
    $('legendActs').innerHTML = Object.values(Data.ACTS).map(a => `<span><i style="background:${a.color}"></i>${a.name}</span>`).join('');
    $('recenterBtn').addEventListener('click', () => setFollow(true));

    // Track an agent: number (with suggestions), random, or the guided tour
    const list = $('agentList');
    for (const a of Data.D.agents) {
      const o = document.createElement('option');
      o.value = String(a.id);
      o.label = `${a.age} · ${a.g === 'F' ? 'female' : 'male'} · ${a.occ || a.role}`;
      list.appendChild(o);
    }
    $('trackBtn').addEventListener('click', e => { e.stopPropagation(); togglePop(); });
    $('trackForm').addEventListener('submit', e => {
      e.preventDefault();
      const v = parseInt($('trackInput').value, 10);
      if (Data.D.index.has(v)) { togglePop(false); track(Data.D.index.get(v), { focus: true }); }
      else $('trackMsg').textContent = `No agent #${$('trackInput').value.trim() || '?'}. Pick one from the suggestions, or try Random.`;
    });
    $('trackRandom').addEventListener('click', () => { togglePop(false); track(randomAgent(), { focus: true }); });
    $('tourLink').addEventListener('click', () => { togglePop(false); Story.start(Data.D.meta.featured[0]); });
    document.addEventListener('click', e => { if (!$('trackPop').hidden && !e.target.closest('#trackPop')) togglePop(false); });

    window.addEventListener('keydown', e => {
      if (e.key === 'Escape') {
        if (!$('trackPop').hidden) return togglePop(false);
        if (!$('about').hidden) { $('about').hidden = true; return; }
        if (S.sel != null && !Story.active) return stopTracking();
      }
      if (e.target.closest('input, select, textarea') || Story.active) return;
      if (e.code === 'Space') { e.preventDefault(); S.playing ? pause() : play(); }
      if (e.key === 'ArrowRight') setTick(S.tick + 12);
      if (e.key === 'ArrowLeft') setTick(S.tick - 12);
    });
  }

  function togglePop(on) {
    const pop = $('trackPop');
    const show = on == null ? pop.hidden : on;
    pop.hidden = !show;
    $('trackBtn').setAttribute('aria-expanded', String(show));
    if (show) { $('trackMsg').textContent = ''; $('trackInput').value = ''; setTimeout(() => $('trackInput').focus(), 0); }
  }

  // a random agent that actually travels on the day being shown, so there is something to follow
  function randomAgent() {
    const R = S.R, day = Data.clock(S.tick).day, pool = [];
    for (let a = 0; a < Data.D.agents.length; a++) {
      if (Data.tripsOf(R, a).some(i => !R.gap[i] && Math.floor(R.t.dep[i] / 288) + 1 === day && R.t.dep[i] >= S.tick - 1)) pool.push(a);
    }
    const from = pool.length ? pool : Data.D.agents.map((_, a) => a);
    return from[Math.floor(Math.random() * from.length)];
  }

  function aboutStats() {
    const s = Data.D.runs.storm, b = Data.D.runs.base;
    if (!s || !b) return;
    const count = R => { const c = Array(8).fill(0); for (let i = 0; i < R.n; i++) c[Math.floor(R.t.arr[i] / 288) + 1]++; return c; };
    const cs = count(s), cb = count(b), days = Data.D.meta.days;
    const parts = Data.D.meta.runs.storm.storm_days.map(d =>
      `on storm ${days[d - 1]} they make <b>${cs[d].toLocaleString()}</b> trips, ${Math.round(100 * (1 - cs[d] / cb[d]))}% fewer than the same ${days[d - 1]} without the storm (${cb[d].toLocaleString()})`);
    const p = d => Math.round(Data.workSchoolShare(s, d));
    const [sd1, sd2] = Data.D.meta.runs.storm.storm_days;               // Tue, Sat
    $('aboutStats').innerHTML = `<b>What happens:</b> ${parts.join('; ')}. The notice says to travel less, not which trips
      to drop: agents cut optional trips first, so <b>work and school</b> rise to <b>${p(sd1)}%</b> of out-of-home trips on storm
      ${days[sd1 - 1]} (${p(sd1 - 1)}% ${days[sd1 - 2]}, ${p(sd1 + 1)}% ${days[sd1]}) and <b>${p(sd2)}%</b> on storm ${days[sd2 - 1]}
      (${p(sd2 + 1)}% ${days[sd2]}).`;
  }

  /* ---------------- scenario ---------------- */
  async function setRun(run, progress) {
    if (S.run === run && S.R) return;
    const R = await Data.loadRun(run, progress, Data.clock(S.tick).day);
    S.run = run; S.R = R;
    document.querySelectorAll('#runToggle [data-run]').forEach(b => b.classList.toggle('active', b.dataset.run === run));
    refreshTimeline();
    if (S.sel != null) await track(S.sel, { day: S.selDay, keepView: true });
    lastDay = 0;
    render();
  }

  function refreshTimeline() {
    const other = Data.D.runs[S.run === 'storm' ? 'base' : 'storm'];
    Timeline.setCurves(S.R.onRoad, other ? other.onRoad : null);
    Timeline.setStorm(S.R.stormDays, S.run === 'storm');
  }

  /* ---------------- tracking ---------------- */
  function pick(a) { if (a != null && !S.kiosk && !Story.active) track(a, { focus: false }); }

  async function track(a, opts = {}) {
    const wasTracking = S.sel != null;
    S.sel = a;
    S.selDay = opts.day || Data.clock(S.tick).day;
    const id = Data.D.agents[a].id;
    S.file = await Data.agentFile(S.run, id);
    if (S.sel !== a) return;                           // a newer choice won
    if (!wasTracking) { document.body.classList.add('tracking'); MapView.resize(); }
    Panel.showAgent(a, S.file, S.R, S.selDay);
    const trips = MapView.showRoute(S.R, a, S.selDay);
    if (opts.focus && trips && trips.length) MapView.focusTrips(S.R, trips, { maxZoom: 14 });
    if (!opts.keepView) setFollow(true);
    Panel.setTick(S.tick);
    render();
    syncUrl(true);
  }

  function stopTracking() {
    if (S.sel == null) return;
    if (Story.active) Story.stop();
    S.sel = null; S.file = null;
    document.body.classList.remove('tracking');
    Panel.clear();
    MapView.showRoute(null);
    MapView.resize();
    MapView.fitCity(true);
    $('recenterBtn').hidden = true;
    render();
    syncUrl(true);
  }

  function setFollow(on) {
    S.follow = on;
    $('recenterBtn').hidden = on || S.sel == null;
    if (S.sel != null) $('recenterBtn').textContent = `⌖ Follow #${Data.D.agents[S.sel].id} again`;
    followAt = 0;
  }

  function showDay(d, jump) {
    S.selDay = d;
    if (S.sel == null) return;
    Panel.showAgent(S.sel, S.file, S.R, d);
    const trips = MapView.showRoute(S.R, S.sel, d);
    if (jump) {
      const day = S.file.days.find(x => x.d === d);
      setTick(day && day.plan ? day.plan.t : (d - 1) * 288 + 72);
      if (trips && trips.length) MapView.focusTrips(S.R, trips, { maxZoom: 14 });
    }
    Panel.setTick(S.tick);
  }

  /* ---------------- time ---------------- */
  function play(speed) {
    if (speed) { S.speed = speed; $('speedSel').value = String(speed); }
    S.playing = true; $('playBtn').textContent = '❚❚'; $('playBtn').setAttribute('aria-label', 'Pause');
  }
  function pause() { S.playing = false; $('playBtn').textContent = '▶'; $('playBtn').setAttribute('aria-label', 'Play'); syncUrl(true); }
  function setTick(t) { S.tick = ((t % Data.TICKS) + Data.TICKS) % Data.TICKS; render(); }
  function scrub(t, phase) {
    if (phase === 'start') { S.wasPlaying = S.playing; pause(); }
    setTick(t);
    if (phase === 'end' && S.wasPlaying) play();
  }

  function loop(now) {
    const dt = Math.min(0.1, (now - (last || now)) / 1000);
    last = now;
    if (S.playing) {
      S.tick += S.speed * dt;
      if (S.tick >= Data.TICKS) S.tick = 0;
    }
    render(dt);
    requestAnimationFrame(loop);
  }

  function render(dt = 0) {
    if (!S.R) return;
    const c = Data.clock(S.tick);
    const scene = { R: S.R, tick: S.tick, sel: S.sel };
    MapView.setScene(scene);
    Timeline.setTick(S.tick);
    $('clockDay').textContent = c.dayName;
    $('clockTime').textContent = c.text;
    $('clockMoving').textContent = scene.moving || 0;
    // storm cues
    const stormNow = S.R.stormDays.includes(c.day);
    const stormDayInOther = Data.D.meta.runs.storm.storm_days.includes(c.day);
    document.body.classList.toggle('is-storm', stormNow);
    const banner = $('stormBanner');
    if (stormNow) {
      banner.hidden = false; banner.classList.remove('calm');
      $('stormTag').textContent = `⛈ Storm day · added to every agent's prompt`;
      $('stormText').textContent = `“${Data.D.meta.runs[S.run].notice}”`;
    } else if (stormDayInOther && S.run === 'base') {
      banner.hidden = false; banner.classList.add('calm');
      $('stormTag').textContent = `No storm · same ${c.dayName}, no notice`;
      $('stormText').textContent = 'The same 1,000 agents, without the storm sentence. Switch to Storm week to compare.';
    } else banner.hidden = true;
    MapView.drawRain(stormNow && !reduceMotion, dt);
    if (!Data.ready(S.R, c.day)) Data.ensureUpTo(S.R, c.day);   // jumped ahead of the background loading
    const now = performance.now();
    // tracking: follow the car with the camera, and roll the panel over at midnight
    if (S.sel != null && !Story.active) {
      if (S.follow && scene.selLL && now - followAt > 400) { MapView.keepInView(scene.selLL); followAt = now; }
      if (c.day !== lastDay && S.playing) showDay(c.day, false);
    }
    lastDay = c.day;
    if (now - panelAt > 250) { Panel.setTick(S.tick); panelAt = now; }
    if (!S.playing) syncUrl(false);
  }

  function syncUrl(force) {
    const now = performance.now();
    if (!force && now - urlAt < 800) return;
    urlAt = now;
    const q = new URLSearchParams(location.search);
    q.set('run', S.run); q.set('t', S.tick.toFixed(1));
    if (S.sel != null) q.set('agent', Data.D.agents[S.sel].id); else q.delete('agent');
    q.delete('story');
    history.replaceState(null, '', '?' + q.toString());
  }

  const api = {
    setRun, pause, play: s => play(s), setTick, fitCity: () => MapView.fitCity(true),
    select: (a, o) => track(a, o), stopTracking, showDay: d => showDay(d, false),
    focusTrips: (trips, z) => MapView.focusTrips(S.R, trips, { maxZoom: z }),
  };

  window.addEventListener('DOMContentLoaded', boot);
  return { S, api };
})();
