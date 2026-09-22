/* Data layer: loads the static files written by scripts/build_data.py and answers
 * "where is agent a at tick t?" for a run. Ticks are 5-minute steps from Monday 00:00 (0..2016). */
const Data = (() => {
  const KX = Math.cos(37.76 * Math.PI / 180);
  const TICKS = 7 * 288;
  const D = { meta: null, agents: null, index: new Map(), streets: null, land: null, bxy: null, runs: {}, files: new Map() };

  async function json(name) {
    const r = await fetch('data/' + name);
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
    return r.json();
  }

  // [x0, y0, dx1, dy1, ...] quantized by q  ->  Float64Array [lon0, lat0, lon1, lat1, ...]
  function decode(arr, q) {
    const out = new Float64Array(arr.length);
    let x = 0, y = 0;
    for (let i = 0; i < arr.length; i += 2) { x += arr[i]; y += arr[i + 1]; out[i] = x / q; out[i + 1] = y / q; }
    return out;
  }

  async function loadCore(progress) {
    progress('Loading 1,000 agents…');
    [D.meta, D.agents] = await Promise.all([json('meta.json'), json('agents.json')]);
    D.agents.forEach((a, i) => D.index.set(a.id, i));
    progress('Drawing San Francisco…');
    const [st, land, b] = await Promise.all([json('streets.json'), json('land.json'), json('buildings.json')]);
    D.streets = { cls: st.cls, e: st.e.map(e => decode(e, st.q)) };
    D.land = land.polys.map(p => ({ sf: p.sf, rings: p.rings.map(r => decode(r, land.q)) }));
    D.bxy = Float64Array.from(b.xy, v => v / b.q);
  }

  /* Route tables come in one file per day of first use (routes_<run>_<day>.json); a day's trips need
   * the files for that day and every earlier one. onChunk(R, day) fires as each file arrives. */
  let onChunk = null;
  function ensureUpTo(R, day) {
    const jobs = [];
    for (let d = 1; d <= day; d++) {
      if (!R.chunks.has(d)) {
        R.chunks.set(d, json(`routes_${R.run}_${d}.json`).then(arr => {
          arr.forEach((r, k) => { R.routes[R.off[d - 1] + k] = r; });
          R.loaded.add(d);
          if (onChunk) onChunk(R, d);
        }));
      }
      jobs.push(R.chunks.get(d));
    }
    return Promise.all(jobs);
  }
  const ready = (R, day) => { for (let d = 1; d <= day; d++) if (!R.loaded.has(d)) return false; return true; };

  async function loadRun(run, progress, day = 1) {
    if (D.runs[run]) { await ensureUpTo(D.runs[run], day); return D.runs[run]; }
    if (progress) progress(`Loading ${D.meta.runs[run].label.toLowerCase()}…`);
    const t = await json(`trips_${run}.json`);
    const n = t.agent.length, nA = D.agents.length;
    const gap = new Uint8Array(n);
    t.gap.forEach(i => { gap[i] = 1; });
    const start = new Int32Array(nA).fill(-1), end = new Int32Array(nA).fill(-1);
    for (let i = 0; i < n; i++) { const a = t.agent[i]; if (start[a] < 0) start[a] = i; end[a] = i + 1; }
    // agents on the road at each tick (difference array), unrecorded stretches excluded
    const diff = new Float32Array(TICKS + 2);
    for (let i = 0; i < n; i++) {
      if (gap[i]) continue;
      diff[Math.max(0, t.dep[i])] += 1;
      diff[Math.min(TICKS + 1, t.arr[i])] -= 1;
    }
    const onRoad = new Float32Array(TICKS + 1);
    let c = 0;
    for (let k = 0; k <= TICKS; k++) { c += diff[k]; onRoad[k] = c; }
    const R = { run, t, routes: new Array(t.route_off[7]), off: t.route_off, chunks: new Map(), loaded: new Set(),
                n, gap, start, end, onRoad, paths: new Map(), stormDays: D.meta.runs[run].storm_days };
    D.runs[run] = R;
    await ensureUpTo(R, day);
    ensureUpTo(R, 7);                                  // the rest of the week, in the background
    return R;
  }

  // Street path of trip i: building -> route edges -> building, with cumulative metres for constant speed.
  function path(R, i) {
    let p = R.paths.get(i);
    if (p) return p;
    const t = R.t, pts = [], codes = R.routes[t.route[i]];
    if (!codes) {                                     // route file still loading: straight line, not cached
      const P = Float64Array.of(D.bxy[2 * t.o[i]], D.bxy[2 * t.o[i] + 1], D.bxy[2 * t.b[i]], D.bxy[2 * t.b[i] + 1]);
      const len = Math.hypot((P[2] - P[0]) * KX * 111320, (P[3] - P[1]) * 110574) || 1;
      return { pts: P, cum: Float64Array.of(0, len), total: len, pending: true };
    }
    const push = (lon, lat) => {
      const m = pts.length;
      if (m && pts[m - 2] === lon && pts[m - 1] === lat) return;
      pts.push(lon, lat);
    };
    push(D.bxy[2 * t.o[i]], D.bxy[2 * t.o[i] + 1]);
    for (const code of codes) {
      const e = D.streets.e[code >> 1], m = e.length / 2, rev = code & 1;
      for (let k = 0; k < m; k++) { const j = rev ? m - 1 - k : k; push(e[2 * j], e[2 * j + 1]); }
    }
    push(D.bxy[2 * t.b[i]], D.bxy[2 * t.b[i] + 1]);
    const m = pts.length / 2, cum = new Float64Array(m);
    for (let k = 1; k < m; k++) {
      const dx = (pts[2 * k] - pts[2 * k - 2]) * KX * 111320, dy = (pts[2 * k + 1] - pts[2 * k - 1]) * 110574;
      cum[k] = cum[k - 1] + Math.hypot(dx, dy);
    }
    p = { pts: Float64Array.from(pts), cum, total: cum[m - 1] || 1 };
    R.paths.set(i, p);
    return p;
  }

  // Position along trip i at fraction f in [0,1]: {lon, lat, lon0, lat0, lon1, lat1} (segment for heading)
  function along(R, i, f) {
    const p = path(R, i), target = f * p.total, cum = p.cum;
    let lo = 0, hi = cum.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (cum[mid] <= target) lo = mid; else hi = mid; }
    const seg = cum[hi] - cum[lo], g = seg > 0 ? (target - cum[lo]) / seg : 0, P = p.pts;
    return {
      lon: P[2 * lo] + g * (P[2 * hi] - P[2 * lo]), lat: P[2 * lo + 1] + g * (P[2 * hi + 1] - P[2 * lo + 1]),
      lon0: P[2 * lo], lat0: P[2 * lo + 1], lon1: P[2 * hi], lat1: P[2 * hi + 1],
    };
  }

  /* State of agent index a at tick: {kind: 'stay'|'move'|'hidden', trip, b (building), act} */
  function state(R, a, tick) {
    const s = R.start[a], t = R.t;
    if (s < 0) return { kind: 'stay', trip: -1, b: D.agents[a].home, act: 1 };
    if (tick < t.dep[s]) return { kind: 'stay', trip: -1, b: t.o[s], act: 1 };
    let lo = s, hi = R.end[a] - 1;                       // last trip with dep <= tick
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (t.dep[mid] <= tick) lo = mid; else hi = mid - 1; }
    if (tick < t.arr[lo]) {
      if (R.gap[lo]) return { kind: 'hidden', trip: lo, b: -1, act: t.act[lo] };
      return { kind: 'move', trip: lo, b: t.b[lo], act: t.act[lo], f: (tick - t.dep[lo]) / Math.max(1, t.arr[lo] - t.dep[lo]) };
    }
    return { kind: 'stay', trip: lo, b: t.b[lo], act: t.act[lo] };
  }

  function tripsOf(R, a) {
    const out = [];
    if (R.start[a] < 0) return out;
    for (let i = R.start[a]; i < R.end[a]; i++) out.push(i);
    return out;
  }

  async function agentFile(run, id) {
    const key = run + '/' + id;
    if (!D.files.has(key)) D.files.set(key, json(`agents/${run}/${id}.json`));
    return D.files.get(key);
  }

  const ACTS = {
    1: { name: 'Home', color: '#10B981' }, 2: { name: 'Work', color: '#3B82F6' }, 3: { name: 'Eat meal', color: '#F59E0B' },
    4: { name: 'Education', color: '#8B5CF6' }, 5: { name: 'Recreation', color: '#EC4899' }, 6: { name: 'Shopping', color: '#06B6D4' },
    7: { name: 'Care', color: '#EF4444' }, 8: { name: 'Community', color: '#84CC16' }, 9: { name: 'Other', color: '#9CA3AF' },
    10: { name: 'Social visit', color: '#F97316' },
  };

  function clock(tick) {
    const k = Math.max(0, Math.min(TICKS - 0.001, tick));
    const day = Math.floor(k / 288), min = Math.floor((k % 288) * 5);
    return { day: day + 1, dayName: D.meta ? D.meta.days[day] : '', hh: String(Math.floor(min / 60)).padStart(2, '0'),
             mm: String(min % 60).padStart(2, '0'), text: `${String(Math.floor(min / 60)).padStart(2, '0')}:${String(min % 60).padStart(2, '0')}` };
  }

  return { D, TICKS, KX, ACTS, loadCore, loadRun, ensureUpTo, ready, path, along, state, tripsOf, agentFile, clock,
           set onChunk(f) { onChunk = f; } };
})();
