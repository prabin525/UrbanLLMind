/* Map: self-hosted basemap (shoreline + streets, no tile server), 1,000 agents on one canvas,
 * the selected agent's route + numbered stops, the COLM venue marker, and rain on storm days. */
const MapView = (() => {
  const SF_BOUNDS = [[37.706, -122.515], [37.833, -122.357]];
  const STREET = [{ color: '#2E5A45', w: 1.6 }, { color: '#22453A', w: 1.1 }, { color: '#18352A', w: 0.7 }];
  let map, streetLayers = [], routeLayer, canvas, ctx, rain, rctx, dpr = 1;
  let sprites = {}, positions = [], onPick = null, onUserPan = null, scene = null, drops = [];

  function ll(arr) {                               // Float64Array lon,lat -> [[lat, lon], ...]
    const out = new Array(arr.length / 2);
    for (let i = 0; i < arr.length; i += 2) out[i / 2] = [arr[i + 1], arr[i]];
    return out;
  }

  function init(pick, userPan) {
    onPick = pick; onUserPan = userPan;
    const D = Data.D;
    map = L.map('map', {
      zoomControl: false, attributionControl: true, zoomSnap: 0.25, zoomDelta: 0.5, minZoom: 11.5, maxZoom: 17.5,
      maxBounds: [[37.60, -122.70], [37.93, -122.18]], maxBoundsViscosity: 0.8, preferCanvas: true, fadeAnimation: false,
    });
    map.attributionControl.setPrefix(false);
    map.attributionControl.addAttribution('Map data © OpenStreetMap contributors · shoreline: U.S. Census Bureau');
    L.control.zoom({ position: 'topright' }).addTo(map);
    const renderer = L.canvas({ padding: 0.6 });

    for (const p of D.land) {
      L.polygon(p.rings.map(ll), { renderer, stroke: false, fillColor: p.sf ? '#0D2119' : '#0A1A13', fillOpacity: 1, interactive: false }).addTo(map);
    }
    for (let c = 2; c >= 0; c--) {
      const lines = [];
      D.streets.e.forEach((e, i) => { if (D.streets.cls[i] === c) lines.push(ll(e)); });
      streetLayers[c] = L.polyline(lines, { renderer, color: STREET[c].color, weight: STREET[c].w, interactive: false, smoothFactor: 1.2 }).addTo(map);
    }
    routeLayer = L.layerGroup().addTo(map);
    const v = D.meta.venue;
    L.marker([v.lat, v.lon], {
      interactive: false, keyboard: false, zIndexOffset: 1000,
      icon: L.divIcon({ className: '', iconSize: [18, 18], iconAnchor: [9, 9],
        html: '<div class="venue-icon" title="COLM 2026 venue: Hilton San Francisco Union Square"></div><div class="venue-label"><b>COLM 2026</b></div>' }),
    }).addTo(map);

    canvas = document.getElementById('agentsCanvas');
    ctx = canvas.getContext('2d');
    rain = document.getElementById('rainCanvas');
    rctx = rain.getContext('2d');
    makeSprites();
    map.on('move zoom resize', () => draw());
    map.on('zoomend', styleStreets);
    map.on('click', e => pickAt(e.containerPoint));
    map.on('dragstart', () => { if (onUserPan) onUserPan(); });
    window.addEventListener('resize', resize);
    fitCity(false);
    resize();
    styleStreets();
  }

  function fitCity(animate) {
    map.fitBounds(SF_BOUNDS, { padding: [12, 12], animate: !!animate });
  }

  function resize() {
    if (!map) return;
    map.invalidateSize({ pan: false });
    dpr = Math.min(2, window.devicePixelRatio || 1);
    const s = map.getSize();
    for (const c of [canvas, rain]) {
      c.width = Math.round(s.x * dpr); c.height = Math.round(s.y * dpr);
      c.style.width = s.x + 'px'; c.style.height = s.y + 'px';
    }
    draw();
  }

  function zoomScale(base, k, lo, hi) {
    return Math.max(lo, Math.min(hi, base * Math.pow(2, (map.getZoom() - 13) * k)));
  }

  function styleStreets() {
    STREET.forEach((s, c) => streetLayers[c] && streetLayers[c].setStyle({ weight: zoomScale(s.w, 0.55, s.w * 0.6, s.w * 4) }));
  }

  // car sprites (drawn facing +x): headlight beam, body, destination-coloured cabin, lights
  function makeSprites() {
    const S = 4, W = 44, H = 16;
    const make = (cabin, body, outline) => {
      const c = document.createElement('canvas');
      c.width = W * S; c.height = H * S;
      const g = c.getContext('2d');
      g.scale(S, S);
      const bodyL = 22, bodyW = 10, x0 = 4, cy = H / 2;
      const grad = g.createLinearGradient(x0 + bodyL, 0, W, 0);
      grad.addColorStop(0, 'rgba(255,255,210,0.55)');
      grad.addColorStop(1, 'rgba(255,255,210,0)');
      g.fillStyle = grad;
      g.beginPath(); g.moveTo(x0 + bodyL, cy - bodyW / 2 + 1); g.lineTo(W, cy - bodyW); g.lineTo(W, cy + bodyW); g.lineTo(x0 + bodyL, cy + bodyW / 2 - 1); g.closePath(); g.fill();
      g.fillStyle = body; g.strokeStyle = outline; g.lineWidth = 1.2;
      g.beginPath(); g.roundRect ? g.roundRect(x0, cy - bodyW / 2, bodyL, bodyW, 3) : g.rect(x0, cy - bodyW / 2, bodyL, bodyW); g.fill(); g.stroke();
      g.fillStyle = cabin; g.fillRect(x0 + bodyL * 0.3, cy - bodyW / 2 + 1.5, bodyL * 0.45, bodyW - 3);
      g.fillStyle = '#EF4444'; g.fillRect(x0, cy - bodyW / 2 + 1, 1.6, 2.4); g.fillRect(x0, cy + bodyW / 2 - 3.4, 1.6, 2.4);
      g.fillStyle = '#FEF08A'; g.fillRect(x0 + bodyL - 1.6, cy - bodyW / 2 + 1, 1.6, 2.4); g.fillRect(x0 + bodyL - 1.6, cy + bodyW / 2 - 3.4, 1.6, 2.4);
      return { c, W, H, bodyL, x0 };
    };
    for (const [k, a] of Object.entries(Data.ACTS)) {
      sprites[k] = make(a.color, '#F1F5F9', '#0B1B14');
      sprites['sel' + k] = make(a.color, '#FFCC33', '#06110C');
    }
  }

  function setScene(s) { scene = s; draw(); }

  function draw() {
    if (!ctx || !scene) return;
    const { R, tick, sel } = scene;
    const D = Data.D, n = D.agents.length, size = map.getSize();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.x, size.y);
    const carL = zoomScale(11, 0.6, 7, 26), dotR = zoomScale(2.1, 0.5, 1.3, 5.5);
    positions = [];
    let selPos = null, moving = 0;
    ctx.globalAlpha = 0.88;
    for (let a = 0; a < n; a++) {
      const st = Data.state(R, a, tick);
      if (st.kind === 'hidden') continue;
      let lon, lat, ang = 0;
      if (st.kind === 'move') {
        moving++;
        const p = Data.along(R, st.trip, st.f);
        lon = p.lon; lat = p.lat;
        const q0 = map.latLngToContainerPoint([p.lat0, p.lon0]), q1 = map.latLngToContainerPoint([p.lat1, p.lon1]);
        ang = Math.atan2(q1.y - q0.y, q1.x - q0.x) || 0;
      } else {
        lon = D.bxy[2 * st.b]; lat = D.bxy[2 * st.b + 1];
      }
      if (a === sel) scene.selLL = [lat, lon];        // recorded even off-screen, for the follow camera
      const pt = map.latLngToContainerPoint([lat, lon]);
      if (pt.x < -30 || pt.y < -30 || pt.x > size.x + 30 || pt.y > size.y + 30) continue;
      positions.push({ a, x: pt.x, y: pt.y });
      if (a === sel) { selPos = { pt, st, ang }; continue; }
      if (st.kind === 'move') drawCar(pt, ang, sprites[st.act], carL);
      else {
        ctx.fillStyle = Data.ACTS[st.act].color;
        ctx.beginPath(); ctx.arc(pt.x, pt.y, dotR, 0, 6.2832); ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    if (selPos) {
      if (selPos.st.kind === 'move') drawCar(selPos.pt, selPos.ang, sprites['sel' + selPos.st.act], carL * 1.6);
      else {
        // a ring, so a numbered stop pin at the same place stays readable inside it
        ctx.strokeStyle = 'rgba(255,204,51,0.35)'; ctx.lineWidth = 8;
        ctx.beginPath(); ctx.arc(selPos.pt.x, selPos.pt.y, 19, 0, 6.2832); ctx.stroke();
        ctx.strokeStyle = '#FFCC33'; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.arc(selPos.pt.x, selPos.pt.y, 19, 0, 6.2832); ctx.stroke();
      }
    }
    scene.moving = moving;
  }

  function drawCar(pt, ang, sp, bodyLen) {
    const k = bodyLen / sp.bodyL, w = sp.W * k, h = sp.H * k;
    ctx.save();
    ctx.translate(pt.x, pt.y);
    ctx.rotate(ang);
    ctx.drawImage(sp.c, -(sp.x0 + sp.bodyL / 2) * k, -h / 2, w, h);
    ctx.restore();
  }

  function pickAt(p) {
    let best = null, bd = 18 * 18;
    for (const q of positions) {
      const d = (q.x - p.x) ** 2 + (q.y - p.y) ** 2;
      if (d < bd) { bd = d; best = q.a; }
    }
    if (onPick) onPick(best);
  }

  /* Selected agent's route for one day: soft glow under the traffic + hairline + numbered stops. */
  function showRoute(R, a, day) {
    routeLayer.clearLayers();
    if (a == null || !R) return;
    const t = R.t, from = (day - 1) * 288, to = day * 288;
    const trips = Data.tripsOf(R, a).filter(i => t.dep[i] < to && t.arr[i] >= from);
    const stops = [];
    trips.forEach((i, k) => {
      if (!R.gap[i] && !Data.path(R, i).pending) {
        const pts = ll(Data.path(R, i).pts);
        L.polyline(pts, { color: '#FFFFFF', weight: 9, opacity: 0.16, interactive: false, lineCap: 'round' }).addTo(routeLayer);
        L.polyline(pts, { color: '#FFFFFF', weight: 2, opacity: 0.95, interactive: false }).addTo(routeLayer);
      }
      const b = t.b[i], lat = Data.D.bxy[2 * b + 1], lon = Data.D.bxy[2 * b];
      const near = stops.find(s => Math.abs(s.lat - lat) < 0.0012 && Math.abs(s.lon - lon) < 0.0015);
      if (near) near.n.push(k + 1); else stops.push({ lat, lon, n: [k + 1] });
    });
    for (const s of stops) {
      L.marker([s.lat, s.lon], { interactive: false, keyboard: false, zIndexOffset: 500,
        icon: L.divIcon({ className: '', iconSize: [0, 0], html: `<div class="stop-icon">${s.n.join('·')}</div>` }) }).addTo(routeLayer);
    }
    return trips;
  }

  function focusTrips(R, trips, opts = {}) {
    const pts = [];
    for (const i of trips) {
      const P = Data.path(R, i).pts;
      for (let k = 0; k < P.length; k += 2) pts.push([P[k + 1], P[k]]);
    }
    const sheet = document.getElementById('panel');              // phone: keep the route above the bottom sheet
    const below = window.innerWidth <= 820 && sheet.classList.contains('open') ? sheet.getBoundingClientRect().height : 0;
    if (pts.length) map.flyToBounds(L.latLngBounds(pts), { paddingTopLeft: [50, 60], paddingBottomRight: [50, 60 + below],
                                                           maxZoom: opts.maxZoom || 14.5, duration: 0.8 });
  }

  /* camera follow: pan only when the tracked agent leaves the middle of the view */
  function keepInView(ll) {
    const p = map.latLngToContainerPoint(ll), s = map.getSize();
    const sheet = document.getElementById('panel');           // phone: the bottom sheet covers part of the map
    const below = window.innerWidth <= 820 ? (sheet.classList.contains('open') ? sheet.getBoundingClientRect().height : 46) : 0;
    const h = s.y - below;
    if (p.x > s.x * 0.22 && p.x < s.x * 0.78 && p.y > h * 0.2 && p.y < h * 0.8) return false;
    map.panBy([p.x - s.x / 2, p.y - h / 2], { animate: true, duration: 0.6 });
    return true;
  }

  function flyTo(lat, lon, zoom) { map.flyTo([lat, lon], zoom || Math.max(map.getZoom(), 14), { duration: 0.8 }); }

  // rain streaks (storm days only); called from the app's animation loop
  function drawRain(active, dt) {
    const s = map.getSize();
    rctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    rctx.clearRect(0, 0, s.x, s.y);
    if (!active) { drops.length = 0; return; }
    const want = Math.round((s.x * s.y) / 5200);
    while (drops.length < want) drops.push({ x: Math.random() * s.x, y: Math.random() * s.y, v: 500 + Math.random() * 400, l: 8 + Math.random() * 10 });
    drops.length = Math.min(drops.length, want);
    rctx.strokeStyle = 'rgba(170, 215, 255, 0.28)';
    rctx.lineWidth = 1;
    rctx.beginPath();
    for (const d of drops) {
      d.y += d.v * dt; d.x -= d.v * dt * 0.25;
      if (d.y > s.y) { d.y = -d.l; d.x = Math.random() * (s.x + 100); }
      rctx.moveTo(d.x, d.y); rctx.lineTo(d.x - d.l * 0.25, d.y + d.l);
    }
    rctx.stroke();
  }

  return { init, setScene, draw, showRoute, focusTrips, flyTo, fitCity, keepInView, drawRain, resize, get map() { return map; } };
})();
