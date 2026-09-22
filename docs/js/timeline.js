/* Week timeline: agents on the road across all 7 days (the city's "heartbeat"), storm days shaded,
 * the other scenario as a ghost line, a playhead, and click/drag scrubbing. */
const Timeline = (() => {
  let canvas, ctx, dpr = 1, onScrub = null;
  let main = null, ghost = null, stormDays = [], stormRun = true, tick = 0, peak = 1;
  const TOP = 16;                                 // label row height (css px)

  function init(c, scrub) {
    canvas = c; ctx = c.getContext('2d'); onScrub = scrub;
    const toTick = e => {
      const r = canvas.getBoundingClientRect();
      return Math.max(0, Math.min(Data.TICKS - 0.01, ((e.clientX - r.left) / r.width) * Data.TICKS));
    };
    canvas.addEventListener('pointerdown', e => {
      canvas.setPointerCapture(e.pointerId);
      onScrub(toTick(e), 'start');
      const move = ev => onScrub(toTick(ev), 'move');
      const up = () => { canvas.removeEventListener('pointermove', move); canvas.removeEventListener('pointerup', up); onScrub(tick, 'end'); };
      canvas.addEventListener('pointermove', move);
      canvas.addEventListener('pointerup', up);
    });
    window.addEventListener('resize', resize);
    resize();
  }

  function resize() {
    if (!canvas) return;
    dpr = Math.min(2, window.devicePixelRatio || 1);
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
    draw();
  }

  function setCurves(m, g) {
    main = m; ghost = g;
    peak = 1;
    for (const arr of [m, g]) if (arr) for (let k = 0; k < arr.length; k++) peak = Math.max(peak, arr[k]);
    draw();
  }
  function setStorm(days, isStormRun) { stormDays = days; stormRun = isStormRun; draw(); }
  function setTick(t) { tick = t; draw(); }

  function curve(arr, W, H, close) {
    const n = arr.length - 1;
    ctx.beginPath();
    ctx.moveTo(0, H);
    for (let x = 0; x <= W; x += 1) {
      const k0 = Math.floor((x / W) * n), k1 = Math.min(n, Math.floor(((x + 1) / W) * n));
      let v = 0;
      for (let k = k0; k <= k1; k++) v = Math.max(v, arr[k]);       // keep peaks at any width
      ctx.lineTo(x, H - (v / peak) * (H - TOP - 4));
    }
    if (close) { ctx.lineTo(W, H); ctx.closePath(); }
  }

  function draw() {
    if (!ctx || !main) return;
    const W = canvas.width / dpr, H = canvas.height / dpr, dayW = W / 7;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(255,255,255,0.03)';
    ctx.fillRect(0, TOP, W, H - TOP);
    // storm days (the same Tue/Sat are outlined in the no-storm run for comparison)
    for (const d of (Data.D.meta.runs.storm.storm_days || [])) {
      const x = (d - 1) * dayW;
      if (stormRun) { ctx.fillStyle = 'rgba(125,211,252,0.13)'; ctx.fillRect(x, TOP, dayW, H - TOP); }
      else { ctx.strokeStyle = 'rgba(143,167,154,0.5)'; ctx.setLineDash([3, 3]); ctx.strokeRect(x + 0.5, TOP + 0.5, dayW - 1, H - TOP - 1); ctx.setLineDash([]); }
    }
    if (ghost) {
      curve(ghost, W, H, false);
      ctx.strokeStyle = 'rgba(200,214,206,0.45)'; ctx.lineWidth = 1; ctx.setLineDash([2, 2]); ctx.stroke(); ctx.setLineDash([]);
    }
    curve(main, W, H, true);
    const grad = ctx.createLinearGradient(0, TOP, 0, H);
    grad.addColorStop(0, 'rgba(255,204,51,0.85)'); grad.addColorStop(1, 'rgba(150,70,10,0.35)');
    ctx.fillStyle = grad; ctx.fill();
    // day separators + labels
    ctx.font = '700 11px -apple-system, Segoe UI, Roboto, Arial, sans-serif';
    ctx.textBaseline = 'top';
    for (let d = 0; d < 7; d++) {
      const x = d * dayW;
      if (d) { ctx.fillStyle = 'rgba(244,240,226,0.14)'; ctx.fillRect(x, 0, 1, H); }
      const storm = (Data.D.meta.runs.storm.storm_days || []).includes(d + 1);
      ctx.fillStyle = storm ? (stormRun ? '#7DD3FC' : '#8FA79A') : '#C9D6CE';
      const label = Data.D.meta.days[d] + (storm ? (stormRun ? ' ⛈' : ' · no storm') : '');
      ctx.fillText(dayW < 70 && storm && !stormRun ? Data.D.meta.days[d] : label, x + 6, 2);
    }
    // playhead
    const px = (tick / Data.TICKS) * W;
    ctx.fillStyle = '#FFFFFF'; ctx.fillRect(px - 1, TOP - 2, 2, H - TOP + 2);
    ctx.beginPath(); ctx.arc(px, TOP - 2, 4, 0, 6.2832); ctx.fill();
  }

  return { init, resize, setCurves, setStorm, setTick, draw };
})();
