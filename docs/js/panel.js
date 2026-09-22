/* Agent panel: profile, day picker, and the day's plan -> decisions -> reflection, each with what the
 * agent recalled from memory and its reasoning, verbatim from the simulation logs. */
const Panel = (() => {
  let el, grip, cb = {}, current = null, userScrollAt = 0, showAll = false;

  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const T0 = () => new Date(Data.D.meta.t0 + 'T00:00:00');

  function init(callbacks) {
    cb = callbacks;
    el = document.getElementById('panelInner');
    grip = document.getElementById('panelGrip');
    el.addEventListener('scroll', () => { userScrollAt = performance.now(); }, { passive: true });
    el.addEventListener('click', onClick);
    grip.addEventListener('click', e => {
      if (e.target.closest('#gripClose')) cb.close();
      else document.getElementById('panel').classList.toggle('open');
    });
  }

  function clear() { current = null; el.innerHTML = ''; setGrip(''); open(false); }

  function setGrip(label) { grip.dataset.label = label; }
  function open(on) { document.getElementById('panel').classList.toggle('open', on); }

  function tickLabel(t) { const c = Data.clock(t); return `${c.dayName} ${c.text}`; }
  function memWhen(s) {                                   // "2025-09-10 06:40" -> "Wed 06:40"
    const [d, hm] = s.split(' ');
    const k = Math.round((new Date(d + 'T00:00:00') - T0()) / 86400000);
    return `${Data.D.meta.days[k] || d} ${hm}`;
  }

  /* ---------------- agent view ---------------- */
  function showAgent(a, file, R, day) {
    const A = Data.D.agents[a];
    current = { a, file, R, day };
    const run = Data.D.meta.runs[R.run];
    const who = `${A.age} · ${A.g === 'F' ? 'female' : 'male'} · ${A.role}${A.occ ? ` (${A.occ})` : ''}`;
    setGrip(`Agent #${A.id} · ${who}`);
    const perDay = Array(8).fill(0);
    Data.tripsOf(R, a).forEach(i => { perDay[Math.floor(R.t.arr[i] / 288) + 1]++; });
    const days = Data.D.meta.days.map((n, k) => {
      const storm = run.storm_days.includes(k + 1);
      return `<button type="button" data-day="${k + 1}" class="${k + 1 === day ? 'active' : ''} ${storm ? 'storm' : ''}">${n}${storm ? ' ⛈' : ''}<small>${perDay[k + 1]} trips</small></button>`;
    }).join('');
    el.innerHTML = `
      <div class="p-head">
        <div><div class="p-title">Agent #${A.id} <small>· ${esc(run.label)}</small></div><div class="p-sub">${esc(who)}</div></div>
        <button type="button" class="btn-ghost" data-act="close" title="Stop tracking (Esc)">✕ Stop</button>
      </div>
      <div class="chips"><span class="chip">${esc(A.inc)}</span><span class="chip">${esc(A.hh)} household</span>
        <span class="chip">${esc(A.veh)}</span><span class="chip">${esc(A.kids)}</span></div>
      ${A.profile ? `<details class="p-profile"><summary>What the model was told about itself</summary><pre>${esc(A.profile)}</pre></details>` : ''}
      <div class="days" role="tablist">${days}</div>
      <div class="p-toolbar"><span>${esc(dayTitle(R, day))}</span>
        <label><input type="checkbox" data-act="all" ${showAll ? 'checked' : ''}> every decision</label></div>
      <div id="cards">${cardsFor(a, file, R, day)}</div>`;
    el.scrollTop = 0;
  }

  function dayTitle(R, day) {
    const storm = Data.D.meta.runs[R.run].storm_days.includes(day);
    const was = Data.D.meta.runs.storm.storm_days.includes(day);
    return `${Data.D.meta.days[day - 1]}${storm ? ' · storm notice in every prompt' : was ? ' · same day, no storm notice' : ''}`;
  }

  function memList(file, ids) {
    if (!ids || !ids.length) return '';
    const items = ids.map(i => {
      const s = file.s[i], k = s[0], j = s.indexOf('|', 2);          // "M|2025-09-10 06:40|text"
      const when = s.slice(2, j), text = s.slice(j + 1);
      return `<li class="${k === 'R' ? 'r' : ''}"><em>${k === 'R' ? 'reflection' : 'memory'} · ${esc(memWhen(when))}</em>${esc(text)}</li>`;
    }).join('');
    return `<details><summary>Remembered (${ids.length})</summary><ul class="mem">${items}</ul></details>`;
  }
  function thoughtBox(file, id) {
    const s = file.s[id];
    return s ? `<details><summary>Thinking</summary><div class="thought">${esc(s)}</div></details>` : '';
  }

  function cardsFor(a, file, R, day) {
    const d = file.days.find(x => x.d === day);
    const t = R.t, parts = [];
    if (!d) return '<div class="note">No logged activity for this day.</div>';
    const byTrip = new Map();
    for (const x of d.dec) if (x.tr != null) byTrip.set(x.tr, x);
    if (d.plan) {
      parts.push({ t: d.plan.t, html: `<div class="card" data-t="${d.plan.t}" style="--c:#FFCC33">
        <div class="card-top"><span class="card-kind">Plan</span><span class="card-time">${tickLabel(d.plan.t)}</span></div>
        <div class="card-main">${esc(file.s[d.plan.x])}</div>${memList(file, d.plan.m)}${thoughtBox(file, d.plan.th)}</div>` });
    }
    let place = null;
    for (const x of d.dec) {
      const act = Data.ACTS[x.a] || { name: `activity ${x.a}`, color: '#9CA3AF' };
      if (x.tr != null) {
        const i = x.tr, mins = (t.arr[i] - t.dep[i]) * 5;
        parts.push({ t: x.t, html: `<div class="card" data-t="${x.t}" data-trip="${i}" style="--c:${act.color}">
          <div class="card-top"><span class="card-kind">Decide · ${esc(act.name)}</span><span class="card-time">${tickLabel(x.t)}</span></div>
          <div class="card-main">${esc(file.s[x.x])}</div>
          <div class="card-move">→ travels to <b>${esc(act.name)}</b>, arrives ${Data.clock(t.arr[i]).text} (${mins} min) · plans to stay ~${esc(x.st)} min</div>
          ${memList(file, x.m)}${thoughtBox(file, x.th)}</div>` });
        place = x.a;
      } else if (showAll) {
        parts.push({ t: x.t, html: `<div class="card stay" data-t="${x.t}" style="--c:${act.color}">
          <div class="card-top"><span class="card-kind">Decide · stay${place === x.a || place == null ? '' : ' · ' + esc(act.name)}</span><span class="card-time">${tickLabel(x.t)}</span></div>
          <div class="card-main">${esc(file.s[x.x])} <span style="color:var(--muted)">(~${esc(x.st)} min)</span></div>
          ${memList(file, x.m)}${thoughtBox(file, x.th)}</div>` });
      }
    }
    // stretches the engine did not record (see meta.notes.gaps)
    Data.tripsOf(R, a).filter(i => R.gap[i] && Math.floor(t.dep[i] / 288) + 1 === day).forEach(i => {
      parts.push({ t: t.dep[i] + 0.5, html: `<div class="note">${Data.clock(t.dep[i]).text}–${Data.clock(t.arr[i]).text}: the engine logged
        no stay for this stretch, so the agent is hidden on the map. Its decisions above say what it intended.</div>` });
    });
    const ref = d.ref;
    if (ref) {
      parts.push({ t: ref.t, html: `<div class="card" data-t="${ref.t}" style="--c:#C4B5FD">
        <div class="card-top"><span class="card-kind">Reflect</span><span class="card-time">end of ${Data.D.meta.days[day - 1]}</span></div>
        <div class="card-main">${esc(file.s[ref.x])}</div>${memList(file, ref.m)}${thoughtBox(file, ref.th)}</div>` });
    }
    parts.sort((p, q) => p.t - q.t);
    return parts.map(p => p.html).join('') || '<div class="note">No decisions logged for this day.</div>';
  }

  function onClick(e) {
    if (e.target.closest('summary') || e.target.closest('.thought') || e.target.closest('.mem')) return;
    const act = e.target.closest('[data-act]');
    if (act) {
      const k = act.dataset.act;
      if (k === 'close') cb.close();
      else if (k === 'all') { showAll = act.checked; if (current) document.getElementById('cards').innerHTML = cardsFor(current.a, current.file, current.R, current.day); }
      return;
    }
    const dayBtn = e.target.closest('[data-day]');
    if (dayBtn) { cb.day(parseInt(dayBtn.dataset.day, 10)); return; }
    const card = e.target.closest('.card');
    if (card) cb.jump(parseFloat(card.dataset.t), card.dataset.trip != null ? parseInt(card.dataset.trip, 10) : null);
  }

  /* highlight the card that is "now" for the current tick */
  function setTick(tick) {
    if (!current) return;
    const cards = el.querySelectorAll('.card');
    let now = null;
    cards.forEach(c => { if (parseFloat(c.dataset.t) <= tick + 1e-6) now = c; });
    cards.forEach(c => c.classList.toggle('now', c === now));
    if (now && performance.now() - userScrollAt > 4000) {
      const r = now.getBoundingClientRect(), p = el.getBoundingClientRect();
      if (r.top < p.top || r.bottom > p.bottom) now.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }

  return { init, showAgent, clear, setTick, open, get current() { return current; } };
})();
