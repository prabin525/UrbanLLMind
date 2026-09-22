/* Guided story: agent #8429 (paper Table A2) through the storm week in seven steps.
 * Every quote is read from the agent's log file at runtime (verbatim, cut only at a sentence end). */
const Story = (() => {
  let card, app, steps = [], k = 0, active = false;
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function excerpt(s, max = 240) {                 // verbatim prefix ending at a sentence boundary
    if (!s || s.length <= max) return s;
    const cut = s.slice(0, max);
    const end = Math.max(cut.lastIndexOf('. '), cut.lastIndexOf('! '), cut.lastIndexOf('? '));
    return (end > 60 ? cut.slice(0, end + 1) : cut.slice(0, cut.lastIndexOf(' '))) + ' …';
  }
  const memText = s => s.slice(s.indexOf('|', 2) + 1);

  function init(appApi) {
    app = appApi;
    card = document.getElementById('storyCard');
    card.addEventListener('click', e => {
      const b = e.target.closest('[data-nav]');
      if (!b) return;
      const n = b.dataset.nav;
      if (n === 'next') go(k + 1); else if (n === 'back') go(k - 1);
      else if (n === 'compare') { stop(); app.stopTracking(); app.setRun('base'); }
      else { stop(); app.stopTracking(); }                       // "Explore": back to the full city
    });
    window.addEventListener('keydown', e => {
      if (!active || e.target.tagName === 'INPUT') return;
      if (e.key === 'ArrowRight') go(k + 1);
      if (e.key === 'ArrowLeft') go(k - 1);
      if (e.key === 'Escape') { stop(); app.stopTracking(); }
    });
  }

  async function start(agentId) {
    const a = Data.D.index.get(agentId);
    await app.setRun('storm');
    const R = Data.D.runs.storm, f = await Data.agentFile('storm', agentId);
    const day = d => f.days.find(x => x.d === d) || { dec: [] };
    const S = i => f.s[i];
    const A = Data.D.agents[a], notice = Data.D.meta.runs.storm.notice;
    const trips = d => Data.tripsOf(R, a).filter(i => Math.floor(R.t.dep[i] / 288) + 1 === d);
    const mon = day(1), tue = day(2), wed = day(3), sat = day(6);
    const firstMove = mon.dec.find(x => x.tr != null);
    const satTrips = trips(6).length;
    const who = `${A.age}-year-old ${A.occ || A.role.toLowerCase()}${/6 to 17|Under 6/.test(A.kids) ? ' with children at home' : ''}`;
    steps = [
      { title: 'Every morning, a plan', tick: mon.plan.t, day: 1, fit: trips(1),
        body: `Agent #${A.id} is a ${esc(who)}, and like every agent here it is an LLM. At the start of each day it writes a rough plan from its profile and memories.`,
        quotes: [{ label: `Plan · Mon ${Data.clock(mon.plan.t).text}`, text: excerpt(S(mon.plan.x)) }] },
      firstMove && { title: 'At every stop, a decision', tick: firstMove.t + 0.2, day: 1, trip: firstMove.tr, play: 1,
        body: 'Each time a stay ends, the model chooses what to do next and for how long. The simulation engine turns that choice into a trip across the city.',
        quotes: [{ label: `Decide · Mon ${Data.clock(firstMove.t).text} → ${Data.ACTS[firstMove.a].name}`, text: excerpt(S(firstMove.x)) }] },
      mon.ref && { title: 'Every night, a reflection', tick: 287.5, day: 1, fit: trips(1),
        body: 'At the end of the day it condenses what happened into a short reflection, which goes into its long-term memory.',
        quotes: [{ label: 'Reflect · end of Monday', text: excerpt(S(mon.ref.x), 300), cls: 'memory' }] },
      tue.plan && { title: 'Tuesday: one sentence changes', tick: tue.plan.t, day: 2, fit: trips(2),
        body: 'On Tuesday the same sentence is added to every agent’s prompt. No storm rules are coded anywhere; the agents decide what it means for them.',
        quotes: [{ label: 'Added to every prompt', text: notice, cls: 'storm' },
                 { label: `Plan · Tue ${Data.clock(tue.plan.t).text}`, text: excerpt(S(tue.plan.x)) }] },
      wed.plan && wed.plan.m.length && { title: 'Wednesday: remembering', tick: wed.plan.t, day: 3, fit: trips(3),
        body: 'Before planning, the agent retrieves its most relevant memories, including yesterday’s reflection.',
        quotes: [{ label: 'Remembered', text: excerpt(memText(S(wed.plan.m[0])), 280), cls: 'memory' },
                 { label: `Plan · Wed ${Data.clock(wed.plan.t).text}`, text: excerpt(S(wed.plan.x)) }] },
      sat.plan && { title: 'Saturday: a routine, then a storm', tick: sat.plan.t, day: 6, fit: satTrips ? trips(6) : null,
        body: `By the end of the week its reflections have become a routine (“On weekdays I…”). On storm Saturday it ${satTrips ? `makes ${satTrips} trip${satTrips > 1 ? 's' : ''}` : 'stays home all day'}.`,
        quotes: [sat.plan.m.length && { label: 'Remembered', text: excerpt(memText(S(sat.plan.m[0])), 280), cls: 'memory' },
                 { label: `Plan · Sat ${Data.clock(sat.plan.t).text}`, text: excerpt(S(sat.plan.x)) }].filter(Boolean) },
      { title: 'Your turn', tick: 288 + 8.5 * 12, day: 2, city: true, last: true,      // Tuesday 08:30
        body: 'Switch to <b>No storm</b> to see the same 1,000 agents’ Tuesday without the notice, or use <b>Track</b> (or tap any car or dot) to follow any agent.',
        quotes: [] },
    ].filter(Boolean);
    active = true;
    await app.select(a, { day: 1, quiet: true });
    go(0);
  }

  async function go(n) {
    if (n < 0 || n >= steps.length) return;
    k = n;
    const s = steps[k];
    render(s);
    app.pause();
    if (s.city) { app.setTick(s.tick); app.fitCity(); return; }
    app.showDay(s.day);
    app.setTick(s.tick);
    if (s.trip != null) app.focusTrips([s.trip], 15);
    else if (s.fit && s.fit.length) app.focusTrips(s.fit, 14);
    if (s.play) app.play(s.play);
  }

  function render(s) {
    const dots = steps.map((_, i) => `<i class="${i === k ? 'on' : ''}"></i>`).join('');
    card.innerHTML = `
      <div class="sc-step">Story · ${k + 1} of ${steps.length}</div>
      <div class="sc-title">${esc(s.title)}</div>
      <div class="sc-body">${s.body}</div>
      ${s.quotes.map(q => `<div class="sc-quote ${q.cls || ''}"><small>${esc(q.label)}</small>“${esc(q.text)}”</div>`).join('')}
      <div class="sc-nav">
        <button type="button" class="btn-ghost" data-nav="back" ${k === 0 ? 'disabled' : ''}>Back</button>
        <span class="sc-dots">${dots}</span>
        ${s.last ? `<span><button type="button" class="btn-ghost" data-nav="compare">Compare: No storm</button>
                    <button type="button" class="btn-gold" data-nav="done">Explore</button></span>`
                 : `<button type="button" class="btn-gold" data-nav="next">Next</button>`}
      </div>`;
    card.hidden = false;
  }

  function stop() { active = false; card.hidden = true; }

  return { init, start, stop, get active() { return active; } };
})();
