/* Pricing page. Nothing here charges anyone — the paid tiers are signposting
   while SmileBERTa is in beta. Prices are provisional; edit PLANS to change
   them, and keep the Free tier's limits in step with docking.py. */
(function () {
  const { $, $$, esc, icon } = SB;

  const PLANS = [
    {
      id: 'free',
      name: 'Free',
      price: '$0',
      cadence: 'always',
      tagline: 'Every tool, no account needed. This is what you get today.',
      cta: 'Start using it',
      href: './#tools',
      status: 'available',
      features: [
        ['All eight tools', true, 'Fragment prediction, classification, linking, builder, docking, structure studio, ADMET, Model Hub.'],
        ['Protein–ligand docking', true, '2 jobs at a time, 40 an hour — enough for a full afternoon of work.'],
        ['ADMET screening', true, 'Up to 200 compounds in one batch.'],
        ['Save your work', true, 'Molecules, docking runs and structures in your workspace.'],
        ['Colab notebooks', true, 'Generated from any Hugging Face model.'],
        ['Cite-everything outputs', true, 'Every value traceable to its paper.'],
      ],
    },
    {
      id: 'researcher',
      name: 'Researcher',
      price: '$5',
      cadence: 'per month',
      tagline: 'For people running this daily, who need it to not queue.',
      cta: 'Coming soon',
      status: 'planned',
      highlight: true,
      features: [
        ['Everything in Free', true],
        ['Priority docking queue', false, 'Your jobs jump ahead of the free queue.'],
        ['Much higher limits', false, 'More concurrent jobs and a far larger hourly allowance.'],
        ['Large batch screening', false, 'Thousands of compounds per ADMET or docking run.'],
        ['API access', false, 'An API key so SmileBERTa can run inside your own pipeline.'],
        ['Results kept longer', false, 'Docking runs survive restarts and stay available.'],
      ],
    },
    {
      id: 'lab',
      name: 'Lab',
      price: '$20',
      cadence: 'per month',
      tagline: 'For a research group sharing one target and one dataset.',
      cta: 'Coming soon',
      status: 'planned',
      features: [
        ['Everything in Researcher', true],
        ['Up to 10 seats', false, 'One invoice for the whole group.'],
        ['Shared workspace', false, 'Molecules and runs visible across the lab.'],
        ['Dedicated compute', false, 'Your jobs do not share a queue with anyone else.'],
        ['Email support', false, 'A person to ask when something looks wrong.'],
        ['Onboarding for your group', false, 'A session to get everyone running.'],
      ],
    },
  ];

  const FAQ = [
    ['Is anything paid right now?',
     'No. SmileBERTa is in beta and every feature on this site is free to use. The paid tiers above ' +
     'describe where it is heading so you can see the shape of it — there is no checkout, no card ' +
     'form and no trial that expires.'],
    ['Will the free tier survive?',
     'Yes. The free tier is meant to stay genuinely useful, not become a crippled demo. The things ' +
     'that will cost money are the ones that cost us money: sustained compute, big batches and ' +
     'dedicated capacity. Running a few dockings and profiling some molecules should stay free.'],
    ['What about teaching and coursework?',
     'Classroom use will be free. If you are running a course or a workshop and the limits get in ' +
     'your way, get in touch and we will raise them for your group.'],
    ['Is my work used to train your models?',
     'Only if you opt in, and only for predictions you explicitly rate. It is off by default. ' +
     'See the <a href="terms#privacy">privacy policy</a> for exactly what is stored.'],
    ['Can I run it myself instead?',
     'Yes — the code is open and the tools it builds on are listed on the ' +
     '<a href="credits">credits page</a>. If your structures are confidential, running locally is ' +
     'the right answer.'],
  ];

  $('#plans').innerHTML = PLANS.map((p) => `
    <div class="plan${p.highlight ? ' featured' : ''}">
      ${p.highlight ? '<div class="plan-flag">Planned flagship</div>' : ''}
      <div class="plan-head">
        <div class="row" style="gap:9px;align-items:center">
          <h3>${esc(p.name)}</h3>
          ${p.status === 'available'
            ? '<span class="badge green">Available now</span>'
            : '<span class="badge amber">Coming soon</span>'}
        </div>
        <div class="plan-price"><span class="amt">${esc(p.price)}</span>
          <span class="cad">${esc(p.cadence)}</span></div>
        <p class="plan-tag">${esc(p.tagline)}</p>
      </div>
      ${p.status === 'available'
        ? `<a class="btn btn-primary btn-block" href="${esc(p.href)}">${esc(p.cta)}</a>`
        : `<button class="btn btn-block" disabled>${esc(p.cta)}</button>`}
      <ul class="plan-feats">
        ${p.features.map(([label, now, detail]) => `
          <li class="${now ? 'on' : 'soon'}">
            <span class="fi">${icon(now ? 'check' : 'zap')}</span>
            <span><b>${esc(label)}</b>${detail ? `<span class="fd">${esc(detail)}</span>` : ''}</span>
          </li>`).join('')}
      </ul>
    </div>`).join('');

  $('#faq').innerHTML = FAQ.map(([q, a]) => `
    <div class="faq-item">
      <button class="faq-q" type="button"><span>${esc(q)}</span><span class="faq-chev">${icon('chevron')}</span></button>
      <div class="faq-a"><p>${a}</p></div>
    </div>`).join('');

  $$('.faq-q').forEach((b) => b.addEventListener('click', () =>
    b.parentElement.classList.toggle('open')));
})();
