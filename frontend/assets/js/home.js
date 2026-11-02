/* Landing page interactions. */
(function () {
  const { icon, TOOLS, LEARN, $, $$ } = SB;
  const TAGS = { 'fragmentation': ['AI model', 'blue'], 'drug_class': ['AI model', 'blue'], 'combine': ['Chemistry', 'violet'], 'builder': ['Force field', 'cyan'], 'docking': ['Physics', 'green'], 'studio': ['Visualization', 'amber'] };
  const LONG = {
    'fragmentation': 'Give it a drug-like molecule and SmileBERTa predicts the fragment most relevant to binding. You get Rule-of-Three checks, a 3D conformer and a side-by-side property comparison.',
    'drug_class': 'Predict the therapeutic class of any compound (anti-infective, antineoplastic, CNS, cardiovascular and more), shown with ranked alternatives and a drug-likeness profile.',
    'combine': 'Link two fragments through amide or ester bridges. Only candidates that pass Lipinski and QED filters are kept.',
    'builder': 'An Avogadro-style 3D builder. Click to grow molecules atom by atom, then relax them with the MMFF94 or UFF force fields and search for conformers.',
    'docking': 'Load any PDB target, define a search box around the pocket, and dock ligands with AutoDock Vina. Inspect poses, affinities and H-bonds.',
    'studio': 'A Chimera-style viewer. Isolate chains, show backbones and surfaces, colour by secondary structure or B-factor, and measure distances and angles.',
  };

  $('#tool-cards').innerHTML = TOOLS.map((t, i) => `
    <a class="card card-hover tool-card reveal" data-delay="${i % 3}" href="${t.href}">
      <div class="icon-tile">${icon(t.icon)}</div>
      <span class="badge ${TAGS[t.href][1]} corner">${TAGS[t.href][0]}</span>
      <h3>${t.title}</h3>
      <p>${LONG[t.href]}</p>
      <span class="go">Open tool ${icon('arrow')}</span>
    </a>`).join('');

  $('#learn-cards').innerHTML = LEARN.map((t, i) => `
    <a class="card card-hover learn-card reveal" data-delay="${i}" href="${t.href}">
      <div class="icon-tile">${icon(t.icon)}</div><h4>${t.title}</h4><p>${t.desc}</p>
    </a>`).join('');

  // Real BPE tokens produced by the SmileBERTa tokenizer for aspirin.
  const tokens = ['CC', '(=', 'O', ')', 'Oc', '1', 'ccccc', '1', 'C', '(=', 'O', ')', 'O'];
  $('#tokens').innerHTML = tokens.map((t, i) => `<span class="tok" style="animation-delay:${0.25 + i * 0.09}s">${t}</span>`).join('');

  SB.initReveal();

  // Toggle "in" on pipeline & flow when visible (CSS-driven animations).
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); } }), { threshold: 0.3 });
  ['#pipeline', '#flow'].forEach((s) => $(s) && io.observe($(s)));

  // Animated counters.
  const cio = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    cio.unobserve(e.target);
    const el = e.target, end = +el.dataset.count, suffix = el.dataset.suffix || '';
    const t0 = performance.now(), dur = 1600;
    const step = (t) => {
      const p = Math.min(1, (t - t0) / dur), v = Math.round(end * (1 - Math.pow(1 - p, 3)));
      el.textContent = v.toLocaleString() + (p === 1 ? suffix : '');
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }), { threshold: 0.5 });
  $$('[data-count]').forEach((el) => cio.observe(el));

  // SMILES ticker typing effect.
  const smi = 'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1';
  let k = 0;
  const ticker = $('#ticker');
  (function type() { ticker.textContent = smi.slice(0, k); k = k >= smi.length + 30 ? 0 : k + 1; setTimeout(type, k > smi.length ? 60 : 45); })();

  // Hero 3D molecule.
  function initMol() {
    if (!window.$3Dmol) return setTimeout(initMol, 100);
    fetch('assets/data/imatinib.sdf').then((r) => r.text()).then((sdf) => {
      const el = $('#hero-mol');
      const v = $3Dmol.createViewer(el, { backgroundColor: 'rgba(0,0,0,0)', backgroundAlpha: 0, antialias: true });
      v.addModel(sdf, 'sdf');
      v.setStyle({}, { stick: { radius: 0.16, colorscheme: 'Jmol' }, sphere: { scale: 0.28, colorscheme: 'Jmol' } });
      v.setStyle({ elem: 'C' }, { stick: { radius: 0.16, color: '#8fb8ff' }, sphere: { scale: 0.28, color: '#8fb8ff' } });
      v.addSurface($3Dmol.SurfaceType.VDW, { opacity: 0.08, color: '#22d3ee' });
      v.zoomTo(); v.zoom(1.15); v.render();
      if (!matchMedia('(prefers-reduced-motion: reduce)').matches) v.spin('y', 0.6);
    }).catch(() => {});
  }
  initMol();

  // Hero background: drifting molecular graph.
  const canvas = $('#hero-canvas');
  const ctx = canvas.getContext('2d');
  let W, H, nodes;
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = canvas.clientWidth; H = canvas.clientHeight;
    canvas.width = W * dpr; canvas.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.min(90, Math.round(W * H / 16000));
    nodes = Array.from({ length: n }, () => ({ x: Math.random() * W, y: Math.random() * H, vx: (Math.random() - 0.5) * 0.25, vy: (Math.random() - 0.5) * 0.25, r: Math.random() * 1.6 + 0.8 }));
  }
  let mouse = { x: -999, y: -999 };
  canvas.parentElement.addEventListener('mousemove', (e) => { const r = canvas.getBoundingClientRect(); mouse = { x: e.clientX - r.left, y: e.clientY - r.top }; });
  function frame() {
    ctx.clearRect(0, 0, W, H);
    for (const n of nodes) {
      n.x += n.vx; n.y += n.vy;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
    }
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j], dx = a.x - b.x, dy = a.y - b.y, d = Math.hypot(dx, dy);
        if (d < 130) { ctx.strokeStyle = `rgba(96,165,250,${(1 - d / 130) * 0.28})`; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
      }
      const a = nodes[i], dm = Math.hypot(a.x - mouse.x, a.y - mouse.y);
      if (dm < 180) { ctx.strokeStyle = `rgba(34,211,238,${(1 - dm / 180) * 0.5})`; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(mouse.x, mouse.y); ctx.stroke(); }
    }
    for (const n of nodes) { ctx.fillStyle = 'rgba(147,197,253,.75)'; ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2); ctx.fill(); }
    requestAnimationFrame(frame);
  }
  resize(); addEventListener('resize', resize);
  if (!matchMedia('(prefers-reduced-motion: reduce)').matches) frame();
})();
