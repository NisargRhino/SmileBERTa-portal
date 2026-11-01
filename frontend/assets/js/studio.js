/* Structure Studio — a Chimera-style structure explorer built on 3Dmol.js. */
(function () {
  const { $, $$, esc, icon, fmt, toast, saveItem } = SB;

  const EXAMPLES = [['1CRN', 'Crambin'], ['4HHB', 'Hemoglobin'], ['1HSG', 'HIV protease'], ['1EMA', 'GFP'], ['6LU7', 'SARS-CoV-2 Mpro'], ['1BNA', 'B-DNA'], ['1UBQ', 'Ubiquitin']];
  const WATERS = ['HOH', 'WAT', 'DOD', 'H2O'];
  const THREE = { ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C', GLN: 'Q', GLU: 'E', GLY: 'G', HIS: 'H', ILE: 'I', LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P', SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V', MSE: 'M', SEC: 'U', DA: 'A', DC: 'C', DG: 'G', DT: 'T', A: 'A', C: 'C', G: 'G', U: 'U' };
  const KD = { ILE: 4.5, VAL: 4.2, LEU: 3.8, PHE: 2.8, CYS: 2.5, MET: 1.9, ALA: 1.8, GLY: -0.4, THR: -0.7, SER: -0.8, TRP: -0.9, TYR: -1.3, PRO: -1.6, HIS: -3.2, GLU: -3.5, GLN: -3.5, ASP: -3.5, ASN: -3.5, LYS: -3.9, ARG: -4.5 };
  const CHAIN_COLORS = ['#60a5fa', '#34d399', '#f472b6', '#fbbf24', '#a78bfa', '#22d3ee', '#fb923c', '#f87171', '#4ade80', '#e879f9'];

  let viewer = null, model = null, atoms = [], byIndex = new Map();
  let struct = null;  // { text, format, pdbId, label, chains, ligands, header, bRange }
  const state = { rep: 'cartoon', color: 'chain', single: '#60a5fa', sidechains: false, ligands: true, waters: false, hydrogens: false,
    surfType: 'none', surfColor: 'white', surfOp: 0.7, hidden: [], layers: [], bg: '#070a12', selected: null };
  let measureK = 2, picks = [], measures = [], activeTab = 'chains';

  // ── Colour helpers ───────────────────────────────────────────────────────
  function lerpColor(a, b, t) {
    const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
    const r = Math.round(((pa >> 16) & 255) + (((pb >> 16) & 255) - ((pa >> 16) & 255)) * t);
    const g = Math.round(((pa >> 8) & 255) + (((pb >> 8) & 255) - ((pa >> 8) & 255)) * t);
    const bl = Math.round((pa & 255) + ((pb & 255) - (pa & 255)) * t);
    return (r << 16) | (g << 8) | bl;
  }
  const hydroFunc = (atom) => { const v = KD[atom.resn]; if (v === undefined) return 0xcbd5e1; return v >= 0 ? lerpColor('#f8fafc', '#f97316', v / 4.5) : lerpColor('#f8fafc', '#3b82f6', -v / 4.5); };
  const spectrumFunc = (atom) => {
    const c = struct && struct.chainRange[atom.chain]; if (!c) return 0x94a3b8;
    const t = (atom.resi - c[0]) / Math.max(1, c[1] - c[0]);
    const stops = ['#3b82f6', '#22d3ee', '#34d399', '#facc15', '#f97316', '#ef4444'];
    const k = Math.min(stops.length - 2, Math.floor(t * (stops.length - 1)));
    return lerpColor(stops[k], stops[k + 1], t * (stops.length - 1) - k);
  };
  function colorSpec(scheme, rep) {
    switch (scheme) {
      case 'ss': return { colorscheme: 'ssJmol' };
      case 'chain': return { colorfunc: (a) => parseInt(chainColor(a.chain).slice(1), 16) };
      case 'spectrum': return rep === 'cartoon' || rep === 'tube' ? { color: 'spectrum' } : { colorfunc: spectrumFunc };
      case 'element': return { colorscheme: 'Jmol' };
      case 'amino': return { colorscheme: 'amino' };
      case 'hydro': return { colorfunc: hydroFunc };
      case 'bfactor': return { colorscheme: { prop: 'b', gradient: 'rwb', min: struct.bRange[1], max: struct.bRange[0] } };
      case 'white': return { color: '#e2e8f0' };
      default: return { color: state.single };
    }
  }
  function chainColor(ch) { if (!struct) return '#94a3b8'; const i = struct.chains.findIndex((c) => c.id === ch); return CHAIN_COLORS[(i < 0 ? 0 : i) % CHAIN_COLORS.length]; }

  // ── Structure analysis ───────────────────────────────────────────────────
  function analyze(text, format) {
    atoms = model.selectedAtoms({});
    byIndex = new Map(atoms.map((a) => [a.index, a]));
    const chains = new Map(), ligands = new Map(); let waters = 0, bmin = Infinity, bmax = -Infinity;
    const chainRange = {};
    for (const a of atoms) {
      if (typeof a.b === 'number') { bmin = Math.min(bmin, a.b); bmax = Math.max(bmax, a.b); }
      if (WATERS.includes(a.resn)) { if (a.atom === 'O') waters++; continue; }
      if (a.hetflag) {
        const k = `${a.resn}:${a.chain}:${a.resi}`;
        const l = ligands.get(k) || { key: k, resn: a.resn, chain: a.chain, resi: a.resi, n: 0 }; l.n++; ligands.set(k, l);
        continue;
      }
      const c = chains.get(a.chain) || { id: a.chain, residues: new Map(), atoms: 0, type: 'protein' };
      c.atoms++;
      if (!c.residues.has(a.resi)) c.residues.set(a.resi, { resn: a.resn, resi: a.resi, ss: 'c' });
      if (a.atom === 'CA' || a.atom === 'P') c.residues.get(a.resi).ss = a.ss || 'c';
      if (a.atom === "C4'" || a.atom === 'P') c.type = 'nucleic';
      chains.set(a.chain, c);
      const r = chainRange[a.chain] || [Infinity, -Infinity]; r[0] = Math.min(r[0], a.resi); r[1] = Math.max(r[1], a.resi); chainRange[a.chain] = r;
    }
    const header = {};
    if (format === 'pdb') {
      for (const line of text.split(/\r?\n/)) {
        if (line.startsWith('HEADER')) header.classification = line.slice(10, 50).trim();
        if (line.startsWith('TITLE')) header.title = ((header.title || '') + ' ' + line.slice(10).trim()).trim();
        if (line.startsWith('EXPDTA')) header.method = line.slice(10).trim();
        if (line.startsWith('REMARK   2 RESOLUTION.')) header.resolution = line.slice(22).trim();
        if (line.startsWith('ATOM')) break;
      }
    }
    return { chains: [...chains.values()].map((c) => ({ ...c, residues: [...c.residues.values()].sort((x, y) => x.resi - y.resi) })), ligands: [...ligands.values()].filter((l) => l.n > 1 || !['NA', 'CL', 'K'].includes(l.resn)), waters, header, chainRange, bRange: [isFinite(bmin) ? bmin : 0, isFinite(bmax) ? bmax : 1], atomCount: atoms.length };
  }

  // ── Rendering ────────────────────────────────────────────────────────────
  function visibleChains() { return struct.chains.map((c) => c.id).filter((c) => !state.hidden.includes(c)); }
  function repStyle(rep, color) {
    switch (rep) {
      case 'cartoon': return { cartoon: { ...color, arrows: true } };
      case 'tube': return { cartoon: { ...color, style: 'trace', thickness: 0.45 } };
      case 'lines': return { line: { ...color } };
      case 'stick': return { stick: { radius: 0.2, ...color } };
      case 'ballstick': return { stick: { radius: 0.12, ...color }, sphere: { scale: 0.25, ...color } };
      case 'sphere': return { sphere: { ...color } };
      case 'cartoonL': return { cartoon: { ...color } };
      default: return {};
    }
  }
  const BACKBONE = ['N', 'CA', 'C', 'O', 'P', "O5'", "C5'", "C4'", "C3'", "O3'"];

  function applyStyles() {
    if (!model) return;
    const vis = visibleChains();
    viewer.setStyle({}, {});
    const prot = { chain: vis, hetflag: false };
    const color = colorSpec(state.color, state.rep);
    if (state.rep === 'backbone') viewer.setStyle({ ...prot, atom: BACKBONE }, { stick: { radius: 0.22, ...color } });
    else if (state.rep !== 'none') viewer.setStyle(prot, repStyle(state.rep, color));
    if (state.sidechains && ['cartoon', 'tube', 'backbone'].includes(state.rep)) viewer.addStyle({ ...prot, not: { atom: ['N', 'C', 'O'] } }, { stick: { radius: 0.14, colorscheme: 'Jmol' } });
    if (state.ligands) viewer.setStyle({ hetflag: true, not: { resn: WATERS } }, { stick: { radius: 0.2, colorscheme: 'greenCarbon' }, sphere: { scale: 0.22, colorscheme: 'greenCarbon' } });
    if (state.waters) viewer.setStyle({ resn: WATERS }, { sphere: { scale: 0.22, color: '#7dd3fc', opacity: 0.8 } });
    state.layers.forEach((l) => { if (l.style !== 'surface') viewer.addStyle(l.sel, l.style === 'cartoon' ? { cartoon: { color: l.color } } : repStyle(l.style, { color: l.color })); });
    if (state.selected) viewer.addStyle({ chain: state.selected.chain, resi: state.selected.resi }, { stick: { radius: 0.2, colorscheme: 'orangeCarbon' } });
    if (!state.hydrogens) viewer.setStyle({ elem: 'H' }, {});
    drawOverlays();
    viewer.render();
  }

  function applySurfaces() {
    if (!model) return;
    viewer.removeAllSurfaces();
    const vis = visibleChains();
    if (state.surfType !== 'none') {
      const c = state.surfColor === 'hydro' ? { colorfunc: hydroFunc } : colorSpec(state.surfColor, 'surface');
      viewer.addSurface($3Dmol.SurfaceType[state.surfType], { opacity: state.surfOp, ...c }, { chain: vis, hetflag: false });
    }
    state.layers.filter((l) => l.style === 'surface').forEach((l) => viewer.addSurface($3Dmol.SurfaceType.MS, { opacity: 0.85, color: l.color }, l.sel));
    viewer.render();
  }

  function drawOverlays() {
    viewer.removeAllShapes(); viewer.removeAllLabels();
    state.layers.filter((l) => l.label).forEach((l) => viewer.addResLabels(l.sel, { fontSize: 11, fontColor: '#e2e8f0', backgroundColor: '#0b1020', backgroundOpacity: 0.75, inFront: true }));
    if (state.selected) viewer.addResLabels({ chain: state.selected.chain, resi: state.selected.resi }, { fontSize: 12, fontColor: '#fde68a', backgroundColor: '#1a1405', backgroundOpacity: 0.85, inFront: true });
    picks.forEach((a) => viewer.addSphere({ center: a, radius: 0.45, color: '#fbbf24', opacity: 0.5 }));
    measures.forEach((m) => {
      for (let i = 0; i < m.atoms.length - 1; i++) viewer.addCylinder({ start: m.atoms[i], end: m.atoms[i + 1], radius: 0.05, color: '#fbbf24', dashed: true, fromCap: 1, toCap: 1 });
      const c = m.atoms.reduce((s, p) => ({ x: s.x + p.x / m.atoms.length, y: s.y + p.y / m.atoms.length, z: s.z + p.z / m.atoms.length }), { x: 0, y: 0, z: 0 });
      viewer.addLabel(m.text, { position: c, fontSize: 12, fontColor: '#fde68a', backgroundColor: '#1a1405', backgroundOpacity: 0.85, inFront: true });
    });
  }

  // ── Panels ───────────────────────────────────────────────────────────────
  function renderChains() {
    $('#chains').innerHTML = struct.chains.map((c) => `<div class="chain-row ${state.hidden.includes(c.id) ? 'off' : ''}">
      <span class="swatch" style="background:${chainColor(c.id)}">${esc(c.id.trim() || '_')}</span>
      <div><b>Chain ${esc(c.id.trim() || '_')}</b> <span class="badge" style="margin-left:4px">${c.type}</span><div class="muted" style="font-size:.74rem">${c.residues.length} residues · ${c.atoms.toLocaleString()} atoms</div></div>
      <div class="row" style="gap:2px"><button class="btn btn-xs btn-ghost" data-ch-vis="${esc(c.id)}" title="Show / hide">${icon(state.hidden.includes(c.id) ? 'eyeOff' : 'eye')}</button><button class="btn btn-xs btn-ghost" data-ch-zoom="${esc(c.id)}" title="Zoom">${icon('search')}</button><button class="btn btn-xs btn-ghost" data-ch-iso="${esc(c.id)}" title="Isolate">${icon('target')}</button></div></div>`).join('')
      + (struct.ligands.length ? `<div class="panel-title" style="margin-top:16px">Ligands &amp; cofactors</div>${struct.ligands.map((l) => `<div class="layer-row"><span class="grow mono">${esc(l.resn)} <span class="muted">${esc(l.chain)}${l.resi}</span></span><span class="muted">${l.n} atoms</span><button class="btn btn-xs btn-ghost" data-lig-zoom="${esc(l.key)}">${icon('search')}</button></div>`).join('')}` : '')
      + (struct.waters ? `<p class="muted" style="font-size:.78rem;margin-top:10px">${struct.waters} water molecules</p>` : '');
    $$('[data-ch-vis]').forEach((b) => b.addEventListener('click', () => { const c = b.dataset.chVis; state.hidden = state.hidden.includes(c) ? state.hidden.filter((x) => x !== c) : [...state.hidden, c]; renderChains(); applyStyles(); applySurfaces(); }));
    $$('[data-ch-zoom]').forEach((b) => b.addEventListener('click', () => viewer.zoomTo({ chain: b.dataset.chZoom }, 600)));
    $$('[data-ch-iso]').forEach((b) => b.addEventListener('click', () => { const c = b.dataset.chIso; state.hidden = struct.chains.map((x) => x.id).filter((x) => x !== c); renderChains(); applyStyles(); applySurfaces(); viewer.zoomTo({ chain: c }, 600); }));
    $$('[data-lig-zoom]').forEach((b) => b.addEventListener('click', () => { const [resn, chain, resi] = b.dataset.ligZoom.split(':'); viewer.zoomTo({ resn, chain, resi: +resi }, 600); }));
  }

  function renderInfo() {
    const h = struct.header, res = struct.chains.reduce((s, c) => s + c.residues.length, 0);
    const ss = { h: 0, s: 0, c: 0 }; struct.chains.forEach((c) => c.residues.forEach((r) => (ss[r.ss] = (ss[r.ss] || 0) + 1)));
    const tot = Math.max(1, ss.h + ss.s + ss.c);
    $('#info').innerHTML = `${h.title ? `<p style="font-size:.86rem;color:var(--text)">${esc(h.title)}</p>` : ''}
      <div class="kv">${struct.pdbId ? `<div>PDB ID</div><div><a href="https://www.rcsb.org/structure/${esc(struct.pdbId)}" target="_blank" rel="noopener">${esc(struct.pdbId)} ↗</a></div>` : ''}
        ${h.classification ? `<div>Classification</div><div>${esc(h.classification)}</div>` : ''}${h.method ? `<div>Method</div><div>${esc(h.method)}</div>` : ''}${h.resolution ? `<div>Resolution</div><div>${esc(h.resolution)}</div>` : ''}
        <div>Atoms</div><div>${struct.atomCount.toLocaleString()}</div><div>Polymer chains</div><div>${struct.chains.length}</div><div>Residues</div><div>${res.toLocaleString()}</div>
        <div>Ligands</div><div>${struct.ligands.length}</div><div>Waters</div><div>${struct.waters}</div><div>B-factor range</div><div>${fmt(struct.bRange[0], 1)} – ${fmt(struct.bRange[1], 1)}</div></div>
      ${res ? `<div class="panel-title" style="margin-top:18px">Secondary structure</div>
      <div style="display:flex;height:10px;border-radius:5px;overflow:hidden"><div style="width:${ss.h / tot * 100}%;background:#f472b6"></div><div style="width:${ss.s / tot * 100}%;background:#fbbf24"></div><div style="width:${ss.c / tot * 100}%;background:#475569"></div></div>
      <div class="row" style="gap:12px;font-size:.76rem;margin-top:6px;color:var(--muted)"><span><b style="color:#f472b6">●</b> Helix ${Math.round(ss.h / tot * 100)}%</span><span><b style="color:#fbbf24">●</b> Sheet ${Math.round(ss.s / tot * 100)}%</span><span><b style="color:#475569">●</b> Coil ${Math.round(ss.c / tot * 100)}%</span></div>` : ''}`;
  }

  function renderSequence() {
    const ch = struct.chains.find((c) => c.id === $('#seq-chain').value) || struct.chains[0];
    if (!ch) { $('#seq-strip').classList.add('hidden'); return; }
    $('#seq-strip').classList.remove('hidden');
    $('#seq').innerHTML = ch.residues.map((r, i) => `<span class="${r.ss === 'h' ? 'h' : r.ss === 's' ? 's' : ''} ${i % 10 === 0 ? 'tick' : ''} ${state.selected && state.selected.chain === ch.id && state.selected.resi === r.resi ? 'sel' : ''}" data-n="${r.resi}" data-resi="${r.resi}" title="${esc(r.resn)} ${r.resi}">${THREE[r.resn] || 'X'}</span>`).join('');
    $$('#seq span').forEach((s) => s.addEventListener('click', () => selectResidue(ch.id, +s.dataset.resi, true)));
    setTimeout(() => viewer.resize(), 0);
  }

  function renderLayers() {
    $('#layers').innerHTML = state.layers.length ? state.layers.map((l, i) => `<div class="layer-row"><span style="width:12px;height:12px;border-radius:3px;background:${l.color}"></span><span class="grow">${esc(l.name)} <span class="muted">· ${l.style}</span></span><button class="btn btn-xs btn-ghost" data-l-zoom="${i}">${icon('search')}</button><button class="btn btn-xs btn-ghost" data-l-del="${i}">${icon('x')}</button></div>`).join('') : '<p class="muted" style="font-size:.8rem">No custom layers.</p>';
    $$('[data-l-del]').forEach((b) => b.addEventListener('click', () => { const l = state.layers.splice(+b.dataset.lDel, 1)[0]; renderLayers(); applyStyles(); if (l.style === 'surface') applySurfaces(); }));
    $$('[data-l-zoom]').forEach((b) => b.addEventListener('click', () => viewer.zoomTo(state.layers[+b.dataset.lZoom].sel, 600)));
  }

  function renderMeasures() {
    $('#measures').innerHTML = measures.map((m, i) => `<div class="meas-row"><span><b>${m.kind}</b> <span class="muted mono" style="font-size:.72rem">${esc(m.desc)}</span></span><span class="mono" style="color:var(--amber)">${esc(m.text)}</span></div>`).join('') || '<p class="muted" style="font-size:.8rem">No measurements yet.</p>';
  }

  // ── Interaction ──────────────────────────────────────────────────────────
  const atomLabel = (a) => `${a.resn || ''} ${a.chain || ''}${a.resi ?? ''} ${a.atom || a.elem}`.trim();
  function selectResidue(chain, resi, zoom) {
    state.selected = { chain, resi };
    applyStyles();
    if (zoom) viewer.zoomTo({ chain, resi }, 500);
    $$('#seq span').forEach((s) => s.classList.toggle('sel', $('#seq-chain').value === chain && +s.dataset.resi === resi));
    const resAtoms = atoms.filter((a) => a.chain === chain && a.resi === resi);
    if (resAtoms.length) {
      const r = resAtoms[0];
      $('#sel-hud').classList.remove('hidden');
      $('#sel-hud').innerHTML = `<b style="color:#fde68a">${esc(r.resn)} ${esc(chain)}${resi}</b> · ${resAtoms.length} atoms · avg B ${fmt(resAtoms.reduce((s, a) => s + (a.b || 0), 0) / resAtoms.length, 1)}${KD[r.resn] !== undefined ? ` · KD hydropathy ${KD[r.resn]}` : ''}`;
    }
  }

  function showAtom(a) {
    $('#atom-panel').classList.remove('hidden');
    const bonds = (a.bonds || []).map((j, k) => { const b = byIndex.get(j) || atoms[j]; if (!b) return null; return { b, order: a.bondOrder ? a.bondOrder[k] : 1, d: Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z) }; }).filter(Boolean);
    $('#atom-info').innerHTML = `<div class="kv"><div>Atom</div><div>${esc(a.atom || a.elem)} (${esc(a.elem)})</div>${a.resn ? `<div>Residue</div><div>${esc(a.resn)} ${esc(a.chain)}${a.resi}</div>` : ''}<div>B-factor</div><div>${fmt(a.b, 2)}</div><div>Secondary str.</div><div>${{ h: 'Helix', s: 'Sheet', c: 'Coil' }[a.ss] || '—'}</div><div>Position</div><div>${fmt(a.x, 2)}, ${fmt(a.y, 2)}, ${fmt(a.z, 2)}</div></div>
      <div class="label" style="margin-top:12px">Bonds (${bonds.length})</div>${bonds.map((x) => `<div class="meas-row"><span>${esc(a.atom || a.elem)} ${'—=≡'[Math.min(2, (x.order || 1) - 1)] || '—'} ${esc(x.b.atom || x.b.elem)} <span class="muted">${x.b.resi !== a.resi ? esc(x.b.resn + ' ' + x.b.resi) : ''}</span></span><span class="mono">${fmt(x.d, 3)} Å</span></div>`).join('') || '<p class="muted" style="font-size:.8rem">No bonds</p>'}`;
  }

  function onClick(a) {
    if (activeTab === 'measure') {
      picks.push({ x: a.x, y: a.y, z: a.z, label: atomLabel(a) });
      $('#mh-count').textContent = picks.length;
      if (picks.length === measureK) {
        const p = picks.map((q) => [q.x, q.y, q.z]);
        const sub = (u, v) => [u[0] - v[0], u[1] - v[1], u[2] - v[2]], dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2], len = (u) => Math.hypot(...u);
        const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        let kind, text;
        if (measureK === 2) { kind = 'Distance'; text = fmt(len(sub(p[0], p[1])), 3) + ' Å'; }
        if (measureK === 3) { const u = sub(p[0], p[1]), v = sub(p[2], p[1]); kind = 'Angle'; text = fmt(Math.acos(Math.max(-1, Math.min(1, dot(u, v) / (len(u) * len(v))))) * 180 / Math.PI, 1) + '°'; }
        if (measureK === 4) { const b1 = sub(p[1], p[0]), b2 = sub(p[2], p[1]), b3 = sub(p[3], p[2]); const n1 = cross(b1, b2), n2 = cross(b2, b3); const m1 = cross(n1, b2.map((x) => x / len(b2))); kind = 'Dihedral'; text = fmt(Math.atan2(dot(m1, n2), dot(n1, n2)) * 180 / Math.PI, 1) + '°'; }
        measures.push({ kind, text, atoms: picks.map(({ x, y, z }) => ({ x, y, z })), desc: picks.map((q) => q.label).join(' · ') });
        picks = []; $('#mh-count').textContent = 0;
        renderMeasures();
      }
      drawOverlays(); viewer.render();
      return;
    }
    showAtom(a);
    if (a.resi !== undefined && !a.hetflag) selectResidue(a.chain, a.resi, false);
    else { state.selected = null; applyStyles(); }
  }

  const tip = SB.h('<div class="tooltip-3d hidden"></div>'); document.body.appendChild(tip);
  document.addEventListener('mousemove', (e) => { tip.style.left = e.clientX + 'px'; tip.style.top = e.clientY + 'px'; });
  document.addEventListener('mouseover', (e) => { if (!e.target.closest('#viewer')) tip.classList.add('hidden'); });

  // ── Loading ──────────────────────────────────────────────────────────────
  async function load(text, format, label, pdbId = null, savedState = null) {
    viewer.clear();
    measures = []; picks = []; state.selected = null; state.layers = []; state.hidden = [];
    if (savedState) Object.assign(state, savedState, { selected: null });
    try { model = viewer.addModel(text, format, { keepH: true }); } catch (e) { toast('Could not parse this file', 'error'); return; }
    struct = { text, format, label, pdbId, ...analyze(text, format) };
    if (!struct.atomCount) { toast('No atoms found in this file', 'error'); return; }
    if (!struct.chains.length && state.rep === 'cartoon') state.rep = 'ballstick';
    syncControls();
    $('#stage-empty').classList.add('hidden');
    $('#title').textContent = label;
    $('#title-badge').classList.remove('hidden'); $('#title-badge').textContent = `${struct.atomCount.toLocaleString()} atoms`;
    viewer.setHoverable({}, true, (a) => { tip.textContent = `${atomLabel(a)} · ${a.elem}${typeof a.b === 'number' ? ' · B ' + a.b.toFixed(1) : ''}`; tip.classList.remove('hidden'); }, () => tip.classList.add('hidden'));
    viewer.setClickable({}, true, onClick);
    renderChains(); renderInfo(); renderLayers(); renderMeasures();
    $('#seq-chain').innerHTML = struct.chains.map((c) => `<option value="${esc(c.id)}">Chain ${esc(c.id.trim() || '_')}</option>`).join('');
    $('#q-chain').innerHTML = '<option value="">Any</option>' + struct.chains.map((c) => `<option>${esc(c.id)}</option>`).join('');
    $('#site-lig').innerHTML = struct.ligands.length ? struct.ligands.map((l) => `<option value="${esc(l.key)}">${esc(l.resn)} ${esc(l.chain)}${l.resi}</option>`).join('') : '<option value="">No ligands</option>';
    renderSequence();
    applyStyles(); applySurfaces();
    viewer.zoomTo(); viewer.render();
  }

  async function fetchPDB(id) {
    id = (id || '').trim().toUpperCase();
    if (!/^[0-9][A-Z0-9]{3}$/.test(id)) return toast('PDB IDs are 4 characters, e.g. 4HHB', 'error');
    const btn = $('#fetch'); btn.classList.add('loading');
    try {
      let r = await fetch(`https://files.rcsb.org/download/${id}.pdb`), fmtName = 'pdb';
      if (!r.ok) { r = await fetch(`https://files.rcsb.org/download/${id}.cif`); fmtName = 'cif'; }
      if (!r.ok) throw new Error(`Structure ${id} was not found in the PDB`);
      await load(await r.text(), fmtName, id, id);
    } catch (e) { toast(e.message, 'error'); } finally { btn.classList.remove('loading'); }
  }

  function syncControls() {
    $$('#rep button').forEach((b) => b.classList.toggle('active', b.dataset.r === state.rep));
    $('#color').value = state.color; $('#single-color').value = state.single; $('#single-color').classList.toggle('hidden', state.color !== 'single');
    $('#sidechains').checked = state.sidechains; $('#ligands').checked = state.ligands; $('#waters').checked = state.waters; $('#hydrogens').checked = state.hydrogens;
    $('#surf-type').value = state.surfType; $('#surf-color').value = state.surfColor; $('#surf-op').value = state.surfOp;
    $('#bg').value = state.bg; viewer.setBackgroundColor(state.bg);
  }

  function buildQuery() {
    const sel = { chain: visibleChains() };
    const ch = $('#q-chain').value; if (ch) sel.chain = ch;
    const resi = $('#q-resi').value.trim();
    if (resi) sel.resi = resi.split(',').map((s) => s.trim()).filter(Boolean);
    const list = (v) => v.split(',').map((s) => s.trim().toUpperCase()).filter(Boolean);
    const rn = list($('#q-resn').value); if (rn.length) sel.resn = rn;
    const an = list($('#q-atom').value); if (an.length) sel.atom = an;
    const el = $('#q-elem').value.split(',').map((s) => s.trim()).filter(Boolean).map((e) => e[0].toUpperCase() + e.slice(1).toLowerCase()); if (el.length) sel.elem = el;
    return sel;
  }
  function describe(sel) { return [sel.chain && !Array.isArray(sel.chain) ? 'chain ' + sel.chain : '', sel.resi ? 'res ' + sel.resi.join(',') : '', sel.resn ? sel.resn.join(',') : '', sel.atom ? sel.atom.join(',') : '', sel.elem ? sel.elem.join(',') : ''].filter(Boolean).join(' · ') || 'all visible'; }

  // ── Init ─────────────────────────────────────────────────────────────────
  function init() {
    if (!window.$3Dmol) return setTimeout(init, 80);
    viewer = $3Dmol.createViewer($('#viewer'), { backgroundColor: state.bg, antialias: true });
    $('#upload').innerHTML = icon('upload'); $('#save').innerHTML = icon('save') + ' Save session';
    $('#examples').innerHTML = EXAMPLES.map(([id, n]) => `<button class="chip" data-id="${id}">${id} · ${n}</button>`).join('');
    $$('#examples .chip').forEach((c) => c.addEventListener('click', () => { $('#pdb-id').value = c.dataset.id; fetchPDB(c.dataset.id); }));
    $('#fetch').addEventListener('click', () => fetchPDB($('#pdb-id').value));
    $('#pdb-id').addEventListener('keydown', (e) => { if (e.key === 'Enter') fetchPDB(e.target.value); });
    $('#upload').addEventListener('click', () => {
      const inp = SB.h('<input type="file" accept=".pdb,.ent,.cif,.mmcif,.sdf,.mol,.mol2,.xyz,.pdbqt" hidden>'); document.body.appendChild(inp);
      inp.addEventListener('change', async () => {
        const f = inp.files[0]; inp.remove(); if (!f) return;
        let ext = f.name.split('.').pop().toLowerCase(); if (ext === 'ent') ext = 'pdb'; if (ext === 'mmcif') ext = 'cif'; if (ext === 'mol') ext = 'sdf';
        load(await SB.readFile(f), ext, f.name);
      });
      inp.click();
    });

    $$('#rep button').forEach((b) => b.addEventListener('click', () => { state.rep = b.dataset.r; syncControls(); applyStyles(); }));
    $('#color').addEventListener('change', (e) => { state.color = e.target.value; syncControls(); applyStyles(); });
    $('#single-color').addEventListener('input', (e) => { state.single = e.target.value; applyStyles(); });
    [['sidechains', 'sidechains'], ['ligands', 'ligands'], ['waters', 'waters'], ['hydrogens', 'hydrogens']].forEach(([id, k]) => $('#' + id).addEventListener('change', (e) => { state[k] = e.target.checked; applyStyles(); }));
    $('#surf-type').addEventListener('change', (e) => { state.surfType = e.target.value; applySurfaces(); });
    $('#surf-color').addEventListener('change', (e) => { state.surfColor = e.target.value; applySurfaces(); });
    $('#surf-op').addEventListener('change', (e) => { state.surfOp = +e.target.value; applySurfaces(); });
    $('#bg').addEventListener('change', (e) => { state.bg = e.target.value; viewer.setBackgroundColor(state.bg); viewer.render(); });
    $('#proj').addEventListener('change', (e) => { if (viewer.setProjection) { viewer.setProjection(e.target.value); viewer.render(); } });
    $('#outline').addEventListener('change', (e) => { viewer.setViewStyle(e.target.checked ? { style: 'outline', color: 'black', width: 0.08 } : { style: 'none' }); viewer.render(); });
    $('#spin').addEventListener('change', (e) => viewer.spin(e.target.checked ? 'y' : false, 0.6));
    $('#fog').addEventListener('change', (e) => { viewer.enableFog(e.target.checked); viewer.render(); });

    $$('[data-tab]').forEach((b) => b.addEventListener('click', () => {
      activeTab = b.dataset.tab;
      $$('[data-tab]').forEach((x) => x.classList.toggle('active', x === b));
      $$('[data-pane]').forEach((p) => p.classList.toggle('hidden', p.dataset.pane !== activeTab));
      $('#measure-hud').classList.toggle('hidden', activeTab !== 'measure');
      picks = []; $('#mh-count').textContent = 0; if (model) { drawOverlays(); viewer.render(); }
    }));
    $$('#m-kind button').forEach((b) => b.addEventListener('click', () => { measureK = +b.dataset.k; $$('#m-kind button').forEach((x) => x.classList.toggle('active', x === b)); $('#mh-kind').textContent = b.textContent.toLowerCase(); picks = []; $('#mh-count').textContent = 0; drawOverlays(); viewer.render(); }));
    $('#m-clear').addEventListener('click', () => { measures = []; picks = []; renderMeasures(); drawOverlays(); viewer.render(); });

    $('#seq-chain').addEventListener('change', renderSequence);
    const needModel = () => { if (!model) { toast('Load a structure first', 'error'); return false; } return true; };
    $('#q-add').addEventListener('click', () => {
      if (!needModel()) return;
      const sel = buildQuery();
      const n = model.selectedAtoms(sel).length; if (!n) return toast('Selection matched no atoms', 'error');
      state.layers.push({ name: describe(sel), sel, style: $('#q-style').value, color: $('#q-color').value, label: $('#q-label').checked });
      renderLayers(); applyStyles(); if ($('#q-style').value === 'surface') applySurfaces();
      toast(`Layer added (${n} atoms)`, 'ok', 1800);
    });
    $('#q-zoom').addEventListener('click', () => needModel() && viewer.zoomTo(buildQuery(), 600));
    $('#q-isolate').addEventListener('click', () => {
      if (!needModel()) return;
      const sel = buildQuery();
      state.rep = 'none'; syncControls();
      state.layers.push({ name: describe(sel) + ' (isolated)', sel, style: 'cartoon', color: '#60a5fa', label: false });
      state.layers.push({ name: describe(sel), sel, style: 'stick', color: $('#q-color').value, label: false });
      renderLayers(); applyStyles(); viewer.zoomTo(sel, 600);
    });
    $('#layers-clear').addEventListener('click', () => { state.layers = []; renderLayers(); applyStyles(); applySurfaces(); });
    $('#site-go').addEventListener('click', () => {
      if (!needModel()) return;
      const key = $('#site-lig').value; if (!key) return toast('This structure has no ligands', 'error');
      const [resn, chain, resi] = key.split(':'); const r = +$('#site-r').value;
      const ligSel = { resn, chain, resi: +resi };
      const sel = { byres: true, within: { distance: r, sel: ligSel }, hetflag: false };
      state.layers.push({ name: `${resn} pocket (${r} Å)`, sel, style: 'stick', color: '#e2e8f0', label: true });
      renderLayers(); applyStyles(); viewer.zoomTo(sel, 700);
      const n = new Set(model.selectedAtoms(sel).map((a) => a.chain + a.resi)).size;
      toast(`${n} residues within ${r} Å of ${resn}`, 'ok');
    });

    $('#reset').addEventListener('click', () => { viewer.zoomTo(); viewer.render(); });
    $('#shot').addEventListener('click', () => { const a = document.createElement('a'); a.href = viewer.pngURI(); a.download = (struct ? struct.label : 'structure') + '.png'; a.click(); });
    $('#full').addEventListener('click', () => { const s = $('#stage'); (document.fullscreenElement ? document.exitFullscreen() : s.requestFullscreen && s.requestFullscreen()); setTimeout(() => viewer.resize(), 200); });
    document.addEventListener('fullscreenchange', () => setTimeout(() => viewer.resize(), 100));
    $('#save').addEventListener('click', () => {
      if (!needModel()) return;
      const text = struct.pdbId ? null : struct.text;
      if (text && text.length > 5.5e6) return toast('File too large to save. Structures fetched by PDB ID can always be saved.', 'error');
      const { selected, ...st } = state;
      saveItem('structure', struct.label + ' session', { pdb_id: struct.pdbId, text, format: struct.format, label: struct.label, state: JSON.parse(JSON.stringify(st, (k, v) => (typeof v === 'function' ? undefined : v))), view: viewer.getView() }, `${struct.label} · ${state.rep} · ${state.layers.length} layers`);
    });
    addEventListener('resize', () => viewer.resize());

    (async () => {
      const id = SB.param('item');
      if (id) {
        try {
          const it = await SB.loadItem(id); const d = it.data;
          if (d.pdb_id) {
            const r = await fetch(`https://files.rcsb.org/download/${d.pdb_id}.${d.format === 'cif' ? 'cif' : 'pdb'}`);
            await load(await r.text(), d.format || 'pdb', d.label || d.pdb_id, d.pdb_id, d.state);
          } else await load(d.text, d.format, d.label, null, d.state);
          if (d.view) { viewer.setView(d.view); viewer.render(); }
          toast('Restored “' + it.name + '”', 'ok');
        } catch (e) { toast(e.message, 'error'); }
        return;
      }
      const pid = SB.param('pdb'); if (pid) { $('#pdb-id').value = pid; fetchPDB(pid); }
    })();
  }
  init();
})();
