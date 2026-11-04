/* Docking page — AutoDock Vina via the SmileBERTa API. */
(function () {
  const { $, $$, api, esc, icon, fmt, toast, depict, saveItem, download } = SB;

  const RECEPTORS = [
    ['1HSG', 'HIV-1 protease'], ['6LU7', 'SARS-CoV-2 Mpro'], ['1M17', 'EGFR kinase'], ['4EY7', 'Acetylcholinesterase'], ['3PTB', 'Trypsin'],
  ];
  const LIGANDS = [
    ['Indinavir', 'CC(C)(C)NC(=O)[C@@H]1CN(Cc2cccnc2)CCN1C[C@@H](O)C[C@@H](Cc1ccccc1)C(=O)N[C@@H]1c2ccccc2C[C@@H]1O'],
    ['Erlotinib', 'COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC'],
    ['Donepezil', 'COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2'],
    ['Benzamidine', 'NC(=N)c1ccccc1'],
    ['Ibuprofen', 'CC(C)Cc1ccc(C(C)C(=O)O)cc1'],
  ];
  const SKIP_HET = new Set(['HOH', 'WAT', 'DOD', 'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'ACT', 'DMS', 'CL', 'NA', 'K', 'MG', 'CA', 'ZN', 'MN', 'NI', 'CD', 'IOD', 'BR', 'FMT', 'MPD', 'TRS', 'BME', 'NO3', 'SCN', 'EPE', 'MES', 'IMD', 'PG4', '1PE', 'P6G', 'CIT', 'TLA', 'HEZ']);

  let viewer = null, receptor = null, recModel = null, poseModel = null;
  let ligMolblock = null, job = null, result = null, currentPose = 0, picking = false;

  // ── PDB parsing ──────────────────────────────────────────────────────────
  function parsePDB(text) {
    const chains = new Map(), hets = new Map(); let title = '', n = 0;
    const atoms = [];
    for (const line of text.split(/\r?\n/)) {
      if (line.startsWith('TITLE')) title += line.slice(10).trim() + ' ';
      if (line.startsWith('ENDMDL')) break;
      const rec = line.slice(0, 6);
      if (rec !== 'ATOM  ' && rec !== 'HETATM') continue;
      const resn = line.slice(17, 20).trim(), ch = line.slice(21, 22), resi = line.slice(22, 26).trim();
      const p = [parseFloat(line.slice(30, 38)), parseFloat(line.slice(38, 46)), parseFloat(line.slice(46, 54))];
      n++;
      if (rec === 'ATOM  ') { const c = chains.get(ch) || { id: ch, residues: new Set(), atoms: 0 }; c.residues.add(resi); c.atoms++; chains.set(ch, c); atoms.push(p); }
      else if (!SKIP_HET.has(resn)) { const k = `${resn} ${ch}${resi}`; const h = hets.get(k) || { key: k, resn, chain: ch, resi, pts: [] }; h.pts.push(p); hets.set(k, h); }
    }
    hets.forEach((h) => {
      const c = [0, 1, 2].map((k) => h.pts.reduce((s, p) => s + p[k], 0) / h.pts.length);
      const ext = [0, 1, 2].map((k) => Math.max(...h.pts.map((p) => p[k])) - Math.min(...h.pts.map((p) => p[k])));
      h.center = c; h.size = ext.map((e) => Math.min(40, Math.max(18, Math.ceil(e + 10))));
    });
    const center = atoms.length ? [0, 1, 2].map((k) => atoms.reduce((s, p) => s + p[k], 0) / atoms.length) : [0, 0, 0];
    return { title: title.trim(), atoms: n, chains: [...chains.values()].map((c) => ({ id: c.id, residues: c.residues.size, atoms: c.atoms })), ligands: [...hets.values()].filter((h) => h.pts.length >= 6), center };
  }

  // ── Viewer ───────────────────────────────────────────────────────────────
  function selectedChains() { return $$('[data-chain]:checked').map((c) => c.dataset.chain); }
  function styleReceptor() {
    if (!recModel) return;
    const st = $('#rec-style .active').dataset.s;
    viewer.removeAllSurfaces();
    recModel.setStyle({}, {});
    const chains = selectedChains();
    const sel = { chain: chains, hetflag: false };
    // Fade the ribbon once a pose is shown so it never hides the ligand.
    if (st === 'cartoon') recModel.setStyle(sel, { cartoon: { colorscheme: 'chain', opacity: poseModel ? 0.45 : 0.92, thickness: 0.4 } });
    if (st === 'lines') recModel.setStyle(sel, { line: { colorscheme: 'Jmol' } });
    if (st === 'surface') { recModel.setStyle(sel, { cartoon: { color: '#334155', opacity: 0.4 } }); viewer.addSurface($3Dmol.SurfaceType.MS, { opacity: 0.75, colorscheme: { prop: 'b', gradient: 'rwb' } }, { model: recModel, chain: chains, hetflag: false }); }
    if ($('#show-het').checked) recModel.setStyle({ hetflag: true, not: { resn: [...SKIP_HET] } }, { stick: { radius: 0.18, colorscheme: 'grayCarbon' } });
    if (poseModel && $('#show-site').checked && result) {
      const res = (result.poses[currentPose].interactions.residues || []).map((r) => r.residue.split(' ')[1]);
      res.forEach((cr) => recModel.setStyle({ chain: cr[0], resi: cr.slice(1), hetflag: false }, { stick: { radius: 0.14, colorscheme: 'whiteCarbon' }, cartoon: st === 'cartoon' ? { colorscheme: 'chain', opacity: 0.45 } : undefined }));
    }
    drawOverlays();
  }
  function getBox() { return { c: ['cx', 'cy', 'cz'].map((k) => +$('#' + k).value), s: ['sx', 'sy', 'sz'].map((k) => +$('#' + k).value) }; }
  function setBox(c, s) { ['cx', 'cy', 'cz'].forEach((k, i) => ($('#' + k).value = (+c[i]).toFixed(2))); if (s) ['sx', 'sy', 'sz'].forEach((k, i) => ($('#' + k).value = Math.round(s[i]))); drawOverlays(); markSteps(); }
  function drawOverlays() {
    if (!viewer) return;
    viewer.removeAllShapes(); viewer.removeAllLabels();
    if (receptor && $('#show-box').checked) {
      const { c, s } = getBox();
      viewer.addBox({ center: { x: c[0], y: c[1], z: c[2] }, dimensions: { w: s[0], h: s[1], d: s[2] }, wireframe: true, color: '#22d3ee' });
      viewer.addBox({ center: { x: c[0], y: c[1], z: c[2] }, dimensions: { w: s[0], h: s[1], d: s[2] }, color: '#22d3ee', opacity: 0.05 });
    }
    if (result && poseModel) {
      const p = result.poses[currentPose];
      (p.interactions.hbonds || []).forEach((hb) => {
        const a = { x: hb.p[0], y: hb.p[1], z: hb.p[2] }, b = { x: hb.l[0], y: hb.l[1], z: hb.l[2] };
        viewer.addCylinder({ start: a, end: b, radius: 0.06, color: '#fbbf24', dashed: true, fromCap: 1, toCap: 1 });
        viewer.addLabel(hb.distance.toFixed(2) + ' Å', { position: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 }, fontSize: 10, fontColor: '#fde68a', backgroundColor: '#1a1405', backgroundOpacity: 0.8, inFront: true });
      });
      if ($('#show-site').checked) (p.interactions.residues || []).slice(0, 14).forEach((r) => {
        const [resn, cr] = r.residue.split(' ');
        const at = recModel.selectedAtoms({ chain: cr[0], resi: cr.slice(1), atom: 'CA' })[0];
        if (at) viewer.addLabel(resn + cr.slice(1), { position: at, fontSize: 10, fontColor: '#e2e8f0', backgroundColor: '#0b1020', backgroundOpacity: 0.7, inFront: true });
      });
    }
    viewer.render();
  }

  async function loadReceptor(text, label, pdbId = null) {
    receptor = { text, label, pdbId, info: parsePDB(text) };
    if (!receptor.info.chains.length) { receptor = null; return toast('No protein chains found in this file', 'error'); }
    viewer.removeAllModels(); poseModel = null; result = null;
    recModel = viewer.addModel(text, 'pdb');
    $('#stage-empty').classList.add('hidden');
    const info = receptor.info;
    $('#rec-info').innerHTML = `<div class="hud-card" style="padding:12px"><div class="row-between"><b>${esc(label)}</b><span class="badge">${info.atoms.toLocaleString()} atoms</span></div>
      ${info.title ? `<div class="muted" style="font-size:.76rem;margin:6px 0 10px;line-height:1.4">${esc(info.title.slice(0, 140))}</div>` : ''}
      <div class="label" style="margin-top:8px">Chains to keep</div><div class="row" style="gap:10px">${info.chains.map((c) => `<label class="check"><input type="checkbox" data-chain="${esc(c.id)}" checked> ${esc(c.id || '_')} <span class="muted" style="font-size:.72rem">(${c.residues})</span></label>`).join('')}</div>
      <div class="label" style="margin-top:10px">Co-crystallised ligands</div>${info.ligands.length ? info.ligands.map((h, i) => `<div class="lig-row"><span class="mono grow">${esc(h.key)}</span><button class="btn btn-xs" data-lig="${i}">Box here</button></div>`).join('') : '<div class="muted" style="font-size:.78rem">None found. Use the protein center or pick a point.</div>'}</div>`;
    $$('[data-chain]').forEach((c) => c.addEventListener('change', styleReceptor));
    $$('[data-lig]').forEach((b) => b.addEventListener('click', () => { const h = info.ligands[+b.dataset.lig]; setBox(h.center, h.size); viewer.zoomTo({ within: { distance: 8, sel: { resn: h.resn, chain: h.chain } } }, 600); }));
    if (info.ligands.length) setBox(info.ligands[0].center, info.ligands[0].size); else setBox(info.center, [26, 26, 26]);
    styleReceptor();
    viewer.zoomTo(); viewer.render();
    $('#hud').classList.remove('hidden'); $('#hud').textContent = `${label} · ${info.chains.length} chain(s) · ${info.ligands.length} ligand(s)`;
    resetResults();
    markSteps();
  }

  async function fetchPDB(id) {
    id = (id || '').trim().toUpperCase();
    if (!/^[0-9][A-Z0-9]{3}$/.test(id)) return toast('PDB IDs are 4 characters, e.g. 1HSG', 'error');
    const btn = $('#fetch'); btn.classList.add('loading');
    try {
      const r = await fetch(`https://files.rcsb.org/download/${id}.pdb`);
      if (!r.ok) throw new Error(`Structure ${id} not found (or only available as mmCIF)`);
      await loadReceptor(await r.text(), id, id);
      toast(`Loaded ${id} from RCSB`, 'ok');
    } catch (e) { toast(e.message, 'error'); } finally { btn.classList.remove('loading'); }
  }

  // ── Ligand ───────────────────────────────────────────────────────────────
  const previewLig = SB.debounce(async () => {
    const s = $('#lig-smiles').value.trim();
    if (!s) { $('#lig-preview').innerHTML = '<span class="muted" style="font-size:.8rem">Ligand preview</span>'; markSteps(); return; }
    if (await SB.validSmiles(s)) depict($('#lig-preview'), s, { width: 280, height: 150 }); else $('#lig-preview').innerHTML = '<span style="color:var(--red);font-size:.8rem">Invalid SMILES</span>';
    markSteps();
  }, 350);

  function markSteps() {
    $('#st-1').classList.toggle('done', !!receptor);
    $('#st-2').classList.toggle('done', !!($('#lig-smiles').value.trim() || ligMolblock));
    $('#st-3').classList.toggle('done', !!receptor);
  }

  // ── Run & poll ───────────────────────────────────────────────────────────
  function resetResults() { result = null; $('#results-panel').classList.add('hidden'); $('#contacts-panel').classList.add('hidden'); }

  async function run() {
    if (!receptor) return toast('Load a receptor first', 'error');
    const smiles = $('#lig-smiles').value.trim();
    if (!smiles && !ligMolblock) return toast('Provide a ligand', 'error');
    const chains = selectedChains(); if (!chains.length) return toast('Select at least one chain', 'error');
    const { c, s } = getBox();
    if (s.some((v) => v < 5 || v > 40)) return toast('Box edges must be between 5 and 40 Å', 'error');
    const btn = $('#run'); btn.classList.add('loading'); btn.innerHTML = '<span class="spinner"></span> Docking…';
    resetResults();
    $('#run-progress').classList.remove('hidden'); $('#run-progress > div').style.width = '2%';
    $('#run-log').classList.remove('hidden'); $('#run-log').textContent = '';
    $('#run-status').innerHTML = 'Submitting job…';
    try {
      const body = { receptor_pdb: receptor.text, chains, center: c, size: s, exhaustiveness: +$('#exh').value, num_modes: +$('#modes').value };
      if (ligMolblock) body.ligand_molblock = ligMolblock; else body.ligand_smiles = smiles;
      const { job_id } = await api('/dock', { body, timeout: 120000, base: SB.DOCK_API, auth: false });
      job = job_id;
      const t0 = Date.now();
      while (true) {
        await SB.sleep(2000);
        const j = await api('/dock/' + job_id, { timeout: 60000, base: SB.DOCK_API, auth: false });
        $('#run-progress > div').style.width = j.progress + '%';
        $('#run-status').innerHTML = `<b style="color:#fff">${esc(j.message)}</b> <span class="muted">· ${Math.round((Date.now() - t0) / 1000)}s</span>`;
        $('#run-log').innerHTML = j.log.map((l) => `<div class="${l.includes('Error') ? 'err' : l.includes('Done') ? 'ok' : ''}">${esc(l)}</div>`).join('');
        $('#run-log').scrollTop = 1e6;
        if (j.status === 'done') { showResults(j.result); toast('Docking complete', 'ok'); break; }
        if (j.status === 'error') { $('#run-status').innerHTML = `<span style="color:var(--red)">${esc(j.error)}</span>`; break; }
      }
    } catch (e) { $('#run-status').innerHTML = `<span style="color:var(--red)">${esc(e.message)}</span>`; }
    finally { btn.classList.remove('loading'); btn.innerHTML = icon('play') + ' Run docking'; $('#run-progress').classList.add('hidden'); }
  }

  function kiText(dg) {
    const ki = Math.exp(dg / 0.5925);
    if (ki < 1e-9) return fmt(ki * 1e12, 1) + ' pM';
    if (ki < 1e-6) return fmt(ki * 1e9, 1) + ' nM';
    if (ki < 1e-3) return fmt(ki * 1e6, 1) + ' µM';
    return fmt(ki * 1e3, 1) + ' mM';
  }

  function showResults(r) {
    result = r; delete result.receptor_pdb;
    const best = r.poses[0];
    $('#results-panel').classList.remove('hidden');
    $('#best').innerHTML = `<div class="muted" style="font-size:.72rem;text-transform:uppercase;letter-spacing:.12em">Best binding affinity</div><div class="v" style="margin:6px 0">${fmt(best.affinity, 1)} <span style="font-size:1rem;color:var(--muted)">kcal/mol</span></div><div class="muted" style="font-size:.8rem">≈ K<sub>i</sub> ${kiText(best.affinity)} at 298 K (rough estimate)</div>`;
    const min = Math.min(...r.poses.map((p) => p.affinity));
    $('#pose-body').innerHTML = r.poses.map((p, i) => `<tr class="pose-row ${i ? '' : 'selected'}" data-i="${i}"><td>${p.mode}</td><td class="num" style="color:#fff">${fmt(p.affinity, 1)}</td><td class="num">${fmt(p.rmsd_lb, 1)} / ${fmt(p.rmsd_ub, 1)}</td><td style="width:60px"><div class="aff-bar" style="width:${Math.max(8, p.affinity / min * 100)}%"></div></td></tr>`).join('');
    $$('.pose-row').forEach((tr) => tr.addEventListener('click', () => showPose(+tr.dataset.i)));
    showPose(0, true);
  }

  function showPose(i, zoom = false) {
    currentPose = i;
    $$('.pose-row').forEach((tr) => tr.classList.toggle('selected', +tr.dataset.i === i));
    if (poseModel) viewer.removeModel(poseModel);
    const p = result.poses[i];
    poseModel = p.sdf ? viewer.addModel(p.sdf, 'sdf') : viewer.addModel(p.pdbqt, 'pdbqt');
    poseModel.setStyle({}, { stick: { radius: 0.22, colorscheme: 'cyanCarbon' }, sphere: { scale: 0.22, colorscheme: 'cyanCarbon' } });
    poseModel.setStyle({ elem: 'H' }, {});
    styleReceptor();
    if (zoom) viewer.zoomTo({ within: { distance: 11, sel: { model: poseModel } } }, 700);
    viewer.render();
    const it = p.interactions;
    $('#contacts-panel').classList.remove('hidden');
    $('#pose-n').textContent = '#' + p.mode;
    $('#contacts').innerHTML = `<div class="stat-grid" style="grid-template-columns:repeat(3,1fr)"><div class="stat"><div class="v">${it.hbonds.length}</div><div class="k">H-bonds</div></div><div class="stat"><div class="v">${it.residues.length}</div><div class="k">Contact residues</div></div><div class="stat"><div class="v">${it.hydrophobic}</div><div class="k">C···C contacts</div></div></div>
      ${it.hbonds.length ? `<div class="label" style="margin-top:14px">Hydrogen bonds (≤ 3.5 Å)</div>${it.hbonds.map((h) => `<div class="lig-row"><span class="grow"><b>${esc(h.residue)}</b> <span class="muted">${esc(h.protein_atom)} ··· ${esc(h.ligand_atom)}</span></span><span class="mono" style="color:var(--amber)">${fmt(h.distance, 2)} Å</span></div>`).join('')}` : ''}
      <div class="label" style="margin-top:14px">Residues within 4 Å</div><div class="row" style="gap:5px">${it.residues.map((r) => `<span class="badge" title="min ${r.min_distance} Å">${esc(r.residue)}</span>`).join('') || '<span class="muted">none</span>'}</div>`;
  }

  function complexPDB() {
    const rec = receptor.text.split(/\r?\n/).filter((l) => l.startsWith('ATOM') && selectedChains().includes(l[21]));
    const AD = { A: 'C', NA: 'N', NS: 'N', OA: 'O', OS: 'O', SA: 'S', HD: 'H', HS: 'H' };
    const lig = result.poses[currentPose].pdbqt.split('\n').filter((l) => l.startsWith('ATOM') || l.startsWith('HETATM')).map((l) => {
      const t = l.slice(77, 79).trim(); const el = (AD[t] || t).toUpperCase();
      return 'HETATM' + l.slice(6, 17) + 'LIG L 999' + l.slice(26, 54) + '  1.00  0.00          ' + el.padStart(2);
    });
    return [...rec, 'TER', ...lig, 'END'].join('\n') + '\n';
  }

  // ── Init ─────────────────────────────────────────────────────────────────
  function init() {
    if (!window.$3Dmol) return setTimeout(init, 80);
    viewer = $3Dmol.createViewer($('#viewer'), { backgroundColor: '#070a12', antialias: true });
    // Wake the docking service early: free instances sleep when idle and take ~1 min to start.
    (async () => {
      const st = $('#run-status');
      st.innerHTML = '<span class="spinner" style="width:12px;height:12px"></span> Waking the docking engine…';
      for (let i = 0; i < 20; i++) {
        try { const r = await fetch(SB.DOCK_API + '/health', { cache: 'no-store' }); if (r.ok) { const h = await r.json(); st.innerHTML = h.docking ? '<span style="color:var(--green)">●</span> Docking engine ready (AutoDock Vina)' : '<span style="color:var(--red)">●</span> Docking engine unavailable'; return; } } catch (e) { /* retry */ }
        await SB.sleep(5000);
      }
      st.innerHTML = '<span style="color:var(--amber)">●</span> Docking engine is not responding yet. You can still try running.';
    })();
    $('#run').innerHTML = icon('play') + ' Run docking';
    $('#save').innerHTML = icon('save') + ' Save';
    $('#upload').innerHTML = icon('upload');
    $('#rec-examples').innerHTML = RECEPTORS.map(([id, n]) => `<button class="chip" data-id="${id}" title="${n}">${id} · ${n}</button>`).join('');
    $$('#rec-examples .chip').forEach((c) => c.addEventListener('click', () => { $('#pdb-id').value = c.dataset.id; fetchPDB(c.dataset.id); }));
    $('#lig-examples').innerHTML = LIGANDS.map(([n, s]) => `<button class="chip" data-s="${esc(s)}">${n}</button>`).join('');
    $$('#lig-examples .chip').forEach((c) => c.addEventListener('click', () => { $('#lig-smiles').value = c.dataset.s; ligMolblock = null; $('#lig-source').innerHTML = ''; previewLig(); }));
    $('#fetch').addEventListener('click', () => fetchPDB($('#pdb-id').value));
    $('#pdb-id').addEventListener('keydown', (e) => { if (e.key === 'Enter') fetchPDB(e.target.value); });
    $('#upload').addEventListener('click', () => {
      const inp = SB.h('<input type="file" accept=".pdb,.ent" hidden>'); document.body.appendChild(inp);
      inp.addEventListener('change', async () => { const f = inp.files[0]; inp.remove(); if (f) loadReceptor(await SB.readFile(f), f.name); });
      inp.click();
    });
    $('#lig-smiles').addEventListener('input', () => { ligMolblock = null; $('#lig-source').innerHTML = ''; previewLig(); });
    ['cx', 'cy', 'cz', 'sx', 'sy', 'sz'].forEach((k) => $('#' + k).addEventListener('input', SB.debounce(drawOverlays, 150)));
    $('#exh').addEventListener('input', (e) => ($('#exh-v').textContent = e.target.value));
    $('#modes').addEventListener('input', (e) => ($('#modes-v').textContent = e.target.value));
    $$('#rec-style button').forEach((b) => b.addEventListener('click', () => { $$('#rec-style button').forEach((x) => x.classList.toggle('active', x === b)); styleReceptor(); viewer.render(); }));
    ['#show-box', '#show-het', '#show-site'].forEach((s) => $(s).addEventListener('change', () => { styleReceptor(); viewer.render(); }));
    $('#fit-box').addEventListener('click', () => receptor && setBox(receptor.info.center, [30, 30, 30]));
    $('#pick-center').addEventListener('click', () => {
      if (!receptor) return toast('Load a receptor first', 'error');
      picking = true; $('#pick-hud').classList.remove('hidden');
      viewer.setClickable({}, true, (atom) => { if (!picking) return; picking = false; $('#pick-hud').classList.add('hidden'); setBox([atom.x, atom.y, atom.z]); toast(`Box centred on ${atom.resn} ${atom.chain}${atom.resi}`, 'ok'); });
      viewer.render();
    });
    $('#pick-cancel').addEventListener('click', (e) => { e.preventDefault(); picking = false; $('#pick-hud').classList.add('hidden'); });
    $('#run').addEventListener('click', run);
    $('#reset-view').addEventListener('click', () => { viewer.zoomTo(); viewer.render(); });
    $('#shot').addEventListener('click', () => { const a = document.createElement('a'); a.href = viewer.pngURI(); a.download = 'docking.png'; a.click(); });
    const dd = $('#results-panel [data-drop]');
    dd.querySelector('button').addEventListener('click', (e) => { e.stopPropagation(); dd.classList.toggle('open'); });
    $$('[data-dl]').forEach((a) => a.addEventListener('click', (e) => {
      e.preventDefault(); dd.classList.remove('open'); if (!result) return;
      const k = a.dataset.dl;
      if (k === 'sdf') download(result.poses.map((p) => p.sdf || '').join(''), 'poses.sdf', 'chemical/x-mdl-sdfile');
      if (k === 'pdbqt') download(result.poses.map((p, i) => `MODEL ${i + 1}\n${p.pdbqt}ENDMDL\n`).join(''), 'poses.pdbqt');
      if (k === 'complex') download(complexPDB(), `complex_pose${currentPose + 1}.pdb`, 'chemical/x-pdb');
      if (k === 'csv') download(SB.toCSV([['mode', 'affinity_kcal_mol', 'rmsd_lb', 'rmsd_ub', 'hbonds', 'contact_residues'], ...result.poses.map((p) => [p.mode, p.affinity, p.rmsd_lb, p.rmsd_ub, p.interactions.hbonds.length, p.interactions.residues.map((r) => r.residue).join(' ')])]), 'docking_scores.csv', 'text/csv');
    }));
    $('#save').addEventListener('click', () => {
      if (!result) return;
      const recText = receptor.pdbId ? null : (receptor.text.length < 5e6 ? receptor.text : null);
      if (!receptor.pdbId && !recText) toast('The uploaded receptor is too large to store; only the results will be saved', 'info');
      saveItem('docking', `${receptor.label} · ${fmt(result.poses[0].affinity, 1)} kcal/mol`,
        { pdb_id: receptor.pdbId, receptor_label: receptor.label, receptor_pdb: recText, chains: selectedChains(), result },
        `${receptor.label} · ${result.ligand_smiles} · best ${fmt(result.poses[0].affinity, 1)} kcal/mol`);
    });
    addEventListener('resize', () => viewer.resize());

    // Restore saved run or hand-off from another tool.
    (async () => {
      const id = SB.param('item');
      if (id) {
        try {
          const it = await SB.loadItem(id); const d = it.data;
          if (d.pdb_id) { $('#pdb-id').value = d.pdb_id; await fetchPDB(d.pdb_id); } else if (d.receptor_pdb) await loadReceptor(d.receptor_pdb, d.receptor_label || 'Saved receptor');
          if (receptor && d.chains) $$('[data-chain]').forEach((c) => (c.checked = d.chains.includes(c.dataset.chain)));
          $('#lig-smiles').value = d.result.ligand_smiles; previewLig();
          setBox(d.result.center, d.result.size);
          if (receptor) showResults(d.result);
        } catch (e) { toast(e.message, 'error'); }
        return;
      }
      const ho = SB.takeHandoff('docking');
      if (ho) {
        if (ho.smiles) $('#lig-smiles').value = ho.smiles;
        if (ho.molblock) { ligMolblock = ho.molblock; $('#lig-source').innerHTML = `<div class="notice info" style="margin-bottom:8px">Using 3D structure from ${esc(ho.name || 'another tool')}</div>`; }
        previewLig();
        toast('Ligand loaded. Now choose a receptor.', 'info');
      }
    })();
  }
  init();
})();
