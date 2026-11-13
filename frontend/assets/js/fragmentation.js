/* Fragment prediction page. */
(function () {
  const { $, $$, api, esc, icon, fmt, toast, depict, depictSVG, loadRDKit, library, saveItem, handoff, copy, download, show3DFromSmiles } = SB;

  const EXAMPLES = [
    ['Oxazolidinone', 'C=CCCC(=O)NCC1CN(c2ccc(N3CCOCC3)c(F)c2)C(=O)O1'],
    ['Thiazolyl oxazolidinone', 'CC(=O)NCC1CN(c2ccc(-c3cncs3)c(F)c2)C(=O)O1'],
    ['β-blocker', 'CC(C)(C)NCC(O)COc1ccc(O)c2c1CCC(=O)N2'],
    ['Peptidomimetic', 'CC(C)C(NC(=O)CCc1ccccc1)C(=O)N1CCCC1'],
    ['Imatinib', 'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1'],
  ];
  $('#examples').innerHTML = EXAMPLES.map(([n, s]) => `<button type="button" class="chip" data-smiles="${esc(s)}">${esc(n)}</button>`).join('');
  $$('#examples .chip').forEach((c) => c.addEventListener('click', () => { $('#smiles').value = c.dataset.smiles; $$('#examples .chip').forEach((x) => x.classList.toggle('active', x === c)); }));

  // Tabs
  $$('[data-tab]').forEach((b) => b.addEventListener('click', () => {
    $$('[data-tab]').forEach((x) => x.classList.toggle('active', x === b));
    $('#single-form').classList.toggle('hidden', b.dataset.tab !== 'single');
    $('#batch-pane').classList.toggle('hidden', b.dataset.tab !== 'batch');
  }));

  let current = null;

  async function highlightSVG(smiles, fragment) {
    try {
      const R = await loadRDKit();
      const m = R.get_mol(smiles), q = R.get_mol(fragment);
      let match = null;
      if (m && q) { const r = JSON.parse(m.get_substruct_match(q) || '{}'); if (r.atoms && r.atoms.length) match = r; }
      m && m.delete(); q && q.delete();
      if (match) return depictSVG(smiles, { width: 420, height: 300, highlight: match.atoms });
    } catch (e) { /* fall through */ }
    return depictSVG(smiles, { width: 420, height: 300 });
  }

  function ro3Row(label, key, inp, frag, limit, digits = 1) {
    const iv = inp ? inp[key] : null, fv = frag ? frag[key] : null;
    const pass = limit === null || fv === null ? null : fv <= limit;
    const delta = iv !== null && fv !== null ? fv - iv : null;
    return `<tr><td>${label}</td><td class="num">${fmt(iv, digits)}</td><td class="num" style="color:#fff">${fmt(fv, digits)}</td>
      <td class="num" style="color:${delta < 0 ? 'var(--green)' : 'var(--muted)'}">${delta === null ? '—' : (delta > 0 ? '+' : '') + fmt(delta, digits)}</td>
      <td class="num">${limit === null ? '—' : '≤ ' + limit}</td>
      <td>${pass === null ? '' : `<span class="badge ${pass ? 'green' : 'amber'}">${pass ? 'Pass' : 'Above'}</span>`}</td></tr>`;
  }

  function renderResult(d, inputSmiles) {
    current = { input: inputSmiles, ...d };
    const a = d.analysis || {}, ia = d.input_analysis || {};
    const ro3Pass = a.molecular_weight <= 300 && a.log_p <= 3 && a.hydrogen_bond_donors <= 3 && a.hydrogen_bond_acceptors <= 3;
    $('#results').innerHTML = `
      <div class="stack fade-in">
        <div class="card">
          <div class="row-between" style="margin-bottom:18px">
            <div class="row" style="gap:8px"><h3 style="margin:0">Prediction</h3>
              <span class="badge blue">Target · ${esc(d.protein || '—')}</span>
              <span class="badge ${ro3Pass ? 'green' : 'amber'}">${ro3Pass ? 'Rule of Three ✓' : 'Outside Rule of Three'}</span>
              ${d.corrected ? `<span class="badge violet" title="The raw model output was not a valid molecule and was repaired">Corrected output</span>` : '<span class="badge cyan">Direct model output</span>'}
            </div>
            <div class="row" style="gap:6px">
              <button class="btn btn-sm" id="add-lib">${icon('plus')} Library</button>
              <button class="btn btn-sm" id="save">${icon('save')} Save</button>
              <button class="btn btn-sm" id="to-dock">${icon('target')} Dock</button>
              <button class="btn btn-sm" id="to-build">${icon('atom')} Builder</button>
              <button class="btn btn-sm" id="to-admet">${icon('gauge')} ADMET</button>
            </div>
          </div>
          <div class="two-col">
            <div>
              <div class="row-between" style="margin-bottom:8px"><span class="panel-title" style="margin:0">Input molecule</span><div class="seg" data-view="in"><button class="active" data-v="2d">2D</button><button data-v="3d">3D</button></div></div>
              <div class="mol-2d" id="in-2d"></div><div class="viewer3d hidden" id="in-3d" style="height:300px"></div>
              <div class="smiles-box" style="margin-top:10px"><span>${esc(inputSmiles)}</span><button class="btn btn-xs btn-ghost" data-copy="${esc(inputSmiles)}">${icon('copy')}</button></div>
            </div>
            <div>
              <div class="row-between" style="margin-bottom:8px"><span class="panel-title" style="margin:0;color:var(--cyan)">Predicted fragment</span><div class="seg" data-view="fr"><button class="active" data-v="2d">2D</button><button data-v="3d">3D</button></div></div>
              <div class="mol-2d" id="fr-2d" style="border-color:rgba(34,211,238,.35)"></div><div class="viewer3d hidden" id="fr-3d" style="height:300px"></div>
              <div class="smiles-box" style="margin-top:10px"><span style="color:#a5f3fc">${esc(d.fragment_smiles)}</span><button class="btn btn-xs btn-ghost" data-copy="${esc(d.fragment_smiles)}">${icon('copy')}</button></div>
            </div>
          </div>
          <p class="kbd-hint" style="margin:12px 0 0">Highlighted atoms in the input show where the predicted fragment occurs in the parent molecule (if it is an exact substructure).</p>
        </div>

        <div class="card">
          <div class="row-between" style="margin-bottom:14px"><h3 style="margin:0">Property comparison</h3><span class="muted" style="font-size:.8rem">Rule of Three (Congreve et al., 2003)</span></div>
          <div class="table-wrap"><table class="table"><thead><tr><th>Property</th><th class="num">Input</th><th class="num">Fragment</th><th class="num">Δ</th><th class="num">Ro3 limit</th><th></th></tr></thead><tbody>
            ${ro3Row('Molecular weight (Da)', 'molecular_weight', ia, a, 300)}
            ${ro3Row('cLogP', 'log_p', ia, a, 3, 2)}
            ${ro3Row('H-bond donors', 'hydrogen_bond_donors', ia, a, 3, 0)}
            ${ro3Row('H-bond acceptors', 'hydrogen_bond_acceptors', ia, a, 3, 0)}
            ${ro3Row('Rotatable bonds', 'rotatable_bonds', ia, a, 3, 0)}
            ${ro3Row('TPSA (Å²)', 'tpsa', ia, a, 60)}
            ${ro3Row('Heavy atoms', 'heavy_atoms', ia, a, null, 0)}
            ${ro3Row('QED', 'qed', ia, a, null, 2)}
          </tbody></table></div>
        </div>

        <div class="card">
          <h3>Provenance</h3>
          <div class="kv">
            <div>Model</div><div>NisargRhino/protein-models</div>
            <div>Raw decoded output</div><div style="word-break:break-all;max-width:420px">${esc(d.raw_output || '—')}</div>
            <div>Validity</div><div>${d.corrected ? (d.correction === 'corrector-model' ? 'Repaired by SMILES corrector model' : 'Snapped to nearest reference fragment') : 'Valid as generated'}</div>
            <div>Fragment formula</div><div>${esc(a.formula || '—')}</div>
            <div>InChIKey</div><div>${esc(a.inchikey || '—')}</div>
          </div>
          <div class="row" style="margin-top:16px">
            <button class="btn btn-sm" id="dl-sdf">${icon('download')} Fragment PDB</button>
            <button class="btn btn-sm" id="dl-json">${icon('download')} Result JSON</button>
          </div>
        </div>
      </div>`;

    highlightSVG(inputSmiles, d.fragment_smiles).then((svg) => ($('#in-2d').innerHTML = svg));
    depict($('#fr-2d'), d.fragment_smiles, { width: 420, height: 300 });

    $$('[data-copy]').forEach((b) => b.addEventListener('click', () => copy(b.dataset.copy)));
    $$('[data-view]').forEach((seg) => seg.addEventListener('click', async (e) => {
      const btn = e.target.closest('button'); if (!btn) return;
      $$('button', seg).forEach((x) => x.classList.toggle('active', x === btn));
      const k = seg.dataset.view, is3d = btn.dataset.v === '3d';
      $(`#${k}-2d`).classList.toggle('hidden', is3d); $(`#${k}-3d`).classList.toggle('hidden', !is3d);
      const el = $(`#${k}-3d`);
      if (is3d && !el.dataset.loaded) {
        el.dataset.loaded = '1';
        try { const { viewer } = await show3DFromSmiles(el, k === 'in' ? inputSmiles : d.fragment_smiles); viewer.spin('y', 0.5); } catch (err) { el.innerHTML = `<div class="empty">${esc(err.message)}</div>`; }
      }
    }));
    $('#add-lib').addEventListener('click', async () => {
      try { await library.add(d.fragment_smiles, d.properties); toast('Added to fragment library', 'ok'); renderLibrary(); } catch (e) { toast(e.message, 'error'); }
    });
    $('#save').addEventListener('click', () => saveItem('fragment', 'Fragment of ' + (a.formula || 'molecule'), { input: inputSmiles, result: d }, `${d.fragment_smiles} · ${d.protein}`));
    $('#to-dock').addEventListener('click', () => handoff('docking', { smiles: d.fragment_smiles, name: 'Predicted fragment' }));
    $('#to-build').addEventListener('click', () => handoff('builder', { smiles: d.fragment_smiles }));
    $('#to-admet').addEventListener('click', () => handoff('admet', { smiles: d.fragment_smiles }));
    $('#dl-sdf').addEventListener('click', () => d.pdb ? download(d.pdb, 'fragment.pdb', 'chemical/x-pdb') : toast('No 3D structure available', 'error'));
    $('#dl-json').addEventListener('click', () => download(JSON.stringify({ input: inputSmiles, ...d }, null, 2), 'fragment_prediction.json', 'application/json'));
  }

  function loadingState() {
    $('#results').innerHTML = `<div class="card stack"><div class="row" style="gap:12px"><span class="spinner"></span><b>Running SmileBERTa…</b><span class="muted" id="load-step" style="font-size:.86rem">Tokenizing SMILES</span></div>
      <div class="two-col"><div class="skeleton" style="height:300px"></div><div class="skeleton" style="height:300px"></div></div><div class="skeleton" style="height:180px"></div></div>`;
    const steps = ['Tokenizing SMILES', 'Running 6 transformer layers', 'Decoding fragment tokens', 'Validating with RDKit', 'Embedding 3D conformer (ETKDG + MMFF94)', 'Still working (the model may be loading on a cold server)'];
    let i = 0;
    return setInterval(() => { const el = $('#load-step'); if (el && i < steps.length - 1) el.textContent = steps[++i]; }, 1600);
  }

  $('#single-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const smiles = $('#smiles').value.trim();
    if (!smiles) return toast('Enter a SMILES string', 'error');
    if (!(await SB.validSmiles(smiles))) return toast('That SMILES string is not valid', 'error');
    const btn = $('#predict-btn'); btn.classList.add('loading'); btn.innerHTML = '<span class="spinner"></span> Predicting…';
    const t = loadingState();
    try {
      const d = await api('/predict_fragment', { body: { smiles, protein: $('#protein').value }, timeout: 180000 });
      renderResult(d, smiles);
    } catch (err) {
      $('#results').innerHTML = `<div class="card empty" style="min-height:300px"><div>${icon('alert')}<h3 style="color:var(--text)">Prediction failed</h3><p>${esc(err.message)}</p></div></div>`;
    } finally { clearInterval(t); btn.classList.remove('loading'); btn.textContent = 'Predict fragment'; }
  });

  // ── Library ────────────────────────────────────────────────────────────────
  async function renderLibrary() {
    const el = $('#library');
    const lib = await library.list();
    if (!lib.length) { el.innerHTML = '<p class="muted" style="font-size:.86rem;margin:0">Fragments you add will appear here. Combine any two of them on the Fragment Linking page.</p>'; return; }
    el.innerHTML = lib.map((f, i) => `<div class="list-item"><div class="thumb" id="lib-${i}"></div><div class="grow"><div class="mono" style="font-size:.74rem;word-break:break-all;color:var(--text-2)">${esc(f.smiles)}</div></div><button class="btn btn-xs btn-ghost" data-rm="${esc(f.smiles)}" aria-label="Remove">${icon('trash')}</button></div>`).join('');
    lib.forEach((f, i) => depictSVG(f.smiles, { width: 140, height: 110 }).then((s) => { const t = $('#lib-' + i); if (t) t.innerHTML = s; }).catch(() => {}));
    $$('[data-rm]', el).forEach((b) => b.addEventListener('click', async () => { await library.remove(b.dataset.rm); renderLibrary(); }));
  }
  renderLibrary();

  // ── Batch ──────────────────────────────────────────────────────────────────
  let batchSmiles = [];
  SB.bindDropzone($('#dropzone'), async (file) => {
    batchSmiles = SB.parseCSV(await SB.readFile(file)).slice(0, 50);
    $('#batch-file').innerHTML = `<b>${esc(file.name)}</b> · ${batchSmiles.length} molecules`;
    $('#batch-run').disabled = !batchSmiles.length;
  }, '.csv,.txt,.smi');
  $('#batch-run').addEventListener('click', async () => {
    const prog = $('#batch-progress'); prog.classList.remove('hidden'); prog.firstElementChild.style.width = '0%';
    $('#batch-run').classList.add('loading');
    const rows = [];
    $('#results').innerHTML = `<div class="card"><div class="row-between" style="margin-bottom:14px"><h3 style="margin:0">Batch results</h3><button class="btn btn-sm" id="batch-dl" disabled>${icon('download')} CSV</button></div><div class="table-wrap"><table class="table"><thead><tr><th>#</th><th>Input</th><th>Fragment</th><th class="num">MW</th><th class="num">cLogP</th><th>Status</th></tr></thead><tbody id="batch-body"></tbody></table></div></div>`;
    const protein = $('#batch-protein').value;
    for (let i = 0; i < batchSmiles.length; i++) {
      const s = batchSmiles[i];
      let row;
      try {
        const d = await api('/predict_fragment', { body: { smiles: s, protein }, timeout: 180000 });
        row = { input: s, fragment: d.fragment_smiles, mw: d.properties.molecular_weight, logp: d.properties.log_p, status: d.corrected ? 'corrected' : 'ok' };
      } catch (e) { row = { input: s, fragment: '', mw: '', logp: '', status: e.message }; }
      rows.push(row);
      $('#batch-body').insertAdjacentHTML('beforeend', `<tr class="fade-in"><td>${i + 1}</td><td class="mono">${esc(row.input)}</td><td class="mono" style="color:#a5f3fc">${esc(row.fragment)}</td><td class="num">${row.mw}</td><td class="num">${row.logp}</td><td>${row.fragment ? `<span class="badge ${row.status === 'ok' ? 'green' : 'violet'}">${row.status}</span>` : `<span class="badge red" title="${esc(row.status)}">error</span>`}</td></tr>`);
      prog.firstElementChild.style.width = ((i + 1) / batchSmiles.length * 100) + '%';
    }
    $('#batch-run').classList.remove('loading');
    const dl = $('#batch-dl'); dl.disabled = false;
    dl.addEventListener('click', () => download(SB.toCSV([['input_smiles', 'fragment_smiles', 'molecular_weight', 'log_p', 'status'], ...rows.map((r) => [r.input, r.fragment, r.mw, r.logp, r.status])]), 'fragment_batch.csv', 'text/csv'));
    toast(`Batch complete: ${rows.filter((r) => r.fragment).length}/${rows.length} succeeded`, 'ok');
  });

  // ── Restore saved item / hand-off ──────────────────────────────────────────
  (async () => {
    const id = SB.param('item');
    if (id) {
      try { const it = await SB.loadItem(id); if (it.data && it.data.result) { $('#smiles').value = it.data.input; renderResult(it.data.result, it.data.input); } } catch (e) { toast(e.message, 'error'); }
    }
    const ho = SB.takeHandoff('fragmentation');
    if (ho && ho.smiles) $('#smiles').value = ho.smiles;
  })();
})();
