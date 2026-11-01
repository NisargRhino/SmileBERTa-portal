/* Drug classification page. */
(function () {
  const { $, $$, api, esc, icon, fmt, toast, depict, saveItem, handoff, copy, download, show3DFromSmiles, propertyTiles, druglikeBadges } = SB;

  const CLASSES = {
    antiinfective: ['Anti-infective', '#34d399', 'Antibacterials, antivirals, antifungals and antiparasitics'],
    antineoplastic: ['Antineoplastic', '#f472b6', 'Oncology agents that inhibit tumour growth'],
    cns: ['Central nervous system', '#a78bfa', 'Analgesics, anaesthetics, antidepressants, antipsychotics'],
    cardio: ['Cardiovascular', '#f87171', 'Antihypertensives, antiarrhythmics, vasodilators'],
    antiinflammatory: ['Anti-inflammatory', '#fbbf24', 'NSAIDs, corticosteroids and immunomodulators'],
    hematologic: ['Hematologic', '#fb7185', 'Anticoagulants, antiplatelets and blood modifiers'],
    gastrointestinal: ['Gastrointestinal', '#fdba74', 'Antacids, antiemetics, motility agents'],
    lipidregulating: ['Lipid-regulating', '#facc15', 'Statins, fibrates and cholesterol modulators'],
    respiratorysystem: ['Respiratory system', '#38bdf8', 'Bronchodilators, antitussives, antihistamines'],
    reproductivecontrol: ['Reproductive control', '#e879f9', 'Hormonal contraceptives and fertility agents'],
    dermatologic: ['Dermatologic', '#2dd4bf', 'Topical and systemic skin therapies'],
    urological: ['Urological', '#60a5fa', 'Bladder, prostate and renal agents'],
  };
  const label = (k) => (CLASSES[k] ? CLASSES[k][0] : k);

  $('#cat-list').innerHTML = Object.entries(CLASSES).map(([k, [n, c, d]]) => `<div style="display:flex;gap:10px;padding:6px 0;border-bottom:1px solid var(--border)"><span style="width:8px;height:8px;border-radius:50%;background:${c};margin-top:7px;flex-shrink:0"></span><div><b style="color:var(--text)">${n}</b><div class="muted" style="font-size:.76rem">${d}</div></div></div>`).join('');

  const EXAMPLES = [
    ['Aspirin', 'CC(=O)Oc1ccccc1C(=O)O'], ['Imatinib', 'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1'],
    ['Ciprofloxacin', 'OC(=O)C1=CN(C2CC2)c2cc(N3CCNCC3)c(F)cc2C1=O'], ['Fluoxetine', 'CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1'],
    ['Atorvastatin', 'CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O'], ['Propranolol', 'CC(C)NCC(O)COc1cccc2ccccc12'],
  ];
  $('#examples').innerHTML = EXAMPLES.map(([n, s]) => `<button type="button" class="chip" data-smiles="${esc(s)}">${n}</button>`).join('');
  $$('#examples .chip').forEach((c) => c.addEventListener('click', () => { $('#smiles').value = c.dataset.smiles; preview(); }));

  const preview = SB.debounce(async () => {
    const s = $('#smiles').value.trim();
    if (!s) { $('#preview').innerHTML = '<span class="muted" style="font-size:.84rem">Structure preview</span>'; return; }
    if (await SB.validSmiles(s)) depict($('#preview'), s, { width: 320, height: 200 });
    else $('#preview').innerHTML = '<span style="color:var(--red);font-size:.84rem">Invalid SMILES</span>';
  }, 350);
  $('#smiles').addEventListener('input', preview);

  $$('[data-tab]').forEach((b) => b.addEventListener('click', () => {
    $$('[data-tab]').forEach((x) => x.classList.toggle('active', x === b));
    $('#single-form').classList.toggle('hidden', b.dataset.tab !== 'single');
    $('#batch-pane').classList.toggle('hidden', b.dataset.tab !== 'batch');
  }));

  function render(d, smiles) {
    const parts = d.prediction.split(/\s+/).filter(Boolean);
    const p = d.analysis || {};
    $('#results').innerHTML = `<div class="stack fade-in">
      <div class="card">
        <div class="row-between" style="margin-bottom:18px"><h3 style="margin:0">Prediction</h3>
          <div class="row" style="gap:6px"><button class="btn btn-sm" id="save">${icon('save')} Save</button><button class="btn btn-sm" id="dock">${icon('target')} Dock</button><button class="btn btn-sm" id="frag">${icon('split')} Fragment</button><button class="btn btn-sm" id="json">${icon('download')}</button></div></div>
        <div class="class-hero">
          <div class="icon-tile" style="width:56px;height:56px">${icon('tag')}</div>
          <div class="grow"><div class="muted" style="font-size:.76rem;text-transform:uppercase;letter-spacing:.12em">Predicted therapeutic class${parts.length > 1 ? 'es' : ''}</div>
            <div class="big grad-text">${parts.map(label).map(esc).join(' · ')}</div></div>
        </div>
        <div class="two-col" style="margin-top:16px">${parts.map((k) => `<div class="class-card"><span class="dot" style="background:${(CLASSES[k] || [])[1] || '#888'};color:${(CLASSES[k] || [])[1] || '#888'}"></span><div><b>${esc(label(k))}</b><div class="muted" style="font-size:.82rem">${esc((CLASSES[k] || [])[2] || '')}</div></div></div>`).join('')}</div>
      </div>
      <div class="two-col">
        <div class="card">
          <h3>Closest label matches</h3>
          <p class="muted" style="font-size:.84rem">The model writes its answer as text. That text is matched to the known labels by edit distance; the closest match is the prediction.</p>
          <div id="bars" style="margin-top:14px">${d.candidates.map((c, i) => `<div class="bar-row"><span style="color:${i ? 'var(--text-2)' : '#fff'};font-weight:${i ? 400 : 600}">${esc(c.name.split(' ').map(label).join(' + '))}</span><div class="bar"><div data-w="${Math.max(4, c.similarity * 100)}"></div></div><span class="num mono" style="font-size:.76rem;color:var(--muted)">${fmt(c.similarity, 2)}</span></div>`).join('')}</div>
          <div class="smiles-box" style="margin-top:14px"><span><span class="muted">raw model output → </span>${esc(d.raw || '∅')}</span></div>
        </div>
        <div class="card">
          <div class="row-between" style="margin-bottom:10px"><h3 style="margin:0">Structure</h3><div class="seg" id="view-seg"><button class="active" data-v="2d">2D</button><button data-v="3d">3D</button></div></div>
          <div class="mol-2d" id="mol-2d" style="min-height:260px"></div><div class="viewer3d hidden" id="mol-3d" style="height:260px"></div>
          <div class="smiles-box" style="margin-top:10px"><span>${esc(smiles)}</span><button class="btn btn-xs btn-ghost" id="copy">${icon('copy')}</button></div>
        </div>
      </div>
      <div class="card"><div class="row-between" style="margin-bottom:14px"><h3 style="margin:0">Drug-likeness profile</h3>${druglikeBadges(p)}</div>${propertyTiles(p)}</div>
    </div>`;
    requestAnimationFrame(() => $$('#bars [data-w]').forEach((b) => (b.style.width = b.dataset.w + '%')));
    depict($('#mol-2d'), smiles, { width: 420, height: 260 });
    $('#copy').addEventListener('click', () => copy(smiles));
    $('#view-seg').addEventListener('click', async (e) => {
      const b = e.target.closest('button'); if (!b) return;
      $$('#view-seg button').forEach((x) => x.classList.toggle('active', x === b));
      const is3d = b.dataset.v === '3d';
      $('#mol-2d').classList.toggle('hidden', is3d); $('#mol-3d').classList.toggle('hidden', !is3d);
      if (is3d && !$('#mol-3d').dataset.loaded) { $('#mol-3d').dataset.loaded = 1; try { (await show3DFromSmiles($('#mol-3d'), smiles)).viewer.spin('y', 0.5); } catch (err) { toast(err.message, 'error'); } }
    });
    $('#save').addEventListener('click', () => saveItem('classification', (p.formula || 'Compound') + ' · ' + parts.map(label).join(', '), { smiles, result: d }, parts.map(label).join(', ')));
    $('#dock').addEventListener('click', () => handoff('docking', { smiles }));
    $('#frag').addEventListener('click', () => handoff('fragmentation', { smiles }));
    $('#json').addEventListener('click', () => download(JSON.stringify({ smiles, ...d }, null, 2), 'classification.json', 'application/json'));
  }

  $('#single-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const smiles = $('#smiles').value.trim();
    if (!smiles) return toast('Enter a SMILES string', 'error');
    const btn = $('#go'); btn.classList.add('loading'); btn.innerHTML = '<span class="spinner"></span> Classifying…';
    $('#results').innerHTML = '<div class="card stack"><div class="row"><span class="spinner"></span><b>Running classifier…</b></div><div class="skeleton" style="height:120px"></div><div class="two-col"><div class="skeleton" style="height:240px"></div><div class="skeleton" style="height:240px"></div></div></div>';
    try { render(await api('/classify_smiles', { body: { smiles }, timeout: 150000 }), smiles); }
    catch (err) { $('#results').innerHTML = `<div class="card empty" style="min-height:300px"><div>${icon('alert')}<h3 style="color:var(--text)">Classification failed</h3><p>${esc(err.message)}</p></div></div>`; }
    finally { btn.classList.remove('loading'); btn.textContent = 'Classify compound'; }
  });

  let batch = [];
  SB.bindDropzone($('#dropzone'), async (f) => {
    batch = SB.parseCSV(await SB.readFile(f)).slice(0, 100);
    $('#batch-file').innerHTML = `<b>${esc(f.name)}</b> · ${batch.length} compounds`;
    $('#batch-run').disabled = !batch.length;
  }, '.csv,.txt,.smi');
  $('#batch-run').addEventListener('click', async () => {
    const btn = $('#batch-run'); btn.classList.add('loading'); btn.innerHTML = '<span class="spinner"></span> Classifying…';
    try {
      const { results } = await api('/batch_classify', { body: { smiles_list: batch }, timeout: 300000 });
      const counts = {};
      results.forEach((r) => r.prediction && r.prediction.split(' ').forEach((k) => (counts[k] = (counts[k] || 0) + 1)));
      const max = Math.max(1, ...Object.values(counts));
      $('#results').innerHTML = `<div class="stack fade-in"><div class="card"><div class="row-between" style="margin-bottom:12px"><h3 style="margin:0">Class distribution</h3><button class="btn btn-sm" id="dl">${icon('download')} CSV</button></div>
        ${Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, n]) => `<div class="bar-row"><span>${esc(label(k))}</span><div class="bar"><div data-w="${n / max * 100}" style="background:${(CLASSES[k] || [])[1] || 'var(--cyan)'}"></div></div><span class="num mono">${n}</span></div>`).join('')}</div>
        <div class="table-wrap"><table class="table"><thead><tr><th>#</th><th>SMILES</th><th>Prediction</th></tr></thead><tbody>${results.map((r, i) => `<tr><td>${i + 1}</td><td class="mono">${esc(r.smiles)}</td><td>${r.prediction ? r.prediction.split(' ').map((k) => `<span class="badge" style="margin:2px">${esc(label(k))}</span>`).join('') : `<span class="badge red">${esc(r.error)}</span>`}</td></tr>`).join('')}</tbody></table></div></div>`;
      requestAnimationFrame(() => $$('#results [data-w]').forEach((b) => (b.style.width = b.dataset.w + '%')));
      $('#dl').addEventListener('click', () => download(SB.toCSV([['smiles', 'prediction', 'error'], ...results.map((r) => [r.smiles, r.prediction, r.error])]), 'classifications.csv', 'text/csv'));
    } catch (err) { toast(err.message, 'error'); }
    finally { btn.classList.remove('loading'); btn.textContent = 'Classify all'; }
  });

  (async () => {
    const id = SB.param('item');
    if (id) { try { const it = await SB.loadItem(id); if (it.data) { $('#smiles').value = it.data.smiles; preview(); render(it.data.result, it.data.smiles); } } catch (e) { toast(e.message, 'error'); } }
    const ho = SB.takeHandoff('drug_class'); if (ho && ho.smiles) { $('#smiles').value = ho.smiles; preview(); }
  })();
})();
