/* Pharmacophore Predictor: per-atom interactions, trained on crystal contacts.
 *
 * The page has one job beyond showing numbers: make it impossible to mistake
 * the pose-free answer for the site-specific one. They are different claims and
 * only one of them is about a particular protein. */
(function () {
  const { $, $$, api, esc, toast, download, depict, saveItem } = SB;

  const EXAMPLES = [
    ['Benzamidine in trypsin', 'NC(=N)c1ccccc1', '3PTB'],
    ['Donepezil in AChE', 'COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2', '4EY7'],
    ['Aspirin (no pocket)', 'CC(=O)Oc1ccccc1C(=O)O', ''],
    ['Imatinib (no pocket)', 'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1', ''],
  ];

  let result = null;

  $('#examples').innerHTML = EXAMPLES
    .map(([n, s, p], i) => `<button class="chip" data-i="${i}">${esc(n)}</button>`).join('');
  $$('#examples .chip').forEach((b) => b.addEventListener('click', () => {
    const [, smi, pdb] = EXAMPLES[+b.dataset.i];
    $('#smi').value = smi;
    $('#pdb').value = pdb;
    run();
  }));

  // ── Availability ──────────────────────────────────────────────────────────
  // The model is a large file and may not be on every host. Rather than let a
  // prediction fail with a 503 after the user has typed something, the page
  // asks first and says plainly when it is not there.
  async function checkStatus() {
    try {
      const s = await api('/pharmacophore/status', null, 'GET');
      if (!s.available) {
        $('#unavailable').hidden = false;
        $('#unavailable-msg').innerHTML =
          `<b>The pharmacophore model is not loaded on this server.</b>
           <div style="margin-top:7px">${esc(s.reason || '')}</div>`;
        return null;
      }
      $('#tool').hidden = false;
      return s;
    } catch (e) {
      $('#unavailable').hidden = false;
      $('#unavailable-msg').textContent = 'Could not reach the API.';
      return null;
    }
  }

  let status = null;
  checkStatus().then((s) => { status = s; });

  // ── Predict ───────────────────────────────────────────────────────────────
  async function run() {
    const smiles = $('#smi').value.trim();
    if (!smiles) { toast('Enter a SMILES string'); return; }
    const pdb = $('#pdb').value.trim();
    const btn = $('#run');
    btn.disabled = true; btn.textContent = 'Predicting…';
    try {
      result = await api('/pharmacophore', { smiles, pdb_id: pdb || undefined });
      render();
      $('#out').hidden = false;
    } catch (e) {
      toast(e.message || 'Prediction failed');
    } finally {
      btn.disabled = false; btn.textContent = 'Predict';
    }
  }
  $('#run').addEventListener('click', run);
  $('#smi').addEventListener('keydown', (e) => { if (e.key === 'Enter') run(); });
  $('#pdb').addEventListener('keydown', (e) => { if (e.key === 'Enter') run(); });

  // ── Render ────────────────────────────────────────────────────────────────
  function render() {
    const r = result;

    // Which claim is this? Stated before any number, in the server's own words.
    const claim = $('#claim');
    claim.classList.toggle('specific', !!r.used_pocket);
    claim.innerHTML = r.used_pocket
      ? `<b>Site-specific prediction</b>Using the crystallised pose, so these
         interactions are predicted for this binding site in particular.`
      : `<b>Ligand-only prediction</b>No pocket was supplied, so this says which
         atoms of this molecule tend to make interactions — not which atoms bind
         any particular protein. Add a PDB entry where this molecule is bound
         for a site-specific answer.`;

    $('#depict').innerHTML = r.svg || '';
    $('#summary').innerHTML = [
      ['Heavy atoms', r.n_heavy_atoms],
      ['Predicted binding atoms', r.pharmacophore_atoms.length],
      ['Pocket used', r.used_pocket ? 'yes' : 'no'],
    ].map(([k, v]) => `<div class="kv-row"><span>${esc(k)}</span><b>${esc(String(v))}</b></div>`).join('');

    // Provenance: the model's own recorded metrics, not copy written here.
    const m = r.model || {};
    const bits = [];
    if (m.trained_on_complexes) bits.push(`Trained on ${Number(m.trained_on_complexes).toLocaleString()} crystal complexes.`);
    if (m.test_macro_auc != null) bits.push(`Held-out macro AUC <code>${esc(String(m.test_macro_auc))}</code> across interaction types.`);
    if (m.ablation_delta != null) {
      bits.push(`Removing the pocket costs <code>${esc(String(m.ablation_delta))}</code> macro AUC against an identical control, which is how we check the protein is being used at all.`);
    }
    bits.push(`Interactions are typed from crystal geometry, so a high score means "atoms like this one, in sites like this one, were observed making this contact".`);
    bits.push(`Given a pose, much of this task is determined by geometry: hand-written rules using the same distances reach about 0.77–0.82 macro AUC with nothing fitted. The model is measured against that, not against zero.`);
    $('#provenance').innerHTML = bits.map((b) => `<div style="margin-bottom:7px">${b}</div>`).join('');

    const atoms = r.per_atom.slice().sort((a, b) => b.score - a.score);
    $('#count').textContent = `${r.pharmacophore_atoms.length} of ${r.n_heavy_atoms} atoms above threshold`;
    $('#atoms').innerHTML = atoms.map((a) => {
      const types = Object.entries(a.interactions)
        .sort((x, y) => y[1] - x[1])
        .map(([t, p]) => `<span class="type">${esc(t)} ${p.toFixed(2)}</span>`).join('');
      return `<div class="atom">
        <div class="atom-top">
          <span class="atom-id">atom ${a.atom} · ${esc(a.element)}</span>
          <span class="atom-score">${a.score.toFixed(3)}</span>
        </div>
        <div class="bar"><i style="width:${Math.round(a.score * 100)}%"></i></div>
        ${types ? `<div class="types">${types}</div>` : ''}
      </div>`;
    }).join('');
  }

  // ── Export ────────────────────────────────────────────────────────────────
  $('#csv').addEventListener('click', () => {
    if (!result) return;
    const rows = [['atom', 'element', 'score', 'interactions']];
    result.per_atom.forEach((a) => rows.push([
      a.atom, a.element, a.score,
      Object.entries(a.interactions).map(([t, p]) => `${t}:${p}`).join(' '),
    ]));
    download('pharmacophore.csv', rows.map((r) => r.join(',')).join('\n'));
  });

  $('#save').addEventListener('click', async () => {
    if (!result) return;
    try {
      await saveItem('fragment', {
        smiles: result.smiles,
        note: result.used_pocket ? 'pharmacophore (site-specific)' : 'pharmacophore (ligand-only)',
        payload: result,
      });
      toast('Saved to workspace');
    } catch (e) { toast(e.message || 'Could not save'); }
  });
})();
