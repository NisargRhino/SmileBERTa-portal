/* ADMET Profiler: published absorption/solubility/drug-likeness models, one molecule or many. */
(function () {
  const { $, $$, api, esc, icon, fmt, toast, download, depict, saveItem,
          takeHandoff, readFile, parseCSV, toCSV, bindDropzone } = SB;

  const EXAMPLES = [
    ['Aspirin', 'CC(=O)Oc1ccccc1C(=O)O'],
    ['Caffeine', 'Cn1cnc2c1c(=O)n(C)c(=O)n2C'],
    ['Ibuprofen', 'CC(C)Cc1ccc(C(C)C(=O)O)cc1'],
    ['Imatinib', 'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1'],
    ['Atorvastatin', 'CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O'],
    ['Quercetin', 'O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12'],
  ];

  let profile = null, batch = [];

  // ── Mode switch ───────────────────────────────────────────────────────────
  $$('.mode-row .chip').forEach((b) => b.addEventListener('click', () => {
    $$('.mode-row .chip').forEach((x) => x.classList.toggle('active', x === b));
    const single = b.dataset.mode === 'single';
    $('#mode-single').hidden = !single;
    $('#mode-batch').hidden = single;
  }));

  // ── Single molecule ───────────────────────────────────────────────────────
  $('#examples').innerHTML = EXAMPLES
    .map(([n, s]) => `<button class="chip" data-smi="${esc(s)}">${esc(n)}</button>`).join('');
  $$('#examples .chip').forEach((b) => b.addEventListener('click', () => {
    $('#smi').value = b.dataset.smi;
    run();
  }));

  $('#run').addEventListener('click', run);
  $('#smi').addEventListener('keydown', (e) => { if (e.key === 'Enter') run(); });

  async function run(smilesOverride) {
    const smiles = (smilesOverride || $('#smi').value || '').trim();
    if (!smiles) { toast('Enter a SMILES string', 'error'); return; }
    const btn = $('#run');
    btn.disabled = true; btn.textContent = 'Profiling…';
    try {
      const res = await api('/admet', { body: { smiles } });
      profile = res.profile;
      render(res);
      $('#out').hidden = false;
    } catch (e) {
      toast(e.message || 'Could not profile that molecule', 'error');
    } finally {
      btn.disabled = false; btn.textContent = 'Profile';
    }
  }

  function render(res) {
    const p = profile;

    // Structure + identity
    if (res.svg) $('#depict').innerHTML = res.svg;
    else depict($('#depict'), p.smiles);

    const sol = p.solubility;
    $('#badges').innerHTML = [
      `<span class="badge">${esc(p.formula)}</span>`,
      `<span class="badge ${p.filters_passed >= 5 ? 'green' : p.filters_passed >= 3 ? 'amber' : 'red'}">${p.filters_passed}/${p.filters_total} filters</span>`,
      `<span class="badge ${p.alert_count ? 'amber' : 'green'}">${p.alert_count} alert${p.alert_count === 1 ? '' : 's'}</span>`,
      `<span class="badge cyan">${esc(sol['class'])}</span>`,
    ].join('');

    $('#ids').innerHTML = [
      ['Canonical SMILES', p.smiles],
      ['InChIKey', p.inchikey || '—'],
      ['Water solubility', `${fmt(sol.log_s, 2)} log S · ${sol.mg_per_ml} mg/mL`],
    ].map(([k, v]) => `<div>${esc(k)}</div><div style="word-break:break-all">${esc(v)}</div>`).join('');

    $('#physchem').innerHTML = p.physchem
      .map((r) => `<div>${esc(r.label)}</div><div>${esc(String(r.value))}${r.unit ? ' ' + esc(r.unit) : ''}</div>`)
      .join('');

    $('#radar').innerHTML = radarSVG(p.radar);
    $('#pk').innerHTML = p.pharmacokinetics.map(endpointRow).join('');
    $('#medchem').innerHTML = p.medchem.map(endpointRow).join('');

    // Filters
    $('#filt-count').textContent = `${p.filters_passed} of ${p.filters_total} passed`;
    $('#filters').innerHTML = p.filters.map((f, i) => `
      <div class="filt" data-i="${i}">
        <div class="filt-head">
          <span class="filt-name">${esc(f.name)}</span>
          <span class="row" style="gap:8px;align-items:center">
            <span class="badge ${f.passed ? 'green' : 'amber'}">${f.violations.length
              ? esc(f.violations.length + ' violation' + (f.violations.length === 1 ? '' : 's'))
              : 'pass'}</span>
            <span class="filt-chev">${icon('chevron')}</span>
          </span>
        </div>
        <div class="filt-body">
          <p class="muted" style="font-size:.77rem;margin:0 0 9px">${esc(f.note)}</p>
          ${f.checks.map((c) => `<div class="check ${c.ok ? 'yes' : 'no'}">${icon(c.ok ? 'check' : 'x')}<span>${esc(c.desc)}</span></div>`).join('')}
          ${f.allowed_violations ? `<p class="muted" style="font-size:.73rem;margin:8px 0 0">Up to ${f.allowed_violations} violation is tolerated by this filter.</p>` : ''}
          <p class="ep-cite" style="margin-top:8px">${esc(f.cite)}</p>
        </div>
      </div>`).join('');
    $$('#filters .filt-head').forEach((h) => h.addEventListener('click', () =>
      h.parentElement.classList.toggle('open')));

    // Alerts
    $('#alert-count').textContent = p.alert_count ? `${p.alert_count} matched` : 'none';
    $('#alerts').innerHTML = p.alerts.length
      ? p.alerts.map((a) => `
        <div class="alert-item">
          <div class="row" style="justify-content:space-between;gap:10px;align-items:baseline">
            <span class="alert-name">${esc(a.name)}</span>
            <span class="badge amber">${esc(a.family)}</span>
          </div>
          <div class="ep-note">${esc(a.blurb)}</div>
          <div class="ep-cite">${esc(a.cite)}</div>
        </div>`).join('')
      : `<p class="muted" style="font-size:.82rem">No PAINS, Brenk or NIH alerts matched.</p>`;

    $('#unavailable').innerHTML = p.unavailable.map((u) => `
      <div class="ep">
        <div class="ep-top"><span class="ep-label">${esc(u.label)}</span>
          <span class="ep-val"><span class="dot neutral"></span>not computed</span></div>
        <div class="ep-note">${esc(u.why)}</div>
      </div>`).join('');

    $('#refs').innerHTML = p.references.map((r) => `
      <div class="ref-item">
        <span class="ref-what">${esc(r.what)}</span> —
        <span class="ref-cite">${esc(r.cite)}${r.doi ? ` · <a href="https://doi.org/${esc(r.doi)}" target="_blank" rel="noopener">doi</a>` : ''}</span>
      </div>`).join('');
  }

  function endpointRow(r) {
    return `
      <div class="ep">
        <div class="ep-top">
          <span class="ep-label">${esc(r.label)}</span>
          <span class="ep-val"><span class="dot ${esc(r.verdict || 'neutral')}"></span>${esc(String(r.value ?? '—'))}</span>
        </div>
        ${r.note ? `<div class="ep-note">${esc(r.note)}</div>` : ''}
        ${r.cite ? `<div class="ep-cite">${esc(r.cite)}</div>` : ''}
      </div>`;
  }

  // ── Radar ─────────────────────────────────────────────────────────────────
  // Six axes. Each value is normalised onto its own published scale, so the
  // favourable band is a ring rather than a circle.
  function radarSVG(axes) {
    const S = 320, C = S / 2, R = 108;
    const norm = (v, [lo, hi]) => Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
    const pt = (i, r) => {
      const a = -Math.PI / 2 + (i * 2 * Math.PI) / axes.length;
      return [C + r * R * Math.cos(a), C + r * R * Math.sin(a)];
    };
    const poly = (rs) => rs.map((r, i) => pt(i, r).map((n) => n.toFixed(1)).join(',')).join(' ');

    const outer = axes.map((a) => norm(a.optimal[1], a.scale));
    const inner = axes.map((a) => norm(a.optimal[0], a.scale));
    const shape = axes.map((a) => norm(a.value, a.scale));

    const grid = [0.25, 0.5, 0.75, 1].map((f) =>
      `<polygon points="${poly(axes.map(() => f))}" fill="none" stroke="var(--border)" stroke-width="1"/>`).join('');
    const spokes = axes.map((a, i) => {
      const [x, y] = pt(i, 1);
      return `<line x1="${C}" y1="${C}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="var(--border)" stroke-width="1"/>`;
    }).join('');
    const labels = axes.map((a, i) => {
      const [x, y] = pt(i, 1.26);
      const anchor = Math.abs(x - C) < 6 ? 'middle' : (x > C ? 'start' : 'end');
      return `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="${anchor}"
        font-size="10.5" font-weight="600" fill="var(--muted)" dominant-baseline="middle">${esc(a.key)}</text>
        <text x="${x.toFixed(1)}" y="${(y + 11).toFixed(1)}" text-anchor="${anchor}"
        font-size="9" fill="var(--muted)" opacity=".75" dominant-baseline="middle">${esc(fmtAxis(a))}</text>`;
    }).join('');

    return `<svg viewBox="0 0 ${S} ${S}" role="img" aria-label="Bioavailability radar">
      ${grid}${spokes}
      <path d="M ${poly(outer).split(' ').join(' L ')} Z M ${poly(inner).split(' ').join(' L ')} Z"
        fill="rgba(52,211,153,.18)" stroke="rgba(52,211,153,.5)" stroke-width="1" fill-rule="evenodd"/>
      <polygon points="${poly(shape)}" fill="rgba(34,211,238,.28)" stroke="var(--cyan)" stroke-width="2"/>
      ${axes.map((a, i) => { const [x, y] = pt(i, norm(a.value, a.scale));
        return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3" fill="var(--cyan)"/>`; }).join('')}
      ${labels}
    </svg>`;
  }

  function fmtAxis(a) {
    const v = a.key === 'SIZE' ? Math.round(a.value) : Number(a.value).toFixed(2);
    return `${v}`;
  }

  // ── Save / export (single) ────────────────────────────────────────────────
  $('#save').addEventListener('click', () => {
    if (!profile) return;
    const sol = profile.solubility;
    saveItem('admet', profile.formula || 'ADMET profile', profile,
      `${profile.filters_passed}/${profile.filters_total} filters · log S ${fmt(sol.log_s, 2)} · ${profile.alert_count} alert${profile.alert_count === 1 ? '' : 's'}`);
  });

  $('#csv').addEventListener('click', () => {
    if (!profile) return;
    const rows = [['property', 'value', 'basis', 'source']];
    profile.physchem.forEach((r) => rows.push([r.label, r.value, 'descriptor', 'RDKit']));
    profile.pharmacokinetics.forEach((r) => rows.push([r.label, r.value, r.basis || '', r.cite || '']));
    profile.medchem.forEach((r) => rows.push([r.label, r.value, r.basis || '', r.cite || '']));
    profile.filters.forEach((f) => rows.push([
      f.name, f.passed ? 'pass' : `fail (${f.violations.join('; ')})`, 'rule', f.cite]));
    profile.alerts.forEach((a) => rows.push([`Alert (${a.family})`, a.name, 'substructure', a.cite]));
    download(toCSV(rows), `admet_${(profile.formula || 'profile').replace(/[^\w]/g, '')}.csv`, 'text/csv');
  });

  // ── Batch ─────────────────────────────────────────────────────────────────
  const COLS = [
    ['input_smiles', 'SMILES', 'smi'], ['mw', 'MW'], ['wlogp', 'logP'], ['tpsa', 'TPSA'],
    ['hbd', 'HBD'], ['hba', 'HBA'], ['rotatable_bonds', 'Rot'], ['log_s', 'log S'],
    ['solubility_class', 'Solubility'], ['gi_absorption', 'GI abs.'], ['bbb', 'BBB'],
    ['qed', 'QED'], ['sa_score', 'SA'], ['lipinski_violations', 'Ro5 viol.'],
    ['filters_passed', 'Filters'], ['alert_count', 'Alerts'],
  ];

  bindDropzone($('#drop'), async (file) => {
    try {
      const text = await readFile(file);
      const list = parseCSV(text);
      if (!list.length) { toast('No SMILES column found in that file', 'error'); return; }
      $('#bsmi').value = list.join('\n');
      toast(`Loaded ${list.length} compound${list.length === 1 ? '' : 's'}`, 'ok');
    } catch (e) { toast('Could not read that file', 'error'); }
  }, '.csv,.smi,.txt');

  $('#brun').addEventListener('click', async () => {
    const list = ($('#bsmi').value || '').split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
    if (!list.length) { toast('Paste some SMILES first', 'error'); return; }
    const btn = $('#brun');
    btn.disabled = true; btn.textContent = 'Screening…';
    try {
      const res = await api('/admet/batch', { body: { smiles_list: list }, timeout: 180000 });
      batch = res.results || [];
      renderBatch();
      $('#bout').hidden = false;
      $('#bcsv').disabled = !batch.length;
    } catch (e) {
      toast(e.message || 'Batch screen failed', 'error');
    } finally {
      btn.disabled = false; btn.textContent = 'Screen compounds';
    }
  });

  function renderBatch() {
    const ok = batch.filter((r) => !r.error).length;
    $('#bcount').textContent = `${ok} profiled${ok < batch.length ? `, ${batch.length - ok} failed` : ''}`;
    $('#btbl').innerHTML = `
      <thead><tr>${COLS.map(([, label]) => `<th>${esc(label)}</th>`).join('')}</tr></thead>
      <tbody>${batch.map((r, i) => r.error
        ? `<tr class="bad"><td class="smi" title="${esc(r.input_smiles)}">${esc(r.input_smiles)}</td>
             <td colspan="${COLS.length - 1}">${esc(r.error)}</td></tr>`
        : `<tr data-i="${i}" style="cursor:pointer">${COLS.map(([k, , cls]) =>
             `<td class="${cls || ''}"${cls === 'smi' ? ` title="${esc(r[k])}"` : ''}>${esc(String(r[k] ?? '—'))}</td>`).join('')}</tr>`
      ).join('')}</tbody>`;
    $$('#btbl tr[data-i]').forEach((tr) => tr.addEventListener('click', () => {
      const r = batch[+tr.dataset.i];
      $$('.mode-row .chip').forEach((x) => x.classList.toggle('active', x.dataset.mode === 'single'));
      $('#mode-single').hidden = false; $('#mode-batch').hidden = true;
      $('#smi').value = r.input_smiles;
      window.scrollTo({ top: 0, behavior: 'smooth' });
      run(r.input_smiles);
    }));
  }

  $('#bcsv').addEventListener('click', () => {
    const rows = [COLS.map(([, label]) => label).concat(['alerts', 'error'])];
    batch.forEach((r) => rows.push(
      COLS.map(([k]) => r[k] ?? '').concat([(r.alerts || []).join('; '), r.error || ''])));
    download(toCSV(rows), 'admet_screen.csv', 'text/csv');
  });

  // ── Handoff from another tool ─────────────────────────────────────────────
  const incoming = takeHandoff('admet');
  if (incoming && incoming.smiles) {
    $('#smi').value = incoming.smiles;
    run();
  } else if (SB.param('smiles')) {
    $('#smi').value = SB.param('smiles');
    run();
  }
})();
