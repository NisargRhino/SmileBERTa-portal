/* Fragment linking page. */
(function () {
  const { $, $$, api, esc, icon, fmt, toast, depict, depictSVG, library, saveItem, handoff, copy, modal, viewer3d, applyMolStyle, download, druglikeBadges } = SB;
  let target = 'a';

  const showSlot = SB.debounce(async (which) => {
    const s = $('#s' + which).value.trim();
    const slot = $('#slot-' + which), pv = $('#p' + which);
    if (s && (await SB.validSmiles(s))) { slot.classList.add('filled'); depict(pv, s, { width: 360, height: 180 }); }
    else { slot.classList.remove('filled'); pv.innerHTML = s ? '<span style="color:var(--red);font-size:.84rem">Invalid SMILES</span>' : '<span class="muted" style="font-size:.84rem">Pick a fragment</span>'; }
  }, 300);
  ['a', 'b'].forEach((w) => { $('#s' + w).addEventListener('input', () => showSlot(w)); $('#s' + w).addEventListener('focus', () => (target = w)); showSlot(w); });

  $('#ex').addEventListener('click', () => { $('#sa').value = 'Cc1ccc(C)c(F)c1'; $('#sb').value = 'CC(=O)CCc1ccccc1'; showSlot('a'); showSlot('b'); });
  $('#swap').addEventListener('click', () => { const t = $('#sa').value; $('#sa').value = $('#sb').value; $('#sb').value = t; showSlot('a'); showSlot('b'); });

  async function renderLibrary() {
    const lib = await library.list();
    const el = $('#library');
    if (!lib.length) { el.innerHTML = `<p class="muted" style="font-size:.86rem;margin:0">Your library is empty. Predict fragments on the <a href="fragmentation.html">Fragment Prediction</a> page and add them here, or type SMILES directly.</p>`; return; }
    el.innerHTML = '<p class="kbd-hint" style="margin:0">Click a fragment to place it in the next empty slot.</p>' + lib.map((f, i) => `<div class="list-item lib-item" data-s="${esc(f.smiles)}"><div class="thumb" id="l${i}"></div><div class="grow mono" style="font-size:.72rem;word-break:break-all">${esc(f.smiles)}</div></div>`).join('');
    lib.forEach((f, i) => depictSVG(f.smiles, { width: 140, height: 110 }).then((s) => { const t = $('#l' + i); if (t) t.innerHTML = s; }).catch(() => {}));
    $$('.lib-item', el).forEach((it) => it.addEventListener('click', () => {
      const w = !$('#sa').value.trim() ? 'a' : !$('#sb').value.trim() ? 'b' : target;
      $('#s' + w).value = it.dataset.s; showSlot(w); target = w === 'a' ? 'b' : 'a';
    }));
  }
  renderLibrary();

  async function view3D(smiles) {
    await modal({ title: '3D conformer', wide: true, body: '<div class="viewer3d" id="m3d" style="height:460px"></div>', actions: [],
      onOpen: async () => { try { const { viewer } = await SB.show3DFromSmiles($('#m3d'), smiles); viewer.spin('y', 0.4); } catch (e) { $('#m3d').innerHTML = `<div class="empty">${esc(e.message)}</div>`; } } });
  }

  $('#go').addEventListener('click', async () => {
    const s1 = $('#sa').value.trim(), s2 = $('#sb').value.trim();
    if (!s1 || !s2) return toast('Provide both fragments', 'error');
    const btn = $('#go'); btn.classList.add('loading'); btn.innerHTML = '<span class="spinner"></span> Linking…';
    $('#results').innerHTML = '<div class="result-grid">' + '<div class="skeleton" style="height:300px"></div>'.repeat(3) + '</div>';
    try {
      const d = await api('/combine', { body: { smiles1: s1, smiles2: s2 } });
      const list = (d.combined_smiles || []).sort((a, b) => ((b.analysis || {}).qed || 0) - ((a.analysis || {}).qed || 0));
      if (!list.length) { $('#results').innerHTML = `<div class="card empty"><div>${icon('info')}<h3 style="color:var(--text)">No drug-like products</h3><p>None of the amide/ester linkages passed the Lipinski + QED filters. Try smaller fragments.</p></div></div>`; return; }
      $('#results').innerHTML = `<div class="row-between" style="margin:8px 0 14px"><h3 style="margin:0">${list.length} linked candidate${list.length > 1 ? 's' : ''}</h3><span class="muted" style="font-size:.84rem">Sorted by QED</span></div>
        <div class="result-grid">${list.map((c, i) => `<div class="card res-card fade-in" style="padding:16px;animation-delay:${i * 60}ms">
          <div class="row-between" style="margin-bottom:8px"><b>Candidate ${i + 1}</b>${c.analysis ? `<span class="badge ${c.analysis.qed >= 0.5 ? 'cyan' : 'amber'}">QED ${fmt(c.analysis.qed, 2)}</span>` : ''}</div>
          <div class="mol-2d" id="c${i}"></div>
          <div class="kv" style="margin-top:10px"><div>MW</div><div>${fmt(c.properties.molecular_weight, 1)} Da</div><div>cLogP</div><div>${fmt(c.properties.log_p, 2)}</div><div>HBD / HBA</div><div>${c.properties.hydrogen_bond_donors} / ${c.properties.hydrogen_bond_acceptors}</div><div>TPSA</div><div>${fmt(c.properties.tpsa, 1)} Å²</div></div>
          <div class="smiles-box" style="margin:10px 0;font-size:.72rem"><span>${esc(c.smiles)}</span></div>
          <div class="row" style="gap:5px"><button class="btn btn-xs" data-act="3d" data-i="${i}">${icon('cube')} 3D</button><button class="btn btn-xs" data-act="copy" data-i="${i}">${icon('copy')}</button><button class="btn btn-xs" data-act="save" data-i="${i}">${icon('save')}</button><button class="btn btn-xs" data-act="dock" data-i="${i}">${icon('target')} Dock</button><button class="btn btn-xs" data-act="build" data-i="${i}">${icon('atom')}</button></div>
        </div>`).join('')}</div>`;
      list.forEach((c, i) => depict($('#c' + i), c.smiles, { width: 320, height: 200 }));
      $$('[data-act]').forEach((b) => b.addEventListener('click', () => {
        const c = list[+b.dataset.i];
        ({ '3d': () => view3D(c.smiles), copy: () => copy(c.smiles), save: () => saveItem('combination', 'Linked candidate ' + (+b.dataset.i + 1), { a: s1, b: s2, smiles: c.smiles, properties: c.properties }, c.smiles), dock: () => handoff('docking', { smiles: c.smiles }), build: () => handoff('builder', { smiles: c.smiles }) })[b.dataset.act]();
      }));
    } catch (err) { $('#results').innerHTML = `<div class="card empty"><div>${icon('alert')}<p>${esc(err.message)}</p></div></div>`; }
    finally { btn.classList.remove('loading'); btn.textContent = 'Link fragments'; }
  });

  (async () => {
    const id = SB.param('item');
    if (id) { try { const it = await SB.loadItem(id); if (it.data) { $('#sa').value = it.data.a || ''; $('#sb').value = it.data.b || ''; showSlot('a'); showSlot('b'); } } catch (e) { toast(e.message, 'error'); } }
  })();
})();
