/* Workspace: saved items per account. */
(function () {
  const { $, $$, api, auth, esc, icon, toast, modal, promptText, download, depictSVG, library } = SB;
  const KINDS = {
    molecule: ['Molecules', 'atom', 'builder.html'], fragment: ['Fragments', 'split', 'fragmentation.html'], classification: ['Classifications', 'tag', 'drug_class.html'],
    combination: ['Linked leads', 'merge', 'combine.html'], docking: ['Docking runs', 'target', 'docking.html'], structure: ['Structures', 'helix', 'studio.html'],
  };
  let items = [], filter = 'all';

  if (!auth.loggedIn()) {
    $('#gate').classList.remove('hidden'); $('#pw').classList.add('hidden'); $('#signout').classList.add('hidden');
    $('#gate .icon-tile').innerHTML = icon('lock');
    return;
  }
  $('#ws').classList.remove('hidden');
  $('#signout').addEventListener('click', () => auth.signOut());

  const ago = (iso) => {
    if (!iso) return '';
    const t = new Date(iso.replace(' ', 'T') + (iso.endsWith('Z') || iso.includes('+') ? '' : 'Z'));
    const s = (Date.now() - t) / 1000;
    if (s < 60) return 'just now'; if (s < 3600) return Math.floor(s / 60) + 'm ago'; if (s < 86400) return Math.floor(s / 3600) + 'h ago';
    return t.toLocaleDateString();
  };

  async function loadAll() {
    try {
      const [me, list] = await Promise.all([api('/me'), api('/items')]);
      items = list.items;
      $('#hello').textContent = `Welcome back, ${me.username}`;
      $('#sub').textContent = `Member since ${me.created_at ? new Date(me.created_at.replace(' ', 'T') + 'Z').toLocaleDateString() : 'today'} · ${items.length} saved item${items.length === 1 ? '' : 's'}`;
      if (!me.persistent_storage) $('#persist-note').innerHTML = '<div class="notice" style="margin-bottom:22px">Heads up: this server is running without a persistent database, so saved items may be cleared when the server redeploys. Export anything important as JSON.</div>';
      $('#stats').innerHTML = Object.entries(KINDS).map(([k, [n, ic]]) => `<div class="stat"><div class="row-between"><div class="v">${me.stats[k] || 0}</div><span style="color:var(--cyan);width:18px;height:18px;display:block">${icon(ic)}</span></div><div class="k">${n}</div></div>`).join('');
      renderTabs(); renderItems();
    } catch (e) { $('#items').innerHTML = `<div class="card empty">${esc(e.message)}</div>`; }
    renderLibrary();
  }

  function renderTabs() {
    const counts = items.reduce((m, i) => ((m[i.kind] = (m[i.kind] || 0) + 1), m), {});
    $('#kinds').innerHTML = `<button data-k="all" class="${filter === 'all' ? 'active' : ''}">All (${items.length})</button>` + Object.entries(KINDS).filter(([k]) => counts[k]).map(([k, [n]]) => `<button data-k="${k}" class="${filter === k ? 'active' : ''}">${n} (${counts[k]})</button>`).join('');
    $$('#kinds button').forEach((b) => b.addEventListener('click', () => { filter = b.dataset.k; renderTabs(); renderItems(); }));
  }

  function renderItems() {
    const q = $('#search').value.trim().toLowerCase();
    const list = items.filter((i) => (filter === 'all' || i.kind === filter) && (!q || (i.name + ' ' + (i.summary || '')).toLowerCase().includes(q)));
    if (!list.length) {
      $('#items').innerHTML = `<div class="card empty" style="grid-column:1/-1;min-height:220px"><div>${icon('folder')}<h3 style="color:var(--text)">${items.length ? 'No matching items' : 'Nothing saved yet'}</h3><p>${items.length ? 'Try a different search or filter.' : 'Use the <b>Save</b> button in any tool to keep results here.'}</p>${items.length ? '' : '<div class="row" style="justify-content:center;margin-top:10px"><a class="btn btn-sm" href="builder.html">Build a molecule</a><a class="btn btn-sm" href="docking.html">Run a docking</a></div>'}</div></div>`;
      return;
    }
    $('#items').innerHTML = list.map((i) => {
      const k = KINDS[i.kind] || ['Item', 'file', 'index.html'];
      return `<div class="card card-hover item-card fade-in"><div class="row-between"><span class="badge cyan"><span style="width:12px;height:12px;display:inline-block">${icon(k[1])}</span>${k[0].replace(/s$/, '')}</span><span class="muted" style="font-size:.74rem">${ago(i.updated_at)}</span></div>
        <div class="name">${esc(i.name)}</div><div class="summary">${esc(i.summary || '')}</div>
        <div class="foot"><a class="btn btn-sm btn-primary" href="${k[2]}?item=${i.id}">Open</a><div class="row" style="gap:2px"><button class="btn btn-xs btn-ghost" data-ren="${i.id}" title="Rename">${icon('edit')}</button><button class="btn btn-xs btn-ghost" data-exp="${i.id}" title="Export JSON">${icon('download')}</button><button class="btn btn-xs btn-ghost" data-del="${i.id}" title="Delete">${icon('trash')}</button></div></div></div>`;
    }).join('');
    $$('[data-ren]').forEach((b) => b.addEventListener('click', async () => {
      const it = items.find((x) => x.id === +b.dataset.ren);
      const name = await promptText('Rename', 'Name', it.name); if (!name) return;
      try { await api('/items/' + it.id, { method: 'PUT', body: { name } }); it.name = name; renderItems(); toast('Renamed', 'ok'); } catch (e) { toast(e.message, 'error'); }
    }));
    $$('[data-exp]').forEach((b) => b.addEventListener('click', async () => {
      try { const it = await api('/items/' + b.dataset.exp); download(JSON.stringify(it, null, 2), it.name.replace(/[^\w.-]+/g, '_') + '.json', 'application/json'); } catch (e) { toast(e.message, 'error'); }
    }));
    $$('[data-del]').forEach((b) => b.addEventListener('click', async () => {
      const it = items.find((x) => x.id === +b.dataset.del);
      const ok = await modal({ title: 'Delete item?', body: `<p class="text-2">“${esc(it.name)}” will be permanently deleted.</p>`, actions: [{ label: 'Cancel', value: false }, { label: 'Delete', danger: true, value: true }] });
      if (!ok) return;
      try { await api('/items/' + it.id, { method: 'DELETE' }); items = items.filter((x) => x.id !== it.id); renderTabs(); renderItems(); toast('Deleted', 'ok'); } catch (e) { toast(e.message, 'error'); }
    }));
  }
  $('#search').addEventListener('input', SB.debounce(renderItems, 150));

  async function renderLibrary() {
    const lib = await library.list();
    if (!lib.length) { $('#library').innerHTML = '<p class="muted">No fragments yet. Add them from <a href="fragmentation.html">Fragment Prediction</a>.</p>'; return; }
    $('#library').innerHTML = lib.map((f, i) => `<div class="card item-card"><div class="mol-2d" id="lf${i}" style="min-height:150px"></div><div class="summary">${esc(f.smiles)}</div><div class="foot"><button class="btn btn-xs" data-dock="${esc(f.smiles)}">${icon('target')} Dock</button><button class="btn btn-xs btn-ghost" data-rm="${esc(f.smiles)}">${icon('trash')}</button></div></div>`).join('');
    lib.forEach((f, i) => depictSVG(f.smiles, { width: 280, height: 150 }).then((s) => { const el = $('#lf' + i); if (el) el.innerHTML = s; }).catch(() => {}));
    $$('[data-rm]', $('#library')).forEach((b) => b.addEventListener('click', async () => { await library.remove(b.dataset.rm); renderLibrary(); }));
    $$('[data-dock]', $('#library')).forEach((b) => b.addEventListener('click', () => SB.handoff('docking', { smiles: b.dataset.dock })));
  }

  $('#pw').addEventListener('click', async () => {
    const vals = await modal({ title: 'Change password', body: `<label class="field"><span>Current password</span><input class="input" type="password" id="pw-cur" autocomplete="current-password"></label><label class="field"><span>New password</span><input class="input" type="password" id="pw-new" autocomplete="new-password" placeholder="At least 8 characters"></label>`,
      actions: [{ label: 'Cancel', value: null }, { label: 'Update password', primary: true, value: (m) => ({ current: m.querySelector('#pw-cur').value, new: m.querySelector('#pw-new').value }) }] });
    if (!vals) return;
    try { await api('/account/password', { body: vals }); toast('Password updated', 'ok'); } catch (e) { toast(e.message, 'error'); }
  });

  loadAll();
})();
