/* Admin dashboard (visible only to usernames listed in ADMIN_USERNAMES on the API). */
(function () {
  const { $, $$, api, auth, esc, icon, toast, modal } = SB;
  let data = null;
  const fmtDate = (iso) => (iso ? new Date(String(iso).replace(' ', 'T') + (/[Z+]/.test(iso) ? '' : 'Z')).toLocaleString() : '—');
  const fmtUp = (s) => (s > 3600 ? Math.floor(s / 3600) + 'h ' : '') + Math.floor((s % 3600) / 60) + 'm';

  function gate(msg) { $('#gate').classList.remove('hidden'); $('#admin').classList.add('hidden'); if (msg) $('#gate-msg').textContent = msg; }

  async function load() {
    if (!auth.loggedIn()) return gate('Sign in with the admin account to view this page.');
    try { data = await api('/admin/overview'); }
    catch (e) { return gate(/admin/i.test(e.message) ? 'This account does not have admin access.' : e.message); }
    $('#gate').classList.add('hidden'); $('#admin').classList.remove('hidden');
    const s = data.server, totalItems = Object.values(data.items_by_kind).reduce((a, b) => a + b, 0);
    $('#persist').innerHTML = s.persistent_storage ? '' : '<div class="notice" style="margin-bottom:22px">The API is using SQLite on an ephemeral disk: all accounts (including this one) are erased on every redeploy. Connect a Postgres database via DATABASE_URL to make them permanent.</div>';
    $('#kpis').innerHTML = [
      [data.users.length, 'Registered users'], [totalItems, 'Saved items'],
      [s.database === 'postgres' ? 'Postgres' : 'SQLite', 'Database'], [fmtUp(s.uptime), `API uptime · ${s.rss_mb ?? '?'} MB RAM`],
    ].map(([v, k]) => `<div class="stat"><div class="v">${esc(v)}</div><div class="k">${esc(k)}</div></div>`).join('');
    const max = Math.max(1, ...Object.values(data.items_by_kind));
    $('#kinds').innerHTML = Object.entries(data.items_by_kind).sort((a, b) => b[1] - a[1]).map(([k, n]) => `<div class="bar-row"><span style="text-transform:capitalize">${esc(k)}</span><div class="bar"><div style="width:${n / max * 100}%"></div></div><span class="num mono">${n}</span></div>`).join('') || '<p class="muted">Nothing saved yet.</p>';
    $('#recent').innerHTML = data.recent_items.map((r) => `<div class="layer-row" style="display:flex;gap:10px;padding:8px 0;border-bottom:1px solid var(--border);font-size:.84rem"><span class="badge cyan">${esc(r.kind)}</span><span class="grow">${esc(r.name)}<div class="muted" style="font-size:.74rem">${esc(r.username)} · ${fmtDate(r.created_at)}</div></span></div>`).join('') || '<p class="muted">No activity yet.</p>';
    renderUsers();
  }

  function renderUsers() {
    const q = $('#q').value.trim().toLowerCase();
    $('#users').innerHTML = data.users.filter((u) => !q || u.username.toLowerCase().includes(q)).map((u) => `<tr>
      <td><b style="color:#fff">${esc(u.username)}</b> ${u.is_admin ? '<span class="badge violet">admin</span>' : ''}</td>
      <td style="font-size:.78rem">${fmtDate(u.created_at)}</td><td class="num">${u.items}</td><td class="num">${u.fragments}</td>
      <td>${u.is_admin ? '' : `<button class="btn btn-xs btn-ghost" data-del="${u.id}" title="Delete user">${icon('trash')}</button>`}</td></tr>`).join('');
    $$('[data-del]').forEach((b) => b.addEventListener('click', async () => {
      const u = data.users.find((x) => x.id === +b.dataset.del);
      const ok = await modal({ title: 'Delete user?', body: `<p class="text-2">This permanently deletes <b>${esc(u.username)}</b> and all ${u.items} saved items.</p>`, actions: [{ label: 'Cancel', value: false }, { label: 'Delete user', danger: true, value: true }] });
      if (!ok) return;
      try { await api('/admin/users/' + u.id, { method: 'DELETE' }); toast('User deleted', 'ok'); load(); } catch (e) { toast(e.message, 'error'); }
    }));
  }

  $('#q').addEventListener('input', SB.debounce(renderUsers, 150));
  $('#refresh').addEventListener('click', load);
  load();
})();
