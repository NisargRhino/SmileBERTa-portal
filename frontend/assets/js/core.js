/* SmileBERTa shared runtime: config, API client, auth, layout, UI helpers, chemistry helpers. */
(function () {
  'use strict';

  // ── Config ────────────────────────────────────────────────────────────────
  const PROD_API = 'https://smileberta-api.onrender.com';
  const isLocal = ['localhost', '127.0.0.1', ''].includes(location.hostname);
  let override = null;
  try { override = localStorage.getItem('sb_api'); } catch (e) { /* ignore */ }
  const API = (override || (isLocal ? 'http://localhost:5001' : PROD_API)).replace(/\/$/, '');
  // Docking runs on its own lightweight service (no PyTorch) so Vina has enough memory.
  // Dedicated docking VPS, not Render: Vina is CPU-bound and the free tier's
  // ~0.1 shared vCPU made a drug-sized dock take ~30 min (now ~35 s).
  // Hostname is sslip.io over the server's IP — update both if the server is
  // ever rebuilt, or point a real subdomain here. See deploy/README.md.
  const PROD_DOCK_API = 'https://5-161-192-134.sslip.io';
  let dockOverride = null;
  try { dockOverride = localStorage.getItem('sb_dock_api'); } catch (e) { /* ignore */ }
  const DOCK_API = (dockOverride || (isLocal ? API : PROD_DOCK_API)).replace(/\/$/, '');

  // ── Storage helpers ───────────────────────────────────────────────────────
  const store = {
    get(k, d = null) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* ignore */ } },
    del(k) { try { localStorage.removeItem(k); } catch (e) { /* ignore */ } },
    json(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch (e) { return d; } },
    setJson(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* ignore */ } },
  };

  // ── Icons (stroke icons, 24×24) ───────────────────────────────────────────
  const P = {
    split: '<path d="M6 3v6a6 6 0 0 0 6 6h0a6 6 0 0 1 6 6"/><path d="M18 3v6a6 6 0 0 1-6 6"/><circle cx="6" cy="3" r="1"/><circle cx="18" cy="3" r="1"/>',
    tag: '<path d="M20.6 13.4 13.4 20.6a2 2 0 0 1-2.8 0L3 13V3h10l7.6 7.6a2 2 0 0 1 0 2.8z"/><circle cx="7.5" cy="7.5" r="1.5"/>',
    merge: '<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="12" r="3"/><path d="M9 6h2a4 4 0 0 1 4 4v0M9 18h2a4 4 0 0 0 4-4v0"/>',
    atom: '<circle cx="12" cy="12" r="1.5"/><ellipse cx="12" cy="12" rx="10" ry="4"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(60 12 12)"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(120 12 12)"/>',
    target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    helix: '<path d="M7 3c0 4 10 5 10 9s-10 5-10 9"/><path d="M17 3c0 4-10 5-10 9s10 5 10 9"/><path d="M8.5 7h7M8.5 17h7M10 12h4"/>',
    folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    book: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5z"/><path d="M4 19.5A2.5 2.5 0 0 0 6.5 22H20v-5"/>',
    chevron: '<path d="m6 9 6 6 6-6"/>',
    arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
    check: '<path d="M20 6 9 17l-5-5"/>',
    x: '<path d="M18 6 6 18M6 6l12 12"/>',
    info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    alert: '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
    save: '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><path d="M17 21v-8H7v8M7 3v5h8"/>',
    download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5M12 15V3"/>',
    upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5M12 3v12"/>',
    copy: '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
    trash: '<path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
    play: '<path d="m6 3 14 9-14 9z"/>',
    zap: '<path d="M13 2 3 14h9l-1 8 10-12h-9z"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    cube: '<path d="M21 16V8a2 2 0 0 0-1-1.7l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.7l7 4a2 2 0 0 0 2 0l7-4a2 2 0 0 0 1-1.7z"/><path d="M3.3 7 12 12l8.7-5M12 22V12"/>',
    camera: '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
    refresh: '<path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/>',
    expand: '<path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>',
    undo: '<path d="M3 7v6h6"/><path d="M21 17a9 9 0 0 0-15-6.7L3 13"/>',
    redo: '<path d="M21 7v6h-6"/><path d="M3 17a9 9 0 0 1 15-6.7L21 13"/>',
    hand: '<path d="M18 11V6a2 2 0 0 0-4 0v5M14 10V4a2 2 0 0 0-4 0v6M10 10.5V6a2 2 0 0 0-4 0v8"/><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.9-6-2.4l-3.6-3.6a2 2 0 0 1 2.8-2.8L7 15"/>',
    pen: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
    eraser: '<path d="m7 21-4.3-4.3a1 1 0 0 1 0-1.4l10-10a1 1 0 0 1 1.4 0l5.6 5.6a1 1 0 0 1 0 1.4L11 21"/><path d="M22 21H7M5 11l9 9"/>',
    link: '<path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7"/><path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7"/>',
    ruler: '<path d="M21.3 15.3 8.7 2.7a1 1 0 0 0-1.4 0L2.7 7.3a1 1 0 0 0 0 1.4l12.6 12.6a1 1 0 0 0 1.4 0l4.6-4.6a1 1 0 0 0 0-1.4z"/><path d="m7.5 10.5 2-2M10.5 13.5l2-2M13.5 16.5l2-2"/>',
    charge: '<circle cx="12" cy="12" r="9"/><path d="M12 8v8M8 12h8"/>',
    eye: '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>',
    eyeOff: '<path d="M17.9 17.9A10 10 0 0 1 12 20c-7 0-11-8-11-8a18.5 18.5 0 0 1 5.1-5.9M9.9 4.2A9 9 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.2 3.2M1 1l22 22"/>',
    search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    user: '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    logout: '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/>',
    menu: '<path d="M3 12h18M3 6h18M3 18h18"/>',
    flask: '<path d="M9 3h6M10 3v6L4.5 18.5A2 2 0 0 0 6.2 21h11.6a2 2 0 0 0 1.7-2.5L14 9V3"/><path d="M7 15h10"/>',
    layers: '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5M2 12l10 5 10-5"/>',
    sparkles: '<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M19 17l.8 2.2L22 20l-2.2.8L19 23l-.8-2.2L16 20l2.2-.8z"/>',
    file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
    gauge: '<path d="M12 15l3.5-3.5"/><path d="M20.3 18A10 10 0 1 0 3.7 18"/>',
    edit: '<path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.1 2.1 0 0 1 3 3L12 15l-4 1 1-4z"/>',
    grid: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
    rotate: '<path d="M21 12a9 9 0 1 1-9-9c2.5 0 4.8 1 6.5 2.7L21 8"/><path d="M21 3v5h-5"/>',
    lock: '<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    cpu: '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
    activity: '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
  };
  function icon(name, cls = '') {
    return `<svg class="${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${P[name] || ''}</svg>`;
  }

  const LOGO = `<svg viewBox="0 0 64 64" aria-hidden="true"><defs><linearGradient id="lg-sb" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3b82f6"/><stop offset=".6" stop-color="#22d3ee"/><stop offset="1" stop-color="#2dd4bf"/></linearGradient></defs>
    <g fill="none" stroke="url(#lg-sb)" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M22 9 33 15.5v13L22 35 11 28.5v-13z"/><path d="M15 17.6v8.8M22 31l7.6-4.4" stroke-width="2.2" opacity=".8"/>
      <path d="M33 15.5 44 10M33 22l11 0M33 28.5 44 34M44 10l10 12M44 22l10 0M44 34l10-12M44 10v24"/>
      <path d="M14 45c9 10 27 10 36 0"/></g>
    <g fill="url(#lg-sb)"><circle cx="44" cy="10" r="3.6"/><circle cx="44" cy="22" r="3.6"/><circle cx="44" cy="34" r="3.6"/><circle cx="54" cy="22" r="3.6"/><circle cx="14" cy="45" r="2.6"/><circle cx="50" cy="45" r="2.6"/></g></svg>`;

  const TOOLS = [
    { href: 'fragmentation', icon: 'split', title: 'Fragment Prediction', desc: 'Predict binding fragments with SmileBERTa' },
    { href: 'drug_class', icon: 'tag', title: 'Drug Classification', desc: 'Therapeutic class from structure' },
    { href: 'combine', icon: 'merge', title: 'Fragment Linking', desc: 'Merge fragments into drug-like leads' },
    { href: 'builder', icon: 'atom', title: 'Molecule Builder', desc: 'Build in 3D, optimise with MMFF94' },
    { href: 'docking', icon: 'target', title: 'Docking', desc: 'Protein–ligand docking with AutoDock Vina' },
    { href: 'studio', icon: 'helix', title: 'Structure Studio', desc: 'Explore chains, backbones, surfaces' },
  ];
  const LEARN = [
    { href: 'learn-fragmentation', icon: 'split', title: 'How fragmentation works', desc: 'Tokenizers, transformers, decoding' },
    { href: 'learn-classification', icon: 'tag', title: 'How classification works', desc: 'From SMILES to therapeutic class' },
    { href: 'learn-docking', icon: 'target', title: 'Docking & scoring', desc: 'Search boxes, Vina scoring, poses' },
    { href: 'learn-forcefields', icon: 'activity', title: 'Force fields', desc: 'MMFF94, UFF and energy minimisation' },
  ];

  // ── HTML helpers ──────────────────────────────────────────────────────────
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d));
  function h(html) { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; }
  function debounce(fn, ms = 250) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
  function param(name) { return new URLSearchParams(location.search).get(name); }

  // ── Auth ──────────────────────────────────────────────────────────────────
  const auth = {
    token: () => store.get('auth_token'),
    username: () => store.get('auth_username'),
    loggedIn: () => !!store.get('auth_token'),
    isAdmin: () => store.get('auth_admin') === '1',
    set(token, username, isAdmin) { store.set('auth_token', token); store.set('auth_username', username); store.set('auth_admin', isAdmin ? '1' : '0'); },
    clear() { store.del('auth_token'); store.del('auth_username'); store.del('auth_admin'); },
    async signOut() {
      const t = auth.token();
      if (t) fetch(API + '/logout', { method: 'POST', headers: { Authorization: 'Bearer ' + t } }).catch(() => {});
      auth.clear();
      location.href = './';
    },
    loginUrl() { return 'login?next=' + encodeURIComponent(location.pathname.split('/').pop() + location.search); },
  };

  // ── API client ────────────────────────────────────────────────────────────
  async function api(path, opts = {}) {
    const { method = opts.body ? 'POST' : 'GET', body, timeout = 120000, raw = false, auth: useAuth = true, base = API } = opts;
    const headers = {};
    if (body !== undefined) headers['Content-Type'] = 'application/json';
    if (useAuth && auth.token()) headers.Authorization = 'Bearer ' + auth.token();
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeout);
    let res;
    try {
      res = await fetch(base + path, { method, headers, body: body !== undefined ? JSON.stringify(body) : undefined, signal: ctrl.signal });
    } catch (e) {
      clearTimeout(timer);
      if (e.name === 'AbortError') throw new Error('The request timed out. The server may be waking up — please try again.');
      throw new Error('Could not reach the SmileBERTa API. It may be waking up (free hosting sleeps when idle) — retry in a moment.');
    }
    clearTimeout(timer);
    if (res.status === 401 && auth.loggedIn() && path !== '/login') {
      auth.clear();
      renderNav.refresh && renderNav.refresh();
    }
    if (raw) { if (!res.ok) throw new Error('Request failed (' + res.status + ')'); return res; }
    let data = null;
    try { data = await res.json(); } catch (e) { /* non-JSON */ }
    if (!res.ok || (data && data.success === false && data.error)) {
      throw new Error((data && data.error) || `Request failed (${res.status})`);
    }
    return data;
  }

  // ── API status pill (handles Render cold starts) ──────────────────────────
  const status = { state: 'unknown', health: null, listeners: [] };
  function setStatus(state, health) {
    status.state = state; if (health) status.health = health;
    $$('.api-pill').forEach((el) => {
      el.className = 'api-pill ' + state;
      el.querySelector('.api-label').textContent = { ok: 'API online', waking: 'Waking server…', down: 'API offline', unknown: 'Connecting…' }[state];
    });
    status.listeners.forEach((fn) => fn(state, status.health));
  }
  async function pingHealth() {
    const slow = setTimeout(() => status.state !== 'ok' && setStatus('waking'), 1800);
    for (let attempt = 0; attempt < 12; attempt++) {
      try {
        const r = await fetch(API + '/health', { cache: 'no-store' });
        if (r.ok) { clearTimeout(slow); setStatus('ok', await r.json()); return; }
      } catch (e) { /* retry */ }
      setStatus('waking');
      await sleep(5000);
    }
    clearTimeout(slow);
    setStatus('down');
  }
  function onStatus(fn) { status.listeners.push(fn); if (status.state !== 'unknown') fn(status.state, status.health); }

  // ── Layout ────────────────────────────────────────────────────────────────
  function menuItems(list) {
    return list.map((t) => `<a href="${t.href}"><span class="mi-icon">${icon(t.icon)}</span><span><span class="mi-title">${t.title}</span><span class="mi-desc">${t.desc}</span></span></a>`).join('');
  }
  function renderNav() {
    const host = document.getElementById('site-nav');
    if (!host) return;
    const page = location.pathname.split('/').pop() || 'index';
    const user = auth.username();
    host.outerHTML = `
    <nav class="nav" id="site-nav"><div class="container container-wide nav-inner">
      <a class="brand" href="./" aria-label="SmileBERTa home">${LOGO}<span>Smile<b>BERT</b>a</span></a>
      <div class="nav-links">
        <div class="nav-drop" data-drop><button type="button" aria-haspopup="true">Tools ${icon('chevron')}</button><div class="nav-menu">${menuItems(TOOLS)}</div></div>
        <div class="nav-drop" data-drop><button type="button" aria-haspopup="true">Learn ${icon('chevron')}</button><div class="nav-menu">${menuItems(LEARN)}<a href="learn"><span class="mi-icon">${icon('book')}</span><span><span class="mi-title">All guides</span><span class="mi-desc">The science behind every tool</span></span></a></div></div>
        <a class="nav-link ${page === 'workspace' ? 'active' : ''}" href="workspace">Workspace</a>
        <a class="nav-link" href="./#about">About</a>
      </div>
      <div class="nav-right">
        <span class="api-pill unknown" title="${esc(API)}"><span class="api-dot"></span><span class="api-label">Connecting…</span></span>
        ${user
          ? `<div class="user-menu nav-drop" data-drop><button type="button" class="avatar" aria-label="Account">${esc(user[0].toUpperCase())}</button>
              <div class="nav-menu"><div style="padding:10px 12px 6px"><div class="muted" style="font-size:.74rem">Signed in as</div><div style="font-weight:600">${esc(user)}</div></div>
              <a href="workspace"><span class="mi-icon">${icon('folder')}</span><span><span class="mi-title">My workspace</span><span class="mi-desc">Saved molecules, runs and structures</span></span></a>
              ${auth.isAdmin() ? `<a href="admin"><span class="mi-icon">${icon('lock')}</span><span><span class="mi-title">Admin dashboard</span><span class="mi-desc">Users, activity and server health</span></span></a>` : ''}
              <a href="#" data-signout><span class="mi-icon">${icon('logout')}</span><span><span class="mi-title">Sign out</span></span></a></div></div>`
          : `<a class="btn btn-ghost btn-sm" href="${auth.loginUrl()}">Sign in</a><a class="btn btn-primary btn-sm" href="login?mode=register">Get started</a>`}
        <button class="nav-burger" type="button" aria-label="Menu">${icon('menu')}</button>
      </div>
    </div></nav>
    <div class="mobile-menu" id="mobile-menu">
      <h6>Tools</h6>${TOOLS.map((t) => `<a href="${t.href}">${t.title}</a>`).join('')}
      <h6>Learn</h6>${LEARN.map((t) => `<a href="${t.href}">${t.title}</a>`).join('')}
      <h6>Account</h6><a href="workspace">Workspace</a>${auth.isAdmin() ? '<a href="admin">Admin dashboard</a>' : ''}${user ? '<a href="#" data-signout>Sign out</a>' : '<a href="login">Sign in</a>'}
    </div>`;
    const canHover = () => matchMedia('(hover: hover)').matches;
    $$('#site-nav [data-drop]').forEach((d) => {
      const btn = d.querySelector('button');
      const hoverMenu = !d.classList.contains('user-menu');
      let closeTimer = null;
      const open = () => { clearTimeout(closeTimer); $$('[data-drop]').forEach((x) => x !== d && x.classList.remove('open')); d.classList.add('open'); };
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        // With hover menus the click should never close what hovering just opened.
        if (hoverMenu && canHover()) { open(); return; }
        if (d.classList.contains('open')) d.classList.remove('open'); else open();
      });
      d.querySelector('.nav-menu').addEventListener('click', (e) => e.stopPropagation());
      if (hoverMenu) {
        d.addEventListener('mouseenter', () => { if (canHover()) open(); });
        d.addEventListener('mouseleave', () => { if (canHover()) { clearTimeout(closeTimer); closeTimer = setTimeout(() => d.classList.remove('open'), 250); } });
      }
    });
    document.addEventListener('click', () => $$('[data-drop]').forEach((x) => x.classList.remove('open')));
    $$('[data-signout]').forEach((a) => a.addEventListener('click', (e) => { e.preventDefault(); auth.signOut(); }));
    const burger = $('.nav-burger');
    burger && burger.addEventListener('click', () => $('#mobile-menu').classList.toggle('open'));
    setStatus(status.state);
  }
  renderNav.refresh = renderNav;

  function renderFooter() {
    const host = document.getElementById('site-footer');
    if (!host) return;
    host.outerHTML = `<footer class="footer"><div class="container">
      <div class="footer-grid">
        <div><a class="brand" href="./">${LOGO}<span>Smile<b>BERT</b>a</span></a>
          <p class="muted" style="margin-top:14px;max-width:320px;font-size:.88rem">Open, AI-powered drug discovery tools — transformer models, physics-based docking and interactive 3D chemistry in the browser.</p></div>
        <div><h6>Tools</h6>${TOOLS.map((t) => `<a href="${t.href}">${t.title}</a>`).join('')}</div>
        <div><h6>Learn</h6>${LEARN.map((t) => `<a href="${t.href}">${t.title}</a>`).join('')}</div>
        <div><h6>Platform</h6><a href="workspace">Workspace</a><a href="login">Sign in</a><a href="./#about">About</a><a href="https://huggingface.co/NisargRhino" target="_blank" rel="noopener">Models on Hugging Face</a></div>
      </div>
      <div class="footer-bottom"><span>© ${new Date().getFullYear()} SmileBERTa. For research and education — not for clinical use.</span><span>Built with RDKit · PyTorch · AutoDock Vina · 3Dmol.js</span></div>
    </div></footer>`;
  }

  function addBackground() {
    if (document.body.dataset.bg === 'none') return;
    document.body.insertAdjacentHTML('afterbegin', '<div class="bg-grid"></div><div class="bg-glow g1"></div><div class="bg-glow g2"></div>');
  }

  function initReveal() {
    const els = $$('.reveal:not(.in)');
    if (!('IntersectionObserver' in window)) { els.forEach((e) => e.classList.add('in')); return; }
    const io = new IntersectionObserver((entries) => entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); } }), { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
    els.forEach((e) => io.observe(e));
  }

  // ── Toasts & modals ───────────────────────────────────────────────────────
  function toast(msg, type = 'info', ms = 3800) {
    let host = $('.toasts');
    if (!host) { host = h('<div class="toasts" role="status" aria-live="polite"></div>'); document.body.appendChild(host); }
    const ic = { ok: 'check', error: 'alert', info: 'info' }[type] || 'info';
    const el = h(`<div class="toast ${type}">${icon(ic, 'ti')}<div>${esc(msg)}</div></div>`);
    host.appendChild(el);
    setTimeout(() => { el.classList.add('out'); setTimeout(() => el.remove(), 320); }, ms);
  }

  function modal({ title, body = '', actions = [{ label: 'OK', value: true, primary: true }], wide = false, onOpen }) {
    return new Promise((resolve) => {
      const back = h(`<div class="modal-back"><div class="modal ${wide ? 'wide' : ''}" role="dialog" aria-modal="true">
        <div class="modal-head row-between"><h3>${esc(title)}</h3><button class="btn btn-ghost btn-icon" data-close aria-label="Close">${icon('x')}</button></div>
        <div class="modal-body"></div><div class="modal-foot"></div></div></div>`);
      const bodyEl = back.querySelector('.modal-body');
      if (typeof body === 'string') bodyEl.innerHTML = body; else bodyEl.appendChild(body);
      const foot = back.querySelector('.modal-foot');
      const close = (v) => { back.remove(); document.removeEventListener('keydown', onKey); resolve(v); };
      actions.forEach((a) => {
        const b = h(`<button class="btn ${a.primary ? 'btn-primary' : a.danger ? 'btn-danger' : ''}">${esc(a.label)}</button>`);
        b.addEventListener('click', () => close(typeof a.value === 'function' ? a.value(back) : a.value));
        foot.appendChild(b);
      });
      if (!actions.length) foot.remove();
      back.addEventListener('click', (e) => { if (e.target === back) close(null); });
      back.querySelector('[data-close]').addEventListener('click', () => close(null));
      const onKey = (e) => { if (e.key === 'Escape') close(null); };
      document.addEventListener('keydown', onKey);
      document.body.appendChild(back);
      onOpen && onOpen(back);
      const first = back.querySelector('input, textarea, select');
      first && setTimeout(() => first.focus(), 50);
    });
  }

  async function promptText(title, label, value = '') {
    return modal({
      title, body: `<label class="field"><span>${esc(label)}</span><input class="input" id="prompt-input" value="${esc(value)}" maxlength="120"></label>`,
      actions: [{ label: 'Cancel', value: null }, { label: 'Save', primary: true, value: (m) => m.querySelector('#prompt-input').value.trim() || null }],
      onOpen: (m) => m.querySelector('#prompt-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') m.querySelector('.btn-primary').click(); }),
    });
  }

  async function requireLoginModal(what = 'save your work') {
    const go = await modal({
      title: 'Sign in to continue',
      body: `<p class="text-2">Create a free account to ${esc(what)}. Everything you save is stored in your personal workspace and available on any device.</p>`,
      actions: [{ label: 'Not now', value: false }, { label: 'Sign in / Register', primary: true, value: true }],
    });
    if (go) location.href = auth.loginUrl();
    return false;
  }

  /** Save an item to the signed-in user's workspace. */
  async function saveItem(kind, defaultName, data, summary) {
    if (!auth.loggedIn()) return requireLoginModal('save ' + kind + 's to your workspace');
    const name = await promptText('Save to workspace', 'Name', defaultName);
    if (!name) return null;
    try {
      const r = await api('/items', { body: { kind, name, data, summary } });
      toast(`Saved “${name}” to your workspace`, 'ok');
      return r.id;
    } catch (e) { toast(e.message, 'error'); return null; }
  }
  async function loadItem(id) { return api('/items/' + encodeURIComponent(id)); }

  // ── Files ─────────────────────────────────────────────────────────────────
  function download(text, filename, mime = 'text/plain') {
    const blob = text instanceof Blob ? text : new Blob([text], { type: mime });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }
  async function copy(text) {
    try { await navigator.clipboard.writeText(text); toast('Copied to clipboard', 'ok', 1800); } catch (e) { toast('Copy failed', 'error'); }
  }
  function readFile(file) { return new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result); r.onerror = rej; r.readAsText(file); }); }
  function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/);
    if (!lines.length) return [];
    const header = lines[0].split(',').map((x) => x.trim().toLowerCase().replace(/"/g, ''));
    const idx = header.findIndex((x) => ['smiles', 'smile', 'canonical_smiles', 'structure'].includes(x));
    const out = [];
    for (let i = idx >= 0 ? 1 : 0; i < lines.length; i++) {
      if (!lines[i].trim()) continue;
      const cols = lines[i].split(',').map((c) => c.trim().replace(/^"|"$/g, ''));
      const s = idx >= 0 ? cols[idx] : cols[0];
      if (s && s.length > 1) out.push(s);
    }
    return out;
  }
  function toCSV(rows) { return rows.map((r) => r.map((c) => '"' + String(c ?? '').replace(/"/g, '""') + '"').join(',')).join('\n'); }
  function bindDropzone(el, onFile, accept) {
    const input = h(`<input type="file" hidden ${accept ? `accept="${accept}"` : ''}>`);
    el.appendChild(input);
    el.addEventListener('click', (e) => { if (e.target !== input) input.click(); });
    input.addEventListener('change', () => input.files[0] && onFile(input.files[0]));
    el.addEventListener('dragover', (e) => { e.preventDefault(); el.classList.add('drag'); });
    el.addEventListener('dragleave', () => el.classList.remove('drag'));
    el.addEventListener('drop', (e) => { e.preventDefault(); el.classList.remove('drag'); e.dataTransfer.files[0] && onFile(e.dataTransfer.files[0]); });
  }

  // ── Fragment library ──────────────────────────────────────────────────────
  const library = {
    async list() {
      if (auth.loggedIn()) { try { return (await api('/get_fragments')).fragments || []; } catch (e) { return []; } }
      return store.json('fragmentLibrary', []);
    },
    async add(smiles, props) {
      if (auth.loggedIn()) { await api('/save_fragment', { body: { smiles, props } }); return; }
      const lib = store.json('fragmentLibrary', []);
      if (lib.find((f) => f.smiles === smiles)) throw new Error('Fragment already in library');
      lib.unshift({ smiles, props: JSON.stringify(props || {}) });
      store.setJson('fragmentLibrary', lib);
    },
    async remove(smiles) {
      if (auth.loggedIn()) { await api('/delete_fragment', { method: 'DELETE', body: { smiles } }); return; }
      store.setJson('fragmentLibrary', store.json('fragmentLibrary', []).filter((f) => f.smiles !== smiles));
    },
  };

  // ── Hand-off between tools (e.g. "Dock this molecule") ────────────────────
  function handoff(target, payload) {
    try { sessionStorage.setItem('sb_handoff_' + target, JSON.stringify(payload)); } catch (e) { /* ignore */ }
    location.href = target;
  }
  function takeHandoff(target) {
    try { const v = sessionStorage.getItem('sb_handoff_' + target); sessionStorage.removeItem('sb_handoff_' + target); return v ? JSON.parse(v) : null; } catch (e) { return null; }
  }

  // ── Chemistry: 2D depiction via RDKit.js (WASM) with server fallback ──────
  const RDKIT_URL = 'https://unpkg.com/@rdkit/rdkit@2025.3.4-1.0.0/dist/';
  let rdkitPromise = null;
  function loadRDKit() {
    if (rdkitPromise) return rdkitPromise;
    rdkitPromise = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = RDKIT_URL + 'RDKit_minimal.js';
      s.onload = () => window.initRDKitModule({ locateFile: () => RDKIT_URL + 'RDKit_minimal.wasm' })
        .then((R) => { try { R.prefer_coordgen(true); } catch (e) { /* older builds */ } resolve(R); }).catch(reject);
      s.onerror = () => reject(new Error('RDKit.js failed to load'));
      document.head.appendChild(s);
    });
    rdkitPromise.catch(() => { rdkitPromise = null; });
    return rdkitPromise;
  }
  const DARK_PALETTE = { '-1': [0.85, 0.88, 0.94], 6: [0.85, 0.88, 0.94], 7: [0.45, 0.62, 1.0], 8: [1.0, 0.42, 0.42], 9: [0.45, 0.9, 0.55], 16: [1.0, 0.85, 0.3], 17: [0.4, 0.92, 0.5], 35: [0.85, 0.45, 0.3], 15: [1.0, 0.6, 0.2], 53: [0.75, 0.4, 0.95], 1: [0.75, 0.78, 0.84] };
  async function depictSVG(smiles, { width = 360, height = 260, highlight = null } = {}) {
    try {
      const R = await loadRDKit();
      const mol = R.get_mol(smiles);
      if (!mol || !mol.is_valid()) { mol && mol.delete(); throw new Error('invalid'); }
      const details = { width, height, bondLineWidth: 1.6, backgroundColour: [0, 0, 0, 0], atomColourPalette: DARK_PALETTE, fixedFontSize: 14, padding: 0.08, clearBackground: false };
      if (highlight) { details.atoms = highlight; details.highlightColour = [0.13, 0.83, 0.93]; }
      const svg = mol.get_svg_with_highlights(JSON.stringify(details));
      mol.delete();
      return svg;
    } catch (e) {
      const res = await api('/depict', { body: { smiles, width, height, dark: true }, raw: true });
      return res.text();
    }
  }
  async function depict(el, smiles, opts = {}) {
    if (!el) return;
    el.innerHTML = '<div class="skeleton" style="width:80%;height:70%;min-height:140px"></div>';
    try { el.innerHTML = await depictSVG(smiles, opts); } catch (e) { el.innerHTML = `<div class="empty">${icon('alert')}<div>Could not draw structure</div></div>`; }
  }
  async function validSmiles(smiles) {
    if (!smiles) return false;
    try { const R = await loadRDKit(); const m = R.get_mol(smiles); const ok = !!m && m.is_valid(); m && m.delete(); return ok; } catch (e) { return true; }
  }
  async function canonical(smiles) {
    try { const R = await loadRDKit(); const m = R.get_mol(smiles); if (!m) return smiles; const s = m.get_smiles(); m.delete(); return s; } catch (e) { return smiles; }
  }

  // ── 3D viewer helpers (3Dmol.js) ──────────────────────────────────────────
  function viewer3d(el, opts = {}) {
    if (!window.$3Dmol) throw new Error('3Dmol.js not loaded');
    el.innerHTML = '';
    return window.$3Dmol.createViewer(el, Object.assign({ backgroundColor: '#070a12', antialias: true, id: el.id ? el.id + '-gl' : undefined }, opts));
  }
  async function show3DFromSmiles(el, smiles, style = 'ballstick') {
    el.innerHTML = '<div class="empty"><span class="spinner"></span><div style="margin-top:10px">Generating 3D conformer…</div></div>';
    const data = await api('/get_3d_structure', { body: { smiles } });
    const v = viewer3d(el);
    v.addModel(data.sdf || data.pdb, data.sdf ? 'sdf' : 'pdb');
    applyMolStyle(v, style);
    v.zoomTo(); v.render(); v.zoom(0.9);
    return { viewer: v, data };
  }
  function applyMolStyle(v, style) {
    const s = {
      ballstick: { stick: { radius: 0.14, colorscheme: 'Jmol' }, sphere: { scale: 0.25, colorscheme: 'Jmol' } },
      stick: { stick: { radius: 0.18, colorscheme: 'Jmol' } },
      sphere: { sphere: { colorscheme: 'Jmol' } },
      line: { line: { colorscheme: 'Jmol' } },
    }[style] || { stick: {} };
    v.setStyle({}, s);
    v.render();
  }

  // ── Molecule property helpers (shared rendering) ──────────────────────────
  function propertyTiles(p, opts = {}) {
    if (!p) return '';
    const t = (v, k, unit = '', cls = '') => `<div class="stat ${cls}"><div class="v">${v}${unit ? `<small>${unit}</small>` : ''}</div><div class="k">${k}</div></div>`;
    const ro3 = opts.ruleOfThree;
    return `<div class="stat-grid">
      ${t(fmt(p.molecular_weight, 1), 'Molecular weight', 'Da', ro3 ? (p.molecular_weight <= 300 ? 'pass' : 'fail') : '')}
      ${t(fmt(p.log_p, 2), 'cLogP', '', ro3 ? (p.log_p <= 3 ? 'pass' : 'fail') : '')}
      ${t(p.hydrogen_bond_donors, 'H-bond donors', '', ro3 ? (p.hydrogen_bond_donors <= 3 ? 'pass' : 'fail') : '')}
      ${t(p.hydrogen_bond_acceptors, 'H-bond acceptors', '', ro3 ? (p.hydrogen_bond_acceptors <= 3 ? 'pass' : 'fail') : '')}
      ${t(fmt(p.tpsa, 1), 'TPSA', 'Å²')}
      ${p.rotatable_bonds !== undefined ? t(p.rotatable_bonds, 'Rotatable bonds', '', ro3 ? (p.rotatable_bonds <= 3 ? 'pass' : 'fail') : '') : ''}
      ${p.qed !== undefined && p.qed !== null ? t(fmt(p.qed, 2), 'QED score') : ''}
      ${p.heavy_atoms !== undefined ? t(p.heavy_atoms, 'Heavy atoms') : ''}
    </div>`;
  }
  function druglikeBadges(p) {
    if (!p) return '';
    const b = [];
    if (p.lipinski_pass !== undefined) b.push(`<span class="badge ${p.lipinski_pass ? 'green' : 'red'}">${icon(p.lipinski_pass ? 'check' : 'x')} Lipinski ${p.lipinski_violations} violation${p.lipinski_violations === 1 ? '' : 's'}</span>`);
    if (p.veber_pass !== undefined) b.push(`<span class="badge ${p.veber_pass ? 'green' : 'amber'}">${icon(p.veber_pass ? 'check' : 'alert')} Veber</span>`);
    if (p.qed !== undefined && p.qed !== null) b.push(`<span class="badge ${p.qed >= 0.5 ? 'cyan' : 'amber'}">QED ${fmt(p.qed, 2)}</span>`);
    if (p.formula) b.push(`<span class="badge">${esc(p.formula)}</span>`);
    return `<div class="row" style="gap:6px">${b.join('')}</div>`;
  }

  // ── Boot ──────────────────────────────────────────────────────────────────
  function boot() {
    // Clean addresses: "/builder" instead of "/builder.html", "/" instead of "/index.html"
    // (keeps any ?query and #section). The static host serves both forms.
    if (/\.html$/.test(location.pathname) && history.replaceState) {
      const clean = location.pathname.replace(/index\.html$/, '').replace(/\.html$/, '');
      history.replaceState(null, '', clean + location.search + location.hash);
    }
    addBackground();
    renderNav();
    renderFooter();
    initReveal();
    pingHealth();
    // Keep the admin flag in sync with the server (it is controlled by ADMIN_USERNAMES on the API).
    if (auth.loggedIn()) api('/me').then((me) => { if (!!me.is_admin !== auth.isAdmin()) { store.set('auth_admin', me.is_admin ? '1' : '0'); renderNav(); } }).catch(() => {});
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();

  window.SB = {
    API, DOCK_API, api, auth, store, icon, LOGO, TOOLS, LEARN, esc, $, $$, h, sleep, fmt, debounce, param,
    toast, modal, promptText, requireLoginModal, saveItem, loadItem, download, copy, readFile, parseCSV, toCSV, bindDropzone,
    library, handoff, takeHandoff, loadRDKit, depictSVG, depict, validSmiles, canonical, viewer3d, show3DFromSmiles, applyMolStyle,
    propertyTiles, druglikeBadges, onStatus, initReveal, status,
  };
})();
