/* Bert — shared conversation engine.
 *
 * One implementation drives both surfaces: the launcher docked in the corner of
 * every page and the full page at /bert. They are the same conversation, not two
 * that happen to look alike — the transcript lives here and is mirrored into
 * sessionStorage, so expanding the panel into the page continues mid-thread and
 * walking to another tool keeps what was said.
 *
 * Two transcripts are kept deliberately:
 *   api  — exactly what the model is sent back each turn (role/content only).
 *   view — what the user sees, tool cards included, so a restored conversation
 *          still shows the work rather than just the prose.
 *
 * Exposes SB.BertChat.{create, state, reset, subscribe}.
 */
(function () {
  'use strict';

  const STORE_KEY = 'sb_bert_thread';
  const KEY_STORE = 'sb_anthropic_key';
  const MAX_RESTORE = 60;

  const state = { api: [], view: [], status: null, busy: false };
  const listeners = new Set();

  function notify() { listeners.forEach((fn) => { try { fn(state); } catch (e) { /* a bad listener must not break the chat */ } }); }
  function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }

  function persist() {
    try {
      sessionStorage.setItem(STORE_KEY, JSON.stringify({
        api: state.api.slice(-MAX_RESTORE), view: state.view.slice(-MAX_RESTORE),
      }));
    } catch (e) { /* private mode, or the thread outgrew the quota — not fatal */ }
  }
  function restore() {
    try {
      const raw = sessionStorage.getItem(STORE_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw);
      if (Array.isArray(saved.api) && Array.isArray(saved.view)) {
        state.api = saved.api; state.view = saved.view;
      }
    } catch (e) { /* unreadable thread: start fresh rather than fail to load */ }
  }
  restore();

  function reset() {
    state.api = []; state.view = [];
    try { sessionStorage.removeItem(STORE_KEY); } catch (e) { /* ignore */ }
    notify();
  }

  const userKey = () => { try { return localStorage.getItem(KEY_STORE) || ''; } catch (e) { return ''; } };
  const setUserKey = (v) => { try { v ? localStorage.setItem(KEY_STORE, v) : localStorage.removeItem(KEY_STORE); } catch (e) { /* ignore */ } };

  // ── Markdown (a deliberate subset: what Bert is told to produce) ───────────
  function inline(text) {
    return SB.esc(text)
      .replace(/`([^`]+)`/g, (m, c) => `<code>${c}</code>`)
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,;:!?]|$)/g, '$1<em>$2</em>')
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
        (m, label, href) => `<a href="${SB.esc(href)}" target="_blank" rel="noopener noreferrer">${label}</a>`)
      // Bare URLs, but not ones already inside an href we just built.
      .replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g,
        (m, pre, href) => `${pre}<a href="${SB.esc(href)}" target="_blank" rel="noopener noreferrer">${SB.esc(href)}</a>`);
  }

  function table(src) {
    const rows = src.split('\n').filter((r) => r.trim().startsWith('|'));
    if (rows.length < 2) return `<p>${inline(src)}</p>`;
    const cells = (r) => r.replace(/^\s*\||\|\s*$/g, '').split('|').map((c) => c.trim());
    const head = cells(rows[0]);
    const body = rows.slice(2).map(cells);
    return '<div class="bert-tablewrap"><table><thead><tr>'
      + head.map((c) => `<th>${inline(c)}</th>`).join('')
      + '</tr></thead><tbody>'
      + body.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`).join('')
      + '</tbody></table></div>';
  }

  function md(src) {
    return String(src).split(/\n{2,}/).map((block) => {
      const b = block.trim();
      if (!b) return '';
      if (/^\|/.test(b) && b.includes('\n')) return table(b);
      if (/^(---+|___+|\*\*\*+)$/.test(b)) return '<hr>';
      if (/^####\s+/.test(b)) return `<h4>${inline(b.replace(/^####\s+/, ''))}</h4>`;
      if (/^###\s+/.test(b)) return `<h3>${inline(b.replace(/^###\s+/, ''))}</h3>`;
      if (/^#{1,2}\s+/.test(b)) return `<h3>${inline(b.replace(/^#{1,2}\s+/, ''))}</h3>`;
      if (/^\d+[.)]\s+/.test(b)) {
        return `<ol>${b.split('\n').map((l) => `<li>${inline(l.replace(/^\s*\d+[.)]\s*/, ''))}</li>`).join('')}</ol>`;
      }
      if (/^[-*+]\s+/.test(b)) {
        return `<ul>${b.split('\n').map((l) => `<li>${inline(l.replace(/^\s*[-*+]\s*/, ''))}</li>`).join('')}</ul>`;
      }
      return `<p>${inline(b).replace(/\n/g, '<br>')}</p>`;
    }).join('');
  }

  // ── Tool presentation ──────────────────────────────────────────────────────
  const TOOL_LABEL = {
    analyze_molecule: 'Analysing molecule', classify_drug: 'Classifying compound',
    predict_fragment: 'Predicting fragment', admet_profile: 'Profiling ADMET',
    link_fragments: 'Linking fragments', dock: 'Docking', inspect_structure: 'Reading structure',
    search_papers: 'Searching literature',
  };
  const TOOL_ICON = {
    analyze_molecule: 'cube', classify_drug: 'layers', predict_fragment: 'sparkles',
    admet_profile: 'gauge', link_fragments: 'link', dock: 'zap', inspect_structure: 'search',
    search_papers: 'file',
  };
  const DONE_LABEL = {
    analyze_molecule: 'Analysed molecule', classify_drug: 'Classified compound',
    predict_fragment: 'Predicted fragment', admet_profile: 'Profiled ADMET',
    link_fragments: 'Linked fragments', dock: 'Docked', inspect_structure: 'Read structure',
    search_papers: 'Searched literature',
  };

  function argSummary(name, args) {
    if (!args) return '';
    if (name === 'dock') return `${args.ligand_smiles || ''} → ${args.pdb_id || ''}`;
    if (name === 'search_papers') return args.query || '';
    if (name === 'inspect_structure') return args.pdb_id || '';
    if (name === 'link_fragments') return `${args.fragment_a || ''} + ${args.fragment_b || ''}`;
    return args.smiles || '';
  }

  // One line a human can read without unfolding the JSON.
  function headline(name, r) {
    if (!r || typeof r !== 'object') return '';
    if (name === 'dock' && r.best_affinity_kcal_per_mol != null) {
      const pocket = r.pocket && r.pocket.from_ligand;
      return `Best pose <b>${SB.esc(String(r.best_affinity_kcal_per_mol))} kcal/mol</b>`
        + (pocket ? ` in the ${SB.esc(pocket)} site` : '')
        + (r.elapsed_seconds ? ` · ${SB.esc(String(r.elapsed_seconds))}s` : '');
    }
    if (name === 'predict_fragment' && r.fragment_smiles) {
      return `Fragment <code>${SB.esc(r.fragment_smiles)}</code>`
        + (r.was_corrected ? ' <span style="color:var(--muted)">(repaired)</span>' : '');
    }
    if (name === 'classify_drug' && r.predicted_class) {
      return `<b>${SB.esc(r.predicted_class)}</b>`
        + (r.match_similarity != null ? ` · similarity ${SB.esc(String(r.match_similarity))}` : '');
    }
    if (name === 'admet_profile') {
      return `Filters ${SB.esc(String(r.filters_passed || '—'))} · ${SB.esc(String(r.alert_count || 0))} structural alert(s)`;
    }
    if (name === 'analyze_molecule' && r.molecular_weight != null) {
      return `MW ${SB.esc(String(r.molecular_weight))} · cLogP ${SB.esc(String(r.log_p))} · QED ${SB.esc(String(r.qed))}`;
    }
    if (name === 'link_fragments') return `${SB.esc(String(r.count || 0))} candidate(s)`;
    if (name === 'search_papers') return `${(r.papers || []).length} paper(s) found`;
    if (name === 'inspect_structure') return `${(r.pockets || []).length} candidate pocket(s)`;
    return '';
  }

  const STARTERS = [
    ['Compare two drugs on one target',
     'Dock aspirin and ibuprofen into COX-2 (PDB 5IKR), compare the scores, then predict the binding fragment of whichever binds better.'],
    ['Profile a compound end to end',
     'Take imatinib: classify it, profile its ADMET, predict its binding fragment, and tell me what the numbers say about it as a drug.'],
    ['Find a lead from a fragment',
     'Predict the fragment of CC(C)(C)NCC(O)COc1ccc(O)c2c1CCC(=O)N2, link it with c1ccc(O)cc1, and profile the best candidate.'],
    ['Start from the literature',
     'What is known about KRAS G12C inhibitors? Find key papers, then dock sotorasib into 6OIM and tell me what the pose looks like.'],
  ];

  // ── A chat surface ─────────────────────────────────────────────────────────
  // host: element to build into. opts.variant: 'panel' | 'page'.
  function create(host, opts = {}) {
    const variant = opts.variant === 'page' ? 'page' : 'panel';
    const { esc, icon } = SB;

    host.innerHTML = `
      <div class="bert-chat is-${variant}">
        <div class="bert-log"></div>
        <div class="bert-foot">
          <form class="bert-composer">
            <textarea rows="1" placeholder="${variant === 'page'
              ? 'Ask for the work — e.g. dock aspirin and ibuprofen into 5IKR, then fragment the better one'
              : 'Ask Bert to run something…'}" autocomplete="off" spellcheck="false"></textarea>
            <button class="bert-send" type="submit" aria-label="Send">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
            </button>
          </form>
          <div class="bert-meta">
            <span class="bert-quota"></span>
            <span><a href="terms">Research use only</a> · predictions, not measurements</span>
          </div>
        </div>
      </div>`;

    const root = host.querySelector('.bert-chat');
    const log = host.querySelector('.bert-log');
    const form = host.querySelector('.bert-composer');
    const input = host.querySelector('textarea');
    const sendBtn = host.querySelector('.bert-send');
    const quotaEl = host.querySelector('.bert-quota');

    const atBottom = () => log.scrollHeight - log.scrollTop - log.clientHeight < 140;
    const scroll = () => { log.scrollTop = log.scrollHeight; };

    function bubble(role) {
      const el = document.createElement('div');
      el.className = `bert-msg ${role}`;
      const av = role === 'user'
        ? (SB.auth.username() ? esc(SB.auth.username()[0].toUpperCase()) : icon('user'))
        : icon('sparkles');
      el.innerHTML = `<div class="av">${av}</div><div class="body"></div>`;
      log.appendChild(el);
      return el.querySelector('.body');
    }

    function toolCard(body, name, args) {
      const el = document.createElement('details');
      el.className = 'bert-tool running';
      el.innerHTML = `<summary><span class="ico">${icon(TOOL_ICON[name] || 'cube')}</span>`
        + `<span class="tname">${esc(TOOL_LABEL[name] || name)}</span>`
        + `<span class="targ">${esc(argSummary(name, args))}</span>`
        + '<span class="tstate">running</span></summary>';
      body.appendChild(el);
      return el;
    }

    function finishCard(el, name, result, error) {
      if (!el) return;
      el.classList.remove('running');
      const tname = el.querySelector('.tname');
      const tstate = el.querySelector('.tstate');
      if (error) {
        el.classList.add('failed');
        tstate.textContent = 'failed';
        el.insertAdjacentHTML('beforeend', `<div class="headline" style="color:var(--red)">${esc(error)}</div>`);
        return;
      }
      if (tname) tname.textContent = DONE_LABEL[name] || tname.textContent;
      tstate.textContent = 'done';
      const line = headline(name, result);
      if (line) el.insertAdjacentHTML('beforeend', `<div class="headline">${line}</div>`);
      el.insertAdjacentHTML('beforeend', `<pre>${esc(JSON.stringify(result, null, 2))}</pre>`);
    }

    function noticeEl(cls, html) {
      const n = document.createElement('div');
      n.className = `bert-notice ${cls}`;
      n.innerHTML = html;
      return n;
    }

    // ── Rendering the stored transcript ──────────────────────────────────────
    function renderAll() {
      log.innerHTML = '';
      // A fresh conversation reads from the top; only a restored one jumps to the end.
      if (!state.view.length) { renderWelcome(); log.scrollTop = 0; return; }
      let body = null;
      let lastRole = null;
      state.view.forEach((entry) => {
        if (entry.type === 'user') { body = bubble('user'); body.innerHTML = md(entry.text); lastRole = 'user'; return; }
        if (lastRole !== 'assistant') { body = bubble('assistant'); lastRole = 'assistant'; }
        if (entry.type === 'text') {
          const d = document.createElement('div');
          d.innerHTML = md(entry.text);
          body.appendChild(d);
        } else if (entry.type === 'tool') {
          const card = toolCard(body, entry.name, entry.input);
          finishCard(card, entry.name, entry.result, entry.error);
        } else if (entry.type === 'error') {
          body.appendChild(noticeEl('err', esc(entry.text) + (entry.hint ? `<br><br>${esc(entry.hint)}` : '')));
        }
      });
      scroll();
    }

    function renderWelcome() {
      const body = bubble('assistant');
      body.innerHTML = `<p>I'm <strong>Bert</strong>, built on Claude. I run SmileBERTa's own tools — docking,
        fragment prediction, ADMET, classification, literature — and chain them together from one request.
        Ask in plain language, or start here:</p><div class="bert-starters"></div>`;
      const box = body.querySelector('.bert-starters');
      STARTERS.forEach(([title, prompt]) => {
        const b = document.createElement('button');
        b.className = 'bert-starter';
        b.type = 'button';
        b.innerHTML = `<b>${esc(title)}</b>${esc(prompt)}`;
        b.addEventListener('click', () => { input.value = prompt; input.focus(); autosize(); });
        box.appendChild(b);
      });
    }

    // ── Availability and quota ───────────────────────────────────────────────
    async function loadStatus() {
      try { state.status = await SB.api('/bert/status', { auth: true, timeout: 25000, retries: 2 }); }
      catch (e) { state.status = null; }
      renderQuota();
      renderGate();
      notify();
    }

    function renderQuota() {
      if (!quotaEl) return;
      if (userKey()) {
        quotaEl.innerHTML = 'Using your own key · <a href="#" data-clearkey>remove</a>';
        const a = quotaEl.querySelector('[data-clearkey]');
        a && a.addEventListener('click', (e) => {
          e.preventDefault(); setUserKey(''); SB.toast('Key removed', 'ok'); loadStatus();
        });
        return;
      }
      const s = state.status;
      quotaEl.textContent = s && s.limit != null ? `${s.remaining} of ${s.limit} free messages left today` : '';
    }

    // A gate notice sits at the top of the log when Bert cannot answer.
    function renderGate() {
      const existing = log.querySelector('[data-gate]');
      if (existing) existing.remove();
      const s = state.status;
      if (!s || userKey()) return;
      let html = '';
      if (!s.sdk_installed) {
        html = '<b>Bert is not installed on this server yet.</b> The Anthropic SDK is missing — deploy the latest build to enable him.';
      } else if (!s.available) {
        html = '<b>Bert needs an API key.</b> This server has no Anthropic key set, so you can use your own. '
          + 'It stays in this browser and is sent only with your own requests — never stored on the server.'
          + keyRow();
      } else if (s.remaining === 0) {
        html = "<b>You've used today's free messages.</b> They reset at midnight UTC. To keep going now, "
          + 'use your own Anthropic key — it stays in this browser.' + keyRow();
      }
      if (!html) return;
      const n = noticeEl(s.sdk_installed ? 'warn' : 'err', html);
      n.dataset.gate = '1';
      log.prepend(n);
      wireKey(n);
    }

    const keyRow = () => '<div class="bert-keyrow"><input type="password" placeholder="sk-ant-..." '
      + 'autocomplete="off" spellcheck="false" data-key><button class="btn btn-sm" data-keysave>Use this key</button></div>';

    function wireKey(scope) {
      const save = scope.querySelector('[data-keysave]');
      const field = scope.querySelector('[data-key]');
      if (!save || !field) return;
      const commit = () => {
        const v = field.value.trim();
        if (!v.startsWith('sk-')) return SB.toast('That does not look like an Anthropic key', 'err');
        setUserKey(v);
        field.value = '';
        SB.toast('Key saved in this browser', 'ok');
        loadStatus();
      };
      save.addEventListener('click', commit);
      field.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); commit(); } });
    }

    // ── Sending ──────────────────────────────────────────────────────────────
    async function send(text) {
      text = (text || '').trim();
      if (state.busy || !text) return;
      state.busy = true;
      sendBtn.disabled = true;
      sendBtn.innerHTML = '<span class="spinner"></span>';
      notify();

      const welcome = log.querySelector('.bert-starters');
      if (welcome) welcome.closest('.bert-msg').remove();
      const gate = log.querySelector('[data-gate]');
      if (gate) gate.remove();

      bubble('user').innerHTML = md(text);
      state.api.push({ role: 'user', content: text });
      state.view.push({ type: 'user', text });
      scroll();

      const body = bubble('assistant');
      const typing = document.createElement('div');
      typing.className = 'bert-typing';
      typing.innerHTML = '<i></i><i></i><i></i>';
      body.appendChild(typing);
      scroll();

      // What this turn produced, appended to the view transcript as it arrives.
      const produced = [];
      const cards = {};
      let answer = '';
      let answerEl = null;
      let failed = false;

      try {
        const headers = { 'Content-Type': 'application/json', 'X-SmileBERTa-Client': SB.clientId() };
        if (SB.auth.token()) headers.Authorization = 'Bearer ' + SB.auth.token();
        const key = userKey();
        if (key) headers['X-Anthropic-Key'] = key;

        const res = await fetch(SB.API + '/bert/chat', {
          method: 'POST', headers, body: JSON.stringify({ messages: state.api }),
        });
        if (!res.ok || !res.body) {
          let m = `Request failed (${res.status})`;
          try { m = (await res.json()).error || m; } catch (e) { /* not JSON */ }
          throw new Error(m);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const chunks = buffer.split('\n\n');
          buffer = chunks.pop();
          for (const chunk of chunks) {
            const ev = chunk.match(/^event:\s*(.+)$/m);
            const dt = chunk.match(/^data:\s*(.+)$/m);
            if (!ev || !dt) continue;
            let p;
            try { p = JSON.parse(dt[1]); } catch (e) { continue; }
            const type = ev[1].trim();
            const stick = atBottom();

            if (type === 'text') {
              typing.remove();
              answer += p.text;
              if (!answerEl) { answerEl = document.createElement('div'); body.appendChild(answerEl); }
              answerEl.innerHTML = md(answer);
            } else if (type === 'tool_start') {
              typing.remove();
              if (answer.trim()) produced.push({ type: 'text', text: answer });
              answer = ''; answerEl = null;    // text after a card starts a new block
              cards[p.id] = toolCard(body, p.name, p.input);
              produced.push({ type: 'tool', name: p.name, input: p.input, result: null, error: null });
              cards[p.id]._entry = produced[produced.length - 1];
            } else if (type === 'tool_end') {
              finishCard(cards[p.id], p.name, p.result, null);
              if (cards[p.id] && cards[p.id]._entry) cards[p.id]._entry.result = p.result;
            } else if (type === 'tool_error') {
              finishCard(cards[p.id], p.name, null, p.error);
              if (cards[p.id] && cards[p.id]._entry) cards[p.id]._entry.error = p.error;
            } else if (type === 'error') {
              failed = true;
              typing.remove();
              body.appendChild(noticeEl('err', esc(p.error) + (p.hint ? `<br><br>${esc(p.hint)}` : '')));
              produced.push({ type: 'error', text: p.error, hint: p.hint });
              if (p.code === 'quota' || p.code === 'config') loadStatus();
            } else if (type === 'done') {
              if (p.remaining != null && state.status) { state.status.remaining = p.remaining; renderQuota(); }
            }
            if (stick) scroll();
          }
        }

        if (answer.trim()) produced.push({ type: 'text', text: answer });

        if (answer.trim()) {
          state.api.push({ role: 'assistant', content: answer });
          state.view.push(...produced);
        } else if (failed) {
          // Keep the failure visible, but do not send a dangling user turn again.
          state.api.pop();
          state.view.push(...produced);
        } else {
          state.api.pop();
          state.view.pop();
        }
      } catch (e) {
        typing.remove();
        const m = e.message || 'Bert could not be reached.';
        body.appendChild(noticeEl('err', esc(m)));
        state.api.pop();
        state.view.push({ type: 'error', text: m });
      } finally {
        state.busy = false;
        sendBtn.disabled = false;
        sendBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>';
        persist();
        notify();
        scroll();
      }
    }

    function autosize() {
      input.style.height = 'auto';
      input.style.height = Math.min(150, input.scrollHeight) + 'px';
    }
    input.addEventListener('input', autosize);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
    });
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const t = input.value;
      input.value = '';
      autosize();
      send(t);
    });

    renderAll();
    loadStatus();

    return {
      root, send, focus: () => input.focus(),
      refresh: () => { renderAll(); renderGate(); },
      clear: () => { reset(); renderAll(); renderGate(); input.focus(); },
      prefill: (t) => { input.value = t; autosize(); input.focus(); },
    };
  }

  SB.BertChat = { create, state, reset, subscribe, md, STARTERS };
})();
