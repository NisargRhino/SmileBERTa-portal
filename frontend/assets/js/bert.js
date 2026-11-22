/* Bert — the SmileBERTa research agent, built on Claude.
   Streams a turn over SSE: text deltas as they are written, plus a card for
   every tool call so the work is visible rather than implied. */
(function () {
  'use strict';

  document.addEventListener('DOMContentLoaded', () => {
    const { $, api, auth, esc, icon, store, API, toast } = SB;
    const log = $('#log'), input = $('#input'), form = $('#composer'), sendBtn = $('#send');

    $('#bert-mark').innerHTML = SB.LOGO;

    // Conversation as the API wants it: the full message list, resent each turn.
    let messages = [];
    let busy = false;
    let quota = null;

    const KEY_STORE = 'sb_anthropic_key';
    const userKey = () => { try { return localStorage.getItem(KEY_STORE) || ''; } catch (e) { return ''; } };

    // ── Markdown (small, deliberate subset) ────────────────────────────────
    function md(src) {
      const blocks = String(src).split(/\n{2,}/).map((block) => {
        const b = block.trim();
        if (!b) return '';
        if (/^\|.+\|/.test(b) && b.includes('\n')) return table(b);
        if (/^###\s+/.test(b)) return `<h3>${inline(b.replace(/^###\s+/, ''))}</h3>`;
        if (/^(\d+)[.)]\s+/m.test(b) && /^\d/.test(b)) {
          return `<ol>${b.split('\n').map((l) => `<li>${inline(l.replace(/^\s*\d+[.)]\s*/, ''))}</li>`).join('')}</ol>`;
        }
        if (/^[-*]\s+/m.test(b) && /^[-*]/.test(b)) {
          return `<ul>${b.split('\n').map((l) => `<li>${inline(l.replace(/^\s*[-*]\s*/, ''))}</li>`).join('')}</ul>`;
        }
        return `<p>${inline(b).replace(/\n/g, '<br>')}</p>`;
      });
      return blocks.join('');
    }
    function table(src) {
      const rows = src.split('\n').filter((r) => r.trim().startsWith('|'));
      if (rows.length < 2) return `<p>${inline(src)}</p>`;
      const cells = (r) => r.replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
      const head = cells(rows[0]);
      const body = rows.slice(2).map((r) => cells(r));
      return `<table><thead><tr>${head.map((h) => `<th>${inline(h)}</th>`).join('')}</tr></thead>`
        + `<tbody>${body.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
    }
    function inline(text) {
      return esc(text)
        .replace(/`([^`]+)`/g, (m, c) => `<code>${c}</code>`)
        .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
        .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
          (m, label, href) => `<a href="${esc(href)}" target="_blank" rel="noopener">${label}</a>`);
    }

    // ── Message bubbles ────────────────────────────────────────────────────
    function bubble(role) {
      const el = document.createElement('div');
      el.className = `msg ${role}`;
      el.innerHTML = `<div class="av">${role === 'user' ? 'You' : 'B'}</div><div class="body"></div>`;
      log.appendChild(el);
      scroll();
      return el.querySelector('.body');
    }
    function scroll() { log.scrollTop = log.scrollHeight; }
    const nearBottom = () => log.scrollHeight - log.scrollTop - log.clientHeight < 120;

    // ── Tool cards ─────────────────────────────────────────────────────────
    const TOOL_LABEL = {
      analyze_molecule: 'Analysing molecule', classify_drug: 'Classifying compound',
      predict_fragment: 'Predicting fragment', admet_profile: 'Profiling ADMET',
      link_fragments: 'Linking fragments', dock: 'Docking', inspect_structure: 'Reading structure',
      search_papers: 'Searching literature',
    };
    const TOOL_ICON = {
      analyze_molecule: 'atom', classify_drug: 'tag', predict_fragment: 'split',
      admet_profile: 'gauge', link_fragments: 'merge', dock: 'target',
      inspect_structure: 'helix', search_papers: 'book',
    };

    function argSummary(name, args) {
      if (!args) return '';
      if (name === 'dock') return `${args.ligand_smiles || ''} → ${args.pdb_id || ''}`;
      if (name === 'search_papers') return args.query || '';
      if (name === 'inspect_structure') return args.pdb_id || '';
      if (name === 'link_fragments') return `${args.fragment_a || ''} + ${args.fragment_b || ''}`;
      return args.smiles || '';
    }

    // One line a human can read without opening the JSON.
    function headline(name, result) {
      if (!result || typeof result !== 'object') return '';
      if (name === 'dock' && result.best_affinity_kcal_per_mol != null) {
        const pocket = result.pocket && result.pocket.from_ligand;
        return `Best pose <b>${result.best_affinity_kcal_per_mol} kcal/mol</b>`
          + (pocket ? ` in the ${esc(pocket)} site` : '')
          + (result.elapsed_seconds ? ` · ${result.elapsed_seconds}s` : '');
      }
      if (name === 'predict_fragment' && result.fragment_smiles) {
        return `Fragment <code>${esc(result.fragment_smiles)}</code>`
          + (result.was_corrected ? ' <span style="color:var(--muted)">(repaired)</span>' : '');
      }
      if (name === 'classify_drug' && result.predicted_class) {
        return `<b>${esc(result.predicted_class)}</b>`
          + (result.match_similarity != null ? ` · similarity ${result.match_similarity}` : '');
      }
      if (name === 'admet_profile') {
        return `Filters ${esc(result.filters_passed || '')} · ${result.alert_count || 0} structural alert(s)`;
      }
      if (name === 'analyze_molecule' && result.molecular_weight) {
        return `MW ${result.molecular_weight} · cLogP ${result.log_p} · QED ${result.qed}`;
      }
      if (name === 'link_fragments') return `${result.count || 0} candidate(s)`;
      if (name === 'search_papers') return `${(result.papers || []).length} paper(s) found`;
      if (name === 'inspect_structure') return `${(result.pockets || []).length} candidate pocket(s)`;
      return '';
    }

    function toolCard(host, id, name, args) {
      const el = document.createElement('details');
      el.className = 'tool running';
      el.dataset.id = id;
      el.innerHTML = `<summary><span class="ico">${icon(TOOL_ICON[name] || 'atom')}</span>`
        + `<span class="tname">${esc(TOOL_LABEL[name] || name)}</span>`
        + `<span class="targ">${esc(argSummary(name, args))}</span>`
        + '<span class="tstate">running…</span></summary>';
      host.appendChild(el);
      scroll();
      return el;
    }

    function finishCard(el, name, result, error) {
      if (!el) return;
      el.classList.remove('running');
      const state = el.querySelector('.tstate');
      if (error) {
        el.classList.add('failed');
        state.textContent = 'failed';
        el.insertAdjacentHTML('beforeend', `<div class="headline" style="color:#fca5a5">${esc(error)}</div>`);
        return;
      }
      state.textContent = 'done';
      const line = headline(name, result);
      if (line) el.insertAdjacentHTML('beforeend', `<div class="headline">${line}</div>`);
      el.insertAdjacentHTML('beforeend', `<pre>${esc(JSON.stringify(result, null, 2))}</pre>`);
    }

    // ── Quota + availability ───────────────────────────────────────────────
    async function loadStatus() {
      try {
        quota = await api('/bert/status', { auth: true });
      } catch (e) {
        quota = null;
      }
      renderNotice();
      renderQuota();
    }

    function renderQuota() {
      const el = $('#quota');
      if (!el) return;
      if (userKey()) {
        el.innerHTML = 'Using your own Anthropic key · <a href="#" id="clear-key">remove</a>';
        const clear = $('#clear-key');
        if (clear) {
          clear.addEventListener('click', (e) => {
            e.preventDefault();
            try { localStorage.removeItem(KEY_STORE); } catch (err) { /* ignore */ }
            toast('Key removed', 'ok');
            loadStatus();
          });
        }
        return;
      }
      if (!quota) { el.textContent = ''; return; }
      el.textContent = `${quota.remaining} of ${quota.limit} free messages left today`;
    }

    function renderNotice() {
      const host = $('#bert-notice');
      if (!host || !quota) return;
      if (!quota.sdk_installed) {
        host.innerHTML = '<div class="notice err">Bert is not installed on this server yet '
          + '(the Anthropic SDK is missing). Deploy the latest build to enable him.</div>';
        return;
      }
      if (!quota.available && !userKey()) {
        host.innerHTML = '<div class="notice warn"><b>Bert needs an API key.</b> '
          + 'This server has no Anthropic key set, so you can use your own. It is kept in this '
          + 'browser and sent only with your own requests — never stored on the server.'
          + '<div class="key-row"><input id="key-input" type="password" placeholder="sk-ant-..." '
          + 'autocomplete="off" spellcheck="false"><button class="btn" id="key-save">Use this key</button></div></div>';
        wireKey();
        return;
      }
      if (quota.remaining === 0 && !userKey()) {
        host.innerHTML = '<div class="notice warn"><b>You have used today\'s free messages.</b> '
          + 'They reset at midnight UTC. To keep going now, use your own Anthropic key — it stays in '
          + 'this browser.<div class="key-row"><input id="key-input" type="password" placeholder="sk-ant-..." '
          + 'autocomplete="off" spellcheck="false"><button class="btn" id="key-save">Use this key</button></div></div>';
        wireKey();
        return;
      }
      host.innerHTML = '';
    }

    function wireKey() {
      const save = $('#key-save'), field = $('#key-input');
      if (!save || !field) return;
      save.addEventListener('click', () => {
        const value = field.value.trim();
        if (!value.startsWith('sk-')) return toast('That does not look like an Anthropic key', 'err');
        try { localStorage.setItem(KEY_STORE, value); } catch (e) { return toast('Could not save the key', 'err'); }
        field.value = '';
        toast('Key saved in this browser', 'ok');
        loadStatus();
      });
    }

    // ── Starters ───────────────────────────────────────────────────────────
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

    function renderStarters() {
      if (messages.length) return;
      const host = document.createElement('div');
      host.className = 'msg bert';
      host.innerHTML = '<div class="av">B</div><div class="body">'
        + '<p>I am Bert, built on Claude. I can run SmileBERTa\'s tools for you — docking, '
        + 'fragment prediction, ADMET, classification — and chain them together in one request. '
        + 'Ask in plain language, or start here:</p><div class="starters"></div></div>';
      log.appendChild(host);
      const box = host.querySelector('.starters');
      STARTERS.forEach(([title, prompt]) => {
        const b = document.createElement('button');
        b.className = 'starter';
        b.type = 'button';
        b.innerHTML = `<b>${esc(title)}</b>${esc(prompt)}`;
        b.addEventListener('click', () => { input.value = prompt; input.focus(); autosize(); });
        box.appendChild(b);
      });
    }

    // ── Sending a turn ─────────────────────────────────────────────────────
    async function send(text) {
      if (busy || !text.trim()) return;
      busy = true;
      sendBtn.disabled = true;
      sendBtn.textContent = 'Working…';

      const starters = log.querySelector('.starters');
      if (starters) starters.closest('.msg').remove();

      md_append('user', text);
      messages.push({ role: 'user', content: text });

      const body = bubble('bert');
      const typing = document.createElement('div');
      typing.className = 'typing';
      typing.innerHTML = '<span></span><span></span><span></span>';
      body.appendChild(typing);

      let answer = '';
      let answerEl = null;
      const cards = {};

      try {
        const headers = { 'Content-Type': 'application/json', 'X-SmileBERTa-Client': clientId() };
        if (auth.token()) headers.Authorization = 'Bearer ' + auth.token();
        const key = userKey();
        if (key) headers['X-Anthropic-Key'] = key;

        const res = await fetch(API + '/bert/chat', {
          method: 'POST', headers, body: JSON.stringify({ messages }),
        });
        if (!res.ok || !res.body) {
          let msg = `Request failed (${res.status})`;
          try { msg = (await res.json()).error || msg; } catch (e) { /* non-JSON */ }
          throw new Error(msg);
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
            const evMatch = chunk.match(/^event:\s*(.+)$/m);
            const dataMatch = chunk.match(/^data:\s*(.+)$/m);
            if (!evMatch || !dataMatch) continue;
            let payload;
            try { payload = JSON.parse(dataMatch[1]); } catch (e) { continue; }
            const type = evMatch[1].trim();

            if (type === 'text') {
              typing.remove();
              answer += payload.text;
              if (!answerEl) { answerEl = document.createElement('div'); body.appendChild(answerEl); }
              answerEl.innerHTML = md(answer);
              if (nearBottom()) scroll();
            } else if (type === 'tool_start') {
              typing.remove();
              answerEl = null;              // later text starts a fresh block below the card
              cards[payload.id] = toolCard(body, payload.id, payload.name, payload.input);
            } else if (type === 'tool_end') {
              finishCard(cards[payload.id], payload.name, payload.result, null);
            } else if (type === 'tool_error') {
              finishCard(cards[payload.id], payload.name, null, payload.error);
            } else if (type === 'error') {
              typing.remove();
              const note = document.createElement('div');
              note.className = 'notice err';
              note.innerHTML = esc(payload.error) + (payload.hint ? `<br><br>${esc(payload.hint)}` : '');
              body.appendChild(note);
              if (payload.code === 'quota' || payload.code === 'config') loadStatus();
            } else if (type === 'done') {
              if (payload.remaining != null && quota) {
                quota.remaining = payload.remaining;
                renderQuota();
              }
            }
          }
        }

        if (answer.trim()) messages.push({ role: 'assistant', content: answer });
        else messages.pop();            // nothing came back; do not keep a dangling user turn
      } catch (e) {
        typing.remove();
        const note = document.createElement('div');
        note.className = 'notice err';
        note.textContent = e.message || 'Bert could not be reached.';
        body.appendChild(note);
        messages.pop();
      } finally {
        busy = false;
        sendBtn.disabled = false;
        sendBtn.textContent = 'Send';
        scroll();
      }
    }

    function md_append(role, text) {
      const body = bubble(role);
      body.innerHTML = md(text);
    }

    function clientId() {
      try {
        let id = localStorage.getItem('sb_client_id');
        if (!id) {
          const buf = new Uint8Array(16);
          crypto.getRandomValues(buf);
          id = Array.from(buf, (b) => b.toString(16).padStart(2, '0')).join('');
          localStorage.setItem('sb_client_id', id);
        }
        return id;
      } catch (e) { return ''; }
    }

    // ── Composer ───────────────────────────────────────────────────────────
    function autosize() {
      input.style.height = 'auto';
      input.style.height = Math.min(160, input.scrollHeight) + 'px';
    }
    input.addEventListener('input', autosize);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
    });
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const text = input.value;
      input.value = '';
      autosize();
      send(text);
    });

    renderStarters();
    loadStatus();
    input.focus();
  });
})();
