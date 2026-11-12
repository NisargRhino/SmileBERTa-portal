/* Model Hub: browse Hugging Face drug-discovery models, generate a runnable notebook. */
(function () {
  const { $, $$, api, esc, toast, copy, download, debounce } = SB;

  let state = { models: [], selected: null, task: 'predict', cat: 'molecular-property', nb: null, name: '' };

  const needsLabel = (t) => t === 'finetune' || t === 'embed';

  async function boot() {
    let meta;
    try {
      meta = await api('/hf/categories');
    } catch (e) {
      $('#list').innerHTML = `<div class="muted" style="padding:20px;text-align:center">Could not reach the API.<br><small>${esc(e.message)}</small></div>`;
      return;
    }

    $('#cats').innerHTML = meta.categories
      .map((c) => `<button class="chip${c.id === state.cat ? ' active' : ''}" data-cat="${esc(c.id)}" title="${esc(c.blurb)}">${esc(c.label)}</button>`)
      .join('');
    $$('#cats .chip').forEach((b) => b.addEventListener('click', () => {
      state.cat = b.dataset.cat;
      $('#q').value = '';
      $$('#cats .chip').forEach((x) => x.classList.toggle('active', x === b));
      loadModels();
    }));

    const tasks = Array.isArray(meta.tasks)
      ? meta.tasks
      : Object.entries(meta.tasks).map(([id, label]) => ({ id, label }));
    $('#tasks').innerHTML = tasks.map(({ id, label }) => `
      <label class="task-opt${id === state.task ? ' sel' : ''}" data-task="${esc(id)}">
        <input type="radio" name="task" value="${esc(id)}"${id === state.task ? ' checked' : ''}>
        <span>${esc(label)}</span>
      </label>`).join('');
    $$('.task-opt').forEach((el) => el.addEventListener('click', () => {
      state.task = el.dataset.task;
      $$('.task-opt').forEach((x) => x.classList.toggle('sel', x === el));
      el.querySelector('input').checked = true;
      $('#labelwrap').style.display = needsLabel(state.task) ? '' : 'none';
      refreshHint();
    }));
    $('#labelwrap').style.display = needsLabel(state.task) ? '' : 'none';

    $('#q').addEventListener('input', debounce(() => loadModels($('#q').value.trim()), 400));
    $('#gen').addEventListener('click', generate);
    $('#dl').addEventListener('click', downloadNb);
    $('#copy').addEventListener('click', () => copy(JSON.stringify(state.nb, null, 1)));

    loadModels();
  }

  async function loadModels(q = '') {
    $('#list').innerHTML = '<div class="muted" style="padding:20px;text-align:center">Searching…</div>';
    let models = [];
    try {
      const params = q ? `q=${encodeURIComponent(q)}` : `category=${encodeURIComponent(state.cat)}`;
      ({ models } = await api(`/hf/models?${params}&limit=24`));
    } catch (e) {
      $('#list').innerHTML = `<div class="muted" style="padding:20px;text-align:center">Search failed.<br><small>${esc(e.message)}</small></div>`;
      return;
    }
    state.models = models;
    if (!models.length) {
      $('#list').innerHTML = '<div class="muted" style="padding:20px;text-align:center">No models found. Try a different search.</div>';
      return;
    }
    $('#list').innerHTML = models.map((m) => `
      <div class="model-card" data-id="${esc(m.id)}">
        <div class="model-id">${esc(m.id)}</div>
        <div class="model-meta">
          <span>↓ ${m.downloads.toLocaleString()}</span>
          <span>♥ ${m.likes.toLocaleString()}</span>
          ${m.pipeline_tag ? `<span>${esc(m.pipeline_tag)}</span>` : ''}
        </div>
        ${m.tags.length ? `<div class="tagline">${m.tags.slice(0, 5).map((t) => `<span class="minitag">${esc(t)}</span>`).join('')}</div>` : ''}
      </div>`).join('');
    $$('.model-card').forEach((c) => c.addEventListener('click', () => {
      state.selected = c.dataset.id;
      $$('.model-card').forEach((x) => x.classList.toggle('sel', x === c));
      refreshHint();
    }));
  }

  function refreshHint() {
    $('#hint').innerHTML = state.selected
      ? `Ready: <span class="mono">${esc(state.selected)}</span> · ${esc(state.task)}`
      : 'Select a model on the left to begin.';
  }

  async function generate() {
    if (!state.selected && state.task !== 'descriptors') {
      toast('Pick a model first', 'error');
      return;
    }
    const btn = $('#gen');
    btn.disabled = true;
    btn.textContent = 'Generating…';
    try {
      const res = await api('/notebook', {
        method: 'POST',
        body: {
          model_id: state.selected || '',
          task: state.task,
          smiles_col: $('#smilescol').value.trim() || 'smiles',
          label_col: $('#labelcol').value.trim() || 'label',
          dataset: $('#dataset').value.trim() || 'your_data.csv',
        },
      });
      state.nb = res.notebook;
      state.name = res.filename;
      renderPreview();
      $('#dl').disabled = false;
      $('#copy').disabled = false;
      toast('Notebook ready', 'ok');
    } catch (e) {
      toast(e.message, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Generate notebook';
    }
  }

  function renderPreview() {
    $('#nbname').textContent = state.name;
    $('#preview').innerHTML = state.nb.cells.map((c) => {
      const src = (Array.isArray(c.source) ? c.source.join('\n') : c.source).trim();
      const md = c.cell_type === 'markdown';
      return `<div class="nb-cell${md ? ' md' : ''}">
        <div class="nb-kind">${md ? 'Markdown' : 'Code'}</div>
        <pre>${esc(src)}</pre>
      </div>`;
    }).join('');
    $('#previewPanel').hidden = false;
  }

  function downloadNb() {
    download(JSON.stringify(state.nb, null, 1), state.name, 'application/x-ipynb+json');
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
