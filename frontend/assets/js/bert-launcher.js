/* Bert — the docked launcher.
 *
 * Injected by core.js on every page except /bert itself, so the agent is
 * reachable from wherever the user already is rather than being a destination
 * they have to leave their work for. Expanding hands the thread to the full
 * page, which picks it up mid-conversation from sessionStorage.
 */
(function () {
  'use strict';

  function boot() {
    if (!window.SB || !SB.BertChat) return;
    // The full page is the same conversation; a bubble on top of it is noise.
    const page = location.pathname.split('/').pop().replace(/\.html$/, '');
    if (page === 'bert') return;
    if (document.getElementById('bert-fab')) return;

    const { icon } = SB;
    const OPEN_KEY = 'sb_bert_open';

    const fab = document.createElement('button');
    fab.id = 'bert-fab';
    fab.className = 'bert-fab';
    fab.type = 'button';
    fab.setAttribute('aria-label', 'Ask Bert, the research agent');
    fab.innerHTML = `${icon('sparkles')}<span class="dot"></span>`;

    const panel = document.createElement('div');
    panel.id = 'bert-panel';
    panel.className = 'bert-panel closed';
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-label', 'Bert, the research agent');
    panel.innerHTML = `
      <div class="bert-head">
        <div class="mark">${icon('sparkles')}</div>
        <div class="who"><div class="n">Bert</div><div class="s">Research agent · built on Claude</div></div>
        <div class="acts">
          <button class="bert-iconbtn" type="button" data-new title="New conversation">${icon('plus')}</button>
          <button class="bert-iconbtn" type="button" data-expand title="Open full page">${icon('expand')}</button>
          <button class="bert-iconbtn" type="button" data-close title="Close">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>
          </button>
        </div>
      </div>
      <div class="bert-host" style="display:flex;flex-direction:column;flex:1;min-height:0"></div>`;

    document.body.appendChild(fab);
    document.body.appendChild(panel);

    let chat = null;
    let open = false;

    function setOpen(next) {
      open = next;
      panel.classList.toggle('closed', !open);
      fab.classList.toggle('hidden', open);
      try { sessionStorage.setItem(OPEN_KEY, open ? '1' : '0'); } catch (e) { /* ignore */ }
      if (open) {
        fab.classList.remove('unread');
        if (!chat) chat = SB.BertChat.create(panel.querySelector('.bert-host'), { variant: 'panel' });
        else chat.refresh();
        setTimeout(() => chat.focus(), 60);
      }
    }

    fab.addEventListener('click', () => setOpen(true));
    panel.querySelector('[data-close]').addEventListener('click', () => setOpen(false));
    panel.querySelector('[data-new]').addEventListener('click', () => chat && chat.clear());
    panel.querySelector('[data-expand]').addEventListener('click', () => { location.href = 'bert'; });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && open && !SB.BertChat.state.busy) setOpen(false);
    });

    // Reopen across navigation so a thread in progress follows the user around.
    let wasOpen = false;
    try { wasOpen = sessionStorage.getItem(OPEN_KEY) === '1'; } catch (e) { /* ignore */ }
    if (wasOpen) setOpen(true);

    // Let any page hand a question straight to Bert: SB.askBert('dock ...').
    SB.askBert = (text) => {
      setOpen(true);
      setTimeout(() => { if (text) chat.prefill(text); }, 80);
    };
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
