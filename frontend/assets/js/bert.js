/* Bert — the full page.
 *
 * The conversation itself lives in bert-chat.js and is shared with the docked
 * launcher, so arriving here from the corner panel continues the same thread
 * rather than starting a new one. This file is only the page around it.
 */
(function () {
  'use strict';

  const CAPS = [
    ['zap', 'Docking', 'AutoDock Vina against any PDB structure'],
    ['sparkles', 'Fragment prediction', "SmileBERTa's own transformer"],
    ['layers', 'Drug classification', 'Therapeutic class from structure'],
    ['gauge', 'ADMET', 'Absorption, solubility, alerts'],
    ['link', 'Fragment linking', 'Join fragments into leads'],
    ['cube', 'Properties', 'RDKit analysis and drug-likeness'],
    ['file', 'Literature', 'Europe PMC — PubMed and preprints'],
  ];

  function start() {
    const { $, icon } = SB;
    if (!SB.BertChat) return false;       // engine not in yet; caller retries

    $('#hero-mark').innerHTML = icon('sparkles');
    $('#head-mark').innerHTML = icon('sparkles');
    $('#new-chat').innerHTML = icon('plus');
    $('#built-on').insertAdjacentHTML('afterbegin', icon('sparkles'));

    $('#caps').innerHTML = CAPS.map(([ic, title, desc]) =>
      `<div class="cap"><span class="ci">${icon(ic)}</span><span><b>${SB.esc(title)}</b><span>${SB.esc(desc)}</span></span></div>`).join('');

    const chat = SB.BertChat.create($('#chat-host'), { variant: 'page' });
    $('#new-chat').addEventListener('click', () => chat.clear());
    chat.focus();
    return true;
  }

  function boot() {
    // core.js injects the engine, so it may land a tick after this file runs.
    if (start()) return;
    let tries = 0;
    const timer = setInterval(() => {
      if (start() || ++tries > 60) clearInterval(timer);
    }, 50);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
