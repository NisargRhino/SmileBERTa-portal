/* Credits & citations. The data below is the single source of truth for what
   SmileBERTa depends on; keep it in step with requirements-render.txt,
   requirements-dock.txt and the CDN tags in the HTML pages. */
(function () {
  const { $, $$, esc, icon, copy, download } = SB;

  // Licence strings are taken from each project's own metadata, not from memory.
  const SECTIONS = [
    {
      id: 'core',
      title: 'Core scientific software',
      blurb: 'The tools that do the actual chemistry. If you publish work that used SmileBERTa, these are the ones to cite — SmileBERTa runs them, it does not replace them.',
      items: [
        {
          name: 'RDKit', version: '2023.09.6', licence: 'BSD-3-Clause',
          used: 'Descriptors, 2D depiction, 3D embedding (ETKDG), MMFF94/UFF optimisation, conformer search, PAINS/Brenk/NIH alert catalogues, QED, and the synthetic-accessibility and natural-product scorers.',
          cite: 'RDKit: Open-source cheminformatics.', url: 'https://www.rdkit.org',
        },
        {
          name: 'AutoDock Vina', version: '1.2.7', licence: 'Apache-2.0',
          used: 'All protein–ligand docking, scoring and pose generation.',
          cite: 'Eberhardt, J.; Santos-Martins, D.; Tillack, A. F.; Forli, S. AutoDock Vina 1.2.0: New Docking Methods, Expanded Force Field, and Python Bindings. J. Chem. Inf. Model. 2021, 61, 3891–3898.',
          doi: '10.1021/acs.jcim.1c00203',
          also: {
            cite: 'Trott, O.; Olson, A. J. AutoDock Vina: Improving the Speed and Accuracy of Docking with a New Scoring Function, Efficient Optimization, and Multithreading. J. Comput. Chem. 2010, 31, 455–461.',
            doi: '10.1002/jcc.21334',
            note: 'The original Vina paper. The authors ask that both be cited.',
          },
        },
        {
          name: 'Open Babel', version: '3.1.1', licence: 'GPL-2.0',
          used: 'Receptor and ligand preparation — protonation, partial charges and conversion to PDBQT before docking.',
          cite: "O'Boyle, N. M.; Banck, M.; James, C. A.; Morley, C.; Vandermeersch, T.; Hutchison, G. R. Open Babel: An Open Chemical Toolbox. J. Cheminform. 2011, 3, 33.",
          doi: '10.1186/1758-2946-3-33',
        },
        {
          name: '3Dmol.js', version: '2.5.2', licence: 'BSD-3-Clause',
          used: 'Every 3D molecular view in the browser — the builder, docking poses, Structure Studio and the homepage.',
          cite: 'Rego, N.; Koes, D. 3Dmol.js: Molecular Visualization with WebGL. Bioinformatics 2015, 31, 1322–1324.',
          doi: '10.1093/bioinformatics/btu829',
        },
        {
          name: 'RDKit.js', version: '2025.3.4', licence: 'BSD-3-Clause',
          used: 'Client-side 2D structure drawing and SMILES validation, so depiction works without a round trip to the server.',
          cite: 'RDKit MinimalLib (JavaScript/WebAssembly build of RDKit).',
          url: 'https://github.com/rdkit/rdkit-js',
        },
        {
          name: 'PyTorch', version: '2.4.1', licence: 'BSD-3-Clause',
          used: 'Inference for both transformer models.',
          cite: 'Paszke, A.; et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems 32, 2019, 8024–8035.',
          url: 'https://pytorch.org',
        },
        {
          name: 'Hugging Face Transformers', version: '4.43.3', licence: 'Apache-2.0',
          used: 'Loading and running the RoBERTa models; the Model Hub also reads the public Hugging Face model API.',
          cite: 'Wolf, T.; et al. Transformers: State-of-the-Art Natural Language Processing. Proc. EMNLP 2020: System Demonstrations, 38–45.',
          doi: '10.18653/v1/2020.emnlp-demos.6',
        },
      ],
    },
    {
      id: 'models',
      title: 'Models',
      blurb: 'The fragment-prediction and drug-classification models are SmileBERTa’s own work. Everything they are built on is credited here.',
      items: [
        {
          name: 'SmileBERTa fragment prediction', version: 'NisargRhino/protein-models', licence: 'See model card',
          used: 'Predicts the binding-relevant fragment of a drug-like molecule. Trained by Nisarg Shah (Emory University).',
          cite: 'Shah, N. SmileBERTa fragment prediction model.',
          url: 'https://huggingface.co/NisargRhino/protein-models',
        },
        {
          name: 'SmileBERTa drug classification', version: 'NisargRhino/drug-classification', licence: 'See model card',
          used: 'Predicts therapeutic class from structure. Trained by Nisarg Shah (Emory University).',
          cite: 'Shah, N. SmileBERTa drug classification model.',
          url: 'https://huggingface.co/NisargRhino/drug-classification',
        },
        {
          name: 'RoBERTa', version: 'architecture', licence: 'MIT (reference implementation)',
          used: 'The masked-language-model architecture both SmileBERTa models are built on.',
          cite: 'Liu, Y.; et al. RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692, 2019.',
          url: 'https://arxiv.org/abs/1907.11692',
        },
        {
          name: 'ChemBERTa', version: 'seyonec / DeepChem', licence: 'MIT',
          used: 'Offered as a featured model in the Model Hub and used in generated notebooks.',
          cite: 'Chithrananda, S.; Grand, G.; Ramsundar, B. ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction. arXiv:2010.09885, 2020.',
          url: 'https://arxiv.org/abs/2010.09885',
        },
      ],
    },
    {
      id: 'data',
      title: 'Data sources',
      items: [
        {
          name: 'RCSB Protein Data Bank', licence: 'CC0 1.0 (structure data)',
          used: 'Every protein structure loaded by Docking and Structure Studio is fetched live from RCSB.',
          cite: 'Berman, H. M.; Westbrook, J.; Feng, Z.; Gilliland, G.; Bhat, T. N.; Weissig, H.; Shindyalov, I. N.; Bourne, P. E. The Protein Data Bank. Nucleic Acids Res. 2000, 28, 235–242.',
          doi: '10.1093/nar/28.1.235',
          note: 'If you use a specific structure, cite its own primary publication as well as the PDB.',
        },
        {
          name: 'Hugging Face Hub', licence: 'Per-model',
          used: 'The Model Hub searches the public Hugging Face model API. Each model carries its own licence — check it before using a model in your own work.',
          cite: 'Hugging Face Hub.', url: 'https://huggingface.co',
        },
      ],
    },
    {
      id: 'methods',
      title: 'Methods behind the ADMET Profiler',
      blurb: 'Every value on the ADMET page comes from one of these published models, and the page names the source next to each number. SmileBERTa implements them; the science is theirs.',
      compact: true,
      items: [
        { name: 'Water solubility (ESOL)', cite: 'Delaney, J. S. ESOL: Estimating Aqueous Solubility Directly from Molecular Structure. J. Chem. Inf. Comput. Sci. 2004, 44, 1000–1005.', doi: '10.1021/ci034243x' },
        { name: 'GI absorption and brain penetration (BOILED-Egg)', cite: 'Daina, A.; Zoete, V. A BOILED-Egg to Predict Gastrointestinal Absorption and Brain Penetration of Small Molecules. ChemMedChem 2016, 11, 1117–1121.', doi: '10.1002/cmdc.201600182' },
        { name: 'Rule of five', cite: 'Lipinski, C. A.; Lombardo, F.; Dominy, B. W.; Feeney, P. J. Experimental and Computational Approaches to Estimate Solubility and Permeability in Drug Discovery and Development Settings. Adv. Drug Deliv. Rev. 2001, 46, 3–26.', doi: '10.1016/S0169-409X(00)00129-0' },
        { name: 'Ghose qualifying range', cite: 'Ghose, A. K.; Viswanadhan, V. N.; Wendoloski, J. J. A Knowledge-Based Approach in Designing Combinatorial or Medicinal Chemistry Libraries for Drug Discovery. J. Comb. Chem. 1999, 1, 55–68.', doi: '10.1021/cc9800071' },
        { name: 'Veber rules', cite: 'Veber, D. F.; Johnson, S. R.; Cheng, H.-Y.; Smith, B. R.; Ward, K. W.; Kopple, K. D. Molecular Properties That Influence the Oral Bioavailability of Drug Candidates. J. Med. Chem. 2002, 45, 2615–2623.', doi: '10.1021/jm020017n' },
        { name: 'Egan egg', cite: 'Egan, W. J.; Merz, K. M.; Baldwin, J. J. Prediction of Drug Absorption Using Multivariate Statistics. J. Med. Chem. 2000, 43, 3867–3877.', doi: '10.1021/jm000292e' },
        { name: 'Muegge filter', cite: 'Muegge, I.; Heald, S. L.; Brittelli, D. Simple Selection Criteria for Drug-like Chemical Matter. J. Med. Chem. 2001, 44, 1841–1846.', doi: '10.1021/jm015507e' },
        { name: 'Lead-likeness', cite: 'Teague, S. J.; Davis, A. M.; Leeson, P. D.; Oprea, T. The Design of Leadlike Combinatorial Libraries. Angew. Chem. Int. Ed. 1999, 38, 3743–3748.', doi: '10.1002/(SICI)1521-3773(19991216)38:24<3743::AID-ANIE3743>3.0.CO;2-U' },
        { name: 'Rule of three (fragments)', cite: 'Congreve, M.; Carr, R.; Murray, C.; Jhoti, H. A Rule of Three for Fragment-Based Lead Discovery? Drug Discov. Today 2003, 8, 876–877.', doi: '10.1016/S1359-6446(03)02831-9' },
        { name: 'Synthetic accessibility', cite: 'Ertl, P.; Schuffenhauer, A. Estimation of Synthetic Accessibility Score of Drug-like Molecules Based on Molecular Complexity and Fragment Contributions. J. Cheminform. 2009, 1, 8.', doi: '10.1186/1758-2946-1-8' },
        { name: 'Natural-product likeness', cite: 'Ertl, P.; Roggo, S.; Schuffenhauer, A. Natural Product-likeness Score and Its Application for Prioritization of Compound Libraries. J. Chem. Inf. Model. 2008, 48, 68–74.', doi: '10.1021/ci700286x' },
        { name: 'QED', cite: 'Bickerton, G. R.; Paolini, G. V.; Besnard, J.; Muresan, S.; Hopkins, A. L. Quantifying the Chemical Beauty of Drugs. Nat. Chem. 2012, 4, 90–98.', doi: '10.1038/nchem.1243' },
        { name: 'PAINS alerts', cite: 'Baell, J. B.; Holloway, G. A. New Substructure Filters for Removal of Pan Assay Interference Compounds (PAINS) from Screening Libraries. J. Med. Chem. 2010, 53, 2719–2740.', doi: '10.1021/jm901137j' },
        { name: 'Brenk alerts', cite: 'Brenk, R.; Schipani, A.; James, D.; Krasowski, A.; Gilbert, I. H.; Frearson, J.; Wyatt, P. G. Lessons Learnt from Assembling Screening Libraries for Drug Discovery for Neglected Diseases. ChemMedChem 2008, 3, 435–444.', doi: '10.1002/cmdc.200700139' },
        { name: 'Topological polar surface area', cite: 'Ertl, P.; Rohde, B.; Selzer, P. Fast Calculation of Molecular Polar Surface Area as a Sum of Fragment-Based Contributions. J. Med. Chem. 2000, 43, 3714–3717.', doi: '10.1021/jm000942e' },
        { name: 'WLOGP and molar refractivity', cite: 'Wildman, S. A.; Crippen, G. M. Prediction of Physicochemical Parameters by Atomic Contributions. J. Chem. Inf. Comput. Sci. 1999, 39, 868–873.', doi: '10.1021/ci990307l' },
      ],
    },
    {
      id: 'methods-other',
      title: 'Other methods used across the site',
      compact: true,
      items: [
        { name: 'MMFF94 force field', cite: 'Halgren, T. A. Merck Molecular Force Field. I. Basis, Form, Scope, Parameterization, and Performance of MMFF94. J. Comput. Chem. 1996, 17, 490–519.', doi: '10.1002/(SICI)1096-987X(199604)17:5/6<490::AID-JCC1>3.0.CO;2-P' },
        { name: 'UFF force field', cite: 'Rappé, A. K.; Casewit, C. J.; Colwell, K. S.; Goddard, W. A.; Skiff, W. M. UFF, a Full Periodic Table Force Field for Molecular Mechanics and Molecular Dynamics Simulations. J. Am. Chem. Soc. 1992, 114, 10024–10035.', doi: '10.1021/ja00051a040' },
        { name: 'ETKDG conformer generation', cite: 'Riniker, S.; Landrum, G. A. Better Informed Distance Geometry: Using What We Know to Improve Conformation Generation. J. Chem. Inf. Model. 2015, 55, 2562–2574.', doi: '10.1021/acs.jcim.5b00654' },
        { name: 'SMILES', cite: 'Weininger, D. SMILES, a Chemical Language and Information System. 1. Introduction to Methodology and Encoding Rules. J. Chem. Inf. Comput. Sci. 1988, 28, 31–36.', doi: '10.1021/ci00057a005' },
        { name: 'InChI', cite: 'Heller, S. R.; McNaught, A.; Pletnev, I.; Stein, S.; Tchekhovskoi, D. InChI, the IUPAC International Chemical Identifier. J. Cheminform. 2015, 7, 23.', doi: '10.1186/s13321-015-0068-4' },
      ],
    },
    {
      id: 'infra',
      title: 'Infrastructure and supporting libraries',
      blurb: 'Not chemistry, but they keep the site running. Listed for completeness and licence transparency.',
      compact: true,
      items: [
        { name: 'NumPy 1.26.4 · BSD-3-Clause', cite: 'Harris, C. R.; et al. Array Programming with NumPy. Nature 2020, 585, 357–362.', doi: '10.1038/s41586-020-2649-2' },
        { name: 'pandas 2.2.2 · BSD-3-Clause', cite: 'McKinney, W. Data Structures for Statistical Computing in Python. Proc. 9th Python in Science Conf., 2010, 56–61.', doi: '10.25080/Majora-92bf1922-00a' },
        { name: 'Flask 3.0.3 · BSD-3-Clause', cite: 'Flask web framework.', url: 'https://flask.palletsprojects.com' },
        { name: 'Werkzeug 3.0.3 · BSD-3-Clause', cite: 'Werkzeug WSGI library — also provides the PBKDF2 password hashing used for accounts.', url: 'https://werkzeug.palletsprojects.com' },
        { name: 'Gunicorn 23.0.0 · MIT', cite: 'Gunicorn WSGI server.', url: 'https://gunicorn.org' },
        { name: 'Flask-Cors 4.0.1 · MIT', cite: 'Flask-Cors.', url: 'https://github.com/corydolphin/flask-cors' },
        { name: 'RapidFuzz 3.10.1 · MIT', cite: 'RapidFuzz — Levenshtein edit distance for fragment similarity and label matching.', url: 'https://github.com/rapidfuzz/RapidFuzz' },
        { name: 'Requests 2.32.3 · Apache-2.0', cite: 'Requests HTTP library.', url: 'https://requests.readthedocs.io' },
        { name: 'Pillow 10.3.0 · HPND', cite: 'Pillow imaging library.', url: 'https://python-pillow.org' },
        { name: 'psycopg2 2.9.9 · LGPL-3.0', cite: 'PostgreSQL adapter for Python.', url: 'https://www.psycopg.org' },
        { name: 'Caddy · Apache-2.0', cite: 'Caddy web server — automatic HTTPS on the docking host.', url: 'https://caddyserver.com' },
        { name: 'Inter and JetBrains Mono · SIL Open Font License 1.1', cite: 'Typefaces used across the site.', url: 'https://fonts.google.com' },
        { name: 'Space Grotesk · SIL Open Font License 1.1', cite: 'Display typeface.', url: 'https://fonts.google.com/specimen/Space+Grotesk' },
      ],
    },
  ];

  // ── Render ────────────────────────────────────────────────────────────────
  const LIC_COLOUR = (l = '') => /GPL-2|GPL-3/.test(l) && !/LGPL/.test(l) ? 'amber'
    : /LGPL/.test(l) ? 'violet'
    : /BSD|MIT|Apache|CC0|HPND|SIL|OFL/.test(l) ? 'green' : '';

  $('#toc').innerHTML = SECTIONS
    .map((s) => `<a class="chip" href="#${esc(s.id)}">${esc(s.title)}</a>`).join('');

  $('#sections').innerHTML = SECTIONS.map((s) => `
    <section class="panel" id="${esc(s.id)}"><div class="panel-body">
      <h2 class="sec-title">${esc(s.title)}</h2>
      ${s.blurb ? `<p class="muted sec-blurb">${esc(s.blurb)}</p>` : ''}
      <div class="${s.compact ? 'cite-list' : 'dep-list'}">
        ${s.items.map((it) => s.compact ? compactItem(it) : fullItem(it)).join('')}
      </div>
    </div></section>`).join('');

  function licBadge(licence) {
    if (!licence) return '';
    return `<span class="badge ${LIC_COLOUR(licence)}">${esc(licence)}</span>`;
  }

  function doiLink(it) {
    if (it.doi) return `<a class="doi" href="https://doi.org/${encodeURI(it.doi)}" target="_blank" rel="noopener">doi:${esc(it.doi)}</a>`;
    if (it.url) return `<a class="doi" href="${esc(it.url)}" target="_blank" rel="noopener">${esc(it.url.replace(/^https?:\/\//, ''))}</a>`;
    return '';
  }

  function fullItem(it) {
    return `
      <div class="dep">
        <div class="dep-head">
          <span class="dep-name">${esc(it.name)}${it.version ? ` <span class="dep-ver">${esc(it.version)}</span>` : ''}</span>
          ${licBadge(it.licence)}
        </div>
        <p class="dep-used">${esc(it.used || '')}</p>
        <p class="dep-cite">${esc(it.cite)} ${doiLink(it)}</p>
        ${it.also ? `<p class="dep-cite also">${esc(it.also.cite)} ${doiLink(it.also)}${it.also.note ? `<br><span class="muted">${esc(it.also.note)}</span>` : ''}</p>` : ''}
        ${it.note ? `<p class="dep-note">${icon('info')} ${esc(it.note)}</p>` : ''}
      </div>`;
  }

  function compactItem(it) {
    return `
      <div class="cite-row">
        <div class="cite-what">${esc(it.name)}</div>
        <div class="cite-ref">${esc(it.cite)} ${doiLink(it)}</div>
      </div>`;
  }

  // ── Export ────────────────────────────────────────────────────────────────
  function plainText() {
    const lines = [
      'SmileBERTa — software, models and methods used',
      'Generated ' + new Date().toISOString().slice(0, 10),
      '',
      'SmileBERTa implements the published methods below. If this work contributed',
      'to a publication, please cite the relevant primary sources, not only SmileBERTa.',
      '',
    ];
    SECTIONS.forEach((s) => {
      lines.push('## ' + s.title, '');
      s.items.forEach((it) => {
        lines.push('- ' + it.name + (it.version ? ' (' + it.version + ')' : '') +
                   (it.licence ? ' [' + it.licence + ']' : ''));
        if (it.used) lines.push('  Used for: ' + it.used);
        lines.push('  ' + it.cite + (it.doi ? ' https://doi.org/' + it.doi : (it.url ? ' ' + it.url : '')));
        if (it.also) lines.push('  ' + it.also.cite + (it.also.doi ? ' https://doi.org/' + it.also.doi : ''));
        lines.push('');
      });
    });
    return lines.join('\n');
  }

  $('#copy-all').addEventListener('click', () => copy(plainText()));
  $('#dl-all').addEventListener('click', () => download(plainText(), 'smileberta-citations.txt', 'text/plain'));
  $('#copy-self').addEventListener('click', () => copy($('#self-cite').textContent.trim()));
})();
