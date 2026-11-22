"""Bert — the SmileBERTa research agent.

Bert is a Claude model wired to the platform's own tools. A user asks for work
in plain language ("dock these two compounds against EGFR and fragment whichever
scores better") and Bert plans it, runs the tools, reads the numbers and reports
back with the caveats that belong to each method.

Design notes:

* Tools call the platform's Python functions directly, never its own HTTP API.
  One process, no round trip, and no chance of the agent authenticating as
  somebody else.
* Tool results are deliberately small. The real endpoints return SVGs, 3D
  conformers and full pose geometry; none of that helps a language model and all
  of it costs tokens. Each tool here returns the numbers and the identifiers,
  and the UI links to the full tool for the pictures.
* Docking is the expensive one (~35 s and a real CPU), so it is capped per
  conversation and still passes through the docking service's own rate limits.
* Nothing Bert does is hidden: every tool call is streamed to the browser as it
  happens, with its arguments and its result.
"""
import importlib.util
import json
import os
import re
import threading
import time
from datetime import datetime, timezone

import requests
from rdkit import Chem

import admet
import db
from chem_tools import ChemError, analyze
from combine_frag import combine_fragments
from fragpred import calculate_properties, predict_fragment_detailed
from predict_drug_classification import classify_detailed

# The SDK is imported lazily, not at module load. It pulls in pydantic and httpx2,
# and this process already holds PyTorch plus a RoBERTa model inside a 512 MB tier —
# so that cost is paid only when somebody actually talks to Bert.
SDK_AVAILABLE = importlib.util.find_spec('anthropic') is not None
_sdk = {'mod': None}


def _anthropic():
    if _sdk['mod'] is None:
        import anthropic  # noqa: PLC0415 — deliberately deferred, see above
        _sdk['mod'] = anthropic
    return _sdk['mod']

# ── Configuration ────────────────────────────────────────────────────────────

SERVER_KEY = os.environ.get('ANTHROPIC_API_KEY', '').strip()
# Opus 5.5 is the default. Bert's turns are short and tool-heavy, so most of the
# spend is tool results rather than reasoning; if the bill needs cutting, set
# BERT_MODEL=claude-sonnet-5-5 (half the price) or claude-haiku-5-5 (a twentieth).
MODEL = os.environ.get('BERT_MODEL', 'claude-opus-5-5').strip()
EFFORT = os.environ.get('BERT_EFFORT', 'medium').strip()

# Free allowance per day, per identity. A turn is one user message plus every
# tool call and model round trip it takes to answer it.
FREE_TURNS_PER_DAY = int(os.environ.get('BERT_FREE_TURNS', '20'))
MAX_TOOL_CALLS = int(os.environ.get('BERT_MAX_TOOL_CALLS', '24'))   # per turn
MAX_DOCKS = int(os.environ.get('BERT_MAX_DOCKS', '4'))              # per turn
MAX_HISTORY = 24                                                      # messages kept
DOCK_API = os.environ.get('DOCK_API', 'https://5-161-192-134.sslip.io').rstrip('/')
DOCK_TIMEOUT = int(os.environ.get('BERT_DOCK_TIMEOUT', '300'))
USER_AGENT = 'SmileBERTa/2.0 (research platform; +https://smileberta-web.onrender.com)'

# Server-side refusal fallback. Dropped automatically if the account does not
# have the beta, so a missing entitlement cannot take the whole agent down.
_FALLBACKS_OK = {'on': True}


class BertError(Exception):
    pass


class QuotaError(BertError):
    pass


# ── Molecule helpers ─────────────────────────────────────────────────────────

def _mol(smiles):
    mol = Chem.MolFromSmiles((smiles or '').strip())
    if mol is None:
        raise BertError(f'Not a valid SMILES string: {smiles!r}')
    return mol


def _round(value, digits=2):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


# ── Tools ────────────────────────────────────────────────────────────────────

def tool_analyze(smiles):
    """Properties and drug-likeness for one molecule."""
    data = analyze(_mol(smiles))
    keep = ('smiles', 'formula', 'molecular_weight', 'log_p', 'tpsa', 'hydrogen_bond_donors',
            'hydrogen_bond_acceptors', 'rotatable_bonds', 'rings', 'aromatic_rings', 'heavy_atoms',
            'fraction_sp3', 'stereocenters', 'qed', 'lipinski_violations', 'lipinski_pass',
            'veber_pass')
    return {k: data[k] for k in keep if k in data}


def tool_classify(smiles):
    """Predicted therapeutic class."""
    result = classify_detailed(smiles)
    candidates = result.get('candidates') or []
    return {
        'smiles': smiles,
        'predicted_class': result.get('prediction'),
        'raw_model_output': result.get('raw'),
        'match_similarity': (candidates[0].get('similarity') if candidates else None),
        'alternatives': [{'label': c.get('name'), 'similarity': c.get('similarity')}
                         for c in candidates[1:5]],
        'caveat': ('The classifier learns statistical patterns, not mechanisms, and can be '
                   'confidently wrong. Rare classes are least reliable. The raw output is matched '
                   'to the nearest valid label by edit distance, so a low similarity means the '
                   'model produced something garbled.'),
    }


def tool_predict_fragment(smiles, target=None):
    """The binding-relevant fragment for a drug-like molecule."""
    _mol(smiles)
    result = predict_fragment_detailed(smiles, protein=target)
    fragment = result.get('fragment')
    out = {
        'parent_smiles': smiles,
        'fragment_smiles': fragment,
        'raw_model_output': result.get('raw'),
        'was_corrected': bool(result.get('corrected')),
        'correction_method': result.get('correction'),
        'target_recorded': target,
    }
    if fragment:
        props = calculate_properties(fragment) or {}
        out['fragment_properties'] = props
        mw, logp = props.get('molecular_weight'), props.get('log_p')
        hbd, hba = props.get('hydrogen_bond_donors'), props.get('hydrogen_bond_acceptors')
        if None not in (mw, logp, hbd, hba):
            out['rule_of_three_pass'] = bool(mw <= 300 and logp <= 3 and hbd <= 3 and hba <= 3)
    out['caveat'] = ('The target is recorded but the current checkpoint is a single shared model, '
                     'so the prediction does not yet change with the target.')
    return out


def tool_admet(smiles):
    """Absorption, solubility, drug-likeness and structural alerts."""
    profile = admet.profile(_mol(smiles))
    return {
        'smiles': profile.get('smiles'),
        'absorption': [{'property': p.get('label'), 'value': p.get('value'),
                        'verdict': p.get('verdict')}
                       for p in (profile.get('pharmacokinetics') or [])],
        'solubility': {k: v for k, v in (profile.get('solubility') or {}).items()
                       if k in ('log_s', 'class', 'mg_per_ml')},
        'filters': [{'name': f.get('name'), 'passed': f.get('passed')}
                    for f in (profile.get('filters') or [])],
        'filters_passed': f"{profile.get('filters_passed')}/{profile.get('filters_total')}",
        'medchem': [{'property': m.get('label'), 'value': m.get('value'),
                     'verdict': m.get('verdict')}
                    for m in (profile.get('medchem') or [])],
        'structural_alerts': [{'family': a.get('family'), 'name': a.get('name')}
                              for a in (profile.get('alerts') or [])],
        'alert_count': profile.get('alert_count'),
        'unavailable_endpoints': [u.get('label') if isinstance(u, dict) else u
                                  for u in (profile.get('unavailable') or [])],
        'note': ('Endpoints listed as unavailable need a trained model and are deliberately not '
                 'estimated. Every value above has a published basis, cited on the ADMET page.'),
    }


def tool_link_fragments(fragment_a, fragment_b):
    """Join two fragments into drug-like candidates."""
    _mol(fragment_a), _mol(fragment_b)
    try:
        smiles_list = combine_fragments(fragment_a, fragment_b)
    except ValueError as e:
        raise BertError(str(e))

    seen, candidates = set(), []
    for smiles in (smiles_list or []):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            continue
        seen.add(canonical)
        data = analyze(mol)
        candidates.append({'smiles': canonical, 'qed': data.get('qed'),
                           'molecular_weight': data.get('molecular_weight'),
                           'log_p': data.get('log_p'),
                           'lipinski_pass': data.get('lipinski_pass')})
    candidates.sort(key=lambda c: (c.get('qed') or 0), reverse=True)
    return {'candidates': candidates[:8], 'count': len(candidates),
            'note': ('Fragments are joined through short amine (C-N) and ester linkers at open '
                     'carbon positions. Ranked by QED; judge them with ADMET before docking.')}


# ── Docking ──────────────────────────────────────────────────────────────────

_SKIP_HET = {'HOH', 'WAT', 'DOD', 'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'ACT', 'DMS', 'CL', 'NA', 'K',
             'MG', 'CA', 'ZN', 'MN', 'NI', 'CD', 'IOD', 'BR', 'FMT', 'MPD', 'TRS', 'BME', 'NO3',
             'SCN', 'EPE', 'MES', 'IMD', 'PG4', '1PE', 'P6G', 'CIT', 'TLA', 'HEZ'}
_PDB_RE = re.compile(r'^[0-9][A-Za-z0-9]{3}$')
_pdb_cache = {}
_pdb_lock = threading.Lock()


def fetch_pdb(pdb_id):
    """Download a structure from the RCSB, with a small in-process cache."""
    pdb_id = (pdb_id or '').strip().upper()
    if not _PDB_RE.match(pdb_id):
        raise BertError(f'{pdb_id!r} is not a PDB id (four characters, e.g. 3PTB).')
    with _pdb_lock:
        if pdb_id in _pdb_cache:
            return pdb_id, _pdb_cache[pdb_id]
    try:
        r = requests.get(f'https://files.rcsb.org/download/{pdb_id}.pdb',
                         headers={'User-Agent': USER_AGENT}, timeout=30)
    except requests.RequestException as e:
        raise BertError(f'Could not reach the RCSB: {e}')
    if r.status_code == 404:
        raise BertError(f'No structure {pdb_id} in the Protein Data Bank.')
    if not r.ok:
        raise BertError(f'RCSB returned {r.status_code} for {pdb_id}.')
    text = r.text
    with _pdb_lock:
        if len(_pdb_cache) > 12:
            _pdb_cache.clear()
        _pdb_cache[pdb_id] = text
    return pdb_id, text


def parse_pocket(pdb_text):
    """Find candidate pockets: the co-crystallised ligands, largest first.

    Mirrors what the docking page does when a user clicks a ligand to snap the
    box to it, so an agent run and a hand-driven run start from the same box.
    """
    hets, protein, title = {}, [], ''
    for line in pdb_text.splitlines():
        if line.startswith('TITLE'):
            title += line[10:].strip() + ' '
        if line.startswith('ENDMDL'):
            break
        rec = line[:6]
        if rec not in ('ATOM  ', 'HETATM'):
            continue
        try:
            point = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        resn = line[17:20].strip()
        if rec == 'ATOM  ':
            protein.append(point)
        elif resn not in _SKIP_HET:
            hets.setdefault(f'{resn} {line[21:22]}{line[22:26].strip()}', []).append(point)

    pockets = []
    for key, pts in hets.items():
        if len(pts) < 6:
            continue
        center = [sum(p[i] for p in pts) / len(pts) for i in range(3)]
        size = [min(40.0, max(18.0, (max(p[i] for p in pts) - min(p[i] for p in pts)) + 10))
                for i in range(3)]
        pockets.append({'ligand': key, 'atoms': len(pts),
                        'center': [round(c, 2) for c in center], 'size': [round(s, 1) for s in size]})
    pockets.sort(key=lambda p: p['atoms'], reverse=True)
    if not pockets and protein:
        center = [sum(p[i] for p in protein) / len(protein) for i in range(3)]
        pockets.append({'ligand': None, 'atoms': 0, 'center': [round(c, 2) for c in center],
                        'size': [26.0, 26.0, 26.0]})
    return {'title': title.strip(), 'pockets': pockets}


def tool_dock(pdb_id, ligand_smiles, pocket_ligand=None, exhaustiveness=8, ctx=None):
    """Dock one ligand into one protein and wait for the result."""
    _mol(ligand_smiles)
    pdb_id, pdb_text = fetch_pdb(pdb_id)
    info = parse_pocket(pdb_text)
    if not info['pockets']:
        raise BertError(f'No usable pocket found in {pdb_id}.')

    pocket = info['pockets'][0]
    if pocket_ligand:
        wanted = pocket_ligand.strip().upper()
        match = next((p for p in info['pockets'] if p['ligand'] and p['ligand'].upper().startswith(wanted)), None)
        if match:
            pocket = match
    payload = {'receptor_pdb': pdb_text, 'ligand_smiles': ligand_smiles,
               'center': pocket['center'], 'size': pocket['size'],
               'exhaustiveness': max(1, min(16, int(exhaustiveness or 8))), 'num_modes': 5}
    headers = {'User-Agent': USER_AGENT, 'Content-Type': 'application/json'}
    if ctx and ctx.get('client_id'):
        headers['X-SmileBERTa-Client'] = ctx['client_id']

    try:
        start = requests.post(f'{DOCK_API}/dock', json=payload, headers=headers, timeout=90)
    except requests.RequestException as e:
        raise BertError(f'Could not reach the docking service: {e}')
    if start.status_code == 429:
        raise BertError('The docking service rate limit was reached. Try again in a few minutes.')
    if not start.ok:
        raise BertError(f'Docking was refused: {_err_text(start)}')
    job_id = (start.json() or {}).get('job_id')
    if not job_id:
        raise BertError('The docking service did not return a job id.')

    deadline = time.time() + DOCK_TIMEOUT
    while time.time() < deadline:
        time.sleep(3)
        try:
            poll = requests.get(f'{DOCK_API}/dock/{job_id}', headers=headers, timeout=30)
        except requests.RequestException:
            continue
        if not poll.ok:
            continue
        job = poll.json() or {}
        status = job.get('status')
        if status == 'done':
            poses = (job.get('result') or {}).get('poses') or []
            if not poses:
                raise BertError('Docking finished but produced no poses.')
            top = poses[0]
            contacts = top.get('interactions') or {}
            return {
                'pdb_id': pdb_id, 'structure_title': info['title'][:120],
                'ligand_smiles': ligand_smiles,
                'pocket': {'from_ligand': pocket['ligand'], 'center': pocket['center'],
                           'size': pocket['size']},
                'best_affinity_kcal_per_mol': top.get('affinity'),
                'poses': [{'mode': p.get('mode'), 'affinity': p.get('affinity'),
                           'rmsd_lb': p.get('rmsd_lb')} for p in poses[:5]],
                'hydrogen_bonds': [{'residue': h.get('residue'), 'distance': h.get('distance')}
                                   for h in (contacts.get('hbonds') or [])][:8],
                'contact_residues': [r.get('residue') for r in (contacts.get('residues') or [])][:12],
                'hydrophobic_contacts': contacts.get('hydrophobic'),
                'elapsed_seconds': job.get('elapsed'),
                'job_id': job_id,
                'caveat': ("The receptor is rigid and Vina's typical error is about 2 kcal/mol. "
                           'Affinities compare related compounds in the same pocket; they are '
                           'predictions, not measurements.'),
            }
        if status == 'error':
            raise BertError(f'Docking failed: {job.get("error") or "unknown error"}')
    raise BertError(f'Docking did not finish within {DOCK_TIMEOUT}s (job {job_id}).')


def _err_text(response):
    try:
        return (response.json() or {}).get('error') or response.text[:200]
    except ValueError:
        return response.text[:200]


# ── Literature ───────────────────────────────────────────────────────────────

def tool_search_papers(query, limit=5):
    """Search Europe PMC — open, no key, covers PubMed and preprints."""
    try:
        r = requests.get('https://www.ebi.ac.uk/europepmc/webservices/rest/search',
                         params={'query': query, 'format': 'json',
                                 'pageSize': max(1, min(10, int(limit or 5))),
                                 'resultType': 'lite', 'sort': 'CITED desc'},
                         headers={'User-Agent': USER_AGENT}, timeout=25)
        r.raise_for_status()
    except requests.RequestException as e:
        raise BertError(f'Literature search failed: {e}')
    results = ((r.json() or {}).get('resultList') or {}).get('result') or []
    papers = []
    for item in results:
        doi, pmid = item.get('doi'), item.get('pmid')
        papers.append({
            'title': item.get('title'),
            'authors': (item.get('authorString') or '')[:120],
            'journal': item.get('journalTitle') or item.get('bookOrReportDetails'),
            'year': item.get('pubYear'),
            'citations': item.get('citedByCount'),
            'doi': doi,
            'url': (f'https://doi.org/{doi}' if doi else
                    (f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/' if pmid else None)),
        })
    return {'query': query, 'papers': papers, 'source': 'Europe PMC'}


# ── Tool registry ────────────────────────────────────────────────────────────

TOOLS = [
    {
        'name': 'analyze_molecule',
        'description': ('Compute physicochemical properties and drug-likeness for one molecule: '
                        'molecular weight, cLogP, TPSA, H-bond donors and acceptors, rotatable '
                        'bonds, QED, and Lipinski/Veber checks. Fast and exact (RDKit, no model). '
                        'Use it whenever you need numbers about a structure.'),
        'input_schema': {'type': 'object', 'additionalProperties': False,
                         'properties': {'smiles': {'type': 'string', 'description': 'SMILES string.'}},
                         'required': ['smiles']},
    },
    {
        'name': 'classify_drug',
        'description': ("Predict a compound's therapeutic class from its structure using "
                        "SmileBERTa's own fine-tuned transformer. Returns the class, the raw model "
                        'output and ranked alternatives. It is a statistical model and can be '
                        'confidently wrong — always report it as a prediction.'),
        'input_schema': {'type': 'object', 'additionalProperties': False,
                         'properties': {'smiles': {'type': 'string'}},
                         'required': ['smiles']},
    },
    {
        'name': 'predict_fragment',
        'description': ("Predict the binding-relevant fragment of a drug-like molecule using "
                        "SmileBERTa's own fragment model. This is the platform's signature tool. "
                        'Use it to find the part of a molecule worth growing into a new lead.'),
        'input_schema': {'type': 'object', 'additionalProperties': False,
                         'properties': {
                             'smiles': {'type': 'string'},
                             'target': {'type': 'string',
                                        'description': 'Optional target name (KRAS, mTOR, AKT1, ERK2). '
                                                       'Recorded only; predictions do not vary by target yet.'}},
                         'required': ['smiles']},
    },
    {
        'name': 'admet_profile',
        'description': ('Profile absorption, blood-brain-barrier access, aqueous solubility, '
                        'drug-likeness filters, PAINS/Brenk structural alerts and synthetic '
                        'accessibility. Every value has a published basis; endpoints needing a '
                        'trained model are returned as unavailable rather than guessed. Use this '
                        'to judge whether a compound could behave like a drug.'),
        'input_schema': {'type': 'object', 'additionalProperties': False,
                         'properties': {'smiles': {'type': 'string'}},
                         'required': ['smiles']},
    },
    {
        'name': 'link_fragments',
        'description': ('Join two fragments into larger lead-like molecules through short amine '
                        'and ester linkers, keeping only candidates that pass Lipinski with '
                        'QED > 0.3. Returns candidates ranked by QED.'),
        'input_schema': {'type': 'object', 'additionalProperties': False,
                         'properties': {'fragment_a': {'type': 'string'}, 'fragment_b': {'type': 'string'}},
                         'required': ['fragment_a', 'fragment_b']},
    },
    {
        'name': 'dock',
        'description': ('Dock a ligand into a protein with AutoDock Vina and return binding '
                        'affinity in kcal/mol, the top poses, hydrogen bonds and contact residues. '
                        'The structure is fetched from the RCSB by PDB id and the search box is '
                        'placed on the largest co-crystallised ligand automatically. Takes roughly '
                        '30-60 seconds, so dock only what the question needs, and dock compounds '
                        'into the same pocket when comparing them.'),
        'input_schema': {'type': 'object', 'additionalProperties': False,
                         'properties': {
                             'pdb_id': {'type': 'string', 'description': 'Four-character PDB id, e.g. 3PTB.'},
                             'ligand_smiles': {'type': 'string'},
                             'pocket_ligand': {'type': 'string',
                                               'description': 'Optional residue name of the co-crystallised '
                                                              'ligand whose site to target, e.g. STI.'},
                             'exhaustiveness': {'type': 'integer', 'minimum': 1, 'maximum': 16}},
                         'required': ['pdb_id', 'ligand_smiles']},
    },
    {
        'name': 'inspect_structure',
        'description': ('Look up a protein structure by PDB id without docking: its title and the '
                        'co-crystallised ligands that mark candidate pockets. Use it to choose a '
                        'pocket before docking, or to answer questions about a target.'),
        'input_schema': {'type': 'object', 'additionalProperties': False,
                         'properties': {'pdb_id': {'type': 'string'}},
                         'required': ['pdb_id']},
    },
    {
        'name': 'search_papers',
        'description': ('Search the published literature through Europe PMC (covers PubMed and '
                        'preprints). Use it to connect a result to real research — a target, a '
                        'scaffold, a mechanism — and always cite what you find with its link.'),
        'input_schema': {'type': 'object', 'additionalProperties': False,
                         'properties': {'query': {'type': 'string'},
                                        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 10}},
                         'required': ['query']},
    },
]


def _run_tool(name, args, ctx):
    if name == 'analyze_molecule':
        return tool_analyze(args['smiles'])
    if name == 'classify_drug':
        return tool_classify(args['smiles'])
    if name == 'predict_fragment':
        return tool_predict_fragment(args['smiles'], args.get('target'))
    if name == 'admet_profile':
        return tool_admet(args['smiles'])
    if name == 'link_fragments':
        return tool_link_fragments(args['fragment_a'], args['fragment_b'])
    if name == 'inspect_structure':
        pdb_id, text = fetch_pdb(args['pdb_id'])
        info = parse_pocket(text)
        return {'pdb_id': pdb_id, 'title': info['title'][:200], 'pockets': info['pockets'][:6]}
    if name == 'dock':
        if ctx['docks'] >= MAX_DOCKS:
            raise BertError(f'This answer has already run {MAX_DOCKS} docking jobs, the limit for '
                             'one request. Ask again to continue with more.')
        ctx['docks'] += 1
        return tool_dock(args['pdb_id'], args['ligand_smiles'], args.get('pocket_ligand'),
                         args.get('exhaustiveness', 8), ctx)
    if name == 'search_papers':
        return tool_search_papers(args['query'], args.get('limit', 5))
    raise BertError(f'Unknown tool {name}')


# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM = """You are Bert, the research agent built into SmileBERTa — a free, browser-based drug-discovery platform created by Nisarg Shah at Emory University.

You help students and researchers run real computational drug-discovery work. You have the platform's own tools: its fragment-prediction and drug-classification transformer models, RDKit property analysis, ADMET profiling, fragment linking, AutoDock Vina docking against any PDB structure, and literature search.

How to work:

- Plan briefly, then act. When someone asks for several things ("dock both of these, then fragment the better one"), do the whole job: run the tools in order, compare the numbers yourself, and carry the result forward without asking permission between steps.
- Prefer doing over describing. If a question can be answered by running a tool, run it.
- Docking is slow (30-60 seconds) and uses real CPU. Dock what the question needs and no more. When comparing compounds, dock them into the same structure and pocket, and say which pocket you used.
- Check your inputs. If a SMILES string looks wrong or a name is ambiguous, say so rather than guessing. You may use well-known SMILES for named drugs from your own knowledge, but say that is where it came from.

How to report:

- Lead with the answer and the number that supports it. Then the detail.
- Use the units and signs correctly: docking affinities are negative, and more negative is stronger.
- Never present a prediction as a measurement. The classifier can be confidently wrong; Vina's typical error is about 2 kcal/mol with a rigid receptor; the fragment model's target selector is recorded but not yet used. State the limitation that actually applies to what you just ran, not a generic disclaimer, and keep it to one sentence.
- Never invent numbers, affinities, citations or papers. Every number you report must come from a tool result in this conversation. If a tool failed or an endpoint is unavailable, say so plainly.
- When you suggest leads, say exactly why — which number makes each one interesting, and what the obvious next experiment is.
- When literature would help the user trust or extend a result, search for it and cite it with its link.
- This is research and education software, not clinical advice.

Format for a terminal-width chat panel: short paragraphs, small markdown tables for comparisons, no headings above level three, and no emoji."""


# ── Quota ────────────────────────────────────────────────────────────────────

def _today():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')


def identities(user_id, client_id, ip):
    """Every bucket a turn counts against. A signed-in user is one identity;
    everyone else is counted by browser *and* by IP, so rotating one does not
    reset the allowance."""
    if user_id:
        return [f'u:{user_id}']
    idents = []
    if client_id:
        idents.append(f'c:{client_id}')
    if ip:
        idents.append(f'ip:{ip}')
    return idents or ['ip:unknown']


def quota_state(user_id, client_id, ip):
    idents = identities(user_id, client_id, ip)
    try:
        used = db.bert_usage(idents, _today())
    except Exception:
        used = 0  # a quota we cannot read must not block the agent
    return {'used': used, 'limit': FREE_TURNS_PER_DAY,
            'remaining': max(0, FREE_TURNS_PER_DAY - used)}


def status(user_id, client_id, ip):
    state = quota_state(user_id, client_id, ip)
    state.update({
        'available': SDK_AVAILABLE and bool(SERVER_KEY),
        'sdk_installed': SDK_AVAILABLE,
        'byok_supported': SDK_AVAILABLE,
        'model': MODEL,
        'name': 'Bert',
        'max_docks_per_turn': MAX_DOCKS,
    })
    return state


# ── Agent loop ───────────────────────────────────────────────────────────────

def _client(user_key):
    if not SDK_AVAILABLE:
        raise BertError('The Anthropic SDK is not installed on this server.')
    key = (user_key or '').strip() or SERVER_KEY
    if not key:
        raise BertError('Bert is not set up on this server: there is no API key. '
                         'You can paste your own Anthropic key to use him now.')
    return _anthropic().Anthropic(api_key=key, timeout=120.0, max_retries=2)


def _trim(history):
    """Keep the conversation bounded. Tool results are the bulk of it, and a
    cut must not orphan a tool_use block from its tool_result."""
    messages = [m for m in history if m.get('role') in ('user', 'assistant') and m.get('content')]
    if len(messages) <= MAX_HISTORY:
        return messages
    cut = messages[-MAX_HISTORY:]
    while cut and cut[0].get('role') != 'user':
        cut = cut[1:]
    for i, msg in enumerate(cut):          # drop a leading orphan tool_result
        content = msg.get('content')
        if isinstance(content, list) and any(
                isinstance(b, dict) and b.get('type') == 'tool_result' for b in content):
            cut = cut[i + 1:]
        break
    return cut or messages[-2:]


def _sse(event, data):
    return f'event: {event}\ndata: {json.dumps(data)}\n\n'


def run(history, user_id=None, client_id=None, ip=None, user_key=None):
    """Stream one turn as server-sent events."""
    byok = bool((user_key or '').strip())
    idents = identities(user_id, client_id, ip)

    if not byok:
        state = quota_state(user_id, client_id, ip)
        if state['remaining'] <= 0:
            yield _sse('error', {
                'error': f"You have used today's {FREE_TURNS_PER_DAY} free Bert messages.",
                'code': 'quota',
                'hint': 'The allowance resets at midnight UTC. You can also paste your own '
                        'Anthropic API key to keep going right now.'})
            return

    try:
        client = _client(user_key)
    except BertError as e:
        yield _sse('error', {'error': str(e), 'code': 'config'})
        return

    messages = _trim(history)
    ctx = {'docks': 0, 'client_id': client_id, 'user_id': user_id}
    tool_calls = in_tokens = out_tokens = 0
    started = time.time()

    try:
        while True:
            request = {
                'model': MODEL,
                'max_tokens': 8000,
                'system': SYSTEM,
                'messages': messages,
                'tools': TOOLS,
                'thinking': {'type': 'adaptive'},
                'output_config': {'effort': EFFORT},
            }
            # Server-side refusal fallback: if a safety classifier declines, the
            # same request is retried on another model inside the same call.
            use_fallbacks = _FALLBACKS_OK['on']
            if use_fallbacks:
                request['betas'] = ['server-side-fallback-2026-07-01']
                request['fallbacks'] = 'default'

            text_seen = False
            try:
                stream_ctx = (client.beta.messages.stream(**request) if use_fallbacks
                              else client.messages.stream(**request))
                with stream_ctx as stream:
                    for event in stream:
                        if event.type == 'content_block_delta' and event.delta.type == 'text_delta':
                            text_seen = True
                            yield _sse('text', {'text': event.delta.text})
                    final = stream.get_final_message()
            except Exception as e:
                if use_fallbacks and _is_beta_error(e):
                    # The account does not have the fallback beta; run without it.
                    _FALLBACKS_OK['on'] = False
                    continue
                raise

            in_tokens += getattr(final.usage, 'input_tokens', 0) or 0
            out_tokens += getattr(final.usage, 'output_tokens', 0) or 0

            if final.stop_reason == 'refusal':
                yield _sse('error', {'error': 'Bert declined to answer that request.', 'code': 'refusal'})
                break
            if final.stop_reason == 'max_tokens' and not text_seen:
                yield _sse('error', {'error': 'The answer was too long to finish. Try a narrower question.',
                                     'code': 'max_tokens'})
                break

            messages.append({'role': 'assistant', 'content': final.content})
            calls = [b for b in final.content if getattr(b, 'type', None) == 'tool_use']
            if final.stop_reason != 'tool_use' or not calls:
                break

            results = []
            for call in calls:
                tool_calls += 1
                if tool_calls > MAX_TOOL_CALLS:
                    results.append({'type': 'tool_result', 'tool_use_id': call.id, 'is_error': True,
                                    'content': f'Tool-call limit ({MAX_TOOL_CALLS}) reached for this '
                                               'request. Summarise what you have and stop.'})
                    continue
                args = call.input if isinstance(call.input, dict) else {}
                yield _sse('tool_start', {'id': call.id, 'name': call.name, 'input': args})
                try:
                    output = _run_tool(call.name, args, ctx)
                    results.append({'type': 'tool_result', 'tool_use_id': call.id,
                                    'content': json.dumps(output, default=str)[:20000]})
                    yield _sse('tool_end', {'id': call.id, 'name': call.name, 'result': output})
                except Exception as e:                       # a failed tool is data, not a crash
                    message = str(e) or e.__class__.__name__
                    results.append({'type': 'tool_result', 'tool_use_id': call.id,
                                    'is_error': True, 'content': message[:1500]})
                    yield _sse('tool_error', {'id': call.id, 'name': call.name, 'error': message[:400]})
            messages.append({'role': 'user', 'content': results})

    except Exception as e:
        yield _sse('error', {'error': _friendly(e), 'code': 'api'})

    # Only a turn that actually reached the model costs anything, so only that
    # turn is counted. A rejected key, an unreachable API or a missing config
    # burns no tokens, and billing it would let a server-side fault eat somebody's
    # whole day of messages — and keep them locked out after the fault is fixed.
    billable = bool(in_tokens or out_tokens)
    if not byok and billable:
        try:
            db.bert_record(idents, _today(), tool_calls, in_tokens, out_tokens)
        except Exception:
            pass
    state = quota_state(user_id, client_id, ip) if not byok else {'remaining': None, 'limit': None}
    yield _sse('done', {'tool_calls': tool_calls, 'seconds': round(time.time() - started, 1),
                        'input_tokens': in_tokens, 'output_tokens': out_tokens,
                        'remaining': state.get('remaining'), 'byok': byok})


def _is_beta_error(exc):
    text = str(exc).lower()
    return 'beta' in text and ('fallback' in text or 'not allowed' in text or 'unsupported' in text)


def _friendly(exc):
    name = exc.__class__.__name__
    text = str(exc)
    if name == 'AuthenticationError':
        return 'That Anthropic API key was rejected. Check it and try again.'
    if name == 'PermissionDeniedError':
        return 'That API key does not have access to this model.'
    if name == 'RateLimitError':
        return 'Anthropic is rate limiting this key. Wait a moment and try again.'
    if name == 'APIConnectionError' or name == 'APITimeoutError':
        return 'Could not reach Anthropic. Check the connection and try again.'
    if name == 'NotFoundError':
        return f'The model {MODEL} was not found for this key.'
    if name == 'BadRequestError':
        return f'The request was rejected: {text[:200]}'
    return f'Bert hit an error: {text[:200] or name}'
