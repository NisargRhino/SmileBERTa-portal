"""Redocking validation — "can this pipeline reproduce a pose we already know?"

A structure solved with a ligand bound is an answer key. Strip the ligand out,
rebuild it from scratch, dock it back into its own pocket and measure how far the
top-scored pose lands from the experimental one. If the pipeline cannot reproduce
a known pose, nothing else it says about that pocket is worth much.

Three rules decide the design, and every one of them costs us a nicer-looking
number:

1. **The ligand is rebuilt, not reused.** We take the crystal ligand's
   connectivity, write a SMILES and embed a fresh conformer (ETKDG + MMFF94) —
   the same path a user's own SMILES takes. Handing Vina the crystal coordinates
   would produce impressively low RMSDs that measure nothing.
2. **The box comes from the crystal ligand, never from the user's box.** The run
   has to test the pipeline and be reproducible between users.
3. **RMSD is symmetry-corrected, heavy-atom, and computed without
   superposition** (``CalcRMS``, not ``GetBestRMS``). The question is whether the
   pose landed in the right place in the pocket, so the two coordinate frames
   must stay fixed.

Nothing here retries, re-boxes or re-seeds to chase a passing verdict. An
"unreliable" verdict is a successful run of this feature.
"""
import collections
import hashlib
import json
import os
import re
import threading
import time
import urllib.request

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

import chem_tools

PDB_URL = 'https://files.rcsb.org/download/{}.pdb'
COMPONENT_URL = 'https://files.rcsb.org/ligands/download/{}_ideal.sdf'
# Generic client name on purpose: nothing about the user belongs in this header.
USER_AGENT = 'SmileBERTa-docking/1.0'
HTTP_TIMEOUT = 30

# Residue names that are solvent, cryoprotectant, buffer, crystallisation additive,
# sugar or ion — never the ligand the structure was solved to show.
EXCLUDE_RESIDUES = {
    'HOH', 'DOD', 'WAT', 'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'PG4', '1PE', 'P6G', 'MES',
    'TRS', 'ACT', 'ACY', 'DMS', 'IMD', 'FMT', 'CIT', 'TLA', 'MPD', 'BME', 'NAG', 'MAN',
    'BMA', 'FUC', 'GAL', 'NDG', 'HEZ', 'NO3', 'SCN', 'EPE', 'CL', 'BR', 'IOD', 'NA',
    'K', 'MG', 'CA', 'ZN', 'MN', 'FE', 'FE2', 'CU', 'NI', 'CD', 'HG', 'CO', 'SR', 'CS',
}
MIN_HEAVY_ATOMS = 8
BOX_PADDING = 8.0          # Å of margin per axis, half of it on each side
BOX_MIN, BOX_MAX = 18.0, 26.0   # matches the guidance already on the docking page
# A ligand heavy atom this close to a protein heavy atom is a bond, not a contact
# (C–S 1.81 Å, C–N 1.47 Å; the closest non-bonded contacts sit near 2.6 Å).
COVALENT_CUTOFF = 2.1
# Verdict bands. 2.0 Å is the conventional success threshold in the redocking
# literature; do not soften these.
RELIABLE_MAX = 2.0
MARGINAL_MAX = 3.0
VERDICTS = {
    'reliable': 'The pipeline reproduced the known binding pose. Results for this pocket '
                'are worth taking seriously.',
    'marginal': 'The pipeline got the right region but not the right pose. Treat scores '
                'here as rough ranking only.',
    'unreliable': 'The pipeline did not reproduce the known pose. Do not trust docking '
                  'results for this pocket without changing the setup.',
}
# The regenerated conformer has to be a genuinely independent starting point.
MIN_START_RMSD = 1.0


class RedockError(ValueError):
    """A reason this structure cannot be validated, phrased for the user."""


# ── Fetching ─────────────────────────────────────────────────────────────────

_fetch_cache = collections.OrderedDict()
_fetch_lock = threading.Lock()
# A fetched PDB is megabytes and this module is imported by the 512 MB API
# service as well as the docking host, so the download cache is kept to a budget
# rather than an entry count.
FETCH_CACHE_BYTES = 8 * 1024 * 1024


def _get(url):
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
        return r.read().decode('utf-8', 'replace')


def _cached_get(url):
    with _fetch_lock:
        if url in _fetch_cache:
            _fetch_cache.move_to_end(url)
            return _fetch_cache[url]
    text = _get(url)
    with _fetch_lock:
        _fetch_cache[url] = text
        _fetch_cache.move_to_end(url)
        while len(_fetch_cache) > 1 and sum(len(v) for v in _fetch_cache.values()) > FETCH_CACHE_BYTES:
            _fetch_cache.popitem(last=False)
    return text


def valid_pdb_id(pdb_id):
    return bool(re.fullmatch(r'[0-9][A-Za-z0-9]{3}', (pdb_id or '').strip()))


def fetch_structure(pdb_id):
    pdb_id = (pdb_id or '').strip().upper()
    if not valid_pdb_id(pdb_id):
        raise RedockError('A four-character PDB ID is required, for example 3PTB')
    try:
        return pdb_id, _cached_get(PDB_URL.format(pdb_id))
    except Exception:
        raise RedockError(f'Could not fetch {pdb_id} from the RCSB PDB '
                          '(it may exist only as mmCIF, which this feature cannot read)')


def fetch_component(comp_id):
    """The chemical component's idealised structure, which carries bond orders.

    The PDB's own coordinates have no bond orders at all, so the component
    definition is what makes a SMILES — and therefore a rebuilt conformer —
    possible.
    """
    try:
        sdf = _cached_get(COMPONENT_URL.format(comp_id.upper()))
    except Exception:
        raise RedockError(f'Could not look up the chemical definition of {comp_id}, '
                          'so this ligand cannot be rebuilt and validation is not available here.')
    tmpl = Chem.MolFromMolBlock(sdf, removeHs=True)
    if tmpl is None:
        raise RedockError(f'The chemical definition of {comp_id} could not be read, '
                          'so validation is not available here.')
    # RemoveAllHs, not removeHs=True: the latter keeps a hydrogen that carries
    # double-bond stereochemistry, which is exactly what benzamidine's amidine
    # NH is. That one leftover atom made the template a 10-atom molecule and no
    # 9-atom crystal ligand could ever match it.
    return Chem.RemoveAllHs(tmpl)


# ── Picking the co-crystallised ligand ───────────────────────────────────────

def _atom(line):
    return {
        'resn': line[17:20].strip().upper(),
        'chain': line[21:22].strip(),
        'resi': line[22:27].strip(),
        'alt': line[16:17],
        'name': line[12:16].strip(),
        'el': (line[76:78].strip() or re.sub(r'[^A-Za-z]', '', line[12:16])[:1]).upper(),
        'xyz': (float(line[30:38]), float(line[38:46]), float(line[46:54])),
        'line': line,
    }


def parse_structure(pdb_text):
    """HETATM residues and protein heavy-atom coordinates, first model only."""
    residues = {}
    protein = []
    title = ''
    for line in pdb_text.splitlines():
        if line.startswith('TITLE'):
            title += line[10:].strip() + ' '
        if line.startswith('ENDMDL'):
            break
        rec = line[:6]
        if rec not in ('ATOM  ', 'HETATM'):
            continue
        try:
            a = _atom(line)
        except ValueError:
            continue
        if a['alt'] not in (' ', 'A'):
            continue
        if a['el'] == 'H' or a['el'] == 'D':
            continue
        if rec == 'ATOM  ':
            protein.append(a['xyz'])
            continue
        if a['resn'] in EXCLUDE_RESIDUES:
            continue
        key = (a['resn'], a['chain'], a['resi'])
        residues.setdefault(key, []).append(a)
    return {'title': title.strip(), 'residues': residues,
            'protein': np.array(protein) if protein else np.zeros((0, 3))}


def _covalent_partners(atoms, protein_xyz):
    if not len(protein_xyz):
        return 0
    lig = np.array([a['xyz'] for a in atoms])
    d = np.linalg.norm(protein_xyz[:, None, :] - lig[None, :, :], axis=2)
    return int(np.count_nonzero(d < COVALENT_CUTOFF))


def _bonded_groups(residues):
    """Group HETATM residues that are bonded to each other into one entity.

    Ligands are not always one residue. 6LU7's inhibitor, for example, is
    deposited as three linked components; treating them separately would pick
    the one fragment that happens not to touch the protein and validate *that*,
    which is worse than refusing. Residues whose heavy atoms come within bonding
    distance of each other are one ligand.
    """
    keys = list(residues)
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    coords = {k: np.array([a['xyz'] for a in residues[k]]) for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if find(a) == find(b):
                continue
            d = np.linalg.norm(coords[a][:, None, :] - coords[b][None, :, :], axis=2)
            if d.min() < COVALENT_CUTOFF:
                parent[find(b)] = find(a)
    groups = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)
    return list(groups.values())


def ligand_candidates(pdb_text):
    """Every ligand entity that could be the one this structure was solved for.

    Mirrored in ``frontend/assets/js/docking.js`` so the button only appears when
    the server would accept the structure. Keep the two in step.
    """
    parsed = parse_structure(pdb_text)
    out = []
    for group in _bonded_groups(parsed['residues']):
        atoms = [a for key in group for a in parsed['residues'][key]]
        if len(atoms) < MIN_HEAVY_ATOMS:
            continue
        # The entity's identity is its largest component; the rest are listed so
        # a multi-component ligand can be reported as one.
        group = sorted(group, key=lambda k: (-len(parsed['residues'][k]), k))
        resn, chain, resi = group[0]
        out.append({
            'comp_id': resn, 'chain': chain, 'resi': resi, 'heavy_atoms': len(atoms),
            'components': [f'{r} {c}{i}' for r, c, i in group],
            'covalent_contacts': _covalent_partners(atoms, parsed['protein']),
            'atoms': atoms,
        })
    # Most heavy atoms wins; chain/residue order breaks ties so the choice is
    # the same for every user and every run.
    out.sort(key=lambda c: (-c['heavy_atoms'], c['chain'], c['resi']))
    return out, parsed


NO_LIGAND = ('This structure has no co-crystallised ligand, so the setup cannot be '
             'validated automatically.')


def choose_ligand(pdb_text):
    """The ligand to redock, plus a note when the choice needed one.

    Raises RedockError with the copy the user should see when there is nothing
    suitable — a structure holding only waters and ions, or only a covalent
    ligand, is a legitimate "cannot validate", not a failure.
    """
    candidates, parsed = ligand_candidates(pdb_text)
    if not candidates:
        raise RedockError(NO_LIGAND)
    free = [c for c in candidates if not c['covalent_contacts']]
    if not free:
        c = candidates[0]
        raise RedockError(
            f"The ligand in this structure ({c['comp_id']}) is covalently bound to the "
            'protein. Redocking a covalent ligand with a non-covalent protocol would not '
            'mean anything, so validation is not available here.')
    # Only a single-component ligand can be rebuilt from its chemical-component
    # definition, so a peptide-like ligand deposited as several linked residues
    # is skipped rather than validated one fragment at a time.
    single = [c for c in free if len(c['components']) == 1]
    if not single:
        c = free[0]
        raise RedockError(
            f"The ligand in this structure is deposited as {len(c['components'])} linked "
            f"components ({', '.join(c['components'])}). Rebuilding it from its chemical "
            'definition is not something this check can do reliably, so validation is not '
            'available here.')
    chosen = single[0]
    notes = []
    skipped_multi = [c for c in free if len(c['components']) > 1]
    if skipped_multi:
        notes.append('Skipped as a multi-component ligand: '
                     + ', '.join('+'.join(c['components']) for c in skipped_multi[:2]) + '.')
    copies = [c for c in single if c['comp_id'] == chosen['comp_id']]
    if len(copies) > 1:
        notes.append(f"{chosen['comp_id']} appears in {len(copies)} chains "
                     f"({', '.join(sorted(c['chain'] for c in copies))}); "
                     f"chain {chosen['chain']} was used.")
    others = [c for c in single[1:] if c['comp_id'] != chosen['comp_id']]
    if others:
        notes.append('Other candidates were smaller: '
                     + ', '.join(f"{c['comp_id']} ({c['heavy_atoms']} atoms)" for c in others[:3]) + '.')
    dropped = [c for c in candidates if c['covalent_contacts']]
    if dropped:
        notes.append('Skipped as covalently bound: '
                     + ', '.join(sorted({c['comp_id'] for c in dropped})) + '.')
    return chosen, parsed, ' '.join(notes)


def component_name(pdb_text, comp_id):
    """The ligand's name from the structure's own HETNAM records."""
    parts = []
    for line in pdb_text.splitlines():
        if line.startswith('HETNAM') and line[11:14].strip().upper() == comp_id.upper():
            parts.append(line[15:].strip())
        elif line.startswith('FORMUL'):
            break
    name = ' '.join(parts).strip()
    return name[:120].lower() if name else comp_id


# ── Box ──────────────────────────────────────────────────────────────────────

def box_for(atoms):
    """Centroid-centred box around the crystal ligand, padded on every side.

    The size is measured from the centre outwards rather than from the bounding
    box, because the centroid of an uneven ligand is not the middle of its
    bounding box: on 1HSG, extent + 8 Å around the centroid leaves 5.4 Å of
    margin on one face of the box and 2.6 Å on the other, and the tight face is
    the one that limits what Vina can sample. Measuring from the centre gives
    the ligand at least half the padding on each side, which is what padding is
    supposed to mean.
    """
    xyz = np.array([a['xyz'] for a in atoms])
    center = xyz.mean(axis=0)
    reach = np.maximum(center - xyz.min(axis=0), xyz.max(axis=0) - center)
    size = np.clip(2 * reach + BOX_PADDING, BOX_MIN, BOX_MAX)
    return [round(float(v), 3) for v in center], [round(float(v), 2) for v in size]


def box_contains(atoms, center, size):
    """Does the (clamped) box still hold every atom of the crystal ligand?

    The 26 Å ceiling keeps validation runs comparable between structures, but a
    ligand long enough to need more than that could not be reproduced inside the
    box at all. Saying so is better than reporting an RMSD that was impossible
    to achieve.
    """
    xyz = np.array([a['xyz'] for a in atoms])
    center, size = np.array(center), np.array(size)
    return bool((xyz.min(axis=0) >= center - size / 2).all()
                and (xyz.max(axis=0) <= center + size / 2).all())


# ── Rebuilding the ligand ────────────────────────────────────────────────────

def crystal_mol(atoms, template):
    """The crystal ligand with real bond orders: PDB coordinates + component graph."""
    block = '\n'.join(a['line'] for a in atoms) + '\nEND\n'
    raw = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=True, proximityBonding=True)
    if raw is None:
        raise RedockError('Could not read the ligand coordinates from this structure, '
                          'so validation is not available here.')
    Chem.FastFindRings(raw)   # ring info for the substructure match, without sanitizing
    try:
        mol = AllChem.AssignBondOrdersFromTemplate(template, raw)
    except Exception:
        raise RedockError(
            'Could not rebuild this ligand from the structure — its coordinates do not '
            'match the standard chemical definition (often missing or disordered atoms), '
            'so validation is not available here.')
    mol = Chem.RemoveHs(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        raise RedockError('Could not rebuild this ligand from the structure, so validation '
                          'is not available here.')
    return mol


# ── RMSD ─────────────────────────────────────────────────────────────────────

def _skeleton(mol):
    """Heavy atoms, every bond single, charges cleared — an element-and-connectivity graph.

    Two things make this the right basis for the comparison. Open Babel perceives
    the pose's bond orders from the PDBQT geometry and RDKit perceives the
    crystal ligand's from the component definition, so insisting the two agree
    would turn a bookkeeping difference into a failed measurement. And chemically
    equivalent atoms *should* be interchangeable: the two oxygens of a carboxylate
    or a flipped phenyl ring are not errors, and flattening bond orders lets the
    symmetry search in CalcRMS see them as the symmetry they are.
    """
    m = Chem.RWMol(Chem.Mol(mol))
    # Drop hydrogens by hand rather than with RemoveHs, which expects a sanitized
    # molecule — poses come back from Open Babel with valences RDKit will not
    # always accept, and refusing to measure them would be the wrong answer.
    for idx in sorted((a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 1), reverse=True):
        m.RemoveAtom(idx)
    for atom in m.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in m.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    out = m.GetMol()
    Chem.FastFindRings(out)   # enough for substructure matching, no sanitization needed
    return out


def pose_rmsd(pose_mol, crystal_mol_):
    """Symmetry-corrected heavy-atom RMSD in the fixed receptor frame.

    ``CalcRMS``, never ``GetBestRMS``: GetBestRMS superposes the two molecules
    first, which answers "is this the same shape?" when the question here is
    "did it land in the same place?".
    """
    probe, ref = _skeleton(pose_mol), _skeleton(crystal_mol_)
    if probe.GetNumAtoms() != ref.GetNumAtoms():
        raise RedockError('The docked pose and the crystal ligand have different atom '
                          'counts, so no honest RMSD can be computed.')
    try:
        return float(rdMolAlign.CalcRMS(probe, ref))
    except Exception:
        raise RedockError('Could not match the docked pose to the crystal ligand, '
                          'so no honest RMSD can be computed.')


def verdict_for(rmsd):
    if rmsd <= RELIABLE_MAX:
        return 'reliable'
    if rmsd <= MARGINAL_MAX:
        return 'marginal'
    return 'unreliable'


# ── Preparing a run ──────────────────────────────────────────────────────────

def identify(pdb_id):
    """Which ligand this structure would be validated on — the cheap half.

    Split out from prepare() so a cached result can be served without looking up
    a chemical component or embedding a conformer first.

    Raises RedockError (with user-facing copy) when the structure cannot be
    validated.
    """
    pdb_id, pdb_text = fetch_structure(pdb_id)
    chosen, parsed, note = choose_ligand(pdb_text)
    return {'pdb_id': pdb_id, 'pdb_text': pdb_text, 'chosen': chosen,
            'title': parsed['title'][:160], 'note': note}


def prepare(pdb_id, ident=None):
    """Everything needed to launch and later score a validation run."""
    ident = ident or identify(pdb_id)
    pdb_id, pdb_text = ident['pdb_id'], ident['pdb_text']
    chosen, note = ident['chosen'], ident['note']
    template = fetch_component(chosen['comp_id'])
    crystal = crystal_mol(chosen['atoms'], template)
    smiles = Chem.MolToSmiles(crystal)
    if not smiles:
        raise RedockError('Could not rebuild this ligand from the structure, so validation '
                          'is not available here.')

    # The conformer docking will build from this SMILES. embed_3d is deterministic
    # (fixed seed), so measuring it here measures the geometry Vina actually gets.
    try:
        rebuilt = chem_tools.embed_3d(chem_tools.mol_from_smiles(smiles))
    except Exception:
        raise RedockError('Could not build a 3D conformer for this ligand, so validation '
                          'is not available here.')
    rebuilt_heavy = Chem.RemoveHs(rebuilt)
    start_rmsd = pose_rmsd(rebuilt_heavy, crystal)
    if start_rmsd <= MIN_START_RMSD:
        # Would mean we are handing Vina something indistinguishable from the answer.
        raise RedockError('Internal check failed: the rebuilt conformer is already on top '
                          'of the crystal pose, which would make this test meaningless.')
    try:
        shape_rmsd = float(rdMolAlign.GetBestRMS(_skeleton(rebuilt_heavy), _skeleton(crystal)))
    except Exception:
        shape_rmsd = None

    center, size = box_for(chosen['atoms'])
    if not box_contains(chosen['atoms'], center, size):
        raise RedockError(
            f"This ligand ({chosen['comp_id']}, {chosen['heavy_atoms']} heavy atoms) is too "
            f'long to fit in a standard {BOX_MAX:g} Å validation box, so the crystal pose '
            'could not be reproduced inside it and validation is not available here.')
    return {
        'pdb_id': pdb_id,
        'title': ident['title'],
        'ligand': {'comp_id': chosen['comp_id'], 'name': component_name(pdb_text, chosen['comp_id']),
                   'chain': chosen['chain'], 'resi': chosen['resi'],
                   'heavy_atoms': chosen['heavy_atoms'], 'smiles': smiles},
        'box': {'center': center, 'size': size},
        'note': note,
        'receptor_pdb': pdb_text,
        'crystal_sdf': Chem.MolToMolBlock(crystal),
        'start_rmsd': round(start_rmsd, 2),
        'conformer_shape_rmsd': round(shape_rmsd, 2) if shape_rmsd is not None else None,
    }


def score(crystal_sdf, poses):
    """Score the poses a dock job returned against the crystal pose.

    Called from the docking worker, so it must stay cheap and must not raise.
    """
    crystal = Chem.MolFromMolBlock(crystal_sdf, removeHs=True, sanitize=False)
    if crystal is None:
        return {'error': 'The crystal pose could not be re-read for comparison.'}
    rows = []
    failures = []
    last_error = None
    for p in poses:
        mol = None
        if p.get('sdf'):
            mol = Chem.MolFromMolBlock(p['sdf'], removeHs=True, sanitize=False)
        if mol is None:
            failures.append(p['mode'])
            continue
        try:
            rows.append({'rank': p['mode'], 'rmsd': round(pose_rmsd(mol, crystal), 2),
                         'score': p['affinity']})
        except RedockError as e:
            failures.append(p['mode'])
            last_error = str(e)
    if not rows:
        return {'error': last_error or 'None of the docked poses could be compared to the crystal pose.'}
    top = min(rows, key=lambda r: r['rank'])
    best = min(rows, key=lambda r: r['rmsd'])
    return {
        'top_pose': {'rmsd': top['rmsd'], 'score': top['score'], 'rank': top['rank']},
        'best_pose': {'rmsd': best['rmsd'], 'score': best['score'], 'rank': best['rank']},
        'verdict': verdict_for(top['rmsd']),
        'message': VERDICTS[verdict_for(top['rmsd'])],
        'poses': rows,
        'unscored_poses': failures,
    }


# ── Cache ────────────────────────────────────────────────────────────────────
# Shared between users: a validation of 3PTB at the same settings is the same
# run for everybody, and the whole point of caching it is that a first-time
# visitor sees a real number rather than a spinner.

_mem_cache = {}
_mem_lock = threading.Lock()

try:
    import db as _db
    _DB_CACHE = bool(_db.DATABASE_URL and _db.IS_POSTGRES)
except Exception:  # psycopg2 or db.py absent (the docking image can run without them)
    _db = None
    _DB_CACHE = False


def cache_stats():
    with _mem_lock:
        n = len(_mem_cache)
    return {'entries': n, 'store': 'postgres' if _DB_CACHE else 'memory'}


def cache_key(pdb_id, comp_id, exhaustiveness, num_poses, vina_version):
    return '|'.join([pdb_id.upper(), comp_id.upper(), str(exhaustiveness), str(num_poses),
                     (vina_version or 'unknown')[:40]])


def cache_get(key):
    with _mem_lock:
        hit = _mem_cache.get(key)
    if hit:
        return hit
    if not _DB_CACHE:
        return None
    try:
        row = _db.redock_cache_get(key)
    except Exception:
        return None
    if not row:
        return None
    try:
        payload = json.loads(row['payload'])
    except Exception:
        return None
    payload['computed_at'] = str(row.get('created_at') or '')[:19]
    with _mem_lock:
        _mem_cache[key] = payload
    return payload


def cache_put(key, payload):
    payload = dict(payload)
    payload.setdefault('computed_at', time.strftime('%Y-%m-%d %H:%M:%S'))
    with _mem_lock:
        if len(_mem_cache) > 200:
            _mem_cache.clear()
        _mem_cache[key] = payload
    if _DB_CACHE:
        try:
            _db.redock_cache_put(key, json.dumps(payload))
        except Exception:
            pass   # the cache is an optimisation, never a reason to lose a result
    return payload


# ── Jobs ─────────────────────────────────────────────────────────────────────
# A validation run is an ordinary docking job with extra bookkeeping, so it goes
# through the same queue (at lower priority) and the same rate limits (at half
# weight — these runs are short and we want people using them).

_runs = {}
_runs_lock = threading.Lock()
RUN_TTL = 6 * 3600


def _remember(job_id, meta):
    now = time.time()
    with _runs_lock:
        for jid in [j for j, v in _runs.items() if now - v['created'] > RUN_TTL]:
            del _runs[jid]
        _runs[job_id] = dict(meta, created=now)


def start(pdb_id, exhaustiveness, num_poses, user_id=None, ip=None, cid=None):
    """Launch (or serve from cache) a validation run. Returns (job_id, cached)."""
    import docking
    exhaustiveness = max(1, min(int(exhaustiveness or 4), 32))
    num_poses = max(1, min(int(num_poses or 9), 20))
    # Identify the ligand first (one HTTP fetch, no chemistry) so a cache hit is
    # genuinely instant; only a miss pays for the component lookup and the embed.
    ident = identify(pdb_id)   # raises RedockError with user-facing copy
    key = cache_key(ident['pdb_id'], ident['chosen']['comp_id'], exhaustiveness, num_poses,
                    docking.VINA_VERSION)
    hit = cache_get(key)
    if hit:
        job_id = 'cached-' + hashlib.sha1(key.encode()).hexdigest()[:12]
        _remember(job_id, {'cached': hit})
        return job_id, True

    prep = prepare(ident['pdb_id'], ident)
    job_id = docking.submit(
        {'receptor_pdb': prep['receptor_pdb'], 'ligand_smiles': prep['ligand']['smiles'],
         'center': prep['box']['center'], 'size': prep['box']['size'],
         'exhaustiveness': exhaustiveness, 'num_modes': num_poses},
        user_id=user_id, ip=ip, cid=cid,
        validation={'crystal_sdf': prep['crystal_sdf']},
        priority=1, rate_weight=0.5)
    meta = {k: v for k, v in prep.items() if k != 'receptor_pdb'}
    meta['settings'] = {'exhaustiveness': exhaustiveness, 'num_poses': num_poses}
    meta['cache_key'] = key
    _remember(job_id, meta)
    return job_id, False


def _payload(meta, extra):
    out = {
        'pdb_id': meta.get('pdb_id'), 'title': meta.get('title'),
        'ligand': meta.get('ligand'), 'box': meta.get('box'),
        'settings': meta.get('settings'), 'note': meta.get('note'),
        'start_rmsd': meta.get('start_rmsd'),
        'conformer_shape_rmsd': meta.get('conformer_shape_rmsd'),
        'crystal_sdf': meta.get('crystal_sdf'),
        'vina': None,
        'thresholds': {'reliable': RELIABLE_MAX, 'marginal': MARGINAL_MAX},
    }
    out.update(extra)
    return out


def status(job_id):
    """The documented status shape, or None when the run is unknown."""
    import docking
    with _runs_lock:
        meta = _runs.get(job_id)
    if meta is None:
        return None
    if 'cached' in meta:
        return dict(meta['cached'], status='done', cached=True)

    job = docking.get_job(job_id)
    if job is None:
        return _payload(meta, {'status': 'error', 'cached': False,
                               'error': 'This validation run expired or the server restarted.'})
    out = _payload(meta, {'status': job['status'], 'cached': False,
                          'progress': job.get('progress', 0),
                          'message': job.get('message'), 'log': job.get('log', []),
                          'elapsed': job.get('elapsed')})
    if job['status'] == 'error':
        out['error'] = job.get('error') or 'Docking failed'
        return out
    if job['status'] != 'done':
        return out

    result = job.get('result') or {}
    val = result.get('validation') or {}
    if val.get('error') or not val.get('top_pose'):
        out['status'] = 'error'
        out['error'] = val.get('error') or 'The docked poses could not be compared to the crystal pose.'
        return out
    top_mode = val['top_pose']['rank']
    docked = next((p.get('sdf') for p in result.get('poses', []) if p['mode'] == top_mode), None)
    out.update({
        'top_pose': {'rmsd': val['top_pose']['rmsd'], 'score': val['top_pose']['score']},
        'best_pose': val['best_pose'], 'verdict': val['verdict'], 'message': val['message'],
        'poses': val['poses'], 'unscored_poses': val.get('unscored_poses') or [],
        'docked_sdf': docked, 'vina': docking.VINA_VERSION,
        'computed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    })
    key = meta.pop('cache_key', None)   # written once, on the poll that first sees 'done'
    if key:
        cache_put(key, {k: v for k, v in out.items()
                        if k not in ('log', 'progress', 'elapsed')})
    return out


# ── Warm-up ──────────────────────────────────────────────────────────────────
# The docking page offers five example structures. Validating them once, in the
# background, is what lets a first-time visitor click the button and get a real
# result instead of a two-minute wait. Each is skipped when it is already cached,
# so with a Postgres cache this runs once in the lifetime of the deployment,
# not once per restart.

WARM_IDS = [s.strip().upper() for s in
            os.environ.get('REDOCK_WARM_IDS', '3PTB,1HSG,1M17,4EY7,6LU7').split(',') if s.strip()]
WARM_EXHAUSTIVENESS = int(os.environ.get('REDOCK_WARM_EXHAUSTIVENESS', '4'))
WARM_POSES = int(os.environ.get('REDOCK_WARM_POSES', '9'))
WARM_DELAY = int(os.environ.get('REDOCK_WARM_DELAY', '45'))


def warm(ids=None, delay=0):
    """Validate the example structures once, one at a time, lowest priority."""
    import docking
    if delay:
        time.sleep(delay)
    for pdb_id in (ids or WARM_IDS):
        if not docking.DOCKING_AVAILABLE:
            return
        try:
            job_id, cached = start(pdb_id, WARM_EXHAUSTIVENESS, WARM_POSES, user_id='warm')
            if cached:
                print(f'[redock] {pdb_id} already cached')
                continue
            while True:
                time.sleep(3)
                st = status(job_id)
                if not st or st['status'] in ('done', 'error'):
                    print(f"[redock] warmed {pdb_id}: "
                          f"{st and (st.get('verdict') or st.get('error'))}")
                    break
        except RedockError as e:
            print(f'[redock] {pdb_id} cannot be validated: {e}')
        except Exception as e:   # pragma: no cover - warm-up must never crash the service
            print(f'[redock] warm-up failed for {pdb_id}: {e}')


def start_warm_thread():
    if os.environ.get('REDOCK_WARM', '1') not in ('1', 'true', 'yes'):
        return
    threading.Thread(target=warm, args=(None, WARM_DELAY), name='redock-warm', daemon=True).start()
