"""Protein–ligand docking with AutoDock Vina.

Pipeline:
  receptor PDB  ──clean──▶ protein atoms ──Open Babel (+polar H, -xr)──▶ receptor.pdbqt
  ligand SMILES ──RDKit ETKDG + MMFF94──▶ 3D mol ──Open Babel──▶ ligand.pdbqt
  vina --receptor --ligand --center --size --exhaustiveness ──▶ poses.pdbqt
  poses ──Open Babel──▶ SDF  +  contact / H-bond analysis against the receptor

Jobs run in a background thread (one at a time — Vina is CPU bound) and the
client polls for progress.
"""
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

import numpy as np
from rdkit import Chem

from chem_tools import embed_3d, mol_from_smiles, ChemError

try:
    from openbabel import pybel, openbabel as ob
    ob.obErrorLog.SetOutputLevel(0)
    OPENBABEL_AVAILABLE = True
except Exception:  # pragma: no cover
    OPENBABEL_AVAILABLE = False

VINA_BIN = os.environ.get('VINA_BIN') or shutil.which('vina')
DOCKING_AVAILABLE = bool(OPENBABEL_AVAILABLE and VINA_BIN)

MAX_BOX = 40.0
MAX_RECEPTOR_ATOMS = 40000
JOB_TTL = 3 * 3600

_jobs = {}
_jobs_lock = threading.Lock()
_run_lock = threading.Semaphore(1)

AA = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
      'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HID', 'HIE', 'HIP', 'CYX', 'MSE', 'SEC', 'PYL'}
METALS = {'ZN', 'MG', 'MN', 'CA', 'FE', 'NI', 'CO', 'CU'}


class DockingError(ValueError):
    pass


# ── Receptor ─────────────────────────────────────────────────────────────────

def clean_receptor(pdb_text, chains=None, keep_metals=True):
    """Keep protein atoms (first model, first altloc) of the selected chains."""
    chains = set(c for c in (chains or []) if c)
    out = []
    seen_model = False
    n = 0
    for line in pdb_text.splitlines():
        rec = line[:6]
        if rec.startswith('MODEL'):
            if seen_model:
                break
            seen_model = True
            continue
        if rec.startswith('ENDMDL'):
            break
        if rec not in ('ATOM  ', 'HETATM'):
            continue
        resn = line[17:20].strip()
        chain = line[21:22]
        if chains and chain not in chains:
            continue
        is_protein = rec == 'ATOM  ' or resn in AA
        is_metal = keep_metals and rec == 'HETATM' and resn in METALS
        if not (is_protein or is_metal):
            continue
        alt = line[16:17]
        if alt not in (' ', 'A'):
            continue
        element = line[76:78].strip() if len(line) >= 78 else ''
        if element == 'H' or (not element and line[12:16].strip().startswith('H')):
            continue  # hydrogens are re-added consistently below
        line = line[:16] + ' ' + line[17:]
        out.append(line)
        n += 1
    if n == 0:
        raise DockingError('No protein atoms found in the receptor (check the chain selection)')
    if n > MAX_RECEPTOR_ATOMS:
        raise DockingError(f'Receptor too large ({n} atoms). Select fewer chains.')
    return '\n'.join(out) + '\nEND\n'


def receptor_to_pdbqt(clean_pdb):
    mol = pybel.readstring('pdb', clean_pdb)
    mol.OBMol.AddHydrogens(True, True, 7.4)  # polar only, protonation at pH 7.4
    text = mol.write('pdbqt', opt={'r': None, 'c': None, 'n': None})
    lines = [l for l in text.splitlines() if l.startswith(('ATOM', 'HETATM', 'TER'))]
    if not lines:
        raise DockingError('Receptor preparation failed')
    return '\n'.join(lines) + '\n'


AD_TYPES = {'A': 'C', 'NA': 'N', 'NS': 'N', 'OA': 'O', 'OS': 'O', 'SA': 'S', 'HD': 'H', 'HS': 'H'}


def parse_pdb_atoms(pdb_text, pdbqt=False):
    atoms = []
    for line in pdb_text.splitlines():
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        name = line[12:16].strip()
        if pdbqt:
            t = line[77:79].strip()
            el = AD_TYPES.get(t, t)
        else:
            el = line[76:78].strip() if len(line) >= 78 else ''
        el = el or re.sub(r'[^A-Za-z]', '', name)[:1]
        atoms.append({'name': name, 'resn': line[17:20].strip(), 'chain': line[21:22].strip(),
                      'resi': line[22:26].strip(), 'el': el.upper(), 'xyz': (x, y, z)})
    return atoms


# ── Ligand ───────────────────────────────────────────────────────────────────

def ligand_to_pdbqt(smiles=None, molblock=None):
    if molblock:
        mol = Chem.MolFromMolBlock(molblock, removeHs=False)
        if mol is None:
            raise DockingError('Could not read the ligand structure')
        if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
            mol = embed_3d(Chem.RemoveHs(mol))
        elif not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
            mol = Chem.AddHs(mol, addCoords=True)
        canonical = Chem.MolToSmiles(Chem.RemoveHs(mol))
    else:
        try:
            base = mol_from_smiles(smiles)
        except ChemError as e:
            raise DockingError(str(e))
        if base.GetNumHeavyAtoms() > 120:
            raise DockingError('Ligand too large for docking (max 120 heavy atoms)')
        mol = embed_3d(base)
        canonical = Chem.MolToSmiles(base)
    obmol = pybel.readstring('mol', Chem.MolToMolBlock(mol))
    text = obmol.write('pdbqt')
    if 'ROOT' not in text:
        raise DockingError('Ligand preparation failed')
    return text, canonical, mol.GetNumHeavyAtoms()


# ── Results ──────────────────────────────────────────────────────────────────

def split_models(pdbqt):
    models, cur = [], []
    for line in pdbqt.splitlines():
        if line.startswith('MODEL'):
            cur = []
        elif line.startswith('ENDMDL'):
            models.append('\n'.join(cur) + '\n')
        else:
            cur.append(line)
    return models


def pose_to_sdf(model_pdbqt):
    try:
        mol = pybel.readstring('pdbqt', model_pdbqt)
        return mol.write('sdf')
    except Exception:
        return None


def analyze_contacts(pose_pdbqt, receptor_atoms, rec_xyz):
    lig = [a for a in parse_pdb_atoms(pose_pdbqt, pdbqt=True) if a['el'] != 'H']
    if not lig:
        return {'residues': [], 'hbonds': [], 'hydrophobic': 0}
    lig_xyz = np.array([a['xyz'] for a in lig])
    d = np.linalg.norm(rec_xyz[:, None, :] - lig_xyz[None, :, :], axis=2)
    close_rec, close_lig = np.where(d <= 4.0)
    residues = {}
    hbonds = []
    hydrophobic = 0
    for ri, li in zip(close_rec, close_lig):
        ra, la = receptor_atoms[ri], lig[li]
        key = f"{ra['resn']} {ra['chain']}{ra['resi']}"
        dist = float(d[ri, li])
        residues[key] = min(residues.get(key, 99.0), dist)
        if ra['el'] in ('N', 'O') and la['el'] in ('N', 'O') and dist <= 3.5:
            hbonds.append({'residue': key, 'protein_atom': ra['name'], 'ligand_atom': la['name'],
                           'distance': round(dist, 2), 'p': ra['xyz'], 'l': la['xyz']})
        elif ra['el'] == 'C' and la['el'] == 'C' and dist <= 4.0:
            hydrophobic += 1
    res_list = sorted(({'residue': k, 'min_distance': round(v, 2)} for k, v in residues.items()),
                      key=lambda r: r['min_distance'])
    return {'residues': res_list, 'hbonds': hbonds, 'hydrophobic': hydrophobic}


# ── Jobs ─────────────────────────────────────────────────────────────────────

def _update(job_id, **kw):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            job.update(kw)


def _log(job_id, msg):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            job['log'].append(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _cleanup_jobs():
    now = time.time()
    with _jobs_lock:
        for jid in [j for j, v in _jobs.items() if now - v['created'] > JOB_TTL]:
            del _jobs[jid]


def submit(params, user_id=None):
    if not DOCKING_AVAILABLE:
        raise DockingError('Docking engine is not installed on this server')
    receptor = params.get('receptor_pdb') or ''
    if len(receptor) < 100:
        raise DockingError('Receptor PDB is required')
    if not (params.get('ligand_smiles') or params.get('ligand_molblock')):
        raise DockingError('Ligand SMILES is required')
    try:
        center = [float(v) for v in params['center']]
        size = [float(v) for v in params['size']]
        assert len(center) == 3 and len(size) == 3
    except Exception:
        raise DockingError('Box center and size must be three numbers each')
    if any(s <= 4 or s > MAX_BOX for s in size):
        raise DockingError(f'Box edges must be between 4 and {MAX_BOX:g} Å')
    exh = max(1, min(int(params.get('exhaustiveness', 8)), 32))
    modes = max(1, min(int(params.get('num_modes', 9)), 20))
    _cleanup_jobs()
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        queued = sum(1 for j in _jobs.values() if j['status'] in ('queued', 'preparing', 'running'))
        _jobs[job_id] = {'id': job_id, 'user_id': user_id, 'status': 'queued', 'progress': 0,
                         'message': 'Waiting in queue' if queued else 'Starting',
                         'log': [], 'created': time.time(), 'result': None, 'error': None}
    t = threading.Thread(target=_run_job, args=(job_id, receptor, params.get('chains'),
                                                 params.get('ligand_smiles'), params.get('ligand_molblock'),
                                                 center, size, exh, modes), daemon=True)
    t.start()
    return job_id


def get_job(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        out = {k: v for k, v in job.items() if k != 'user_id'}
        out['log'] = list(job['log'])
        out['elapsed'] = round(time.time() - job['created'], 1)
        return out


def _run_job(job_id, receptor_pdb, chains, smiles, molblock, center, size, exh, modes):
    with _run_lock:
        workdir = tempfile.mkdtemp(prefix='dock_')
        try:
            # Free the transformer models first: on a 512 MB instance they and Vina don't fit together.
            if os.environ.get('MODEL_CACHE_SIZE', '1') == '1':
                try:
                    import models
                    models.unload_all()
                except Exception:
                    pass
            _update(job_id, status='preparing', progress=2, message='Preparing receptor')
            clean = clean_receptor(receptor_pdb, chains)
            rec_atoms = parse_pdb_atoms(clean)
            _log(job_id, f'Receptor cleaned: {len(rec_atoms)} heavy atoms kept')
            rec_pdbqt = receptor_to_pdbqt(clean)
            _log(job_id, 'Polar hydrogens added at pH 7.4, AutoDock atom types assigned')

            _update(job_id, progress=6, message='Preparing ligand')
            lig_pdbqt, canonical, heavy = ligand_to_pdbqt(smiles, molblock)
            torsions = len(re.findall(r'^BRANCH', lig_pdbqt, re.M))
            _log(job_id, f'Ligand embedded (ETKDG + MMFF94): {heavy} heavy atoms, {torsions} rotatable torsions')

            rec_path = os.path.join(workdir, 'receptor.pdbqt')
            lig_path = os.path.join(workdir, 'ligand.pdbqt')
            out_path = os.path.join(workdir, 'out.pdbqt')
            with open(rec_path, 'w') as f:
                f.write(rec_pdbqt)
            with open(lig_path, 'w') as f:
                f.write(lig_pdbqt)

            cmd = [VINA_BIN, '--receptor', rec_path, '--ligand', lig_path,
                   '--center_x', f'{center[0]:.3f}', '--center_y', f'{center[1]:.3f}', '--center_z', f'{center[2]:.3f}',
                   '--size_x', f'{size[0]:.2f}', '--size_y', f'{size[1]:.2f}', '--size_z', f'{size[2]:.2f}',
                   '--exhaustiveness', str(exh), '--num_modes', str(modes), '--energy_range', '4',
                   '--cpu', os.environ.get('VINA_CPU', '1'), '--seed', '42', '--out', out_path]
            _update(job_id, status='running', progress=10, message='Vina: computing grid maps & Monte-Carlo search')
            _log(job_id, f'Search box center ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) '
                         f'size {size[0]:.0f}×{size[1]:.0f}×{size[2]:.0f} Å, exhaustiveness {exh}')
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            stars = 0
            tail = []
            start = time.time()
            while True:
                ch = proc.stdout.read(1)
                if not ch:
                    break
                tail.append(ch)
                if ch == '*':
                    stars += 1
                    _update(job_id, progress=min(97, 10 + int(stars / 51 * 87)))
                if time.time() - start > 1500:
                    proc.kill()
                    raise DockingError('Docking timed out — try a smaller box or lower exhaustiveness')
            proc.wait()
            output = ''.join(tail)
            if proc.returncode != 0 or not os.path.exists(out_path):
                msg = output.strip().splitlines()[-1] if output.strip() else 'Vina failed'
                raise DockingError(msg[:300])

            with open(out_path) as f:
                poses_pdbqt = f.read()
            rec_xyz = np.array([a['xyz'] for a in rec_atoms])
            poses = []
            for i, model in enumerate(split_models(poses_pdbqt)):
                m = re.search(r'REMARK VINA RESULT:\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', model)
                if not m:
                    continue
                poses.append({'mode': i + 1, 'affinity': float(m.group(1)),
                              'rmsd_lb': float(m.group(2)), 'rmsd_ub': float(m.group(3)),
                              'pdbqt': model, 'sdf': pose_to_sdf(model),
                              'interactions': analyze_contacts(model, rec_atoms, rec_xyz)})
            if not poses:
                raise DockingError('Vina produced no poses')
            _log(job_id, f'Done: {len(poses)} poses, best affinity {poses[0]["affinity"]:.2f} kcal/mol '
                         f'({time.time() - start:.0f}s)')
            _update(job_id, status='done', progress=100, message='Complete',
                    result={'ligand_smiles': canonical, 'center': center, 'size': size,
                            'exhaustiveness': exh, 'poses': poses, 'receptor_pdb': clean})
        except (DockingError, ChemError) as e:
            _log(job_id, 'Error: ' + str(e))
            _update(job_id, status='error', error=str(e), message='Failed')
        except Exception as e:  # pragma: no cover
            _log(job_id, 'Unexpected error: ' + str(e))
            _update(job_id, status='error', error='Unexpected error: ' + str(e), message='Failed')
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
