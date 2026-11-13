"""Protein–ligand docking with AutoDock Vina.

Pipeline:
  receptor PDB  ──clean──▶ protein atoms ──Open Babel (+polar H, -xr)──▶ receptor.pdbqt
  ligand SMILES ──RDKit ETKDG + MMFF94──▶ 3D mol ──Open Babel──▶ ligand.pdbqt
  vina --receptor --ligand --center --size --exhaustiveness ──▶ poses.pdbqt
  poses ──Open Babel──▶ SDF  +  contact / H-bond analysis against the receptor

Jobs are handed to a fixed worker pool (DOCK_CONCURRENCY threads, each running
Vina on VINA_CPU cores) through a FIFO queue, and the client polls for progress
and queue position.
"""
import collections
import hashlib
import os
import queue
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
VINA_VERSION = None
if VINA_BIN:
    try:
        VINA_VERSION = subprocess.run([VINA_BIN, '--version'], capture_output=True, text=True, timeout=20).stdout.strip()[:80] or None
    except Exception:
        VINA_VERSION = None
DOCKING_AVAILABLE = bool(OPENBABEL_AVAILABLE and VINA_VERSION)

MAX_BOX = 40.0
MAX_RECEPTOR_ATOMS = 40000
JOB_TTL = 3 * 3600

# ── Throughput knobs ─────────────────────────────────────────────────────────
# Tuned for a dedicated-core host. On a 4-core box the default (2 jobs x 2 cores)
# keeps every core busy while one job's single-threaded prep (Open Babel, ETKDG)
# overlaps with another job's Vina search.
DOCK_CONCURRENCY = max(1, int(os.environ.get('DOCK_CONCURRENCY', '2')))
VINA_CPU = os.environ.get('VINA_CPU', '2')
# Nice level for the Vina child. The old value of 19 existed only because Vina
# would starve gunicorn's health checks on a 0.1-vCPU free instance; on dedicated
# cores it just makes docking slower for no benefit.
DOCK_NICE = int(os.environ.get('DOCK_NICE', '0'))
VINA_TIMEOUT = int(os.environ.get('VINA_TIMEOUT', '900'))
# Per-caller fairness, in two layers.
#
# A single IP bucket is wrong for the audience: a whole class on one university
# NAT shares one address, so a per-IP cap of a few jobs locks out twenty people
# who are each behind it legitimately. But the IP is the only thing a client
# cannot forge, so it still has to be the backstop.
#
# Layer 1 — the caller: the account id if known, otherwise a random id the
#   browser keeps in local storage. Tight limits, because this tracks one person
#   at one desk. A determined abuser can rotate the id, which is what layer 2
#   is for.
# Layer 2 — the IP: much higher limits, sized so a whole lab fits under it while
#   a single machine hammering the queue still does not.
MAX_JOBS_PER_USER = max(1, int(os.environ.get('MAX_JOBS_PER_USER', '3')))
MAX_JOBS_PER_CLIENT = max(1, int(os.environ.get('MAX_JOBS_PER_CLIENT', '2')))
MAX_JOBS_PER_IP = max(1, int(os.environ.get('MAX_JOBS_PER_IP', '8')))
# Rolling-window cap, so sequential submissions are limited too, not just
# concurrent ones.
RATE_WINDOW_SEC = max(60, int(os.environ.get('RATE_WINDOW_SEC', '3600')))
MAX_JOBS_PER_WINDOW_USER = max(1, int(os.environ.get('MAX_JOBS_PER_WINDOW_USER', '100')))
MAX_JOBS_PER_WINDOW_CLIENT = max(1, int(os.environ.get('MAX_JOBS_PER_WINDOW_CLIENT', '40')))
MAX_JOBS_PER_WINDOW_IP = max(1, int(os.environ.get('MAX_JOBS_PER_WINDOW_IP', '400')))
MAX_QUEUE = max(1, int(os.environ.get('MAX_QUEUE', '60')))
RECEPTOR_CACHE_SIZE = max(0, int(os.environ.get('RECEPTOR_CACHE_SIZE', '24')))

# identity ─▶ deque of recent submission timestamps, for the rolling window.
_recent = {}
_recent_lock = threading.Lock()

_jobs = {}
_jobs_lock = threading.Lock()
_queue = queue.Queue()
_workers_started = False
_workers_lock = threading.Lock()
# clean-PDB sha1 ─▶ receptor pdbqt. Receptor prep is deterministic and costs
# several seconds, and users dock many ligands against the same target.
_rec_cache = collections.OrderedDict()
_rec_cache_lock = threading.Lock()


def _ensure_workers():
    """Start the worker pool on first submit (not at import — gunicorn may preload)."""
    global _workers_started
    with _workers_lock:
        if _workers_started:
            return
        for i in range(DOCK_CONCURRENCY):
            threading.Thread(target=_worker, name=f'dock-worker-{i}', daemon=True).start()
        _workers_started = True


def _worker():
    while True:
        args = _queue.get()
        try:
            _run_job(*args)
        except Exception:  # pragma: no cover - _run_job handles its own errors
            pass
        finally:
            _queue.task_done()


def _cached_receptor_pdbqt(clean_pdb):
    if not RECEPTOR_CACHE_SIZE:
        return receptor_to_pdbqt(clean_pdb), False
    key = hashlib.sha1(clean_pdb.encode()).hexdigest()
    with _rec_cache_lock:
        if key in _rec_cache:
            _rec_cache.move_to_end(key)
            return _rec_cache[key], True
    pdbqt = receptor_to_pdbqt(clean_pdb)
    with _rec_cache_lock:
        _rec_cache[key] = pdbqt
        _rec_cache.move_to_end(key)
        while len(_rec_cache) > RECEPTOR_CACHE_SIZE:
            _rec_cache.popitem(last=False)
    return pdbqt, False

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


def rss_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return round(int(line.split()[1]) / 1024)
    except Exception:
        return None


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


def client_ip(request):
    """The caller's IP, read from the proxy header.

    Caddy *appends* the real peer address to any X-Forwarded-For the client
    sent, so the last entry is the one the proxy added and the only one a
    client cannot forge. Taking the first entry would let anyone bypass the
    rate limit by sending their own header.
    """
    xff = request.headers.get('X-Forwarded-For', '')
    if xff:
        parts = [p.strip() for p in xff.split(',') if p.strip()]
        if parts:
            return parts[-1]
    return request.remote_addr or 'unknown'


_CLIENT_ID_RE = re.compile(r'^[A-Za-z0-9_-]{8,64}$')


def client_id(request):
    """A per-browser id the client supplies, used only to subdivide an IP.

    This is not a security boundary — anyone can rotate it — which is exactly
    why the IP backstop exists. Its job is to stop twenty people behind one
    campus NAT from sharing one bucket.
    """
    cid = (request.headers.get('X-SmileBERTa-Client') or '').strip()
    return cid if _CLIENT_ID_RE.match(cid) else None


def _buckets(user_id, ip, cid):
    """The (identity, concurrent limit, window limit) pairs a job must satisfy.

    Checked in order, narrowest first, so the message a user sees is about
    their own usage rather than the shared backstop whenever possible.
    """
    out = []
    if user_id is not None:
        out.append((('u', user_id), MAX_JOBS_PER_USER, MAX_JOBS_PER_WINDOW_USER))
    elif cid:
        out.append((('c', cid), MAX_JOBS_PER_CLIENT, MAX_JOBS_PER_WINDOW_CLIENT))
    # The IP backstop always applies, including to signed-in callers.
    out.append((('ip', ip or 'unknown'), MAX_JOBS_PER_IP, MAX_JOBS_PER_WINDOW_IP))
    return out


def _check_window(ident, limit):
    """Rolling-window rate limit. Raises DockingError when the caller is over it."""
    now = time.time()
    cutoff = now - RATE_WINDOW_SEC
    with _recent_lock:
        # Drop identities that have gone quiet, so this cannot grow without bound.
        for key in [k for k, v in _recent.items() if not v or v[-1] < cutoff]:
            if key != ident:
                del _recent[key]
        hits = _recent.setdefault(ident, collections.deque())
        while hits and hits[0] < cutoff:
            hits.popleft()
        if len(hits) >= limit:
            wait = int((hits[0] + RATE_WINDOW_SEC - now) / 60) + 1
            raise DockingError(
                f'Rate limit reached: {limit} docking jobs per '
                f'{RATE_WINDOW_SEC // 60} minutes. Try again in about {wait} minute'
                f'{"s" if wait != 1 else ""}.')
        hits.append(now)


def _release_window(ident):
    """Undo the slot taken by _check_window when the job never gets queued."""
    with _recent_lock:
        hits = _recent.get(ident)
        if hits:
            hits.pop()


def submit(params, user_id=None, ip=None, cid=None):
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
    buckets = _buckets(user_id, ip, cid)
    # Rolling window first: it is the cheap check and the one that stops a
    # script submitting thousands of jobs back to back. Every bucket must pass;
    # charged slots are refunded if a later check refuses the job.
    charged = []
    try:
        for ident, _max_active, max_window in buckets:
            _check_window(ident, max_window)
            charged.append(ident)
    except DockingError:
        for ident in charged:
            _release_window(ident)
        raise

    # The job is attributed to the narrowest bucket, which is the one that
    # should see it as "yours" when counting concurrent jobs.
    own = buckets[0][0]
    job_id = uuid.uuid4().hex[:12]
    active = ('queued', 'preparing', 'running')
    try:
        with _jobs_lock:
            pending = [j for j in _jobs.values() if j['status'] in active]
            if len(pending) >= MAX_QUEUE:
                raise DockingError('The docking queue is full right now — please try again in a few minutes')
            for ident, max_active, _mw in buckets:
                mine = sum(1 for j in pending if ident in j.get('idents', ()))
                if mine >= max_active:
                    shared = ident[0] == 'ip' and len(buckets) > 1
                    raise DockingError(
                        f'{"This network already has" if shared else "You already have"} {mine} '
                        f'docking job{"s" if mine != 1 else ""} running or queued '
                        f'(limit {max_active}). Wait for one to finish.')
            ahead = len(pending)
            _jobs[job_id] = {'id': job_id, 'user_id': user_id, 'ident': own,
                             'idents': tuple(b[0] for b in buckets), 'status': 'queued',
                             'progress': 0,
                             'message': f'Waiting in queue (position {ahead})' if ahead else 'Starting',
                             'log': [], 'created': time.time(), 'result': None, 'error': None}
    except DockingError:
        # Do not charge the caller a rate-limit slot for a job we refused.
        for ident in charged:
            _release_window(ident)
        raise
    _ensure_workers()
    _queue.put((job_id, receptor, params.get('chains'), params.get('ligand_smiles'),
                params.get('ligand_molblock'), center, size, exh, modes))
    return job_id


def queue_stats():
    with _jobs_lock:
        running = sum(1 for j in _jobs.values() if j['status'] in ('preparing', 'running'))
        queued = sum(1 for j in _jobs.values() if j['status'] == 'queued')
    return {'running': running, 'queued': queued, 'concurrency': DOCK_CONCURRENCY,
            'vina_cpu': VINA_CPU, 'receptors_cached': len(_rec_cache)}


def get_job(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        # 'ident' holds the account id or the client IP — never send it to a
        # caller, who may not be the person who submitted the job.
        out = {k: v for k, v in job.items() if k not in ('user_id', 'ident', 'idents')}
        out['log'] = list(job['log'])
        out['elapsed'] = round(time.time() - job['created'], 1)
        if job['status'] == 'queued':
            ahead = sum(1 for j in _jobs.values()
                        if j['status'] == 'queued' and j['created'] < job['created'])
            out['queue_position'] = ahead + 1
            out['message'] = f'Waiting in queue (position {ahead + 1})' if ahead else 'Starting shortly'
        return out


VINA_MEM_MB = int(os.environ.get('VINA_MEM_MB', '1200'))


def _limit_child():
    """Optional nice level + a hard address-space cap so Vina can never take the container down."""
    if DOCK_NICE:
        os.nice(DOCK_NICE)
    try:
        import resource
        cap = VINA_MEM_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except Exception:
        pass


def _run_job(job_id, receptor_pdb, chains, smiles, molblock, center, size, exh, modes):
    workdir = tempfile.mkdtemp(prefix='dock_')
    try:
        # If this runs inside the ML API, free the transformer models first.
        import sys
        if 'models' in sys.modules:
            try:
                sys.modules['models'].unload_all()
            except Exception:
                pass
        _update(job_id, status='preparing', progress=2, message='Preparing receptor')
        clean = clean_receptor(receptor_pdb, chains)
        rec_atoms = parse_pdb_atoms(clean)
        _log(job_id, f'Receptor cleaned: {len(rec_atoms)} heavy atoms kept')
        rec_pdbqt, cached = _cached_receptor_pdbqt(clean)
        _log(job_id, 'Receptor reused from cache (same structure and chains)' if cached else
                     'Polar hydrogens added at pH 7.4, AutoDock atom types assigned')

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
               '--cpu', str(VINA_CPU), '--seed', '42', '--out', out_path]
        _update(job_id, status='running', progress=10, message='Vina: computing grid maps & Monte-Carlo search')
        _log(job_id, f'Search box center ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) '
                     f'size {size[0]:.0f}×{size[1]:.0f}×{size[2]:.0f} Å, exhaustiveness {exh}')
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                preexec_fn=_limit_child)
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
            if time.time() - start > VINA_TIMEOUT:
                proc.kill()
                raise DockingError('Docking timed out — try a smaller box or lower exhaustiveness')
        proc.wait()
        output = ''.join(tail)
        if proc.returncode != 0 or not os.path.exists(out_path):
            lines = [l for l in output.strip().splitlines() if l.strip()]
            msg = ' | '.join(lines[-3:]) if lines else f'Vina failed (exit code {proc.returncode})'
            if proc.returncode < 0 or 'bad_alloc' in output or 'memory' in output.lower():
                msg = f'Vina ran out of memory (limit {VINA_MEM_MB} MB). Try a smaller box. [{msg[:160]}]'
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
