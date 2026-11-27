"""Build the corpus: candidates in, annotated complexes and a manifest out.

Resumable by construction. A full build touches a hundred thousand entries over
many hours and *will* be interrupted — by a dropped connection, a throttled API
or a closed laptop. Every entry's outcome is appended to a JSONL shard as it
completes, and a restart skips what is already recorded, so an interrupted
build costs the entry it was working on and nothing else.

Rejections are written to the same shards with their reason. "How many
structures did we lose, and to what" belongs in the manifest: a corpus whose
exclusions are invisible cannot be audited, and the filter counts are among the
first things a reviewer will ask for.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter

from pharmacophore import complexes as C
from pharmacophore import interactions as I
from pharmacophore import sources

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_DIR = os.environ.get('SB_CORPUS_DIR', os.path.join(ROOT, 'corpus'))


def shard_path(release, index):
    return os.path.join(release, 'records', f'{index:04d}.jsonl')


def done_ids(release):
    """PDB IDs already processed, from the shards on disk."""
    seen = set()
    records_dir = os.path.join(release, 'records')
    if not os.path.isdir(records_dir):
        return seen
    for name in sorted(os.listdir(records_dir)):
        if not name.endswith('.jsonl'):
            continue
        with open(os.path.join(records_dir, name)) as f:
            for line in f:
                try:
                    seen.add(json.loads(line)['pdb_id'])
                except Exception:
                    continue
    return seen


def process(pdb_id):
    """Every outcome for one PDB entry: annotated complexes and/or rejections.

    Exceptions become rejections rather than propagating: one malformed entry
    must not take down a build that has been running for hours. Wall-clock
    runaways are handled by the batch supervisor, not here — a signal cannot
    interrupt a C extension mid-call, and trying killed workers and hung the
    pool instead.
    """
    return _process(pdb_id)


def _process(pdb_id):
    out = []
    try:
        found = C.select(pdb_id)
    except C.Rejected as e:
        return [{'pdb_id': pdb_id, 'rejected': e.reason, 'detail': e.detail}]
    except Exception as e:                        # pragma: no cover - defensive
        return [{'pdb_id': pdb_id, 'rejected': 'error', 'detail': str(e)[:120]}]

    protein_mol = None
    for rec in found:
        if rec.get('rejected'):
            out.append({'pdb_id': pdb_id, 'comp_id': rec.get('comp_id'),
                        'rejected': rec['rejected'], 'detail': rec.get('detail', '')})
            continue
        try:
            # Prepared once and reused for every ligand in this entry.
            if protein_mol is None:
                protein_mol = I.protein_molecule(rec['pdb_text'])
            ifp = I.fingerprint(rec['ligand_mol'], protein_mol=protein_mol,
                                pocket=rec['pocket'])
        except Exception as e:
            out.append({'pdb_id': pdb_id, 'comp_id': rec['comp_id'],
                        'rejected': 'interaction_typing_failed', 'detail': str(e)[:120]})
            continue
        if not ifp['contacts']:
            # A ligand in a pocket with no detected interaction is either a
            # crystallisation artefact or a typing failure. Either way it
            # carries no signal, and an all-zero label row would teach the
            # model that this pocket wants nothing.
            out.append({'pdb_id': pdb_id, 'comp_id': rec['comp_id'],
                        'rejected': 'no_interactions_detected', 'detail': ''})
            continue
        out.append({
            'pdb_id': pdb_id,
            'resolution': rec['resolution'],
            'comp_id': rec['comp_id'],
            'chain': rec['chain'],
            'resi': rec['resi'],
            'uniprot': rec['uniprot'],
            'smiles': rec['smiles'],
            'inchikey': rec['inchikey'],
            'heavy_atoms': rec['heavy_atoms'],
            'pocket_residues': [f'{r[0]} {r[1]}{r[2]}' for r in rec['pocket']],
            'n_pocket_residues': len(rec['pocket']),
            'per_atom': {str(k): v for k, v in ifp['per_atom'].items()},
            'contacts': ifp['contacts'],
            'by_type': ifp['by_type'],
            'pocket_profile': I.pocket_feature_profile(ifp),
            'n_interactions': len(ifp['contacts']),
            'rejected': None,
        })
    return out


BATCH_SIZE = int(os.environ.get('SB_BATCH_SIZE', '40'))
# A batch that overruns this is killed by the operating system, which is the
# only thing that reliably stops a C extension mid-call. Sized so a normal
# batch finishes comfortably inside it.
BATCH_TIMEOUT = int(os.environ.get('SB_BATCH_TIMEOUT', '900'))


def run_batch(release, ids, shard):
    """Process `ids` and append their outcomes to one shard. Runs in its own process."""
    path = shard_path(release, shard)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as handle:
        for pdb_id in ids:
            for record in process(pdb_id):
                handle.write(json.dumps(record) + '\n')
            handle.flush()


def build(release, ids, workers=4, log=print):
    """Supervise batch subprocesses.

    Deliberately boring. An earlier version used multiprocessing.Pool with a
    signal-based per-entry timeout; it hung twice, because SIGALRM cannot
    interrupt RDKit or ProLIF mid-call and firing it near a pipe write kills the
    worker, after which imap_unordered waits for a result that never comes.

    Here each batch is an independent process that the OS can terminate
    unconditionally. A killed batch loses only its in-flight entries, which are
    simply retried on the next pass, because the records on disk are what define
    progress.
    """
    os.makedirs(os.path.join(release, 'records'), exist_ok=True)
    sources.sifts_mapping()            # one download before any child starts
    already = done_ids(release)
    todo = [i for i in ids if i not in already]
    log(f'{len(ids)} candidates, {len(already)} already done, {len(todo)} to process')
    if not todo:
        return summarise(release)

    existing = len([n for n in os.listdir(os.path.join(release, 'records'))
                    if n.endswith('.jsonl')])
    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    log(f'{len(batches)} batches of up to {BATCH_SIZE}, {workers} at a time')

    started = time.time()
    completed = timed_out = 0
    ids_file_dir = os.path.join(release, 'batches')
    os.makedirs(ids_file_dir, exist_ok=True)

    def launch(index, batch):
        ids_path = os.path.join(ids_file_dir, f'{index:05d}.json')
        with open(ids_path, 'w') as f:
            json.dump(batch, f)
        return subprocess.Popen(
            [sys.executable, '-m', 'pharmacophore.build', '--release', release,
             '--run-batch', ids_path, '--shard', str(existing + index)],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env={**os.environ, 'PYTHONPATH': ROOT})

    running = {}
    queue = list(enumerate(batches))
    while queue or running:
        while queue and len(running) < workers:
            index, batch = queue.pop(0)
            running[launch(index, batch)] = (index, time.time())
        time.sleep(2)
        for proc in list(running):
            index, began = running[proc]
            if proc.poll() is not None:
                del running[proc]
                completed += 1
            elif time.time() - began > BATCH_TIMEOUT:
                proc.kill()
                del running[proc]
                completed += 1
                timed_out += 1
                log(f'  batch {index} exceeded {BATCH_TIMEOUT}s and was killed')
        if completed and completed % 5 == 0:
            entries = len(done_ids(release)) - len(already)
            rate = entries / max(time.time() - started, 1e-6) * 60
            left = (len(todo) - entries) / max(rate, 1e-9)
            log(f'  {entries}/{len(todo)} entries  {rate:.0f}/min  ~{left/60:.1f} h left'
                + (f'  ({timed_out} batches killed)' if timed_out else ''))
            time.sleep(3)          # so the modulo does not log repeatedly
    log(f'done: {completed} batches, {timed_out} killed')
    return summarise(release)


def iter_records(release):
    records_dir = os.path.join(release, 'records')
    for name in sorted(os.listdir(records_dir)):
        if not name.endswith('.jsonl'):
            continue
        with open(os.path.join(records_dir, name)) as f:
            for line in f:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def summarise(release):
    """Manifest numbers: what came in, what was kept, what was lost and why."""
    kept, rejected = [], Counter()
    interactions = Counter()
    per_atom_positive = total_atoms = 0
    entries = set()
    for rec in iter_records(release):
        entries.add(rec['pdb_id'])
        if rec.get('rejected'):
            rejected[rec['rejected']] += 1
            continue
        kept.append(rec)
        interactions.update(rec['by_type'])
        per_atom_positive += len(rec['per_atom'])
        total_atoms += rec['heavy_atoms']

    uniprots = {r['uniprot'] for r in kept}
    ligands = {r['inchikey'] for r in kept}
    manifest = {
        'entries_processed': len(entries),
        'complexes_kept': len(kept),
        'rejected': dict(rejected.most_common()),
        'distinct_proteins': len(uniprots),
        'distinct_ligands': len(ligands),
        'interactions_total': sum(interactions.values()),
        'interactions_by_type': dict(interactions.most_common()),
        'mean_interactions_per_complex': round(sum(interactions.values()) / max(len(kept), 1), 2),
        'ligand_atoms_total': total_atoms,
        'ligand_atoms_with_an_interaction': per_atom_positive,
        # The label sparsity, which decides how the model has to be trained and
        # which metrics are meaningful: accuracy is useless at this rate.
        'positive_atom_rate': round(per_atom_positive / max(total_atoms, 1), 4),
        'sources': sources.credits_block(),
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(release, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', default=os.path.join(RELEASE_DIR, 'v1'))
    ap.add_argument('--limit', type=int, help='cap candidate entries (pilot runs)')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument('--ids', nargs='*', help='explicit PDB IDs instead of a search')
    ap.add_argument('--summarise-only', action='store_true')
    ap.add_argument('--shuffle-seed', type=int, default=20261126)
    ap.add_argument('--run-batch', help='internal: process the ids in this file')
    ap.add_argument('--shard', type=int, default=0)
    args = ap.parse_args()

    if args.run_batch:
        with open(args.run_batch) as f:
            run_batch(args.release, json.load(f), args.shard)
        return

    if args.summarise_only:
        print(json.dumps(summarise(args.release), indent=2))
        return

    if args.ids:
        ids = args.ids
    else:
        ids = sources.candidate_ids(os.path.join(args.release, 'candidate_ids.json'))
        # Shuffled with a fixed seed. RCSB returns candidates in ID order, so
        # processing them as given means the first hours are spent on 100D,
        # 101D, 102L and the rest of the alphabetical head - disproportionately
        # nucleic-acid and legacy entries, which is why 42 of the first 150 were
        # mmCIF-only. Shuffling costs nothing, makes partial progress a
        # representative sample of the whole PDB, and the seed keeps the order
        # reproducible across restarts so resume still works.
        random.Random(args.shuffle_seed).shuffle(ids)
        if args.limit:
            ids = ids[:args.limit]
    manifest = build(args.release, ids, workers=args.workers)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'sources'}, indent=2))


if __name__ == '__main__':
    main()
