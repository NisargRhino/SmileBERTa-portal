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
import multiprocessing as mp
import os
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

    Returns a list of plain dicts, ready to serialise. Exceptions are captured
    as rejections rather than raised, because one malformed entry must not take
    down a build that has been running for six hours.
    """
    out = []
    try:
        found = C.select(pdb_id)
    except C.Rejected as e:
        return [{'pdb_id': pdb_id, 'rejected': e.reason, 'detail': e.detail}]
    except Exception as e:                        # pragma: no cover - defensive
        return [{'pdb_id': pdb_id, 'rejected': 'error', 'detail': str(e)[:120]}]

    for rec in found:
        if rec.get('rejected'):
            out.append({'pdb_id': pdb_id, 'comp_id': rec.get('comp_id'),
                        'rejected': rec['rejected'], 'detail': rec.get('detail', '')})
            continue
        try:
            ifp = I.fingerprint(rec['ligand_mol'], rec['pdb_text'])
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


def build(release, ids, workers=4, shard_size=500, log=print):
    os.makedirs(os.path.join(release, 'records'), exist_ok=True)
    sources.sifts_mapping()   # one download, before the workers fork
    already = done_ids(release)
    todo = [i for i in ids if i not in already]
    log(f'{len(ids)} candidates, {len(already)} already done, {len(todo)} to process')
    if not todo:
        return summarise(release)

    index = len([n for n in os.listdir(os.path.join(release, 'records'))
                 if n.endswith('.jsonl')])
    started = time.time()
    written = 0
    handle = open(shard_path(release, index), 'a')
    try:
        with mp.Pool(workers) as pool:
            for n, records in enumerate(pool.imap_unordered(process, todo, chunksize=1), 1):
                for rec in records:
                    handle.write(json.dumps(rec) + '\n')
                written += 1
                if written % shard_size == 0:
                    handle.close()
                    index += 1
                    handle = open(shard_path(release, index), 'a')
                if n % 25 == 0:
                    handle.flush()
                    rate = n / max(time.time() - started, 1e-6)
                    remaining = (len(todo) - n) / max(rate, 1e-9) / 60
                    log(f'  {n}/{len(todo)}  {rate*60:.0f}/min  ~{remaining:.0f} min left')
    finally:
        handle.close()
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
    args = ap.parse_args()

    if args.summarise_only:
        print(json.dumps(summarise(args.release), indent=2))
        return

    if args.ids:
        ids = args.ids
    else:
        ids = sources.candidate_ids(os.path.join(args.release, 'candidate_ids.json'))
        if args.limit:
            ids = ids[:args.limit]
    manifest = build(args.release, ids, workers=args.workers)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'sources'}, indent=2))


if __name__ == '__main__':
    main()
