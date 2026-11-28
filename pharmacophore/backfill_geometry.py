"""Add pocket geometry to complexes built before the corpus stored it.

The corpus kept only residue names for most of its life, so the records built
before `complexes.pocket_geometry` existed carry no coordinates. The geometric
model needs them, and re-running the whole build to get them would cost far
more than re-deriving them for the complexes that were actually kept.

Written to a **sidecar** rather than back into the records. The build appends to
those shards continuously, and rewriting a file another process is appending to
is how corpora get silently corrupted. `dataset.py` merges the sidecar in when
it loads, keyed by the complex, so the two can be produced independently and in
any order.

Resumable like the build: every complex resolved is appended immediately and a
restart skips what is already there.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

from pharmacophore import complexes as C

SIDECAR_DIR = 'geometry'


def key_of(record):
    return '%s|%s|%s|%s' % (record.get('pdb_id'), record.get('comp_id'),
                            record.get('chain'), record.get('resi'))


def iter_records(release):
    records_dir = os.path.join(release, 'records')
    if not os.path.isdir(records_dir):
        return
    for name in sorted(os.listdir(records_dir)):
        if not name.endswith('.jsonl'):
            continue
        with open(os.path.join(records_dir, name)) as handle:
            for line in handle:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def already_done(release):
    done = set()
    directory = os.path.join(release, SIDECAR_DIR)
    if not os.path.isdir(directory):
        return done
    for name in sorted(os.listdir(directory)):
        if not name.endswith('.jsonl'):
            continue
        with open(os.path.join(directory, name)) as handle:
            for line in handle:
                try:
                    done.add(json.loads(line)['key'])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def needed(release):
    """{pdb_id: [keys]} for kept complexes that have no geometry yet."""
    done = already_done(release)
    wanted = defaultdict(list)
    for record in iter_records(release):
        if record.get('rejected'):
            continue
        if record.get('pocket_atoms'):
            continue            # built after the change, nothing to do
        key = key_of(record)
        if key in done:
            continue
        wanted[record['pdb_id']].append(key)
    return wanted, len(done)


def run(release, limit=None, shard=0, of=1, log=print):
    wanted, done_count = needed(release)
    ids = sorted(wanted)
    if of > 1:
        # Partitioned by position in the sorted list rather than by hash, so the
        # shards are equal in length and every id belongs to exactly one of
        # them. Each writes its own sidecar file, so they never contend.
        ids = ids[shard::of]
    if limit:
        ids = ids[:limit]
    total = sum(len(v) for v in wanted.values())
    log('%d entries to re-derive, %d complexes, %d already in the sidecar'
        % (len(ids), total, done_count))
    if not ids:
        return 0

    directory = os.path.join(release, SIDECAR_DIR)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, '%04d.jsonl' % shard)

    written = failed = 0
    started = time.time()
    with open(path, 'a') as handle:
        for index, pdb_id in enumerate(ids, 1):
            try:
                found = C.select(pdb_id)
            except Exception:
                failed += 1
                continue
            for rec in found:
                if rec.get('rejected'):
                    continue
                key = key_of(rec)
                if key not in wanted[pdb_id]:
                    continue
                try:
                    ligand_xyz, pocket_atoms = C.pocket_geometry(
                        rec['pocket'], rec['ligand_mol'])
                except Exception:
                    failed += 1
                    continue
                handle.write(json.dumps({'key': key, 'ligand_xyz': ligand_xyz,
                                         'pocket_atoms': pocket_atoms}) + '\n')
                written += 1
            handle.flush()
            if index % 200 == 0:
                rate = index / max(time.time() - started, 1) * 60
                left = (len(ids) - index) / max(rate, 1e-6)
                log('  %d/%d entries  %.0f/min  %d written  %d failed  ~%.1f h left'
                    % (index, len(ids), rate, written, failed, left))
    log('done: %d complexes written, %d failed' % (written, failed))
    return written


def load_sidecar(release):
    """{key: {ligand_xyz, pocket_atoms}} for dataset.py to merge in."""
    out = {}
    directory = os.path.join(release, SIDECAR_DIR)
    if not os.path.isdir(directory):
        return out
    for name in sorted(os.listdir(directory)):
        if not name.endswith('.jsonl'):
            continue
        with open(os.path.join(directory, name)) as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get('key'):
                    out[entry['key']] = entry
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', default='corpus/v1')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--of', type=int, default=1)
    args = ap.parse_args()
    sys.exit(0 if run(args.release, args.limit, args.shard, args.of) >= 0 else 1)
