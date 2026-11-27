"""Three held-out regimes, because one axis is not enough here.

The same ligand appears bound to many homologous proteins and the same protein
appears bound to hundreds of analogues. Hold out ligands only and a model is
tested on new chemistry against proteins it has seen a thousand times. Hold out
proteins only and it is tested with ligand series it memorised. Both are easy
to publish and neither measures what the model will be asked to do.

| Regime          | Held out                         | Answers                                  |
| --------------- | -------------------------------- | ---------------------------------------- |
| `unseen_ligand` | ligand scaffold                  | new chemistry on a known target           |
| `unseen_protein`| protein cluster                  | a target never seen before               |
| `both_unseen`   | both                             | the honest number, and the one to publish |

`both_unseen` will be much the worst. That gap is the finding.

Protein clustering here is by UniProt accession, which is a *weaker* guarantee
than sequence-identity clustering: two accessions can be close homologues
sharing a pocket, and holding out one while training on the other leaks. The
file records that limitation rather than implying otherwise, and
`cluster_by_identity` is where MMseqs2 or CD-HIT slots in when it is available.
"""
import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from pharmacophore.build import iter_records

RDLogger.DisableLog('rdApp.*')

SEED = 20261126
RATIOS = (0.8, 0.1, 0.1)
NAMES = ('train', 'validation', 'test')
REGIMES = ('unseen_ligand', 'unseen_protein', 'both_unseen')


def scaffold_of(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None


def load_complexes(release):
    """Kept records, deduplicated to one per (protein, ligand).

    The same complex is often deposited several times — different crystals of
    the same pair, or the same pair in several entries. Those are one
    observation of a binding mode, and leaving them in would put copies of the
    same thing on both sides of a split.
    """
    seen, out = {}, []
    for rec in iter_records(release):
        if rec.get('rejected'):
            continue
        key = (rec['uniprot'], rec['inchikey'])
        if key in seen:
            # Keep the better-resolved copy.
            if rec['resolution'] < seen[key]['resolution']:
                seen[key].update(rec)
            continue
        rec = dict(rec, scaffold=scaffold_of(rec['smiles']))
        seen[key] = rec
        out.append(rec)
    return out


def cluster_by_identity(records):
    """Protein groups. UniProt accession today; sequence identity when available.

    **This is the weak link in the split.** Accession-level grouping does not
    catch close homologues under different accessions, so `unseen_protein` is
    optimistic by an unknown amount. Clustering sequences at 30% identity with
    MMseqs2 is the fix, and the manifest says which was used so a reader is not
    misled about what was held out.
    """
    groups = defaultdict(list)
    for i, rec in enumerate(records):
        groups[rec['uniprot']].append(i)
    return groups, 'uniprot_accession'


def _assign_groups(groups, n_items, ratios, seed):
    """Whole groups to splits, largest first, filling whichever is furthest behind."""
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))
    quota = [r * n_items for r in ratios]
    sizes = [0, 0, 0]
    assigned = {}
    rng = random.Random(seed)
    for key, members in order:
        deficits = [quota[s] - sizes[s] for s in range(3)]
        best = max(range(3), key=lambda s: (deficits[s], rng.random()))
        assigned[key] = best
        sizes[best] += len(members)
    return assigned


def build_splits(release, ratios=RATIOS, seed=SEED):
    records = load_complexes(release)
    if not records:
        raise SystemExit('no kept records yet — let the build run first')

    protein_groups, protein_method = cluster_by_identity(records)
    scaffold_groups = defaultdict(list)
    for i, rec in enumerate(records):
        key = rec['scaffold'] if rec['scaffold'] else f'__acyclic__{i}'
        scaffold_groups[key].append(i)

    protein_split = _assign_groups(protein_groups, len(records), ratios, seed)
    ligand_split = _assign_groups(scaffold_groups, len(records), ratios, seed + 1)

    protein_of = {i: protein_split[rec['uniprot']] for i, rec in enumerate(records)}
    ligand_of = {}
    for key, members in scaffold_groups.items():
        for i in members:
            ligand_of[i] = ligand_split[key]

    assignments = {}
    for regime in REGIMES:
        out = {}
        for i in range(len(records)):
            if regime == 'unseen_ligand':
                out[i] = ligand_of[i]
            elif regime == 'unseen_protein':
                out[i] = protein_of[i]
            else:
                # Both held out: test only where ligand *and* protein are both
                # held out, train only where neither is. Anything mixed is
                # discarded rather than quietly used, because a pair that is new
                # on one axis only is not what this regime claims to measure.
                if ligand_of[i] == 2 and protein_of[i] == 2:
                    out[i] = 2
                elif ligand_of[i] == 1 and protein_of[i] == 1:
                    out[i] = 1
                elif ligand_of[i] == 0 and protein_of[i] == 0:
                    out[i] = 0
                else:
                    out[i] = None
        assignments[regime] = out

    os.makedirs(os.path.join(release, 'splits'), exist_ok=True)
    report = {'seed': seed, 'ratios': list(ratios), 'complexes': len(records),
              'protein_grouping': protein_method,
              'protein_groups': len(protein_groups),
              'scaffold_groups': len(scaffold_groups), 'regimes': {}}

    for regime, mapping in assignments.items():
        path = os.path.join(release, 'splits', f'{regime}.csv')
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['pdb_id', 'comp_id', 'uniprot', 'inchikey', 'smiles',
                             'scaffold', 'split'])
            counts = Counter()
            for i, rec in enumerate(records):
                where = mapping[i]
                if where is None:
                    counts['discarded'] += 1
                    continue
                counts[NAMES[where]] += 1
                writer.writerow([rec['pdb_id'], rec['comp_id'], rec['uniprot'],
                                 rec['inchikey'], rec['smiles'], rec['scaffold'] or '',
                                 NAMES[where]])
        with open(path, 'rb') as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:16]
        leaks = _leak_check(records, mapping, regime)
        report['regimes'][regime] = {'counts': dict(counts), 'sha256_16': digest,
                                     'leaks': leaks, 'path': os.path.relpath(path)}
        assert not any(leaks.values()), f'{regime} leaks: {leaks}'

    with open(os.path.join(release, 'splits', 'README.json'), 'w') as f:
        json.dump(report, f, indent=2)
    return report


def _leak_check(records, mapping, regime):
    """What must not appear on both sides, per regime."""
    train_scaffolds, test_scaffolds = set(), set()
    train_proteins, test_proteins = set(), set()
    for i, rec in enumerate(records):
        where = mapping[i]
        if where == 0:
            train_scaffolds.add(rec['scaffold'])
            train_proteins.add(rec['uniprot'])
        elif where == 2:
            test_scaffolds.add(rec['scaffold'])
            test_proteins.add(rec['uniprot'])
    shared_scaffolds = len((train_scaffolds & test_scaffolds) - {None, ''})
    shared_proteins = len(train_proteins & test_proteins)
    return {
        'shared_scaffolds': shared_scaffolds if regime != 'unseen_protein' else 0,
        'shared_proteins': shared_proteins if regime != 'unseen_ligand' else 0,
    }


def load_split(release, regime):
    path = os.path.join(release, 'splits', f'{regime}.csv')
    with open(path) as f:
        return {(r['pdb_id'], r['comp_id']): r['split'] for r in csv.DictReader(f)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', default='corpus/v1')
    args = ap.parse_args()
    print(json.dumps(build_splits(args.release), indent=2))
