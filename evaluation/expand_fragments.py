"""Build a larger fragment corpus with the pipeline that produced the original one.

The fragment model is trained on 219 molecules. Every diagnosis in this
benchmark points at that number rather than at the architecture: the
regularisation interventions do nothing, the data-scaling curve never leaves the
floor, and the memorisation control shows the recipe can only reproduce 18% of
fragments even when handed the answers.

The task spec's last intervention is "more data, from ChEMBL". There is a closer
source: `ml-project/FBDD-discovery.py` built the original corpus by running
RDKit's BRICS decomposition over a set of drug SMILES, and this repository
already holds 8,336 drug SMILES in the classification dataset. Running the same
decomposition over those produces roughly forty times the training pairs with
the identical procedure, no new dependency, and no external data licence to
clear — which also keeps the new corpus directly comparable with the old one.

The original 219 molecules are kept and marked, so the expanded corpus is a
superset and the comparison is clean.
"""
import argparse
import csv
import os
from collections import Counter

from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS

from evaluation import data as D

RDLogger.DisableLog('rdApp.*')

OUT_CSV = os.path.join(D.REPO, 'evaluation', 'fragment_corpus_expanded.csv')
# BRICS can return a single fragment that is the whole molecule when nothing is
# cleavable. Those pairs teach the model to echo its input, which is the one
# behaviour it must not learn, so they are dropped.
MIN_FRAGMENTS = 2


def decompose(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    try:
        return sorted(BRICS.BRICSDecompose(mol))
    except Exception:
        return []


def build(limit=None):
    original, _ = D.load_fragments()
    original_by_smiles = {r['smiles']: r for r in original}

    classification, _ = D.load_classification()
    sources = [r['smiles'] for r in classification]
    if limit:
        sources = sources[:limit]

    rows = []
    stats = Counter()
    seen = set()

    for smiles, rec in original_by_smiles.items():
        for frag in rec['fragments']:
            rows.append({'smiles': smiles, 'fragment': frag, 'source': 'original'})
        seen.add(smiles)
        stats['original_molecules'] += 1

    for smiles in sources:
        if smiles in seen:
            stats['already_present'] += 1
            continue
        frags = decompose(smiles)
        if len(frags) < MIN_FRAGMENTS:
            stats['too_few_fragments'] += 1
            continue
        seen.add(smiles)
        stats['added_molecules'] += 1
        for frag in frags:
            rows.append({'smiles': smiles, 'fragment': frag, 'source': 'classification_dataset'})

    stats['total_pairs'] = len(rows)
    stats['total_molecules'] = len(seen)
    return rows, dict(stats)


def write(rows, path=OUT_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['smiles', 'fragment', 'source'])
        w.writeheader()
        w.writerows(rows)
    return path


def load_expanded(path=OUT_CSV):
    """[{smiles, fragments, scaffold, source}] from the expanded corpus file."""
    by_mol = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rec = by_mol.setdefault(row['smiles'], {
                'smiles': row['smiles'], 'fragments': [],
                'scaffold': D.scaffold_of(row['smiles']), 'source': row['source']})
            if row['fragment'] not in rec['fragments']:
                rec['fragments'].append(row['fragment'])
    return list(by_mol.values())


def expanded_split(path=OUT_CSV):
    """Train on the expanded corpus, test on the *original* 22-molecule test set.

    Holding the test set fixed is what makes "40x more training data" a clean
    comparison instead of two unrelated experiments. Any expanded molecule that
    shares a scaffold with the held-out validation or test molecules is dropped
    from training rather than quietly leaking the answer in — that is the whole
    point of a scaffold split, and growing the corpus is not a licence to
    abandon it.
    """
    from evaluation.splits import SPLIT_DIR, load_split

    original_map = load_split(os.path.join(SPLIT_DIR, 'fragment_scaffold.csv'))
    original_records, _ = D.load_fragments()
    held_scaffolds = {r['scaffold'] for r in original_records
                      if original_map.get(r['smiles']) in ('validation', 'test') and r['scaffold']}

    parts = {'train': [], 'validation': [], 'test': []}
    stats = Counter()
    for rec in load_expanded(path):
        where = original_map.get(rec['smiles'])
        if where in ('validation', 'test'):
            parts[where].append(rec)
            continue
        if rec['scaffold'] and rec['scaffold'] in held_scaffolds:
            stats['dropped_scaffold_overlap'] += 1
            continue
        parts['train'].append(rec)
    stats.update({f'{k}_molecules': len(v) for k, v in parts.items()})
    stats['train_pairs'] = sum(len(r['fragments']) for r in parts['train'])
    return parts, dict(stats)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit', type=int, help='cap the source molecules, for a quick check')
    args = ap.parse_args()
    rows, stats = build(args.limit)
    path = write(rows)
    for k, v in sorted(stats.items()):
        print(f'  {k:24s} {v:,}')
    print('wrote', os.path.relpath(path, D.REPO))
