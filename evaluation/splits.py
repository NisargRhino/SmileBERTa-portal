"""Train / validation / test splits, built once and committed to disk.

Two splits are built for each dataset, and both are reported in the benchmark:

* **scaffold** — no Bemis–Murcko scaffold appears on both sides. This is the
  honest one.
* **random** — the conventional shuffle, kept only so the gap between the two
  can be measured. That gap is the clearest available estimate of how much the
  model was memorising scaffolds rather than learning chemistry.

Why scaffold grouping matters here: 8,336 compounds share just 2,530 scaffolds,
so a random split scatters close analogues of the same scaffold across train and
test and the model can score well by recognising a scaffold it has already seen.

The assignment is written to CSV and loaded from there everywhere. Regenerating
a split on the fly with a different seed would make every number in this repo
incomparable with every other, which is the failure this file exists to prevent.
"""
import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

from evaluation import data as D

SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splits')
RATIOS = (0.8, 0.1, 0.1)   # train / validation / test
SEED = 20261126


def group_by_scaffold(records):
    """scaffold -> [record index]; every acyclic molecule is its own group.

    MurckoScaffoldSmiles returns `''` for a molecule with no ring system. There
    are 543 of those here, and treating them as one group would force 6.5% of the
    dataset into a single split. They share no scaffold with anything (they have
    none), so each is its own group.
    """
    groups = defaultdict(list)
    for i, rec in enumerate(records):
        scaffold = rec['scaffold']
        key = scaffold if scaffold else f'__acyclic__{i}'
        groups[key].append(i)
    return groups


def _label_vector(rec):
    return rec.get('labels') or []


def scaffold_split(records, ratios=RATIOS, stratify=True):
    """Assign whole scaffold groups to train/val/test, largest groups first.

    Largest-first keeps the split sizes close to target: a late 200-molecule
    group would otherwise overshoot whichever split it landed in.

    When `stratify`, the destination for each group is the split that is
    currently furthest below its quota for the labels that group carries, rather
    than simply the emptiest split. Perfect stratification is impossible once
    whole scaffold groups must move together — urological has 48 compounds in
    total — so this gets close and the real per-class counts are recorded.
    """
    groups = group_by_scaffold(records)
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    n = len(records)
    quota = [r * n for r in ratios]
    assigned = [None] * n
    sizes = [0, 0, 0]

    label_totals = Counter(l for rec in records for l in _label_vector(rec))
    label_quota = [{lab: cnt * r for lab, cnt in label_totals.items()} for r in ratios]
    label_have = [Counter() for _ in ratios]

    for _scaffold, idxs in order:
        if stratify and label_totals:
            group_labels = Counter(l for i in idxs for l in _label_vector(records[i]))

            def deficit(s):
                # How badly does split s still need the labels in this group?
                # Normalised by the label's own quota so a rare class (48
                # compounds) can outweigh a common one (2,923).
                score = 0.0
                for lab, cnt in group_labels.items():
                    q = label_quota[s].get(lab, 0.0) or 1.0
                    score += cnt * (label_quota[s].get(lab, 0.0) - label_have[s][lab]) / q
                # Size deficit breaks ties and keeps the overall split sizes right.
                return score + 0.25 * (quota[s] - sizes[s]) / max(quota[s], 1)

            target = max(range(len(ratios)), key=deficit)
        else:
            target = max(range(len(ratios)), key=lambda s: quota[s] - sizes[s])
        for i in idxs:
            assigned[i] = target
            for lab in _label_vector(records[i]):
                label_have[target][lab] += 1
        sizes[target] += len(idxs)
    return assigned


def random_split(records, ratios=RATIOS, seed=SEED):
    """The conventional shuffle. Reported only as the contrast to the scaffold split."""
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n = len(records)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    assigned = [None] * n
    for rank, i in enumerate(idx):
        assigned[i] = 0 if rank < n_train else (1 if rank < n_train + n_val else 2)
    return assigned


NAMES = ('train', 'validation', 'test')


def verify_scaffold_split(records, assigned):
    """No scaffold may appear in two splits. Returns the list of violations."""
    where = defaultdict(set)
    for rec, split in zip(records, assigned):
        if rec['scaffold']:        # acyclic molecules have no scaffold to share
            where[rec['scaffold']].add(split)
    return [s for s, splits in where.items() if len(splits) > 1]


def split_report(records, assigned, kind):
    per_split_labels = [Counter() for _ in NAMES]
    sizes = Counter()
    scaffolds = [set() for _ in NAMES]
    for rec, s in zip(records, assigned):
        sizes[s] += 1
        for lab in _label_vector(rec):
            per_split_labels[s][lab] += 1
        scaffolds[s].add(rec['scaffold'])
    report = {
        'kind': kind,
        'total': len(records),
        'sizes': {NAMES[s]: sizes[s] for s in range(3)},
        'fractions': {NAMES[s]: round(sizes[s] / max(len(records), 1), 4) for s in range(3)},
        'distinct_scaffolds': {NAMES[s]: len(scaffolds[s]) for s in range(3)},
        'shared_scaffolds_train_test': len(
            (scaffolds[0] & scaffolds[2]) - {''}),
    }
    if any(per_split_labels):
        report['per_label'] = {
            lab: {NAMES[s]: per_split_labels[s][lab] for s in range(3)}
            for lab in D.CATEGORIES
            if any(per_split_labels[s][lab] for s in range(3))
        }
    return report


def write_split(path, records, assigned, extra_columns=()):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['smiles', 'split', 'scaffold', *extra_columns])
        for rec, s in zip(records, assigned):
            w.writerow([rec['smiles'], NAMES[s], rec['scaffold'],
                        *[rec[c] if not isinstance(rec.get(c), list) else ' '.join(rec[c])
                          for c in extra_columns]])
    return file_digest(path)


def file_digest(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def load_split(path):
    """{smiles: 'train'|'validation'|'test'} from a committed split file."""
    with open(path) as f:
        return {r['smiles']: r['split'] for r in csv.DictReader(f)}


def build_all():
    out = {'seed': SEED, 'ratios': list(RATIOS), 'files': {}, 'reports': {}}

    cls, cls_dropped = D.load_classification()
    frg, frg_dropped = D.load_fragments()
    out['dropped'] = {'classification': cls_dropped, 'fragments': frg_dropped}

    for name, records, extra in (
            ('classification', cls, ('labels',)),
            ('fragment', frg, ('fragments',))):
        for kind, assigned in (('scaffold', scaffold_split(records, stratify=(name == 'classification'))),
                               ('random', random_split(records))):
            path = os.path.join(SPLIT_DIR, f'{name}_{kind}.csv')
            digest = write_split(path, records, assigned, extra)
            report = split_report(records, assigned, kind)
            if kind == 'scaffold':
                violations = verify_scaffold_split(records, assigned)
                report['scaffold_violations'] = len(violations)
                assert not violations, f'scaffold leaked across splits: {violations[:3]}'
            out['files'][f'{name}_{kind}'] = {'path': os.path.relpath(path, D.REPO),
                                              'sha256_16': digest}
            out['reports'][f'{name}_{kind}'] = report

    meta = os.path.join(SPLIT_DIR, 'README.json')
    with open(meta, 'w') as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    print(json.dumps(build_all(), indent=2))
