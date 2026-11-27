"""Baselines, and the ablation that decides whether the project works.

`BENCHMARK.md` found a random forest on fingerprints beating a 44 M-parameter
transformer by 0.33 macro-F1 on an honest split. That is why this file exists
and why it runs before any network is trained.

Four arms:

* **frequency prior** — each atom gets its element-and-hybridisation base rate.
  A surprising amount of pharmacophore signal is "carboxylate oxygens make salt
  bridges", and a model has to beat that to have learned anything.
* **random forest, ligand only** — atom descriptors, no protein at all.
* **random forest, ligand + pocket** — the same, with pocket descriptors.
* **the difference between those two** is the ablation. Most ligands appear in
  exactly one complex, so a model can reproduce the labels by memorising
  molecules and never consulting the protein. If the pocket adds nothing here,
  the pocket-conditioned half of the project does not work, and that has to be
  known before a GPU is rented rather than after.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

from pharmacophore import dataset as D
from pharmacophore import metrics as M
from pharmacophore.interactions import INTERACTIONS
from pharmacophore.splits import REGIMES, load_split

SEED = 20261126

# See pharmacophore.train for why this floor exists: the promiscuous subset is
# 12 complexes in unseen_ligand and over a thousand in unseen_protein.
MIN_SUBSET_ATOMS = 1000


def frequency_prior(train_examples, test_examples):
    """Per-atom-type base rates learned from training, applied to test.

    The atom type is element plus hybridisation plus aromaticity — enough to
    express "aromatic carbons stack" and "charged oxygens form salt bridges"
    without any model at all.
    """
    counts = defaultdict(lambda: np.zeros(len(INTERACTIONS)))
    totals = defaultdict(int)
    for example in train_examples:
        for i in range(example['n_heavy']):
            key = tuple(example['atom_features'][i][:len(D.ELEMENTS) + 1
                                                    + len(D.HYBRIDISATIONS) + 1].tolist())
            counts[key] += example['labels'][i]
            totals[key] += 1
    global_rate = np.zeros(len(INTERACTIONS))
    global_total = 0
    for key, total in totals.items():
        global_rate += counts[key]
        global_total += total
    global_rate = global_rate / max(global_total, 1)

    scores = []
    for example in test_examples:
        for i in range(example['n_heavy']):
            key = tuple(example['atom_features'][i][:len(D.ELEMENTS) + 1
                                                    + len(D.HYBRIDISATIONS) + 1].tolist())
            if totals.get(key):
                scores.append(counts[key] / totals[key])
            else:
                scores.append(global_rate)
    return np.array(scores) if scores else np.zeros((0, len(INTERACTIONS)))


def random_forest(train_examples, test_examples, with_pocket=True, n_estimators=200):
    """One-vs-rest random forest over per-atom descriptors."""
    from sklearn.ensemble import RandomForestClassifier

    x_train, y_train, _ = D.flatten(train_examples, with_pocket=with_pocket)
    x_test, _y_test, _ = D.flatten(test_examples, with_pocket=with_pocket)
    scores = np.zeros((x_test.shape[0], len(INTERACTIONS)))
    for i in range(len(INTERACTIONS)):
        column = y_train[:, i]
        if column.sum() == 0:
            continue          # never seen in training: leave at zero, honestly
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                       random_state=SEED, min_samples_leaf=2,
                                       class_weight='balanced_subsample')
        model.fit(x_train, column)
        if len(model.classes_) > 1:
            scores[:, i] = model.predict_proba(x_test)[:, 1]
    return scores


def run(release, regime='both_unseen', log=print):
    split_map = load_split(release, regime)
    train = D.load_examples(release, split_map, 'train')
    test = D.load_examples(release, split_map, 'test')
    log(f'[{regime}] {len(train)} train complexes, {len(test)} test complexes')
    if not train or not test:
        return {'regime': regime, 'error': 'not enough data in this regime yet'}

    _x, y_test, _g = D.flatten(test)
    results = {}

    results['frequency_prior'] = M.summarise(y_test, frequency_prior(train, test))
    log('  frequency prior      macro AUC %s' % results['frequency_prior']['macro_auc'])

    scores_without = random_forest(train, test, with_pocket=False)
    results['random_forest_ligand_only'] = M.summarise(y_test, scores_without)
    log('  RF ligand only       macro AUC %s' % results['random_forest_ligand_only']['macro_auc'])

    scores_with = random_forest(train, test, with_pocket=True)
    results['random_forest_with_pocket'] = M.summarise(y_test, scores_with)
    log('  RF ligand + pocket   macro AUC %s' % results['random_forest_with_pocket']['macro_auc'])

    results['_ablation'] = M.ablation_verdict(results['random_forest_with_pocket'],
                                              results['random_forest_ligand_only'])

    # Same ablation on the ligands that bind more than one protein — the only
    # complexes where the pocket is in contention. Reported beside the
    # corpus-wide number, which is left untouched.
    try:
        promiscuous = D.promiscuous_ligands(release)
        mask = D.atom_mask(test, promiscuous)
        n_complexes = sum(1 for e in test if e.get('inchikey') in promiscuous)
        if mask.sum() >= MIN_SUBSET_ATOMS:
            with_pocket = M.summarise(y_test[mask], scores_with[mask])
            without = M.summarise(y_test[mask], scores_without[mask])
            results['multi_protein_ligands'] = {
                'atoms': int(mask.sum()),
                'complexes': n_complexes,
                'share_of_test_atoms': round(float(mask.mean()), 4),
                'random_forest_with_pocket': with_pocket,
                'random_forest_ligand_only': without,
                '_ablation': M.ablation_verdict(with_pocket, without),
            }
            log('  promiscuous subset   %d atoms over %d complexes, %s'
                % (int(mask.sum()), n_complexes,
                   results['multi_protein_ligands']['_ablation']['reason']))
        else:
            log('  promiscuous subset   too small to report (%d atoms over %d '
                'complexes)' % (int(mask.sum()), n_complexes))
    except Exception as exc:
        log('  promiscuous subset failed: %s' % exc)

    log('  ablation: %s (%s)' % (results['_ablation']['verdict'], results['_ablation']['reason']))
    results['_split'] = {'regime': regime, 'train_complexes': len(train),
                         'test_complexes': len(test),
                         'train_labels': D.label_statistics(train),
                         'test_labels': D.label_statistics(test)}
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', default='corpus/v1')
    ap.add_argument('--regimes', nargs='+', default=list(REGIMES))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    everything = {regime: run(args.release, regime) for regime in args.regimes}
    path = args.out or os.path.join(args.release, 'results', 'baselines.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(everything, f, indent=2)
    print('wrote', path)
