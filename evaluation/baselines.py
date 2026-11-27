"""Baselines — the bar a 44 M-parameter transformer has to clear.

Run these before reading any transformer number. If a random forest on 2048-bit
fingerprints matches the transformer on the scaffold split, that is a finding
worth publishing, not an embarrassment: it would say the transformer needs more
data rather than more tuning, which changes what to do next.

Classification: random forest on ECFP4, logistic regression on ECFP4, and a
majority-class baseline.
Fragments: nearest-neighbour retrieval by Tanimoto, and the most common fragment
in the training set.
"""
import argparse
import json
import os

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

from evaluation import data as D
from evaluation import metrics as M
from evaluation.splits import SPLIT_DIR, load_split

RDLogger.DisableLog('rdApp.*')
SEED = 20261126


def ecfp4(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurise(records):
    return np.array([ecfp4(r['smiles']) for r in records], dtype=np.uint8)


def partition(records, split_map):
    out = {'train': [], 'validation': [], 'test': []}
    for rec in records:
        where = split_map.get(rec['smiles'])
        if where:
            out[where].append(rec)
    return out


def _label_matrix(records):
    return np.array([[1 if c in r['labels'] else 0 for c in D.CATEGORIES]
                     for r in records], dtype=np.uint8)


def _rank_from_scores(scores):
    """Categories best-first, for the shared top-k definition in metrics.py."""
    return [[D.CATEGORIES[i] for i in np.argsort(-row)] for row in scores]


def _predict_sets(scores, threshold=0.5):
    """Multi-label decision: everything over the threshold, else the argmax.

    Falling back to the argmax matters — with 12 classes and a long tail, a
    thresholded one-vs-rest model abstains completely on a lot of compounds, and
    an abstention would be scored as a miss on every class rather than as the
    single wrong guess it really is.
    """
    out = []
    for row in scores:
        chosen = {D.CATEGORIES[i] for i, v in enumerate(row) if v >= threshold}
        if not chosen:
            chosen = {D.CATEGORIES[int(np.argmax(row))]}
        out.append(chosen)
    return out


def classification_baselines(split_kind='scaffold'):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    records, _ = D.load_classification()
    split_map = load_split(os.path.join(SPLIT_DIR, f'classification_{split_kind}.csv'))
    part = partition(records, split_map)
    X = {k: featurise(v) for k, v in part.items()}
    Y = {k: _label_matrix(v) for k, v in part.items()}
    y_true = [set(r['labels']) for r in part['test']]
    results = {}

    # ── Majority class ──────────────────────────────────────────────────────
    # Per the spec, reported per class. It predicts the single commonest label
    # for everything, which makes its per-class recall 0 everywhere else — that
    # zero column is the point of including it.
    train_counts = Y['train'].sum(axis=0)
    majority = D.CATEGORIES[int(np.argmax(train_counts))]
    maj_scores = np.tile(train_counts / max(train_counts.sum(), 1), (len(part['test']), 1))
    results['majority_class'] = M.classification_metrics(
        y_true, [{majority}] * len(part['test']), _rank_from_scores(maj_scores))
    results['majority_class']['note'] = f'always predicts {majority}'

    # ── Random forest, one-vs-rest over the 12 categories ───────────────────
    rf_scores = np.zeros((len(part['test']), len(D.CATEGORIES)))
    for i, _cat in enumerate(D.CATEGORIES):
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED,
                                     class_weight='balanced_subsample', min_samples_leaf=1)
        clf.fit(X['train'], Y['train'][:, i])
        rf_scores[:, i] = clf.predict_proba(X['test'])[:, 1] if len(clf.classes_) > 1 else 0.0
    results['random_forest_ecfp4'] = M.classification_metrics(
        y_true, _predict_sets(rf_scores), _rank_from_scores(rf_scores))

    # ── Logistic regression, one-vs-rest ────────────────────────────────────
    lr_scores = np.zeros((len(part['test']), len(D.CATEGORIES)))
    for i, _cat in enumerate(D.CATEGORIES):
        if Y['train'][:, i].sum() == 0:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0,
                                 random_state=SEED, n_jobs=-1)
        clf.fit(X['train'], Y['train'][:, i])
        lr_scores[:, i] = clf.predict_proba(X['test'])[:, 1] if len(clf.classes_) > 1 else 0.0
    results['logistic_regression_ecfp4'] = M.classification_metrics(
        y_true, _predict_sets(lr_scores), _rank_from_scores(lr_scores))

    results['_split'] = {'kind': split_kind,
                         'sizes': {k: len(v) for k, v in part.items()}}
    return results


def fragment_baselines(split_kind='scaffold'):
    records, _ = D.load_fragments()
    split_map = load_split(os.path.join(SPLIT_DIR, f'fragment_{split_kind}.csv'))
    part = partition(records, split_map)
    train, test = part['train'], part['test']
    results = {}

    # ── Nearest-neighbour retrieval ─────────────────────────────────────────
    # Find the most similar training molecule by Tanimoto and return one of its
    # fragments. This is the baseline that matters: if a 44 M transformer cannot
    # beat "look up the nearest molecule you have already seen", it has not
    # learned anything a lookup table does not already contain.
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(r['smiles']), 2, nBits=2048)
                 for r in train]
    nn_records = []
    for rec in test:
        mol = Chem.MolFromSmiles(rec['smiles'])
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        neighbour = train[int(np.argmax(sims))]
        # The neighbour's largest fragment: with no further information it is the
        # most informative single answer, and it matches what the tool returns.
        pick = max(neighbour['fragments'],
                   key=lambda f: (M.strip_dummies(f).GetNumAtoms() if M.strip_dummies(f) else 0))
        nn_records.append({'smiles': rec['smiles'], 'references': rec['fragments'],
                           'raw': M.normalise_fragment(pick),
                           'prediction': M.normalise_fragment(pick),
                           'correction': 'nearest-neighbour'})
    results['nearest_neighbour_tanimoto'] = M.fragment_metrics(nn_records)

    # ── Most common training fragment ───────────────────────────────────────
    counts = {}
    for rec in train:
        for f in rec['fragments']:
            norm = M.normalise_fragment(f)
            if norm:
                counts[norm] = counts.get(norm, 0) + 1
    commonest = max(counts, key=counts.get) if counts else ''
    mc_records = [{'smiles': rec['smiles'], 'references': rec['fragments'],
                   'raw': commonest, 'prediction': commonest, 'correction': 'constant'}
                  for rec in test]
    results['most_common_fragment'] = M.fragment_metrics(mc_records)
    results['most_common_fragment']['note'] = f'always predicts {commonest!r}'

    results['_split'] = {'kind': split_kind, 'sizes': {k: len(v) for k, v in part.items()}}
    return results


def run(split_kinds=('scaffold', 'random')):
    out = {}
    for kind in split_kinds:
        out[kind] = {'classification': classification_baselines(kind),
                     'fragment': fragment_baselines(kind)}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(SPLIT_DIR), 'results', 'baselines.json'))
    args = ap.parse_args()
    results = run()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    for kind, block in results.items():
        print(f'\n=== {kind} split ===')
        for name, res in block['classification'].items():
            if name.startswith('_'):
                continue
            print('  %-28s macro-F1 %.3f  micro-F1 %.3f  top-3 %.3f  subset-acc %.3f'
                  % (name, res['macro_f1'], res['micro_f1'],
                     res.get('top_3_accuracy', 0), res['subset_accuracy']))
        for name, res in block['fragment'].items():
            if name.startswith('_'):
                continue
            print('  %-28s exact %.3f  tanimoto %.3f  substructure %.3f'
                  % (name, res['exact_match_rate'], res['mean_tanimoto_to_reference'],
                     res['substructure_rate']))
    print(f'\nwrote {args.out}')
