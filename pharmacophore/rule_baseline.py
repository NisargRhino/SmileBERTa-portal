"""How much of the geometric model's score is just the label definition?

The labels in this corpus are geometric. A salt bridge is "a charged ligand
atom within 5.0 A of an oppositely charged protein atom"; a hydrophobic contact
is two apolar atoms within about 4.5 A. The geometric model is given, per atom,
the distances to its nearest pocket atoms together with their element and
residue class — which is most of the information those definitions consume.

So a high score from that model is not by itself evidence of chemistry learned.
It could be the model recovering the rule that generated the labels. This
module writes those rules out by hand, with no fitting of any kind, and scores
them the same way. The gap between the rules and the trained model is the part
that is actually learned; if there is no gap, the forward task is close to
trivial once a pose is known and the paper has to say so.

This control exists because the alternative is publishing 0.95 and letting a
reviewer notice.
"""
import argparse
import json
import os

import numpy as np

from pharmacophore import dataset as D
from pharmacophore import metrics as M
from pharmacophore.interactions import INTERACTIONS
from pharmacophore.splits import REGIMES, load_split

# Offsets into a neighbour feature row, mirroring dataset.neighbour_features.
_RBF = len(D.RBF_CENTRES)
_EL = len(D.NEIGHBOUR_ELEMENTS)
IDX_ELEMENT = _RBF
IDX_OTHER_ELEMENT = _RBF + _EL
IDX_HYDROPHOBIC = _RBF + _EL + 1
IDX_AROMATIC = IDX_HYDROPHOBIC + 1
IDX_CATIONIC = IDX_HYDROPHOBIC + 2
IDX_ANIONIC = IDX_HYDROPHOBIC + 3
IDX_POLAR = IDX_HYDROPHOBIC + 4


def _distances(neighbours):
    """Recover an approximate distance per neighbour from its RBF expansion.

    The expansion is a set of Gaussians at known centres, so the centre with
    the largest response is the nearest grid point to the true distance. Exact
    inversion is unnecessary — the rules below only need "about how far".
    """
    rbf = neighbours[..., :_RBF]
    return D.RBF_CENTRES[np.argmax(rbf, axis=-1)]


def rule_scores(example):
    """Hand-written per-atom scores, no parameters fitted anywhere."""
    neighbours = example.get('neighbour_features')
    mask = example.get('neighbour_mask')
    n = example['n_heavy']
    scores = np.zeros((n, len(INTERACTIONS)), dtype=np.float32)
    if neighbours is None:
        return scores

    distance = _distances(neighbours)
    valid = mask.astype(bool)
    big = 99.0
    distance = np.where(valid, distance, big)

    atom = example['atom_features']
    # Formal charge sign, as encoded by dataset.atom_features. Looked up by
    # name rather than a magic index so a feature re-ordering cannot silently
    # make this read the wrong column.
    charge = np.zeros(n, dtype=np.float32)
    if hasattr(D, 'CHARGE_INDEX'):
        charge = atom[:, D.CHARGE_INDEX]

    def nearest_where(flag_index, cutoff):
        """Closeness of the nearest neighbour carrying a flag, 0 if none."""
        has = neighbours[..., flag_index] > 0.5
        d = np.where(has & valid, distance, big)
        nearest = d.min(axis=1)
        return np.clip((cutoff - nearest) / cutoff, 0, 1)

    for j, name in enumerate(INTERACTIONS):
        if name == 'Hydrophobic':
            scores[:, j] = nearest_where(IDX_HYDROPHOBIC, 4.5)
        elif name == 'PiStacking':
            scores[:, j] = nearest_where(IDX_AROMATIC, 5.5)
        elif name == 'PiCation':
            scores[:, j] = nearest_where(IDX_AROMATIC, 5.5)
        elif name == 'Cationic':
            scores[:, j] = nearest_where(IDX_ANIONIC, 5.0)
        elif name == 'Anionic':
            scores[:, j] = nearest_where(IDX_CATIONIC, 5.0)
        elif name in ('HBDonor', 'HBAcceptor'):
            # Any nitrogen or oxygen close enough to hydrogen bond.
            polar = (neighbours[..., IDX_ELEMENT + 1] > 0.5) | \
                    (neighbours[..., IDX_ELEMENT + 2] > 0.5)
            d = np.where(polar & valid, distance, big)
            scores[:, j] = np.clip((3.5 - d.min(axis=1)) / 3.5, 0, 1)
        else:
            scores[:, j] = nearest_where(IDX_POLAR, 5.0) * 0.1
    return scores


def run(release, regime, log=print):
    split_map = load_split(release, regime)
    test = D.load_examples(release, split_map, 'test')
    usable = [e for e in test if e.get('neighbour_features') is not None]
    log('[%s] %d test complexes, %d with geometry' % (regime, len(test), len(usable)))
    if len(usable) < 10:
        return {'regime': regime, 'error': 'not enough geometry in this regime'}

    truth = np.vstack([e['labels'][:e['n_heavy']] for e in usable])
    scores = np.vstack([rule_scores(e) for e in usable])
    summary = M.summarise(truth, scores)
    log('  hand-written rules   macro AUC %s over %d atoms'
        % (summary['macro_auc'], summary['atoms']))
    return {'regime': regime, 'test': summary,
            'complexes': len(usable),
            'note': 'no parameters fitted; these are the label definitions '
                    'applied to the model\'s own input features'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', default='corpus/v1')
    ap.add_argument('--regimes', nargs='+', default=list(REGIMES))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    results = {r: run(args.release, r) for r in args.regimes}
    path = args.out or os.path.join(args.release, 'results', 'rule_baseline.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote', path)
