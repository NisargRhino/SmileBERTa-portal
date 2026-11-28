"""Train the geometric pocket model and settle whether 3D conditioning works.

`train.py` trains the descriptor model, where the pocket is six composition
fractions projected and added to every atom alike. Across three cycles that arm
gained between -0.013 and +0.003 macro AUC over having no pocket at all, while a
random forest on the same six numbers gained +0.040. The information was in the
features; the conditioning could not deliver it.

This trains the same task with each atom cross-attending over its own nearest
pocket atoms, described by distance and chemistry. The comparison that matters
is not against the random forest but against this model's own no-pocket arm,
which is why `use_pocket` masks the neighbourhood rather than removing the
module: both arms have identical parameter counts, depth and schedule, so a
difference between them is the conditioning and nothing else.

Reported beside the descriptor model, never replacing it. A conditioning
mechanism that fails is a result worth publishing, and deleting the failed arm
once a better one exists would be rewriting history.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from pharmacophore import dataset as D
from pharmacophore import metrics as M
from pharmacophore import model as Model
from pharmacophore.interactions import INTERACTIONS
from pharmacophore.splits import REGIMES, load_split

SEED = 20261126
MIN_SUBSET_ATOMS = 1000


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batches(examples, batch_size, atom_dim, shuffle=False):
    order = list(range(len(examples)))
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [examples[i] for i in order[start:start + batch_size]]
        yield Model.collate_geometric(chunk, atom_dim, D.NEIGHBOUR_DIM, D.N_NEIGHBOURS)


@torch.no_grad()
def predict(net, examples, atom_dim, dev, batch_size=32, shuffle_pocket=False):
    """Per-atom probabilities in input order."""
    net.eval()
    scores, truth = [], []
    pool = [(e.get('neighbour_features'), e.get('neighbour_mask'))
            for e in examples if e.get('neighbour_features') is not None]
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        if shuffle_pocket and pool:
            # Each ligand paired with a stranger's neighbourhood. A model that
            # reads the pocket must collapse here; one that ignores it will not
            # move, which is exactly how the descriptor model was caught.
            swapped = []
            for e in chunk:
                nf, nm = random.choice(pool)
                rows = e['n_heavy']
                swapped.append(dict(e,
                                    neighbour_features=_fit(nf, rows),
                                    neighbour_mask=_fit(nm, rows)))
            chunk = swapped
        atoms, nb, nmask, mask, labels = Model.collate_geometric(
            chunk, atom_dim, D.NEIGHBOUR_DIM, D.N_NEIGHBOURS)
        logits = net(atoms.to(dev), nb.to(dev), nmask.to(dev), mask.to(dev))
        probs = torch.sigmoid(logits).cpu().numpy()
        m = mask.numpy()
        for i in range(len(chunk)):
            scores.append(probs[i][m[i]])
            truth.append(labels.numpy()[i][m[i]])
    if not scores:
        return np.zeros((0, len(INTERACTIONS))), np.zeros((0, len(INTERACTIONS)))
    return np.vstack(truth), np.vstack(scores)


def _fit(array, rows):
    """Tile or trim a donor neighbourhood to the recipient's atom count."""
    if array is None:
        return None
    if array.shape[0] >= rows:
        return array[:rows]
    repeats = int(np.ceil(rows / max(array.shape[0], 1)))
    return np.concatenate([array] * repeats, axis=0)[:rows]


def train_one(train_examples, val_examples, *, use_pocket=True, epochs=30, lr=3e-4,
              batch_size=32, patience=5, dev=None, log=print):
    set_seed()
    dev = dev or Model.device()
    atom_dim = train_examples[0]['atom_features'].shape[1]
    net = Model.GeometricPocketAtomModel(atom_dim, D.NEIGHBOUR_DIM,
                                         use_pocket=use_pocket).to(dev)
    pos_weight = Model.positive_weights(train_examples).to(dev)
    optimiser = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)

    best = {'macro_auc': -1.0, 'epoch': -1, 'state': None}
    history, bad = [], 0
    for epoch in range(1, epochs + 1):
        net.train()
        started, total, steps = time.time(), 0.0, 0
        for atoms, nb, nmask, mask, labels in _batches(train_examples, batch_size,
                                                       atom_dim, shuffle=True):
            atoms, nb, nmask = atoms.to(dev), nb.to(dev), nmask.to(dev)
            mask, labels = mask.to(dev), labels.to(dev)
            loss = Model.masked_bce(net(atoms, nb, nmask, mask), labels, mask, pos_weight)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
            total += float(loss.detach())
            steps += 1
        y_true, y_score = predict(net, val_examples, atom_dim, dev)
        val = M.summarise(y_true, y_score)
        macro = val['macro_auc'] or 0.0
        history.append({'epoch': epoch, 'train_loss': round(total / max(steps, 1), 4),
                        'val_macro_auc': macro,
                        'seconds': round(time.time() - started, 1)})
        log('    epoch %2d  loss %.4f  val macro-AUC %.4f  (%.0fs)'
            % (epoch, total / max(steps, 1), macro, time.time() - started))
        if macro > best['macro_auc'] + 1e-4:
            best = {'macro_auc': macro, 'epoch': epoch,
                    'state': {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log('    early stop: validation macro-AUC flat for %d epochs' % patience)
                break
    if best['state']:
        net.load_state_dict(best['state'])
    return net, {'history': history, 'best_epoch': best['epoch'],
                 'best_val_macro_auc': round(best['macro_auc'], 4),
                 'parameters': net.parameter_count()}


def run(release, regime, epochs=30, log=print):
    split_map = load_split(release, regime)
    train_examples = D.load_examples(release, split_map, 'train')
    val_examples = D.load_examples(release, split_map, 'validation')
    test_examples = D.load_examples(release, split_map, 'test')

    coverage = {k: D.geometry_coverage(v) for k, v in
                (('train', train_examples), ('validation', val_examples),
                 ('test', test_examples))}
    log('[%s] %d train / %d val / %d test complexes'
        % (regime, len(train_examples), len(val_examples), len(test_examples)))
    log('  geometry coverage: train %.1f%%  val %.1f%%  test %.1f%%'
        % tuple(100 * coverage[k]['fraction'] for k in ('train', 'validation', 'test')))

    # Without geometry every atom gets an all-zero neighbourhood, so the pocket
    # arm silently becomes the no-pocket arm and the ablation reads 0.0 for a
    # reason that has nothing to do with the architecture.
    if coverage['train']['fraction'] < 0.5 or coverage['test']['fraction'] < 0.5:
        return {'regime': regime, 'geometry_coverage': coverage,
                'error': 'not enough geometry yet — run pharmacophore.backfill_geometry'}
    if min(len(train_examples), len(val_examples), len(test_examples)) < 5:
        return {'regime': regime, 'error': 'not enough complexes in this regime yet'}

    dev = Model.device()
    atom_dim = train_examples[0]['atom_features'].shape[1]
    out = {'regime': regime, 'device': str(dev), 'geometry_coverage': coverage,
           'counts': {'train': len(train_examples), 'validation': len(val_examples),
                      'test': len(test_examples)}}
    log('  device: %s' % dev)

    log('  arm: with 3D pocket')
    net, hist = train_one(train_examples, val_examples, use_pocket=True,
                          epochs=epochs, dev=dev, log=log)
    y_true, y_score = predict(net, test_examples, atom_dim, dev)
    out['with_pocket'] = {'training': hist, 'test': M.summarise(y_true, y_score)}

    log('  arm: shuffled neighbourhoods (must collapse)')
    y_true_s, y_score_s = predict(net, test_examples, atom_dim, dev, shuffle_pocket=True)
    out['shuffled_pocket'] = {'test': M.summarise(y_true_s, y_score_s)}

    log('  arm: no pocket (ablation)')
    net_np, hist_np = train_one(train_examples, val_examples, use_pocket=False,
                                epochs=epochs, dev=dev, log=log)
    y_true_n, y_score_n = predict(net_np, test_examples, atom_dim, dev)
    out['no_pocket'] = {'training': hist_np, 'test': M.summarise(y_true_n, y_score_n)}

    out['_ablation'] = M.ablation_verdict(out['with_pocket']['test'], out['no_pocket']['test'])
    out['_shuffle_check'] = M.ablation_verdict(out['with_pocket']['test'],
                                               out['shuffled_pocket']['test'])
    log('  ablation : %s' % out['_ablation']['reason'])
    log('  shuffled : %s' % out['_shuffle_check']['reason'])

    try:
        promiscuous = D.promiscuous_ligands(release)
        mask = D.atom_mask(test_examples, promiscuous)
        n_complexes = sum(1 for e in test_examples if e.get('inchikey') in promiscuous)
        if mask.sum() >= MIN_SUBSET_ATOMS:
            subset = {'atoms': int(mask.sum()), 'complexes': n_complexes,
                      'share_of_test_atoms': round(float(mask.mean()), 4),
                      'with_pocket': M.summarise(y_true[mask], y_score[mask]),
                      'no_pocket': M.summarise(y_true_n[mask], y_score_n[mask])}
            subset['_ablation'] = M.ablation_verdict(subset['with_pocket'],
                                                     subset['no_pocket'])
            out['multi_protein_ligands'] = subset
            log('  promiscuous subset (%d atoms over %d complexes): %s'
                % (subset['atoms'], n_complexes, subset['_ablation']['reason']))
    except Exception as exc:
        log('  promiscuous subset failed: %s' % exc)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', default='corpus/v1')
    ap.add_argument('--regimes', nargs='+', default=list(REGIMES))
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    results = {regime: run(args.release, regime, args.epochs) for regime in args.regimes}
    path = args.out or os.path.join(args.release, 'results', 'model_geometric.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote', path)
