"""Train and evaluate the pocket-conditioned model, with the ablations as a gate.

Model selection is on **validation** macro-AUC only. The test set of each regime
is touched once, at the end, which is the rule `MODEL_HONEST_EVAL_TASK.md` set
and the reason its numbers were worth publishing.

Every run reports the no-pocket and shuffled-pocket arms alongside the model.
They are not optional extras: most ligands appear in exactly one complex, so a
model can reproduce the labels from the molecule alone and look excellent while
knowing nothing about proteins. The ablations are how that is caught, and they
run on the same split with the same seed so the comparison means something.
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


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batches(examples, batch_size, atom_dim, pocket_dim, shuffle=False):
    order = list(range(len(examples)))
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [examples[i] for i in order[start:start + batch_size]]
        yield Model.collate(chunk, atom_dim, pocket_dim)


@torch.no_grad()
def predict(net, examples, atom_dim, pocket_dim, dev, batch_size=64, shuffle_pocket=False):
    """Per-atom probabilities for every example, in input order."""
    net.eval()
    scores, truth = [], []
    pool = [e['pocket_features'] for e in examples]
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        if shuffle_pocket:
            # Each ligand paired with someone else's pocket. Performance should
            # collapse; if it does not, the pocket is not being used or
            # something is leaking through the ligand.
            chunk = [dict(e, pocket_features=random.choice(pool)) for e in chunk]
        atoms, pocket, mask, labels = Model.collate(chunk, atom_dim, pocket_dim)
        logits = net(atoms.to(dev), pocket.to(dev), mask.to(dev))
        probs = torch.sigmoid(logits).cpu().numpy()
        m = mask.numpy()
        for i in range(len(chunk)):
            scores.append(probs[i][m[i]])
            truth.append(labels.numpy()[i][m[i]])
    if not scores:
        return np.zeros((0, len(INTERACTIONS))), np.zeros((0, len(INTERACTIONS)))
    return np.vstack(truth), np.vstack(scores)


def train_one(train_examples, val_examples, *, use_pocket=True, epochs=30, lr=3e-4,
              batch_size=32, patience=5, dev=None, log=print):
    set_seed()
    dev = dev or Model.device()
    atom_dim = train_examples[0]['atom_features'].shape[1]
    pocket_dim = train_examples[0]['pocket_features'].shape[0]
    net = Model.PocketConditionedAtomModel(atom_dim, pocket_dim,
                                           use_pocket=use_pocket).to(dev)
    pos_weight = Model.positive_weights(train_examples).to(dev)
    optimiser = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)

    best = {'macro_auc': -1.0, 'epoch': -1, 'state': None}
    history, bad = [], 0
    for epoch in range(1, epochs + 1):
        net.train()
        started, total, steps = time.time(), 0.0, 0
        for atoms, pocket, mask, labels in batches(train_examples, batch_size,
                                                   atom_dim, pocket_dim, shuffle=True):
            atoms, pocket, mask, labels = (atoms.to(dev), pocket.to(dev),
                                           mask.to(dev), labels.to(dev))
            loss = Model.masked_bce(net(atoms, pocket, mask), labels, mask, pos_weight)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
            total += float(loss.detach())
            steps += 1
        y_true, y_score = predict(net, val_examples, atom_dim, pocket_dim, dev)
        val = M.summarise(y_true, y_score)
        macro_auc = val['macro_auc'] or 0.0
        history.append({'epoch': epoch, 'train_loss': round(total / max(steps, 1), 4),
                        'val_macro_auc': macro_auc,
                        'seconds': round(time.time() - started, 1)})
        log('    epoch %2d  loss %.4f  val macro-AUC %.4f  (%.0fs)'
            % (epoch, total / max(steps, 1), macro_auc, time.time() - started))
        if macro_auc > best['macro_auc'] + 1e-4:
            best = {'macro_auc': macro_auc, 'epoch': epoch,
                    'state': {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log('    early stop: validation macro-AUC has not improved for %d epochs' % patience)
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
    log('[%s] %d train / %d val / %d test complexes'
        % (regime, len(train_examples), len(val_examples), len(test_examples)))
    if min(len(train_examples), len(val_examples), len(test_examples)) < 5:
        return {'regime': regime,
                'error': 'not enough complexes in this regime yet — let the corpus grow'}

    dev = Model.device()
    log('  device: %s' % dev)
    atom_dim = train_examples[0]['atom_features'].shape[1]
    pocket_dim = train_examples[0]['pocket_features'].shape[0]
    out = {'regime': regime, 'device': str(dev),
           'counts': {'train': len(train_examples), 'validation': len(val_examples),
                      'test': len(test_examples)},
           'label_statistics': D.label_statistics(train_examples)}

    log('  arm: with pocket')
    net, hist = train_one(train_examples, val_examples, use_pocket=True,
                          epochs=epochs, dev=dev, log=log)
    y_true, y_score = predict(net, test_examples, atom_dim, pocket_dim, dev)
    out['with_pocket'] = {'training': hist, 'test': M.summarise(y_true, y_score)}

    log('  arm: shuffled pockets (must collapse)')
    y_true_s, y_score_s = predict(net, test_examples, atom_dim, pocket_dim, dev,
                                  shuffle_pocket=True)
    out['shuffled_pocket'] = {'test': M.summarise(y_true_s, y_score_s)}

    log('  arm: no pocket (ablation)')
    net_np, hist_np = train_one(train_examples, val_examples, use_pocket=False,
                                epochs=epochs, dev=dev, log=log)
    y_true_n, y_score_n = predict(net_np, test_examples, atom_dim, pocket_dim, dev)
    out['no_pocket'] = {'training': hist_np, 'test': M.summarise(y_true_n, y_score_n)}

    out['_ablation'] = M.ablation_verdict(out['with_pocket']['test'],
                                          out['no_pocket']['test'])
    out['_shuffle_check'] = M.ablation_verdict(out['with_pocket']['test'],
                                               out['shuffled_pocket']['test'])
    log('  ablation : %s' % out['_ablation']['reason'])
    log('  shuffled : %s' % out['_shuffle_check']['reason'])
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', default='corpus/v1')
    ap.add_argument('--regimes', nargs='+', default=list(REGIMES))
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    results = {regime: run(args.release, regime, args.epochs) for regime in args.regimes}
    path = args.out or os.path.join(args.release, 'results', 'model.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote', path)
