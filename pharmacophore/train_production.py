"""Train the model that gets served, and save it with its own honest numbers.

The evaluation runs exist to measure; this one exists to ship. It trains on
train plus validation and never touches test, then writes a checkpoint carrying
the test metrics *from the evaluation run* — not from anything this script
measured, because a model trained on validation has no clean validation left
and any number it computed would be flattering.

Refuses to write a checkpoint if the evaluation says the model failed its
ablation gate, unless explicitly overridden. Shipping a pocket-conditioned
predictor that does not use the pocket would be advertising something the
measurements say is not there.
"""
import argparse
import json
import os

from pharmacophore import dataset as D
from pharmacophore import model as Model
from pharmacophore import train_geometric as TG
from pharmacophore.splits import load_split


def run(release, regime='unseen_protein', epochs=30, out=None, force=False,
        log=print):
    results_path = os.path.join(release, 'results', 'model_geometric.json')
    evaluation = {}
    if os.path.exists(results_path):
        with open(results_path) as handle:
            evaluation = (json.load(handle) or {}).get(regime) or {}
    ablation = (evaluation.get('_ablation') or {})
    verdict = ablation.get('verdict')
    log('evaluation verdict for %s: %s (delta %s)'
        % (regime, verdict, ablation.get('delta')))

    if verdict != 'uses_the_pocket' and not force:
        log('refusing to write a checkpoint: the evaluation says this model '
            'does not use the pocket, so serving it as a pocket-aware '
            'pharmacophore predictor would claim something the measurements '
            'do not support. Pass --force to override deliberately.')
        return None

    split_map = load_split(release, regime)
    train_examples = D.load_examples(release, split_map, 'train')
    val_examples = D.load_examples(release, split_map, 'validation')
    # Train on train+validation: model selection already happened during
    # evaluation, and the extra 10% is free accuracy for the served model.
    # Test is untouched, which is what keeps the reported numbers meaningful.
    examples = train_examples + val_examples
    coverage = D.geometry_coverage(examples)
    log('training on %d complexes (%.1f%% with geometry), holding out nothing '
        'but test' % (len(examples), 100 * coverage['fraction']))
    if not examples:
        log('no training data')
        return None

    # A small validation slice carved from training, only so early stopping has
    # something to watch. It is not a measurement.
    cut = max(1, int(0.95 * len(examples)))
    net, history = TG.train_one(examples[:cut], examples[cut:], use_pocket=True,
                                epochs=epochs, log=log)

    out = out or os.path.join(release, 'models', 'geometric.pt')
    test = (evaluation.get('with_pocket') or {}).get('test') or {}
    Model.save_checkpoint(
        net, out,
        atom_dim=examples[0]['atom_features'].shape[1],
        neighbour_dim=D.NEIGHBOUR_DIM,
        kind='geometric',
        metadata={
            'release': release,
            'regime_evaluated': regime,
            'complexes': len(examples),
            # From the evaluation run, on a test set this model never saw.
            'test_macro_auc': test.get('macro_auc'),
            'ablation_delta': ablation.get('delta'),
            'ablation_verdict': verdict,
            'best_val_macro_auc_during_production_fit': history.get('best_val_macro_auc'),
            'note': 'test metrics come from the evaluation run; this model was '
                    'fitted on train+validation and measured nothing itself',
        })
    log('wrote %s' % out)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', default='corpus/v1')
    ap.add_argument('--regime', default='unseen_protein')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--out', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    run(args.release, args.regime, args.epochs, args.out, args.force)
