"""The §4 diagnosis: learning curve, data-scaling curve, and one intervention at a time.

Each experiment writes its own JSON into `evaluation/results/` so a run can be
interrupted and resumed, and so every number in BENCHMARK.md can be traced back
to the run that produced it.

Checkpoints are large and are never committed: point `SB_CHECKPOINT_DIR` at
scratch space.

Ordering follows the spec: early stopping first, then dropout and weight decay,
then class weighting for the long tail, then freezing the lower layers. Nothing
is combined, because a combined change tells you only that *something* helped.
"""
import argparse
import json
import os
import time

from evaluation import evaluate as E
from evaluation import train as T

RESULTS = T.RESULTS
CHECKPOINTS = os.environ.get('SB_CHECKPOINT_DIR', T.CHECKPOINTS)


def _write(name, payload):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print('    wrote', os.path.relpath(path, T.D.REPO))
    return path


def _evaluate(task, model_dir, split_kind, subset='test'):
    if task == 'classification':
        res, _rows = E.evaluate_classification(model_dir, split_kind, subset, contaminated=False)
        summary = {k: res[k] for k in ('macro_f1', 'micro_f1', 'top_3_accuracy',
                                       'subset_accuracy', 'any_correct', 'hamming_loss')}
    else:
        res, _rows = E.evaluate_fragments(model_dir, split_kind, subset, contaminated=False)
        summary = {k: res[k] for k in ('validity_rate_raw', 'exact_match_rate',
                                       'mean_tanimoto_to_reference', 'substructure_rate',
                                       'rule_of_three_pass_rate')}
    return res, summary


def run_one(task, split_kind, name, *, train_fraction=1.0, log=print, **train_kwargs):
    """Train one configuration, evaluate it on test, and return the full record."""
    part = T.load_task(task, split_kind)
    train_records = part['train']
    if train_fraction < 1.0:
        # Deterministic prefix of a seeded shuffle, so 10% is a subset of 25%.
        import random
        order = list(range(len(train_records)))
        random.Random(T.SEED).shuffle(order)
        keep = order[:max(1, int(round(train_fraction * len(order))))]
        train_records = [train_records[i] for i in sorted(keep)]

    train_pairs = T.pairs_for(task, train_records)
    val_pairs = T.pairs_for(task, part['validation'])
    weights = None
    if train_kwargs.pop('class_weighting', False):
        if task != 'classification':
            raise ValueError('class weighting applies to the classifier only')
        per_record = T.class_weights(train_records)
        weights = per_record   # one pair per record for classification

    log('  [%s/%s] %s: %d train pairs, %d val pairs'
        % (task, split_kind, name, len(train_pairs), len(val_pairs)))
    t0 = time.time()
    model, history = T.train_model(train_pairs, val_pairs, weights=weights, log=log,
                                   **train_kwargs)
    model_dir = os.path.join(CHECKPOINTS, f'{task}_{split_kind}_{name}')
    T.save(model, model_dir)
    res, summary = _evaluate(task, model_dir, split_kind)
    record = {
        'task': task, 'split': split_kind, 'name': name,
        'train_fraction': train_fraction,
        'train_pairs': len(train_pairs), 'train_compounds': len(train_records),
        'config': {k: v for k, v in train_kwargs.items()},
        'class_weighting': weights is not None,
        'training': history,
        'test': summary,
        'full_test_metrics': res,
        'checkpoint': model_dir,
        'minutes': round((time.time() - t0) / 60, 1),
    }
    log('  -> %s' % json.dumps(summary))
    return record


def learning_curve(task, epochs, log=print):
    """Train/validation loss per epoch with early stopping **off**.

    Early stopping would hide the upturn, and the epoch where validation loss
    turns upward is exactly what this curve is for.

    This run does triple duty, and deliberately so: it trains the full
    scaffold-split training set for a fixed number of epochs and then restores
    the weights from the best validation epoch. That restored model *is* the
    early-stopped model, so this same record is the scaffold headline and the
    spec's intervention 1. Running it three times would burn an hour to produce
    three copies of one number.
    """
    record = run_one(task, 'scaffold', 'learning_curve', epochs=epochs,
                     early_stopping=False, log=log)
    hist = record['training']['history']
    turn = None
    for i in range(1, len(hist)):
        if hist[i]['val_loss'] > hist[i - 1]['val_loss']:
            turn = hist[i - 1]['epoch']
            break
    record['validation_loss_turns_upward_after_epoch'] = turn
    record['doubles_as'] = ['headline_scaffold', 'intervention_1_early_stopping']
    _write(f'learning_curve_{task}', record)
    return record


def headline(task, epochs, scaffold_record=None, log=print):
    """The honest pair: the same recipe trained under each split.

    The gap between them is the overfitting measurement the spec asks for, on
    models whose training data we control — unlike the shipped checkpoints,
    whose training set is unknown and overlaps any test set defined afterwards.
    """
    out = {}
    out['scaffold'] = scaffold_record or run_one(task, 'scaffold', 'headline',
                                                 epochs=epochs, log=log)
    out['random'] = run_one(task, 'random', 'headline', epochs=epochs, log=log)
    key = 'macro_f1' if task == 'classification' else 'exact_match_rate'
    out['gap'] = {'metric': key,
                  'random': out['random']['test'][key],
                  'scaffold': out['scaffold']['test'][key],
                  'difference': round(out['random']['test'][key] - out['scaffold']['test'][key], 4)}
    _write(f'headline_{task}', out)
    return out


def data_scaling(task, epochs, fractions=(0.1, 0.25, 0.5), full_record=None, log=print):
    """Performance at 10/25/50/100% of the scaffold-split training set.

    Still climbing at 100% means the bottleneck is data volume, not architecture
    or regularisation — which changes what to do next.
    """
    points = []
    for frac in fractions:
        points.append(run_one(task, 'scaffold', f'scale_{int(frac * 100)}',
                              train_fraction=frac, epochs=epochs, log=log))
    # 100% is the full scaffold run that has already been trained.
    points.append(full_record or run_one(task, 'scaffold', 'scale_100',
                                         train_fraction=1.0, epochs=epochs, log=log))
    payload = {'task': task, 'points': points,
               'still_climbing': _still_climbing(task, points)}
    _write(f'data_scaling_{task}', payload)
    return payload


def _still_climbing(task, points):
    key = 'macro_f1' if task == 'classification' else 'mean_tanimoto_to_reference'
    if len(points) < 2:
        return None
    last, prev = points[-1]['test'][key], points[-2]['test'][key]
    return bool(last > prev + 0.005)


def interventions(task, epochs, early_stopped_record=None, log=print):
    """The spec's list, in order, each measured on its own."""
    # Step 1 (early stopping) is the learning-curve run with its best-validation
    # weights restored, so it is passed in rather than retrained.
    steps = [
        ('2_dropout_weight_decay', dict(epochs=epochs, early_stopping=True, patience=2,
                                        dropout=0.2, weight_decay=0.01)),
    ]
    if task == 'classification':
        steps.append(('3_class_weighting', dict(epochs=epochs, early_stopping=True,
                                                patience=2, class_weighting=True)))
    steps.append(('4_freeze_lower_layers', dict(epochs=epochs, early_stopping=True,
                                                patience=2, freeze_below_top=2)))
    out = []
    if early_stopped_record is not None:
        step1 = dict(early_stopped_record)
        step1['name'] = '1_early_stopping'
        step1['note'] = ('the learning-curve run with best-validation weights restored, '
                         'which is what early stopping selects')
        out.append(step1)
    else:
        out.append(run_one(task, 'scaffold', '1_early_stopping',
                           epochs=epochs, early_stopping=True, patience=2, log=log))
    for name, kwargs in steps:
        out.append(run_one(task, 'scaffold', name, log=log, **kwargs))
    _write(f'interventions_{task}', {'task': task, 'steps': out})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('experiment', choices=('headline', 'learning_curve', 'data_scaling',
                                           'interventions', 'all'))
    ap.add_argument('--task', choices=('classification', 'fragment'), required=True)
    ap.add_argument('--epochs', type=int, default=6)
    args = ap.parse_args()

    if args.experiment == 'all':
        # One pass, sharing the full scaffold run between the three stages that
        # would otherwise each retrain it.
        curve = learning_curve(args.task, args.epochs)
        headline(args.task, args.epochs, scaffold_record=curve)
        data_scaling(args.task, args.epochs, full_record=curve)
        interventions(args.task, args.epochs, early_stopped_record=curve)
        return
    if args.experiment == 'headline':
        headline(args.task, args.epochs)
    if args.experiment == 'learning_curve':
        learning_curve(args.task, args.epochs)
    if args.experiment == 'data_scaling':
        data_scaling(args.task, args.epochs)
    if args.experiment == 'interventions':
        interventions(args.task, args.epochs)


if __name__ == '__main__':
    main()
