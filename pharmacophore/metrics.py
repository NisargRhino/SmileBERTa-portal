"""Metrics for per-atom interaction prediction.

Accuracy is not reported here, at all. Around 70% of ligand atoms carry no
interaction and the rarer types sit below 1%, so a model that predicts nothing
anywhere scores in the high nineties on the rarest type and around 70% overall.
Every number below is per interaction type, with its support attached, for the
same reason the classification benchmark reports per class: an average over
twelve types hides that two of them are being answered and ten are not.

Threshold-free metrics (AUC, average precision) come first because they do not
depend on a cutoff chosen after the fact. Precision, recall and F1 follow at a
stated threshold.
"""
import numpy as np

from pharmacophore.interactions import INTERACTIONS


def _safe_auc(y_true, scores):
    """ROC AUC without sklearn's exception when a class is absent."""
    pos, neg = int(y_true.sum()), int((1 - y_true).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks over ties, or AUC is biased by the tie-breaking order —
    # which matters here because a constant predictor ties everything.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[y_true == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _average_precision(y_true, scores):
    pos = int(y_true.sum())
    if pos == 0:
        return None
    order = np.argsort(-scores)
    y = y_true[order]
    cumulative = np.cumsum(y)
    precision = cumulative / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / pos)


def per_type_metrics(y_true, y_score, threshold=0.5, names=None):
    """Per interaction type: AUC, average precision, P/R/F1 and support."""
    names = names or INTERACTIONS
    out = {}
    for i, name in enumerate(names):
        truth = y_true[:, i].astype(int)
        scores = y_score[:, i].astype(float)
        predicted = (scores >= threshold).astype(int)
        tp = int(((truth == 1) & (predicted == 1)).sum())
        fp = int(((truth == 0) & (predicted == 1)).sum())
        fn = int(((truth == 1) & (predicted == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[name] = {
            'support': int(truth.sum()),
            'base_rate': round(float(truth.mean()), 5),
            'auc': _round(_safe_auc(truth, scores)),
            'average_precision': _round(_average_precision(truth, scores)),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'predicted_positive': int(predicted.sum()),
        }
    return out


def _round(value, nd=4):
    return None if value is None else round(value, nd)


def summarise(y_true, y_score, threshold=0.5, names=None):
    """Per-type metrics plus the only aggregates that are honest here.

    Macro averages are taken over the types that actually occur in this test
    set: averaging in a type with zero support would quietly reward a model for
    a question nobody asked it.
    """
    names = names or INTERACTIONS
    per_type = per_type_metrics(y_true, y_score, threshold, names)
    present = [m for m in per_type.values() if m['support'] > 0]
    aucs = [m['auc'] for m in present if m['auc'] is not None]
    aps = [m['average_precision'] for m in present if m['average_precision'] is not None]

    any_true = (y_true.sum(axis=1) > 0).astype(int)
    any_score = y_score.max(axis=1)
    return {
        'atoms': int(y_true.shape[0]),
        'types_present': len(present),
        'per_type': per_type,
        'macro_auc': _round(float(np.mean(aucs)) if aucs else None),
        'macro_average_precision': _round(float(np.mean(aps)) if aps else None),
        'macro_f1': round(float(np.mean([m['f1'] for m in present])), 4) if present else None,
        # "Does this atom interact at all" — the coarse question, useful because
        # it is what a highlighted pharmacophore actually shows a user.
        'any_interaction': {
            'base_rate': round(float(any_true.mean()), 4),
            'auc': _round(_safe_auc(any_true, any_score)),
            'average_precision': _round(_average_precision(any_true, any_score)),
        },
    }


def compare(results, key='macro_auc'):
    """Rank named results, for the ablation table."""
    rows = [(name, res.get(key)) for name, res in results.items()]
    rows.sort(key=lambda r: (r[1] is None, -(r[1] or 0)))
    return rows


def ablation_verdict(with_pocket, without_pocket, key='macro_auc', margin=0.02):
    """Did the protein actually contribute?

    The spec makes this a build gate. Most ligands appear in exactly one
    complex, so a model can reproduce the labels by memorising molecules and
    never consulting the protein — which would make the pocket-conditioned half
    of the project an illusion. If removing the pocket costs less than `margin`,
    that is the answer.
    """
    a, b = with_pocket.get(key), without_pocket.get(key)
    if a is None or b is None:
        return {'verdict': 'unknown', 'reason': f'{key} missing from one arm'}
    delta = a - b
    return {
        'metric': key, 'with_pocket': round(a, 4), 'without_pocket': round(b, 4),
        'delta': round(delta, 4),
        'verdict': 'uses_the_pocket' if delta >= margin else 'ignores_the_pocket',
        'reason': ('removing the pocket costs %.4f, at or above the %.2f margin' % (delta, margin))
        if delta >= margin else
        ('removing the pocket costs only %.4f, below the %.2f margin: the model is '
         'predicting from the ligand alone' % (delta, margin)),
    }
