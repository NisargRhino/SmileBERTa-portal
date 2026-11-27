"""Metrics for both models.

Nothing here reduces a model to one number. The classification dataset runs from
2,923 antiinfectives down to 48 urologicals, and a single accuracy figure would
hide that the rarest class is being predicted from 35 training examples — which
is the opposite of what this benchmark is for. Every classification result is
reported per class, with its support attached.

The fragment metrics include one the shipped pipeline has never measured:
`substructure_rate`, the fraction of predicted fragments that are genuinely part
of the molecule they came from. A "fragment" that is not a substructure of its
own input is a hallucination, and the nearest-reference fallback can produce one
whenever the model's raw output does not parse.
"""
from collections import Counter, defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors

from evaluation.data import CATEGORIES, canonical, is_substructure, strip_dummies

RDLogger.DisableLog('rdApp.*')


# ── Classification ───────────────────────────────────────────────────────────

def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def classification_metrics(y_true, y_pred, y_rank=None, categories=CATEGORIES):
    """Per-class P/R/F1 + support, macro/micro F1, Hamming loss, subset accuracy.

    `y_true` / `y_pred` are lists of label *sets* (multi-label: 1,350 of the
    8,336 compounds carry more than one category).

    `y_rank`, when given, is a list of category rankings best-first, used for
    top-3 accuracy. Ranking by category rather than by output string is what
    makes top-3 comparable between the transformer (which ranks label strings by
    edit distance) and the fingerprint baselines (which rank by probability).
    """
    per_class = {}
    macro_f1 = 0.0
    total_tp = total_fp = total_fn = 0
    for cat in categories:
        tp = sum(1 for t, p in zip(y_true, y_pred) if cat in t and cat in p)
        fp = sum(1 for t, p in zip(y_true, y_pred) if cat not in t and cat in p)
        fn = sum(1 for t, p in zip(y_true, y_pred) if cat in t and cat not in p)
        precision, recall, f1 = _prf(tp, fp, fn)
        per_class[cat] = {'precision': round(precision, 4), 'recall': round(recall, 4),
                          'f1': round(f1, 4), 'support': tp + fn,
                          'predicted': tp + fp}
        macro_f1 += f1
        total_tp, total_fp, total_fn = total_tp + tp, total_fp + fp, total_fn + fn
    macro_f1 /= max(len(categories), 1)
    micro_p, micro_r, micro_f1 = _prf(total_tp, total_fp, total_fn)

    n = max(len(y_true), 1)
    subset_accuracy = sum(1 for t, p in zip(y_true, y_pred) if set(t) == set(p)) / n
    # Hamming loss over the 12-dimensional label indicator.
    wrong_bits = sum(len(set(t) ^ set(p)) for t, p in zip(y_true, y_pred))
    hamming = wrong_bits / (n * len(categories))
    # "Any correct" is the forgiving view: did the model name at least one of the
    # compound's real categories? Reported next to subset accuracy so the gap
    # between them shows how much of the error is multi-label bookkeeping.
    any_correct = sum(1 for t, p in zip(y_true, y_pred) if set(t) & set(p)) / n

    out = {
        'n': len(y_true),
        'per_class': per_class,
        'macro_f1': round(macro_f1, 4),
        'micro_f1': round(micro_f1, 4),
        'micro_precision': round(micro_p, 4),
        'micro_recall': round(micro_r, 4),
        'subset_accuracy': round(subset_accuracy, 4),
        'any_correct': round(any_correct, 4),
        'hamming_loss': round(hamming, 4),
    }
    if y_rank is not None:
        for k in (1, 3):
            hits = sum(1 for t, r in zip(y_true, y_rank) if set(r[:k]) & set(t))
            out[f'top_{k}_accuracy'] = round(hits / n, 4)

    # Multi-label subset reported separately: it is the harder problem and
    # averaging it into the single-label majority hides how much harder.
    multi = [(t, p) for t, p in zip(y_true, y_pred) if len(t) > 1]
    if multi:
        mt, mp = [t for t, _ in multi], [p for _, p in multi]
        out['multi_label_subset'] = {
            'n': len(multi),
            'subset_accuracy': round(sum(1 for t, p in zip(mt, mp) if set(t) == set(p)) / len(multi), 4),
            'any_correct': round(sum(1 for t, p in zip(mt, mp) if set(t) & set(p)) / len(multi), 4),
            'hamming_loss': round(sum(len(set(t) ^ set(p)) for t, p in zip(mt, mp))
                                  / (len(multi) * len(categories)), 4),
        }
    return out


def confusion_matrix(y_true, y_pred, categories=CATEGORIES):
    """Counts over single-label compounds only.

    A confusion matrix needs one true class per row. The 1,350 multi-label
    compounds have no single truth, so they are excluded here and measured by
    Hamming loss and subset accuracy instead; `n` records how many were used.
    """
    matrix = {t: Counter() for t in categories}
    used = 0
    for t, p in zip(y_true, y_pred):
        if len(t) != 1:
            continue
        used += 1
        truth = next(iter(t))
        predicted = sorted(p)[0] if p else '(none)'
        matrix[truth][predicted] += 1
    return {'n': used,
            'matrix': {t: {p: c for p, c in matrix[t].most_common()} for t in categories}}


# ── Fragment prediction ──────────────────────────────────────────────────────

def _fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def tanimoto(a_smiles, b_smiles):
    a, b = Chem.MolFromSmiles(a_smiles or ''), Chem.MolFromSmiles(b_smiles or '')
    if a is None or b is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(_fp(a), _fp(b))


def passes_rule_of_three(smiles):
    """Congreve's Rule of Three, the fragment-screening convention.

    MW < 300, cLogP ≤ 3, H-bond donors ≤ 3, acceptors ≤ 3, rotatable bonds ≤ 3.
    """
    mol = Chem.MolFromSmiles(smiles or '')
    if mol is None:
        return False
    return (Descriptors.MolWt(mol) < 300
            and Descriptors.MolLogP(mol) <= 3
            and Descriptors.NumHDonors(mol) <= 3
            and Descriptors.NumHAcceptors(mol) <= 3
            and Descriptors.NumRotatableBonds(mol) <= 3)


def normalise_fragment(smiles):
    """Canonical, dummy-free form, for comparing a prediction with a reference.

    BRICS references carry `[n*]` attachment points and the model never emits
    them, so comparing the two raw would score zero by construction rather than
    by performance.
    """
    if not smiles:
        return None
    mol = strip_dummies(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def fragment_metrics(records):
    """`records`: {smiles, references, raw, prediction, correction}.

    `raw` is the model's own output before any repair; `prediction` is what the
    pipeline would actually hand a user. Both are measured, because the gap
    between them is how much work the corrector and the nearest-reference
    fallback are doing on the model's behalf.
    """
    n = max(len(records), 1)
    valid_raw = 0
    intervened = 0
    exact = 0
    exact_raw = 0
    tanimotos = []
    ro3 = 0
    substructure = 0
    substructure_by_path = defaultdict(lambda: [0, 0])
    empty_raw = 0

    for rec in records:
        raw, pred = rec.get('raw'), rec.get('prediction')
        refs = [normalise_fragment(f) for f in rec['references']]
        refs = [r for r in refs if r]

        if not (raw or '').strip():
            empty_raw += 1
        if raw and Chem.MolFromSmiles(raw) is not None:
            valid_raw += 1
            if normalise_fragment(raw) in refs:
                exact_raw += 1
        else:
            intervened += 1

        norm = normalise_fragment(pred)
        if norm and norm in refs:
            exact += 1
        best = max((tanimoto(pred, r) for r in refs), default=0.0)
        tanimotos.append(best)
        if passes_rule_of_three(pred):
            ro3 += 1
        is_sub = is_substructure(pred, rec['smiles']) if pred else False
        substructure += 1 if is_sub else 0
        path = rec.get('correction') or 'model'
        substructure_by_path[path][0] += 1 if is_sub else 0
        substructure_by_path[path][1] += 1

    tanimotos.sort()
    return {
        'n': len(records),
        'validity_rate_raw': round(valid_raw / n, 4),
        'intervention_rate': round(intervened / n, 4),
        'empty_output_rate': round(empty_raw / n, 4),
        'exact_match_rate': round(exact / n, 4),
        'exact_match_rate_raw_only': round(exact_raw / n, 4),
        'mean_tanimoto_to_reference': round(sum(tanimotos) / n, 4),
        'median_tanimoto_to_reference': round(tanimotos[len(tanimotos) // 2], 4) if tanimotos else 0.0,
        'rule_of_three_pass_rate': round(ro3 / n, 4),
        # The hallucination line: a predicted fragment that is not part of the
        # molecule it was derived from.
        'substructure_rate': round(substructure / n, 4),
        'hallucination_rate': round(1 - substructure / n, 4),
        'substructure_by_path': {k: {'rate': round(v[0] / max(v[1], 1), 4), 'n': v[1]}
                                 for k, v in substructure_by_path.items()},
    }
