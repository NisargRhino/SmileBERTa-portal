"""Evaluate a checkpoint on a committed split.

Both shipped models are `RobertaForMaskedLM` used as position-wise translators:
the SMILES goes in, every output position is decoded with argmax, and the
resulting string is read back as an answer — a fragment SMILES, or a therapeutic
class label that is then snapped onto the closest known label by edit distance.
This file reproduces that path exactly rather than a tidier one, because the
number that matters is what the deployed pipeline actually does.

**On the shipped checkpoints and the scaffold split.** The published models were
trained on an unknown split, so their training set almost certainly overlaps any
test set defined afterwards. Their numbers here are therefore contaminated, and
every result carries `contaminated: true` to say so. They are reported because
"what does the model users are hitting today actually score" is a real question —
but the honest headline comes from the models retrained on the scaffold-split
training set in `train.py`.
"""
import argparse
import json
import os

import torch
from rdkit import Chem, RDLogger
from transformers import RobertaForMaskedLM, RobertaTokenizer

from evaluation import data as D
from evaluation import metrics as M
from evaluation.baselines import partition
from evaluation.splits import SPLIT_DIR, load_split

RDLogger.DisableLog('rdApp.*')

FRAGMENT_MODEL = 'NisargRhino/protein-models'
CLASSIFICATION_MODEL = 'NisargRhino/drug-classification'
MAX_LENGTH = 128


def device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load(name_or_path):
    tokenizer = RobertaTokenizer.from_pretrained(name_or_path)
    model = RobertaForMaskedLM.from_pretrained(name_or_path)
    model.eval().to(device())
    return model, tokenizer


@torch.no_grad()
def decode_batch(model, tokenizer, smiles_list, batch_size=32):
    """The argmax-decode path from fragpred.py / predict_drug_classification.py."""
    out = []
    dev = device()
    for start in range(0, len(smiles_list), batch_size):
        chunk = smiles_list[start:start + batch_size]
        enc = tokenizer(chunk, max_length=MAX_LENGTH, padding='max_length',
                        truncation=True, return_tensors='pt')
        enc = {k: v.to(dev) for k, v in enc.items() if k in ('input_ids', 'attention_mask')}
        logits = model(**enc).logits
        ids = torch.argmax(logits, dim=-1).cpu()
        out.extend(tokenizer.decode(row, skip_special_tokens=True).strip() for row in ids)
    return out


# ── Classification ───────────────────────────────────────────────────────────

def _levenshtein(a, b):
    """Edit distance, matching what predict_drug_classification.py uses."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def snap(raw, options):
    """Closest known label string by edit distance — the shipped rule."""
    best, best_d = options[0], None
    for opt in options:
        d = _levenshtein(raw, opt)
        if best_d is None or d < best_d:
            best, best_d = opt, d
    return best, best_d


def category_ranking(raw, options):
    """The 12 categories ranked best-first for this raw output.

    A category scores as well as the best option string containing it, so the
    ranking is over categories rather than over the 38 label strings. That makes
    top-3 here mean the same thing as top-3 for the fingerprint baselines.
    """
    best = {}
    for opt in options:
        d = _levenshtein(raw, opt)
        for cat in opt.split():
            if cat in D.CATEGORIES and (cat not in best or d < best[cat]):
                best[cat] = d
    return sorted(D.CATEGORIES, key=lambda c: best.get(c, 10 ** 6))


def evaluate_classification(model_path=CLASSIFICATION_MODEL, split_kind='scaffold',
                            subset='test', contaminated=True, limit=None):
    records, _ = D.load_classification()
    split_map = load_split(os.path.join(SPLIT_DIR, f'classification_{split_kind}.csv'))
    part = partition(records, split_map)[subset]
    if limit:
        part = part[:limit]
    options = D.label_options()

    model, tokenizer = load(model_path)
    raws = decode_batch(model, tokenizer, [r['smiles'] for r in part])

    y_true, y_pred, y_rank, rows = [], [], [], []
    snapped_exact = 0
    for rec, raw in zip(part, raws):
        label, dist = snap(raw, options)
        pred = {c for c in label.split() if c in D.CATEGORIES}
        y_true.append(set(rec['labels']))
        y_pred.append(pred)
        y_rank.append(category_ranking(raw, options))
        snapped_exact += 1 if dist == 0 else 0
        rows.append({'smiles': rec['smiles'], 'true': sorted(rec['labels']),
                     'raw': raw, 'snapped': label, 'edit_distance': dist})

    out = M.classification_metrics(y_true, y_pred, y_rank)
    out['confusion'] = M.confusion_matrix(y_true, y_pred)
    # How often the model's free-form output was already a valid label, before
    # edit-distance snapping rescued it.
    out['raw_output_was_exact_label'] = round(snapped_exact / max(len(part), 1), 4)
    out['model'] = model_path
    out['split'] = f'classification_{split_kind}'
    out['subset'] = subset
    out['contaminated'] = contaminated
    return out, rows


# ── Fragments ────────────────────────────────────────────────────────────────

def evaluate_fragments(model_path=FRAGMENT_MODEL, split_kind='scaffold', subset='test',
                       contaminated=True, use_nearest_reference=True, limit=None):
    """Includes the nearest-reference repair path, since that is what users get.

    The remote SMILES corrector is deliberately not called: it is a third-party
    service whose availability would make this benchmark unreproducible. Runs
    record `correction: 'nearest-reference'` where the shipped pipeline would
    have tried the corrector first, and `intervention_rate` counts every case
    where the raw output did not parse.
    """
    records, _ = D.load_fragments()
    split_map = load_split(os.path.join(SPLIT_DIR, f'fragment_{split_kind}.csv'))
    part = partition(records, split_map)[subset]
    if limit:
        part = part[:limit]

    reference_fragments = []
    if use_nearest_reference:
        import csv as _csv
        with open(os.path.join(D.REPO, 'unique_smile5.csv')) as f:
            reference_fragments = [r['SMILES'] for r in _csv.DictReader(f) if r.get('SMILES')]

    model, tokenizer = load(model_path)
    raws = decode_batch(model, tokenizer, [r['smiles'] for r in part])

    rows = []
    for rec, raw in zip(part, raws):
        valid = bool(raw) and Chem.MolFromSmiles(raw) is not None
        if valid:
            prediction, correction = raw, None
        elif reference_fragments:
            prediction = _closest_reference(raw, reference_fragments)
            correction = 'nearest-reference'
        else:
            prediction, correction = raw, 'none-available'
        rows.append({'smiles': rec['smiles'], 'references': rec['fragments'],
                     'raw': raw, 'prediction': prediction, 'correction': correction})

    out = M.fragment_metrics(rows)
    out['model'] = model_path
    out['split'] = f'fragment_{split_kind}'
    out['subset'] = subset
    out['contaminated'] = contaminated
    return out, rows


def _closest_reference(raw, references):
    """fragpred.closest_reference_smiles: nearest valid fragment by edit distance."""
    best, best_sim = None, -1.0
    for ref in references:
        max_len = max(len(raw), len(ref), 1)
        sim = 1 - _levenshtein(raw, ref) / max_len
        if sim > best_sim:
            best, best_sim = ref, sim
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', choices=('classification', 'fragment', 'both'), default='both')
    ap.add_argument('--splits', nargs='+', default=['scaffold', 'random'])
    ap.add_argument('--classification-model', default=CLASSIFICATION_MODEL)
    ap.add_argument('--fragment-model', default=FRAGMENT_MODEL)
    ap.add_argument('--label', default='shipped')
    ap.add_argument('--subset', default='test')
    ap.add_argument('--not-contaminated', action='store_true',
                    help='set for a model trained on this split\'s own training set')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--out-dir', default=os.path.join(os.path.dirname(SPLIT_DIR), 'results'))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for kind in args.splits:
        block = {}
        if args.task in ('classification', 'both'):
            res, rows = evaluate_classification(args.classification_model, kind, args.subset,
                                                not args.not_contaminated, args.limit)
            block['classification'] = res
            _dump(os.path.join(args.out_dir, f'predictions_classification_{args.label}_{kind}.json'), rows)
            print('[%s/%s] classification: macro-F1 %.3f micro-F1 %.3f top-3 %.3f subset-acc %.3f'
                  % (args.label, kind, res['macro_f1'], res['micro_f1'],
                     res['top_3_accuracy'], res['subset_accuracy']))
        if args.task in ('fragment', 'both'):
            res, rows = evaluate_fragments(args.fragment_model, kind, args.subset,
                                           not args.not_contaminated, True, args.limit)
            block['fragment'] = res
            _dump(os.path.join(args.out_dir, f'predictions_fragment_{args.label}_{kind}.json'), rows)
            print('[%s/%s] fragment: validity %.3f exact %.3f tanimoto %.3f substructure %.3f'
                  % (args.label, kind, res['validity_rate_raw'], res['exact_match_rate'],
                     res['mean_tanimoto_to_reference'], res['substructure_rate']))
        results[kind] = block

    path = os.path.join(args.out_dir, f'evaluation_{args.label}.json')
    _dump(path, results)
    print('wrote', path)


def _dump(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


if __name__ == '__main__':
    main()
