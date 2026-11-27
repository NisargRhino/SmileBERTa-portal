"""Named failure cases for the write-up.

A benchmark that reports only aggregates lets a reader nod along without ever
seeing what a wrong answer looks like. These are specific, checkable molecules,
run through the same path the site uses.

Ciprofloxacin is here because the platform already documents it as a known
failure on `/learn-classification` ("it labels the antibiotic ciprofloxacin as
antineoplastic"). The claim should be verified by the benchmark rather than
repeated on faith — so this script records what the model actually says today.
"""
import argparse
import json
import os

from rdkit import Chem, RDLogger

from evaluation import data as D
from evaluation import evaluate as E
from evaluation import metrics as M

RDLogger.DisableLog('rdApp.*')
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# SMILES taken from the compounds themselves, canonicalised on load. Each entry
# says why it is interesting, which is what the write-up prints next to it.
CASES = [
    ('ciprofloxacin', 'O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O',
     'documented failure: a fluoroquinolone antibiotic the classifier has called antineoplastic'),
    ('aspirin', 'CC(=O)Oc1ccccc1C(=O)O',
     'small, common, unambiguous anti-inflammatory'),
    ('caffeine', 'Cn1c(=O)c2c(ncn2C)n(C)c1=O',
     'CNS stimulant with no therapeutic class in the usual sense'),
    ('imatinib', 'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1',
     'kinase inhibitor, antineoplastic'),
    ('atorvastatin', 'CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O',
     'lipid-regulating, one of the smaller classes'),
    ('tamsulosin', 'CCOc1ccccc1OCCN[C@@H](C)Cc1ccc(OC)c(S(N)(=O)=O)c1',
     'urological, the rarest class (48 compounds in total)'),
    ('salbutamol', 'CC(C)(C)NCC(O)c1ccc(O)c(CO)c1',
     'respiratory, a class the models recall poorly'),
]


def in_dataset(records_by_smiles, smiles):
    rec = records_by_smiles.get(smiles)
    return sorted(rec['labels']) if rec else None


def run(classification_model=E.CLASSIFICATION_MODEL, fragment_model=E.FRAGMENT_MODEL,
        label='shipped'):
    cls_records, _ = D.load_classification()
    by_smiles = {r['smiles']: r for r in cls_records}
    split_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'splits', 'classification_scaffold.csv')
    from evaluation.splits import load_split
    split_map = load_split(split_file)

    canonical_cases = [(name, D.canonical(smi), why) for name, smi, why in CASES]
    options = D.label_options()

    model, tokenizer = E.load(classification_model)
    raws = E.decode_batch(model, tokenizer, [smi for _n, smi, _w in canonical_cases])

    rows = []
    for (name, smi, why), raw in zip(canonical_cases, raws):
        snapped, distance = E.snap(raw, options)
        predicted = sorted(c for c in snapped.split() if c in D.CATEGORIES)
        truth = in_dataset(by_smiles, smi)
        rows.append({
            'name': name, 'smiles': smi, 'why': why,
            'in_dataset': truth is not None,
            'dataset_split': split_map.get(smi),
            'true_labels': truth,
            'raw_output': raw,
            'snapped_label': snapped,
            'edit_distance': distance,
            'predicted': predicted,
            # None, not False, when the compound is not in the dataset: there is
            # no ground truth to be right or wrong about.
            'correct': (bool(set(predicted) & set(truth)) if truth is not None else None),
            # Did the model emit a real label, or did edit-distance snapping
            # manufacture one from noise? Aspirin's raw output is
            # 'aetaamataliatf', which snaps to a label that happens to be right.
            'raw_was_a_valid_label': raw.strip() in options,
        })

    fmodel, ftokenizer = E.load(fragment_model)
    fraws = E.decode_batch(fmodel, ftokenizer, [smi for _n, smi, _w in canonical_cases])
    for row, raw in zip(rows, fraws):
        valid = bool(raw) and Chem.MolFromSmiles(raw) is not None
        row['fragment'] = {
            'raw_output': raw,
            'valid_smiles': valid,
            'is_substructure_of_input': D.is_substructure(raw, row['smiles']) if valid else False,
            'rule_of_three': M.passes_rule_of_three(raw) if valid else False,
        }
    return {'label': label, 'classification_model': classification_model,
            'fragment_model': fragment_model, 'cases': rows}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--classification-model', default=E.CLASSIFICATION_MODEL)
    ap.add_argument('--fragment-model', default=E.FRAGMENT_MODEL)
    ap.add_argument('--label', default='shipped')
    args = ap.parse_args()
    out = run(args.classification_model, args.fragment_model, args.label)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, 'named_examples.json' if args.label == 'shipped'
                        else f'named_examples_{args.label}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    for row in out['cases']:
        verdict = 'n/a' if row['correct'] is None else ('OK' if row['correct'] else 'MISS')
        print('%-14s true=%-28s pred=%-22s %s' % (
            row['name'], ','.join(row['true_labels'] or ['(not in dataset)']),
            ','.join(row['predicted']), verdict))
        print('               raw=%r  fragment_raw=%r valid=%s substructure=%s'
              % (row['raw_output'][:40], row['fragment']['raw_output'][:40],
                 row['fragment']['valid_smiles'], row['fragment']['is_substructure_of_input']))
    print('wrote', path)
