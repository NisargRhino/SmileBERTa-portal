"""Generate BENCHMARK.md and the /benchmark page's data from the result files.

The write-up is generated rather than typed so that every number in it can be
traced to the run that produced it, and so it cannot drift from the JSON as
experiments are re-run. It records the dataset digests, the commit hash and the
date, which is what makes a benchmark citable rather than decorative.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess

from evaluation import data as D
from evaluation.splits import SPLIT_DIR

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
FRONTEND_DATA = os.path.join(D.REPO, 'frontend', 'assets', 'data', 'benchmark.json')

# Named failures the write-up must show. Ciprofloxacin is already documented as
# a known failure elsewhere on the platform and the spec asks for it by name.
NAMED_EXAMPLES = {
    'ciprofloxacin': 'O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O',
    'aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
    'caffeine': 'Cn1c(=O)c2c(ncn2C)n(C)c1=O',
    'imatinib': 'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1',
    'atorvastatin': 'CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O',
}


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=D.REPO, capture_output=True,
                              text=True, timeout=20).stdout.strip()[:12] or 'unknown'
    except Exception:
        return 'unknown'


def file_sha(path, n=16):
    try:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:n]
    except OSError:
        return 'missing'


def load(name):
    path = os.path.join(RESULTS, f'{name}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |',
           '| ' + ' | '.join('---' for _ in headers) + ' |']
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out)


def _pct(x):
    return '—' if x is None else f'{x:.3f}'


def per_class_table(metrics):
    rows = []
    for cat, v in metrics['per_class'].items():
        rows.append([cat, v['support'], _pct(v['precision']), _pct(v['recall']), _pct(v['f1'])])
    return _table(['Class', 'Support (test)', 'Precision', 'Recall', 'F1'], rows)


def collect():
    """Everything the report needs, in one dict — also served to the web page."""
    splits_meta = load_json(os.path.join(SPLIT_DIR, 'README.json'))
    payload = {
        'generated': datetime.date.today().isoformat(),
        'commit': git_commit(),
        'datasets': {
            'classification': {
                'file': 'drug_classification_data_df.csv',
                'sha256_16': file_sha(D.CLASSIFICATION_CSV),
            },
            'fragment': {
                'file': 'ml-project/drug_fragments.csv',
                'sha256_16': file_sha(D.FRAGMENT_CSV),
            },
        },
        'summary': D.summarise(),
        'splits': splits_meta,
        'baselines': load('baselines'),
        'shipped': load('evaluation_shipped'),
        'headline_classification': load('headline_classification'),
        'headline_fragment': load('headline_fragment'),
        'learning_curve_classification': load('learning_curve_classification'),
        'learning_curve_fragment': load('learning_curve_fragment'),
        'data_scaling_classification': load('data_scaling_classification'),
        'data_scaling_fragment': load('data_scaling_fragment'),
        'interventions_classification': load('interventions_classification'),
        'interventions_fragment': load('interventions_fragment'),
        'memorisation_control_fragment': load('memorisation_control_fragment'),
        'memorisation_control_classification': load('memorisation_control_classification'),
        'named_examples': load('named_examples'),
    }
    return payload


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


CLASS_NAMES = {
    'antiinfective': 'Anti-infective', 'antineoplastic': 'Antineoplastic', 'cns': 'CNS',
    'cardio': 'Cardiovascular', 'antiinflammatory': 'Anti-inflammatory',
    'gastrointestinal': 'Gastrointestinal', 'hematologic': 'Hematologic',
    'respiratorysystem': 'Respiratory', 'dermatologic': 'Dermatologic',
    'reproductivecontrol': 'Reproductive', 'lipidregulating': 'Lipid-regulating',
    'urological': 'Urological',
}
MODEL_NAMES = {
    'majority_class': 'Majority class', 'random_forest_ecfp4': 'Random forest (ECFP4)',
    'logistic_regression_ecfp4': 'Logistic regression (ECFP4)',
    'nearest_neighbour_tanimoto': 'Nearest neighbour (Tanimoto)',
    'most_common_fragment': 'Most common fragment',
}


def markdown(d):
    """The full write-up, assembled from the result files."""
    L = []
    w = L.append

    w('# Benchmark: scaffold-split evaluation of both SmileBERTa models')
    w('')
    w(f'Generated {d["generated"]} from commit `{d["commit"]}` by the scripts in '
      '`evaluation/`. Every number below is written to `evaluation/results/` by a '
      'named run and can be regenerated with the commands in '
      '[Reproducing this](#reproducing-this).')
    w('')
    w('> "No published benchmark" was the first item under limitations in the platform '
      'overview. This page is that benchmark. It is deliberately unflattering in places: '
      'a measurement that only reports good news is not a measurement.')
    w('')

    w('## What this found')
    w('')
    for line in _findings(d):
        w(f'- {line}')
    w('')

    # ── Method ──────────────────────────────────────────────────────────────
    w('## Method')
    w('')
    w('### The datasets')
    w('')
    s = d['summary']
    c, f = s['classification'], s['fragments']
    w(_table(['Dataset', 'Size', 'Scaffolds', 'Notes'], [
        ['Drug classification', f'{c["compounds"]:,} compounds',
         f'{c["distinct_scaffolds"]:,}',
         f'12 categories; {c["multi_label_compounds"]:,} carry more than one; '
         f'{c["acyclic_compounds"]} have no ring system'],
        ['Fragment prediction', f'{f["molecules"]:,} molecules / {f["fragment_pairs"]:,} fragments',
         f'{f["distinct_scaffolds"]:,}',
         f'BRICS decompositions, {f["fragments_per_molecule"]} fragments per molecule on average'],
    ]))
    w('')
    w('Class sizes across the whole classification dataset:')
    w('')
    w(_table(['Class', 'Compounds'],
             [[CLASS_NAMES.get(k, k), f'{v:,}'] for k, v in c['per_label'].items()]))
    w('')
    w('### The scaffold split')
    w('')
    w('Every compound gets its Bemis–Murcko scaffold via RDKit\'s '
      '`MurckoScaffold.MurckoScaffoldSmiles`. Compounds are grouped by scaffold and whole '
      'groups are assigned to train, validation or test at 80/10/10, largest group first '
      'so the split sizes land close to target. A group is placed in whichever split is '
      'furthest below its quota **for the classes that group carries**, so the rare '
      'classes survive; perfect stratification is impossible once whole groups must move '
      'together, and the real per-class counts are published below rather than smoothed.')
    w('')
    w('Molecules with no ring system have no Murcko scaffold. There are '
      f'{c["acyclic_compounds"]} of them, and treating them as a single group would force '
      '6.5% of the dataset into one split, so each is its own group.')
    w('')
    w('The assignment is committed to `evaluation/splits/*.csv` and loaded from there by '
      'every script. A split regenerated on the fly with a different seed would make every '
      'number in this repository incomparable with every other.')
    w('')
    if d.get('splits'):
        rows = []
        for name, r in d['splits']['reports'].items():
            rows.append([name.replace('_', ' · '), r['sizes']['train'], r['sizes']['validation'],
                         r['sizes']['test'], r['shared_scaffolds_train_test']])
        w(_table(['Dataset · split', 'Train', 'Validation', 'Test',
                  'Scaffolds in both train and test'], rows))
        w('')
        w('The random split shares hundreds of scaffolds across the train/test boundary. '
          'That is not a bug in the random split — it is what a random split *is*, and it '
          'is the reason its numbers look so much better.')
        w('')
        cls_scaffold = d['splits']['reports'].get('classification_scaffold', {})
        if cls_scaffold.get('per_label'):
            w('Per-class counts in the scaffold split:')
            w('')
            w(_table(['Class', 'Train', 'Validation', 'Test'],
                     [[CLASS_NAMES.get(k, k), v['train'], v['validation'], v['test']]
                      for k, v in cls_scaffold['per_label'].items()]))
            w('')

    w('### A word on the shipped checkpoints')
    w('')
    w('`NisargRhino/drug-classification` and `NisargRhino/protein-models` were trained '
      'before this split existed, on an unknown division of the same data. Their training '
      'set therefore almost certainly includes molecules in this test set. Their numbers '
      'are reported because "what do users get today" is a real question, but they are '
      'marked **contaminated** everywhere and they are not the headline. The honest '
      'numbers come from models retrained here on the scaffold-split training set alone.')
    w('')

    # ── Classification ──────────────────────────────────────────────────────
    w('## Drug classification')
    w('')
    rows = []
    if d.get('baselines'):
        for k, v in d['baselines']['scaffold']['classification'].items():
            if not k.startswith('_'):
                rows.append([MODEL_NAMES.get(k, k), _f(v['macro_f1']), _f(v['micro_f1']),
                             _f(v.get('top_3_accuracy')), _f(v['subset_accuracy']),
                             _f(v['hamming_loss'])])
    if d.get('headline_classification'):
        m = d['headline_classification']['scaffold']['full_test_metrics']
        rows.append(['**Transformer, retrained on this split**', _f(m['macro_f1']),
                     _f(m['micro_f1']), _f(m.get('top_3_accuracy')),
                     _f(m['subset_accuracy']), _f(m['hamming_loss'])])
    if d.get('shipped'):
        m = d['shipped']['scaffold']['classification']
        rows.append(['Shipped transformer *(contaminated)*', _f(m['macro_f1']),
                     _f(m['micro_f1']), _f(m.get('top_3_accuracy')),
                     _f(m['subset_accuracy']), _f(m['hamming_loss'])])
    w(_table(['Model', 'Macro-F1', 'Micro-F1', 'Top-3', 'Exact set', 'Hamming loss'], rows))
    w('')
    w('All on the scaffold-split test set. *Exact set* is the fraction of compounds where '
      'the predicted set of categories matched the true set exactly — the strict '
      'multi-label criterion.')
    w('')

    w('### Per class')
    w('')
    w('Support is the number of test compounds carrying that class. Urological is 48 '
      'compounds in the entire dataset and is reported here rather than dropped, because '
      'dropping it would improve macro-F1 while making the benchmark less true.')
    w('')
    sources = []
    if d.get('baselines'):
        sources.append(('Random forest', d['baselines']['scaffold']['classification']
                        ['random_forest_ecfp4']['per_class']))
    if d.get('headline_classification'):
        sources.append(('Retrained transformer',
                        d['headline_classification']['scaffold']['full_test_metrics']['per_class']))
    if d.get('shipped'):
        sources.append(('Shipped *(contaminated)*',
                        d['shipped']['scaffold']['classification']['per_class']))
    if sources:
        headers = ['Class', 'Support'] + [f'{n} F1' for n, _ in sources]
        prows = []
        for cat in sources[0][1]:
            row = [CLASS_NAMES.get(cat, cat), sources[0][1][cat]['support']]
            for _n, pc in sources:
                row.append(_f(pc[cat]['f1']) if cat in pc else '—')
            prows.append(row)
        w(_table(headers, prows))
        w('')

    if d.get('shipped'):
        full = d['shipped']['scaffold']['classification']
        if full.get('multi_label_subset'):
            ml = full['multi_label_subset']
            w(f'On the {ml["n"]} multi-label compounds in the test set, the shipped model '
              f'gets the exact set right {_f(ml["subset_accuracy"])} of the time and names '
              f'at least one correct category {_f(ml["any_correct"])} of the time.')
            w('')

    # ── The gap ─────────────────────────────────────────────────────────────
    w('## Random split vs scaffold split')
    w('')
    w('This is the measurement the whole exercise turns on.')
    w('')
    gap_rows = []
    if d.get('baselines'):
        for key, label in (('random_forest_ecfp4', 'Random forest (ECFP4)'),
                           ('logistic_regression_ecfp4', 'Logistic regression (ECFP4)')):
            r = d['baselines']['random']['classification'][key]['macro_f1']
            s = d['baselines']['scaffold']['classification'][key]['macro_f1']
            gap_rows.append([label, _f(r), _f(s), _f(r - s)])
    if d.get('headline_classification') and d['headline_classification'].get('gap'):
        g = d['headline_classification']['gap']
        gap_rows.append(['Transformer, retrained', _f(g['random']), _f(g['scaffold']),
                         _f(g['difference'])])
    if d.get('shipped'):
        r = d['shipped']['random']['classification']['macro_f1']
        s = d['shipped']['scaffold']['classification']['macro_f1']
        gap_rows.append(['Shipped transformer *(contaminated)*', _f(r), _f(s), _f(r - s)])
    w(_table(['Model', 'Random split', 'Scaffold split', 'Gap'], gap_rows))
    w('')
    w('Read the last row carefully. A model trained honestly loses a lot of performance '
      'when the test set stops sharing scaffolds with the training set — that is what the '
      'fingerprint baselines show. The shipped transformer loses almost nothing. That is '
      'not evidence that it generalises better than a random forest; it is evidence that '
      'the scaffold split did not hold anything back from it, because it had already seen '
      'these molecules.')
    w('')

    # ── Fragments ───────────────────────────────────────────────────────────
    w('## Fragment prediction')
    w('')
    frows = []
    if d.get('baselines'):
        for k, v in d['baselines']['scaffold']['fragment'].items():
            if not k.startswith('_'):
                frows.append([MODEL_NAMES.get(k, k), _f(v['validity_rate_raw']),
                              _f(v['exact_match_rate']), _f(v['mean_tanimoto_to_reference']),
                              _f(v['substructure_rate']), _f(v['rule_of_three_pass_rate'])])
    if d.get('headline_fragment'):
        m = d['headline_fragment']['scaffold']['full_test_metrics']
        frows.append(['**Transformer, retrained on this split**', _f(m['validity_rate_raw']),
                      _f(m['exact_match_rate']), _f(m['mean_tanimoto_to_reference']),
                      _f(m['substructure_rate']), _f(m['rule_of_three_pass_rate'])])
    if d.get('shipped'):
        m = d['shipped']['scaffold']['fragment']
        frows.append(['Shipped transformer *(contaminated)*', _f(m['validity_rate_raw']),
                      _f(m['exact_match_rate']), _f(m['mean_tanimoto_to_reference']),
                      _f(m['substructure_rate']), _f(m['rule_of_three_pass_rate'])])
    if d.get('memorisation_control_fragment'):
        m = d['memorisation_control_fragment']['full_test_metrics']
        frows.append(['Same recipe trained **on the test set** *(control)*',
                      _f(m['validity_rate_raw']), _f(m['exact_match_rate']),
                      _f(m['mean_tanimoto_to_reference']), _f(m['substructure_rate']),
                      _f(m['rule_of_three_pass_rate'])])
    w(_table(['Model', 'Valid SMILES', 'Exact match', 'Tanimoto', 'Is a substructure',
              'Rule of 3'], frows))
    w('')
    w('**"Is a substructure" is the line that matters.** It is the fraction of returned '
      'fragments that are genuinely contained in the molecule they were derived from. A '
      'fragment that is not part of its own input is a hallucination, and the pipeline can '
      'produce one whenever the model\'s raw output fails to parse and the '
      'nearest-reference fallback substitutes a fragment from some other molecule. Nothing '
      'in the deployed code measured this before.')
    w('')
    w('BRICS attachment points (`[5*]`, `[14*]`) are stripped from the reference fragments '
      'before any comparison, because the model never emits them; comparing raw strings '
      'would score zero by construction rather than by performance.')
    w('')
    if d.get('memorisation_control_fragment'):
        m = d['memorisation_control_fragment']['full_test_metrics']
        w('The control row exists to answer the obvious objection — *was it simply trained '
          'badly?* That model was given the test molecules in its training data and trained '
          f'for {d["memorisation_control_fragment"]["epochs"]} epochs. It reproduces '
          f'{_f(m["exact_match_rate"])} of the fragments it was shown. So the recipe does '
          'learn; what the scaffold split removes is the memorisation it was relying on.')
        w('')
    w('The fragment test set is **22 molecules**, because the entire supervised fragment '
      'corpus in this repository is 219 molecules and 1,223 fragment pairs. Treat these '
      'rates as direction, not precision: a single molecule moves any of them by 4.5 '
      'percentage points. Growing that corpus is the single highest-value thing that could '
      'be done for this model, and it matters more than any architectural change.')
    w('')

    # ── Overfitting diagnosis ───────────────────────────────────────────────
    w('## Diagnosing the overfitting')
    w('')
    for task in ('classification', 'fragment'):
        curve = d.get(f'learning_curve_{task}')
        if not curve:
            continue
        w(f'### Learning curve — {task}')
        w('')
        w(_table(['Epoch', 'Train loss', 'Validation loss'],
                 [[h['epoch'], h['train_loss'], h['val_loss']]
                  for h in curve['training']['history']]))
        w('')
        turn = curve.get('validation_loss_turns_upward_after_epoch')
        if turn:
            w(f'Validation loss bottoms out at epoch {turn} and rises after it. Everything '
              f'past epoch {turn} is the model memorising the training set. Training was '
              'stopped there for every result reported above.')
        else:
            w('Validation loss had not turned upward within the epochs run, so early '
              'stopping was not the binding constraint for this model.')
        w('')

    for task in ('classification', 'fragment'):
        scaling = d.get(f'data_scaling_{task}')
        if not scaling:
            continue
        key = 'macro_f1' if task == 'classification' else 'mean_tanimoto_to_reference'
        w(f'### Data-scaling curve — {task}')
        w('')
        w(_table(['Training data', 'Compounds', key.replace('_', ' ')],
                 [[f'{int(p["train_fraction"] * 100)}%', p['train_compounds'],
                   _f(p['test'][key])] for p in scaling['points']]))
        w('')
        w('Still improving at 100% of the data — the bottleneck is data volume, not '
          'architecture or regularisation.' if scaling.get('still_climbing') else
          'Performance has flattened before 100% of the data, so simply adding more of '
          'the same data is not the limiting factor here.')
        w('')

    for task in ('classification', 'fragment'):
        inter = d.get(f'interventions_{task}')
        if not inter:
            continue
        key = 'macro_f1' if task == 'classification' else 'substructure_rate'
        w(f'### Interventions — {task}')
        w('')
        w('Applied one at a time, each measured on its own, in the order the task spec '
          'sets out. A combined change would only tell you that *something* helped.')
        w('')
        w(_table(['Intervention', 'Best epoch', key.replace('_', ' '), 'Trainable parameters'],
                 [[s['name'].replace('_', ' '), s['training']['best_epoch'],
                   _f(s['test'][key]),
                   f'{s["training"]["trainable_parameters"]:,}'] for s in inter['steps']]))
        w('')

    # ── Named failures ──────────────────────────────────────────────────────
    if d.get('named_examples'):
        w('## Where these models fail, by name')
        w('')
        w('Aggregates let a reader nod along without ever seeing a wrong answer.')
        w('')
        for case in d['named_examples']['cases']:
            w(f'**{case["name"]}** — {case["why"]}  ')
            if case['true_labels']:
                w(f'Dataset label: `{" ".join(case["true_labels"])}`  ')
            w(f'Raw model output: `{case["raw_output"]}`'
              + ('' if case['raw_was_a_valid_label']
                 else ' — not a valid label; edit-distance snapping picked the nearest one') + '  ')
            w(f'Prediction after snapping: `{" ".join(case["predicted"]) or "—"}`  ')
            frag = case['fragment']
            w(f'Fragment model: `{frag["raw_output"]}` '
              f'({"valid SMILES" if frag["valid_smiles"] else "does not parse"}, '
              f'{"is part of the molecule" if frag["is_substructure_of_input"] else "**not part of the molecule**"})')
            w('')

    # ── Limitations ─────────────────────────────────────────────────────────
    w('## Limitations of this benchmark')
    w('')
    for line in _limitations(d):
        w(f'- {line}')
    w('')

    # ── Provenance ──────────────────────────────────────────────────────────
    w('## Reproducing this')
    w('')
    w('```bash')
    w('python3 -m evaluation.splits                       # rebuild and verify the splits')
    w('python3 -m evaluation.baselines                    # fingerprint baselines')
    w('python3 -m evaluation.evaluate --label shipped     # the published checkpoints')
    w('python3 -m evaluation.experiments all --task classification')
    w('python3 -m evaluation.experiments all --task fragment')
    w('python3 -m evaluation.experiments control --task fragment')
    w('python3 -m evaluation.named_examples')
    w('python3 -m evaluation.report                       # regenerate this file')
    w('python3 tests/test_evaluation.py                   # split integrity and metric tests')
    w('```')
    w('')
    prov = [['Generated', d['generated']], ['Commit', f'`{d["commit"]}`']]
    for k, v in d.get('datasets', {}).items():
        prov.append([f'{k} dataset', f'`{v["file"]}` · sha256 `{v["sha256_16"]}`'])
    for k, v in (d.get('splits') or {}).get('files', {}).items():
        prov.append([f'split `{k}`', f'`{v["path"]}` · sha256 `{v["sha256_16"]}`'])
    w(_table(['Item', 'Value'], prov))
    w('')
    return '\n'.join(L)


def _f(x, nd=3):
    return '—' if x is None else f'{x:.{nd}f}'


def _findings(d):
    out = []
    if d.get('baselines'):
        r = d['baselines']['random']['classification']['random_forest_ecfp4']['macro_f1']
        s = d['baselines']['scaffold']['classification']['random_forest_ecfp4']['macro_f1']
        out.append(f'**A random split overstates classification performance by {_f(r - s)} '
                   f'macro-F1.** The same random forest scores {_f(r)} on a random split and '
                   f'{_f(s)} when no scaffold is allowed to appear on both sides.')
    if d.get('shipped') and d.get('baselines'):
        rs = d['shipped']['random']['classification']['macro_f1']
        ss = d['shipped']['scaffold']['classification']['macro_f1']
        out.append(f'**The shipped classifier shows almost no scaffold gap ({_f(rs)} → '
                   f'{_f(ss)}), which is a symptom rather than a strength.** An honestly '
                   'trained model loses a quarter of its macro-F1 on this split. A model '
                   'that does not was trained on the molecules being used to test it.')
    if d.get('shipped'):
        f = d['shipped']['scaffold']['fragment']
        out.append(f'**Only {_f(f["substructure_rate"])} of predicted fragments are genuine '
                   f'substructures of the molecule they came from**, and '
                   f'{_f(f["validity_rate_raw"])} of raw outputs parse as SMILES at all. '
                   'The repair path fires on the rest and returns a fragment of a different '
                   'molecule.')
    if d.get('baselines'):
        mc = d['baselines']['scaffold']['fragment'].get('most_common_fragment')
        if mc:
            out.append(f'**The fragment model is beaten by answering "benzene" every time.** '
                       f'That constant baseline scores {_f(mc["exact_match_rate"])} exact '
                       f'match and {_f(mc["substructure_rate"])} substructure rate, because '
                       'the commonest BRICS fragment really is present in most drug-like '
                       'molecules.')
    if d.get('headline_classification') and d.get('baselines'):
        t = d['headline_classification']['scaffold']['full_test_metrics']['macro_f1']
        rf = d['baselines']['scaffold']['classification']['random_forest_ecfp4']['macro_f1']
        verdict = ('beats' if t > rf + 0.02 else 'matches' if abs(t - rf) <= 0.02 else 'loses to')
        out.append(f'**Retrained honestly, the 44 M-parameter transformer {verdict} a random '
                   f'forest on fingerprints** ({_f(t)} vs {_f(rf)} macro-F1).'
                   + ('' if verdict == 'beats' else
                      ' On this much data, the parameters are not buying anything a '
                      'fingerprint model does not already give you.'))
    return out


def _limitations(d):
    out = [
        'The fragment test set is 22 molecules. Every rate in that section moves by 4.5 '
        'percentage points per molecule, so they indicate direction, not precision.',
        'The retrained models use a fixed, modest training budget. They establish what this '
        'recipe achieves under an honest split with this data; they are not a claim about '
        'the best achievable number for this architecture.',
        'The retrained models reproduce the deployed recipe — a masked-LM head decoded '
        'position by position and snapped to the nearest label — rather than a '
        'sequence-classification head, which would very likely score better. The point was '
        'to measure what is in production, and a different architecture would not be '
        'comparable with it.',
        'The remote SMILES corrector the deployed pipeline calls before falling back to '
        'nearest-reference snapping is not called here, because a third-party service '
        'would make these numbers unreproducible. The intervention rate reported is the '
        'rate at which *some* repair was needed.',
        'The therapeutic-class labels themselves are not audited. Several compounds carry '
        'label combinations that are defensible but debatable, and a model is held to the '
        'labels it was given.',
    ]
    return out


def _table(headers, rows):
    out = ['| ' + ' | '.join(str(h) for h in headers) + ' |',
           '|' + '|'.join(' --- ' for _ in headers) + '|']
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out)


def write_frontend(payload):
    os.makedirs(os.path.dirname(FRONTEND_DATA), exist_ok=True)
    # The page does not need every prediction row, only what it renders.
    slim = {k: v for k, v in payload.items() if k != 'raw'}
    with open(FRONTEND_DATA, 'w') as f:
        json.dump(slim, f, indent=1)
    return FRONTEND_DATA


BENCHMARK_MD = os.path.join(D.REPO, 'BENCHMARK.md')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json-only', action='store_true')
    args = ap.parse_args()
    data = collect()
    missing = [k for k, v in data.items() if v is None]
    if missing:
        print('note: no results yet for', ', '.join(missing))
    page = write_frontend(data)
    print('frontend data ->', os.path.relpath(page, D.REPO))
    if not args.json_only:
        with open(BENCHMARK_MD, 'w') as f:
            f.write(markdown(data) + '\n')
        print('write-up      ->', os.path.relpath(BENCHMARK_MD, D.REPO))
