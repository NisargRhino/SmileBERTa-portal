"""Tests for the honest-evaluation pipeline.

The benchmark's credibility rests on two claims that are easy to get wrong and
invisible when they are: that no scaffold straddles train and test, and that the
metrics mean what their names say. Both are tested here against hand-built cases
with known answers, so a regression shows up as a failing test rather than as a
number nobody can check.

    python3 tests/test_evaluation.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import data as D
from evaluation import metrics as M
from evaluation import splits as S
from evaluation.evaluate import _levenshtein, category_ranking, snap


def rec(smiles, labels=(), scaffold=None):
    return {'smiles': smiles, 'labels': list(labels),
            'scaffold': D.scaffold_of(smiles) if scaffold is None else scaffold}


class ScaffoldSplitTests(unittest.TestCase):
    def test_a_scaffold_never_straddles_train_and_test(self):
        records, _ = D.load_classification()
        assigned = S.scaffold_split(records)
        self.assertEqual(S.verify_scaffold_split(records, assigned), [])

    def test_committed_split_file_is_leak_free(self):
        # The file on disk is what every other script loads, so it is the one
        # that has to be clean — not just a split regenerated in memory.
        path = os.path.join(S.SPLIT_DIR, 'classification_scaffold.csv')
        if not os.path.exists(path):
            self.skipTest('run evaluation/splits.py first')
        mapping = S.load_split(path)
        records, _ = D.load_classification()
        by_split = {}
        for r in records:
            where = mapping.get(r['smiles'])
            if where and r['scaffold']:
                by_split.setdefault(r['scaffold'], set()).add(where)
        straddling = [s for s, w in by_split.items() if len(w) > 1]
        self.assertEqual(straddling, [], f'{len(straddling)} scaffolds in two splits')

    def test_random_split_does_leak_which_is_the_point(self):
        # The contrast only means something if the random split really does
        # share scaffolds across the boundary. If this ever stopped being true
        # the published gap would be measuring nothing.
        records, _ = D.load_classification()
        assigned = S.random_split(records)
        self.assertGreater(len(S.verify_scaffold_split(records, assigned)), 50)

    def test_acyclic_molecules_are_not_one_giant_group(self):
        records = [rec('CCO'), rec('CCCO'), rec('CCCCO'), rec('c1ccccc1C')]
        groups = S.group_by_scaffold(records)
        acyclic = [k for k in groups if k.startswith('__acyclic__')]
        self.assertEqual(len(acyclic), 3)
        self.assertTrue(all(len(groups[k]) == 1 for k in acyclic))

    def test_split_sizes_land_near_target(self):
        records, _ = D.load_classification()
        assigned = S.scaffold_split(records)
        report = S.split_report(records, assigned, 'scaffold')
        for name, target in zip(S.NAMES, S.RATIOS):
            self.assertAlmostEqual(report['fractions'][name], target, delta=0.02)

    def test_every_class_including_the_rarest_appears_in_every_split(self):
        # Dropping urological (48 compounds) would make macro-F1 look better.
        # The spec forbids it, so the split has to keep it present everywhere.
        records, _ = D.load_classification()
        assigned = S.scaffold_split(records)
        report = S.split_report(records, assigned, 'scaffold')
        for cat in D.CATEGORIES:
            counts = report['per_label'][cat]
            for name in S.NAMES:
                self.assertGreater(counts[name], 0, f'{cat} missing from {name}')

    def test_split_is_reproducible(self):
        records, _ = D.load_classification()
        self.assertEqual(S.scaffold_split(records), S.scaffold_split(records))
        self.assertEqual(S.random_split(records), S.random_split(records))


class ClassificationMetricTests(unittest.TestCase):
    def test_perfect_prediction(self):
        y = [{'cns'}, {'cardio'}, {'cns', 'cardio'}]
        out = M.classification_metrics(y, [set(s) for s in y])
        self.assertEqual(out['micro_f1'], 1.0)
        self.assertEqual(out['subset_accuracy'], 1.0)
        self.assertEqual(out['hamming_loss'], 0.0)

    def test_support_counts_are_the_true_counts(self):
        y_true = [{'cns'}, {'cns'}, {'urological'}]
        y_pred = [{'cns'}, {'cardio'}, {'cns'}]
        out = M.classification_metrics(y_true, y_pred)
        self.assertEqual(out['per_class']['cns']['support'], 2)
        self.assertEqual(out['per_class']['urological']['support'], 1)
        self.assertEqual(out['per_class']['urological']['recall'], 0.0)
        # cns: 1 true positive, 1 false negative, 1 false positive
        self.assertAlmostEqual(out['per_class']['cns']['precision'], 0.5)
        self.assertAlmostEqual(out['per_class']['cns']['recall'], 0.5)

    def test_macro_f1_punishes_ignoring_a_rare_class(self):
        # A model that only ever says the common class gets a respectable micro
        # score and a terrible macro score. That asymmetry is why the spec
        # insists on per-class reporting.
        y_true = [{'cns'}] * 90 + [{'urological'}] * 10
        y_pred = [{'cns'}] * 100
        out = M.classification_metrics(y_true, y_pred)
        # 90 of 100 predictions right, so micro-F1 is exactly 0.9 — a number
        # that reads as a good model. Macro-F1 sees one of twelve classes
        # answered at all.
        self.assertAlmostEqual(out['micro_f1'], 0.9)
        self.assertLess(out['macro_f1'], 0.1)

    def test_multi_label_subset_is_reported_separately(self):
        y_true = [{'cns'}, {'cns', 'cardio'}]
        y_pred = [{'cns'}, {'cns'}]
        out = M.classification_metrics(y_true, y_pred)
        self.assertEqual(out['subset_accuracy'], 0.5)
        self.assertEqual(out['multi_label_subset']['n'], 1)
        self.assertEqual(out['multi_label_subset']['subset_accuracy'], 0.0)
        self.assertEqual(out['multi_label_subset']['any_correct'], 1.0)

    def test_top_k_uses_the_ranking(self):
        y_true = [{'urological'}]
        y_pred = [{'cns'}]
        ranking = [['cns', 'cardio', 'urological'] + [c for c in D.CATEGORIES
                                                      if c not in ('cns', 'cardio', 'urological')]]
        out = M.classification_metrics(y_true, y_pred, ranking)
        self.assertEqual(out['top_1_accuracy'], 0.0)
        self.assertEqual(out['top_3_accuracy'], 1.0)

    def test_confusion_matrix_excludes_multi_label_compounds(self):
        y_true = [{'cns'}, {'cns', 'cardio'}]
        y_pred = [{'cardio'}, {'cns'}]
        cm = M.confusion_matrix(y_true, y_pred)
        self.assertEqual(cm['n'], 1)
        self.assertEqual(cm['matrix']['cns'], {'cardio': 1})


class SnappingTests(unittest.TestCase):
    """The deployed classifier snaps free-form output onto a known label."""

    def test_levenshtein(self):
        self.assertEqual(_levenshtein('cns', 'cns'), 0)
        self.assertEqual(_levenshtein('cardio', 'cardi'), 1)
        self.assertEqual(_levenshtein('', 'cns'), 3)

    def test_misspelled_output_snaps_to_the_real_label(self):
        # 'dermatittgit' is a real observed output from the shipped checkpoint.
        label, distance = snap('dermatittgit', D.label_options())
        self.assertEqual(label, 'dermatologic')
        self.assertGreater(distance, 0)

    def test_ranking_puts_the_closest_category_first(self):
        ranking = category_ranking('antiinfectiv', D.label_options())
        self.assertEqual(ranking[0], 'antiinfective')
        self.assertEqual(len(ranking), len(D.CATEGORIES))


class FragmentMetricTests(unittest.TestCase):
    def test_dummy_atoms_are_stripped_before_comparison(self):
        # BRICS references carry [n*] attachment points and the model never
        # emits them. Comparing raw would score zero by construction.
        self.assertEqual(M.normalise_fragment('[5*]N1CCN([5*])CC1'),
                         M.normalise_fragment('C1CNCCN1'))

    def test_substructure_detection(self):
        parent = 'CCN1CCN(Cc2ccccc2)CC1'
        self.assertTrue(D.is_substructure('[5*]N1CCN([5*])CC1', parent))
        self.assertTrue(D.is_substructure('c1ccccc1', parent))
        self.assertFalse(D.is_substructure('c1ccc2ccccc2c1', parent))   # naphthalene is not in it

    def test_hallucination_is_counted(self):
        records = [
            {'smiles': 'CCN1CCN(Cc2ccccc2)CC1', 'references': ['c1ccccc1'],
             'raw': 'c1ccccc1', 'prediction': 'c1ccccc1', 'correction': None},
            # A fragment from some other molecule entirely: the nearest-reference
            # fallback can produce exactly this.
            {'smiles': 'CCN1CCN(Cc2ccccc2)CC1', 'references': ['c1ccccc1'],
             'raw': 'not-a-smiles', 'prediction': 'c1ccc2ccccc2c1',
             'correction': 'nearest-reference'},
        ]
        out = M.fragment_metrics(records)
        self.assertEqual(out['substructure_rate'], 0.5)
        self.assertEqual(out['hallucination_rate'], 0.5)
        self.assertEqual(out['intervention_rate'], 0.5)
        self.assertEqual(out['validity_rate_raw'], 0.5)
        self.assertEqual(out['substructure_by_path']['nearest-reference']['rate'], 0.0)

    def test_exact_match_counts_any_of_the_molecules_fragments(self):
        # A molecule has several BRICS fragments; matching any of them is a hit.
        records = [{'smiles': 'CCN1CCN(Cc2ccccc2)CC1',
                    'references': ['[5*]N1CCN([5*])CC1', 'c1ccccc1'],
                    'raw': 'c1ccccc1', 'prediction': 'c1ccccc1', 'correction': None}]
        self.assertEqual(M.fragment_metrics(records)['exact_match_rate'], 1.0)

    def test_rule_of_three(self):
        self.assertTrue(M.passes_rule_of_three('c1ccccc1'))
        self.assertFalse(M.passes_rule_of_three(
            'CC(C)Cc1ccc(C(C)C(=O)OCCOCCOCCOCCOCCOCCO)cc1'))   # too heavy, too floppy

    def test_tanimoto_bounds(self):
        self.assertAlmostEqual(M.tanimoto('c1ccccc1', 'c1ccccc1'), 1.0)
        self.assertLess(M.tanimoto('c1ccccc1', 'CCCCCCCC'), 0.3)
        self.assertEqual(M.tanimoto('c1ccccc1', 'not-a-smiles'), 0.0)


class ExpandedCorpusTests(unittest.TestCase):
    """The expanded fragment corpus must not quietly undo the scaffold split."""

    @classmethod
    def setUpClass(cls):
        from evaluation import expand_fragments as X
        cls.X = X
        if not os.path.exists(X.OUT_CSV):
            raise unittest.SkipTest('run evaluation/expand_fragments.py first')

    def test_the_expanded_corpus_contains_the_original(self):
        original, _ = D.load_fragments()
        expanded = {r['smiles'] for r in self.X.load_expanded()}
        missing = [r['smiles'] for r in original if r['smiles'] not in expanded]
        self.assertEqual(missing, [], f'{len(missing)} original molecules lost')

    def test_test_set_is_unchanged(self):
        # The whole point of the comparison is that only the training data moved.
        parts, _ = self.X.expanded_split()
        mapping = S.load_split(os.path.join(S.SPLIT_DIR, 'fragment_scaffold.csv'))
        original_test = {s for s, where in mapping.items() if where == 'test'}
        self.assertEqual({r['smiles'] for r in parts['test']}, original_test)

    def test_no_training_molecule_shares_a_scaffold_with_the_test_set(self):
        parts, _ = self.X.expanded_split()
        held = {r['scaffold'] for r in parts['test'] + parts['validation'] if r['scaffold']}
        leaked = [r['smiles'] for r in parts['train'] if r['scaffold'] and r['scaffold'] in held]
        self.assertEqual(leaked, [], f'{len(leaked)} training molecules leak a held-out scaffold')

    def test_single_fragment_molecules_are_excluded(self):
        # A molecule BRICS cannot cleave yields one "fragment" that is the whole
        # molecule. Training on that teaches the model to echo its input.
        for rec in self.X.load_expanded():
            if rec['source'] == 'classification_dataset':
                self.assertGreaterEqual(len(rec['fragments']), self.X.MIN_FRAGMENTS)


class DataTests(unittest.TestCase):
    def test_classification_dataset_matches_the_spec(self):
        records, dropped = D.load_classification()
        self.assertEqual(len(records), 8336)
        self.assertEqual(sum(1 for r in records if len(r['labels']) > 1), 1350)
        counts = {c: sum(1 for r in records if c in r['labels']) for c in D.CATEGORIES}
        self.assertEqual(counts['antiinfective'], 2923)
        self.assertEqual(counts['urological'], 48)
        self.assertEqual(dropped, {})

    def test_fragment_dataset_groups_by_molecule(self):
        records, _ = D.load_fragments()
        self.assertEqual(len(records), 219)
        self.assertEqual(sum(len(r['fragments']) for r in records), 1223)
        self.assertTrue(all(len(set(r['fragments'])) == len(r['fragments']) for r in records))

    def test_smiles_are_canonical(self):
        records, _ = D.load_classification()
        sample = records[:200]
        self.assertTrue(all(r['smiles'] == D.canonical(r['smiles']) for r in sample))


if __name__ == '__main__':
    unittest.main(verbosity=2)
