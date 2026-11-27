"""Tests for the pharmacophore corpus pipeline.

The labels this pipeline produces are the entire value of the corpus, and three
of the four bugs found while building it were silent: hydrogens missing so no
hydrogen bond was ever detected, Open Babel renumbering Asp189 to Asp171, and a
polyethylene glycol molecule entering the corpus as a ligand. None of them
raised an error. They were only visible against complexes whose binding mode is
already known, which is what most of these tests are.

Network tests use the on-disk cache and skip when a structure is not available,
so the suite stays runnable offline and during an RCSB outage.

    python3 tests/test_pharmacophore.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem, RDLogger

from pharmacophore import complexes as C
from pharmacophore import interactions as I
from pharmacophore import protonate as P
from pharmacophore import sources as S

RDLogger.DisableLog('rdApp.*')


def cached(pdb_id):
    """The structure if it is already on disk, else skip — never a live fetch."""
    try:
        return S.fetch(f'{S.RCSB_FILES}/{pdb_id.upper()}.pdb', timeout=20)
    except Exception as e:
        raise unittest.SkipTest(f'{pdb_id} not cached and not reachable: {str(e)[:60]}')


class ArtefactFilterTests(unittest.TestCase):
    """Crystallisation additives must not enter the corpus as ligands."""

    def test_polyethylene_glycol_is_rejected(self):
        # 2PE, the exact molecule that got through the first pilot run.
        peg = Chem.MolFromSmiles('OCCOCCOCCOCCOCCOCCOCCOCCO')
        self.assertEqual(C.is_artefact('2PE', peg), 'known_artefact')

    def test_an_unlisted_acyclic_additive_is_still_rejected(self):
        # The point of the ring test: catching additives nobody listed.
        for smiles in ('OCC(O)CO',            # glycerol
                       'OCCO',                # ethylene glycol
                       'CC(C)(O)CC(C)(C)O'):  # MPD
            mol = Chem.MolFromSmiles(smiles)
            self.assertEqual(C.is_artefact('ZZZ', mol), 'acyclic_likely_additive', smiles)

    def test_real_ligands_survive(self):
        for name, smiles in (
                ('benzamidine', 'NC(=N)c1ccccc1'),
                ('donepezil', 'COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2'),
                ('imatinib', 'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1')):
            self.assertIsNone(C.is_artefact('XXX', Chem.MolFromSmiles(smiles)), name)

    def test_cofactors_are_excluded_from_the_ligand_role(self):
        self.assertIn('HEM', C.COFACTORS)
        self.assertIn('ATP', C.COFACTORS)


class ProtonationTests(unittest.TestCase):
    """Hydrogens must appear without the residue numbering moving."""

    def test_residue_numbering_survives(self):
        # Open Babel renumbers on output: this turned trypsin's Asp189 into
        # Asp171, which would have corrupted every residue reference downstream.
        pdb = cached('3PTB')
        before = {(l[17:20].strip(), l[22:26].strip())
                  for l in pdb.splitlines() if l.startswith('ATOM')}
        after_text, added = P.protonated_pdb(pdb)
        after = {(l[17:20].strip(), l[22:26].strip())
                 for l in after_text.splitlines()
                 if l.startswith('ATOM') and (l[76:78].strip() or 'X') != 'H'}
        self.assertGreater(added, 100)
        self.assertEqual(before, after)
        self.assertIn(('ASP', '189'), after)

    def test_hydrogens_sit_with_their_own_residue(self):
        # Writing every hydrogen at the end of the file splits residues across
        # segments, and ProLIF then refuses the structure outright.
        text, _ = P.protonated_pdb(cached('3PTB'))
        seen, previous = set(), None
        for line in text.splitlines():
            if not line.startswith('ATOM'):
                continue
            key = (line[21:22], line[22:27])
            if key != previous:
                self.assertNotIn(key, seen, f'residue {key} is not contiguous')
                seen.add(key)
                previous = key

    def test_ligand_ionises(self):
        # A neutral amidine in the deposited file is an amidinium at pH 7.4,
        # and without the charge its salt bridge is invisible.
        neutral = Chem.MolFromSmiles('NC(=N)c1ccccc1')
        self.assertEqual(Chem.GetFormalCharge(neutral), 0)
        self.assertEqual(Chem.GetFormalCharge(P.protonated_mol(neutral)), 1)


class InteractionTests(unittest.TestCase):
    """Labels are checked against binding modes that are textbook."""

    def _fingerprint(self, pdb_id):
        pdb = cached(pdb_id)
        candidates, _ = C.ligand_candidates(pdb)
        self.assertTrue(candidates, f'{pdb_id}: no ligand found')
        best = candidates[0]
        mol = C.ligand_mol(best, best['comp_id'])
        return I.fingerprint(mol, pdb), best

    def test_3ptb_finds_asp189(self):
        # Benzamidine's amidinium to Asp189 is the most cited interaction in
        # this complex and the platform already documents the structure.
        result, best = self._fingerprint('3PTB')
        self.assertEqual(best['comp_id'], 'BEN')
        residues = {c['residue'].split('.')[0] for c in result['contacts']}
        self.assertIn('ASP189', residues)

    def test_1hsg_finds_the_catalytic_aspartate(self):
        result, best = self._fingerprint('1HSG')
        self.assertEqual(best['comp_id'], 'MK1')
        residues = {c['residue'].split('.')[0] for c in result['contacts']}
        self.assertTrue({'ASP25', 'ASP29'} & residues, residues)

    def test_contacts_reference_real_ligand_atoms(self):
        result, _ = self._fingerprint('3PTB')
        for contact in result['contacts']:
            for idx in contact['ligand_atoms']:
                self.assertLess(idx, result['n_heavy'])
                self.assertGreaterEqual(idx, 0)

    def test_feature_vector_shape_and_sparsity(self):
        result, _ = self._fingerprint('3PTB')
        vector = I.feature_vector(result)
        self.assertEqual(vector.shape, (result['n_heavy'], len(I.INTERACTIONS)))
        # Most atoms interact with nothing; a metric that ignores this would be
        # reporting the base rate rather than the model.
        self.assertLess(vector.sum(), vector.size / 2)

    def test_pocket_profile_covers_every_interaction_type(self):
        result, _ = self._fingerprint('3PTB')
        profile = I.pocket_feature_profile(result)
        self.assertEqual(sorted(profile), sorted(I.INTERACTIONS))


class ProvenanceTests(unittest.TestCase):
    def test_every_source_declares_a_licence_and_citation(self):
        for entry in S.credits_block():
            for field in ('name', 'url', 'licence', 'citation', 'used_for', 'redistribution'):
                self.assertTrue(entry.get(field), f'{entry["id"]} is missing {field}')

    def test_share_alike_is_flagged(self):
        # ChEMBL's CC BY-SA propagates to anything derived from it, which
        # constrains how a release may be published.
        chembl = S.SOURCES['chembl']
        self.assertIn('share-alike', chembl['redistribution'])


class ResolutionGateTests(unittest.TestCase):
    """The quality gate silently passed everything for a whole build.

    `REMARK   2 RESOLUTION.    2.00 ANGSTROMS.` splits into
    ['REMARK', '2', 'RESOLUTION.', '2.00', 'ANGSTROMS.'] and the first token
    that parses as a float is the remark number. Every entry came out at
    exactly 2.0 A, so nothing was ever above the 2.5 A limit. The corpus
    survived only because the candidate query filters on resolution at the RCSB
    end; the gate itself was doing nothing.
    """

    @staticmethod
    def _header(remark):
        return ('EXPDTA    X-RAY DIFFRACTION\n' + remark +
                '\nATOM      1  N   ALA A   1      0.000   0.000   0.000')

    def test_the_remark_number_is_not_mistaken_for_the_resolution(self):
        text = self._header('REMARK   2 RESOLUTION.    1.08 ANGSTROMS.')
        self.assertAlmostEqual(C.entry_is_usable(text), 1.08)

    def test_a_low_resolution_entry_is_rejected(self):
        text = self._header('REMARK   2 RESOLUTION.    2.60 ANGSTROMS.')
        with self.assertRaises(C.Rejected) as caught:
            C.entry_is_usable(text)
        message = caught.exception.args[0]
        self.assertTrue(message.startswith('resolution_too_low'), message)
        # The real figure must reach the rejection detail, or the manifest
        # cannot say how far outside the limit the entry was.
        self.assertIn('2.6', message)

    def test_a_borderline_entry_is_kept(self):
        text = self._header('REMARK   2 RESOLUTION.    2.50 ANGSTROMS.')
        self.assertAlmostEqual(C.entry_is_usable(text), 2.50)

    def test_nmr_has_no_resolution(self):
        text = self._header('REMARK   2 RESOLUTION. NOT APPLICABLE.')
        with self.assertRaises(C.Rejected) as caught:
            C.entry_is_usable(text)
        self.assertEqual(caught.exception.args[0], 'no_resolution')


class PromiscuousSubsetTests(unittest.TestCase):
    """The ablation is only sensitive where the pocket is in contention."""

    def test_atom_mask_follows_flatten_order(self):
        from pharmacophore import dataset as D
        examples = [
            {'inchikey': 'AAA', 'n_heavy': 3},     # promiscuous
            {'inchikey': 'BBB', 'n_heavy': 2},     # single-protein
            {'inchikey': 'AAA', 'n_heavy': 1},
        ]
        mask = D.atom_mask(examples, {'AAA'})
        self.assertEqual(list(mask), [True, True, True, False, False, True])
        self.assertEqual(int(mask.sum()), 4)

    def test_a_ligand_in_one_protein_is_not_promiscuous(self):
        from pharmacophore import dataset as D
        examples = [{'inchikey': 'BBB', 'n_heavy': 2}]
        self.assertEqual(int(D.atom_mask(examples, {'AAA'}).sum()), 0)


class FigureHonestyTests(unittest.TestCase):
    """A figure must not put results from two different splits on one axis."""

    def test_a_stale_model_regime_is_dropped(self):
        from pharmacophore import figures as F
        model = {'unseen_ligand': {'counts': {'test': 12}},
                 'both_unseen': {'counts': {'test': 77}}}
        baselines = {'unseen_ligand': {'_split': {'test_complexes': 1745}},
                     'both_unseen': {'_split': {'test_complexes': 77}}}
        kept = F._drop_stale_regimes(model, baselines, log=lambda *a: None)
        self.assertNotIn('unseen_ligand', kept)
        self.assertIn('both_unseen', kept)


if __name__ == '__main__':
    unittest.main(verbosity=2)
