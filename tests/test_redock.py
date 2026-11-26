"""Tests for redocking validation (redock.py).

Two tiers:

* Unit tests — no network, no Vina. Run them with `python3 tests/test_redock.py`
  or `pytest tests/test_redock.py`.
* Integration tests — fetch real structures from the RCSB PDB and run AutoDock
  Vina end to end. They are skipped unless `REDOCK_INTEGRATION=1` is set and a
  `vina` binary and Open Babel are importable, because they need both and take
  minutes:

      REDOCK_INTEGRATION=1 python3 tests/test_redock.py
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

import redock

INTEGRATION = os.environ.get('REDOCK_INTEGRATION') == '1'


def _rotate(mol, angle_deg, axis, about):
    """A copy of `mol` with its conformer rotated about an axis through `about`."""
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    t = math.radians(angle_deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)
    for i in range(out.GetNumAtoms()):
        p = np.array(list(conf.GetAtomPosition(i))) - about
        q = R @ p + about
        conf.SetAtomPosition(i, q.tolist())
    return out


def _translate(mol, delta):
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i in range(out.GetNumAtoms()):
        p = np.array(list(conf.GetAtomPosition(i))) + np.asarray(delta, dtype=float)
        conf.SetAtomPosition(i, p.tolist())
    return out


def _naive_rmsd(a, b):
    """Atom-by-atom RMSD in index order — what you get with no symmetry handling."""
    ca, cb = a.GetConformer(), b.GetConformer()
    d = [np.array(list(ca.GetAtomPosition(i))) - np.array(list(cb.GetAtomPosition(i)))
         for i in range(a.GetNumAtoms())]
    return float(np.sqrt(np.mean([float(v @ v) for v in d])))


def _embed(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    assert AllChem.EmbedMolecule(mol, params) == 0
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    return Chem.RemoveHs(mol)


def _pdb_line(serial, name, resn, chain, resi, xyz, element, record='HETATM'):
    return (f'{record:<6}{serial:>5} {name:<4}{"":1}{resn:>3} {chain}{resi:>4}{"":4}'
            f'{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2}')


def _structure(ligand_atoms, protein_atoms=(), resn='LIG', chain='A', resi=1):
    """A minimal PDB: a few protein ATOMs plus one HETATM residue."""
    lines, serial = [], 1
    for i, xyz in enumerate(protein_atoms):
        lines.append(_pdb_line(serial, 'CA', 'ALA', 'A', 100 + i, xyz, 'C', record='ATOM'))
        serial += 1
    for i, xyz in enumerate(ligand_atoms):
        lines.append(_pdb_line(serial, f'C{i + 1}', resn, chain, resi, xyz, 'C'))
        serial += 1
    return '\n'.join(lines) + '\nEND\n'


class RmsdTests(unittest.TestCase):
    """The RMSD function is the whole feature: if it lies, the verdict lies."""

    def test_identical_poses_are_zero(self):
        mol = _embed('CCO')
        self.assertAlmostEqual(redock.pose_rmsd(mol, Chem.Mol(mol)), 0.0, places=6)

    def test_known_offset_is_not_superposed(self):
        # CalcRMS, unlike GetBestRMS, must not align the two molecules first: a
        # pose 3 Å out of the pocket is 3 Å wrong, not a perfect match.
        mol = _embed('CCO')
        moved = _translate(mol, (3.0, 0.0, 0.0))
        self.assertAlmostEqual(redock.pose_rmsd(moved, mol), 3.0, places=3)
        self.assertLess(rdMolAlign.GetBestRMS(redock._skeleton(moved), redock._skeleton(mol)), 1e-3)

    def test_benzene_ring_flip_is_zero(self):
        # Rotating benzene 60° about its own normal maps the ring onto itself.
        # Chemically it is the same pose, so symmetry correction must report ~0
        # where an atom-by-atom comparison reports well over an ångström.
        mol = _embed('c1ccccc1')
        conf = np.array(mol.GetConformer().GetPositions())
        centroid = conf.mean(axis=0)
        # Ring normal: the direction of least variance in the (planar) ring.
        normal = np.linalg.svd(conf - centroid)[2][-1]
        flipped = _rotate(mol, 60.0, normal, centroid)
        self.assertGreater(_naive_rmsd(flipped, mol), 1.0)
        self.assertLess(redock.pose_rmsd(flipped, mol), 0.1)

    def test_para_substituted_ring_flip_is_zero(self):
        # The 180° flip of a para-disubstituted ring is the classic case people
        # mean by "a flipped phenyl is not an error".
        mol = _embed('Cc1ccc(O)cc1')
        ring = np.array([list(mol.GetConformer().GetAtomPosition(a.GetIdx()))
                         for a in mol.GetAtoms() if a.GetIsAromatic()])
        centroid = ring.mean(axis=0)
        axis = ring[0] - centroid   # roughly the C1–C4 axis
        flipped = _rotate(mol, 180.0, axis, centroid)
        self.assertLess(redock.pose_rmsd(flipped, mol), 0.4)

    def test_mismatched_molecules_are_refused(self):
        with self.assertRaises(redock.RedockError):
            redock.pose_rmsd(_embed('CCO'), _embed('CCCO'))


class VerdictTests(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(redock.verdict_for(0.8), 'reliable')
        self.assertEqual(redock.verdict_for(2.0), 'reliable')     # boundary is inclusive
        self.assertEqual(redock.verdict_for(2.01), 'marginal')
        self.assertEqual(redock.verdict_for(3.0), 'marginal')
        self.assertEqual(redock.verdict_for(3.01), 'unreliable')
        self.assertEqual(redock.verdict_for(11.4), 'unreliable')

    def test_thresholds_are_the_published_convention(self):
        self.assertEqual((redock.RELIABLE_MAX, redock.MARGINAL_MAX), (2.0, 3.0))

    def test_every_verdict_has_copy(self):
        for v in ('reliable', 'marginal', 'unreliable'):
            self.assertIn(v, redock.VERDICTS)
            self.assertTrue(redock.VERDICTS[v].strip())


class LigandChoiceTests(unittest.TestCase):
    def test_waters_and_ions_offer_nothing(self):
        pdb = ('\n'.join(
            [_pdb_line(i, 'O', 'HOH', 'A', 200 + i, (i * 3.0, 0, 0), 'O') for i in range(1, 12)]
            + [_pdb_line(50, 'ZN', 'ZN', 'A', 300, (5, 5, 5), 'ZN'),
               _pdb_line(51, 'S', 'SO4', 'A', 301, (9, 9, 9), 'S')]) + '\nEND\n')
        candidates, _ = redock.ligand_candidates(pdb)
        self.assertEqual(candidates, [])
        with self.assertRaises(redock.RedockError) as cm:
            redock.choose_ligand(pdb)
        self.assertEqual(str(cm.exception), redock.NO_LIGAND)

    def test_small_het_residues_are_not_ligands(self):
        # Seven heavy atoms: below the 8-atom floor, so not a ligand.
        pdb = _structure([(i * 1.5, 0, 0) for i in range(7)], resn='XYZ')
        self.assertEqual(redock.ligand_candidates(pdb)[0], [])

    def test_largest_candidate_wins(self):
        small = [_pdb_line(i, f'C{i}', 'SML', 'A', 1, (i * 1.5, 0, 0), 'C') for i in range(1, 10)]
        big = [_pdb_line(20 + i, f'C{i}', 'BIG', 'B', 2, (i * 1.5, 20, 0), 'C') for i in range(1, 16)]
        candidates, _ = redock.ligand_candidates('\n'.join(small + big) + '\nEND\n')
        self.assertEqual([c['comp_id'] for c in candidates], ['BIG', 'SML'])
        chosen, _parsed, note = redock.choose_ligand('\n'.join(small + big) + '\nEND\n')
        self.assertEqual((chosen['comp_id'], chosen['heavy_atoms']), ('BIG', 15))
        self.assertIn('SML', note)

    def test_covalent_ligand_is_refused_not_quietly_docked(self):
        lig = [(0, 0, 0), (1.5, 0, 0), (3.0, 0, 0), (4.5, 0, 0),
               (6.0, 0, 0), (7.5, 0, 0), (9.0, 0, 0), (10.5, 0, 0)]
        # A protein atom 1.8 Å away is a bond, not a contact.
        bonded = _structure(lig, protein_atoms=[(-1.8, 0, 0)])
        candidates, _ = redock.ligand_candidates(bonded)
        self.assertTrue(candidates[0]['covalent_contacts'])
        with self.assertRaises(redock.RedockError) as cm:
            redock.choose_ligand(bonded)
        self.assertIn('covalently bound', str(cm.exception))

    def test_a_close_contact_is_not_a_bond(self):
        lig = [(i * 1.5, 0, 0) for i in range(8)]
        near = _structure(lig, protein_atoms=[(-3.4, 0, 0)])
        candidates, _ = redock.ligand_candidates(near)
        self.assertFalse(candidates[0]['covalent_contacts'])
        chosen, _parsed, _note = redock.choose_ligand(near)
        self.assertEqual(chosen['comp_id'], 'LIG')

    def test_repeated_ligand_notes_the_chain_used(self):
        a = [_pdb_line(i, f'C{i}', 'LIG', 'A', 1, (i * 1.5, 0, 0), 'C') for i in range(1, 10)]
        b = [_pdb_line(20 + i, f'C{i}', 'LIG', 'B', 1, (i * 1.5, 30, 0), 'C') for i in range(1, 10)]
        chosen, _parsed, note = redock.choose_ligand('\n'.join(a + b) + '\nEND\n')
        self.assertEqual(chosen['chain'], 'A')
        self.assertIn('2 chains', note)

    def test_linked_residues_are_one_ligand(self):
        # Two five-atom residues bonded to each other are one ten-atom ligand,
        # not two residues below the floor.
        a = [_pdb_line(i, f'C{i}', 'AAA', 'C', 1, (i * 1.5, 0, 0), 'C') for i in range(1, 6)]
        b = [_pdb_line(10 + i, f'C{i}', 'BBB', 'C', 2, (7.5 + i * 1.5, 0, 0), 'C') for i in range(1, 6)]
        candidates, _ = redock.ligand_candidates('\n'.join(a + b) + '\nEND\n')
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['heavy_atoms'], 10)
        self.assertEqual(len(candidates[0]['components']), 2)

    def test_multi_component_ligand_is_refused(self):
        # A ligand deposited as several linked components cannot be rebuilt from
        # one chemical-component definition. 6LU7's inhibitor is the real case:
        # validating whichever fragment happens not to touch the protein would be
        # worse than refusing.
        a = [_pdb_line(i, f'C{i}', 'AAA', 'C', 1, (i * 1.5, 0, 0), 'C') for i in range(1, 6)]
        b = [_pdb_line(10 + i, f'C{i}', 'BBB', 'C', 2, (7.5 + i * 1.5, 0, 0), 'C') for i in range(1, 6)]
        with self.assertRaises(redock.RedockError) as cm:
            redock.choose_ligand('\n'.join(a + b) + '\nEND\n')
        self.assertIn('linked components', str(cm.exception))

    def test_covalency_is_judged_on_the_whole_entity(self):
        # One fragment of a linked ligand touching the protein makes the whole
        # ligand covalent — this is exactly how 6LU7 slipped through before.
        a = [_pdb_line(i, f'C{i}', 'AAA', 'C', 1, (i * 1.5, 0, 0), 'C') for i in range(1, 6)]
        b = [_pdb_line(10 + i, f'C{i}', 'BBB', 'C', 2, (7.5 + i * 1.5, 0, 0), 'C') for i in range(1, 6)]
        prot = [_pdb_line(90, 'CA', 'CYS', 'A', 145, (1.5 - 1.8, 0, 0), 'S', record='ATOM')]
        candidates, _ = redock.ligand_candidates('\n'.join(prot + a + b) + '\nEND\n')
        self.assertTrue(candidates[0]['covalent_contacts'])
        with self.assertRaises(redock.RedockError) as cm:
            redock.choose_ligand('\n'.join(prot + a + b) + '\nEND\n')
        self.assertIn('covalently bound', str(cm.exception))

    def test_single_component_ligand_wins_over_a_bigger_linked_one(self):
        a = [_pdb_line(i, f'C{i}', 'AAA', 'C', 1, (i * 1.5, 0, 0), 'C') for i in range(1, 9)]
        b = [_pdb_line(20 + i, f'C{i}', 'BBB', 'C', 2, (10.5 + i * 1.5, 0, 0), 'C') for i in range(1, 9)]
        solo = [_pdb_line(50 + i, f'C{i}', 'SOL', 'A', 9, (i * 1.5, 40, 0), 'C') for i in range(1, 10)]
        chosen, _parsed, note = redock.choose_ligand('\n'.join(a + b + solo) + '\nEND\n')
        self.assertEqual(chosen['comp_id'], 'SOL')
        self.assertIn('multi-component', note)

    def test_hydrogens_do_not_count_as_heavy_atoms(self):
        lig = [_pdb_line(i, f'C{i}', 'LIG', 'A', 1, (i * 1.5, 0, 0), 'C') for i in range(1, 8)]
        lig += [_pdb_line(50 + i, f'H{i}', 'LIG', 'A', 1, (i * 1.5, 1.0, 0), 'H') for i in range(1, 6)]
        self.assertEqual(redock.ligand_candidates('\n'.join(lig) + '\nEND\n')[0], [])


class BoxTests(unittest.TestCase):
    def test_centre_is_the_ligand_centroid(self):
        atoms = [{'xyz': (0.0, 0.0, 0.0)}, {'xyz': (4.0, 2.0, -2.0)}]
        center, _size = redock.box_for(atoms)
        self.assertEqual(center, [2.0, 1.0, -1.0])

    def test_small_ligand_gets_the_minimum_box(self):
        atoms = [{'xyz': (0.0, 0.0, 0.0)}, {'xyz': (2.0, 2.0, 2.0)}]
        _center, size = redock.box_for(atoms)
        self.assertEqual(size, [redock.BOX_MIN] * 3)

    def test_large_ligand_is_clamped(self):
        atoms = [{'xyz': (0.0, 0.0, 0.0)}, {'xyz': (40.0, 40.0, 40.0)}]
        _center, size = redock.box_for(atoms)
        self.assertEqual(size, [redock.BOX_MAX] * 3)

    def test_padding_is_added_to_the_extent(self):
        atoms = [{'xyz': (0.0, 0.0, 0.0)}, {'xyz': (14.0, 0.0, 0.0)}]
        _center, size = redock.box_for(atoms)
        self.assertEqual(size[0], 14.0 + redock.BOX_PADDING)

    def test_every_atom_keeps_its_margin_when_the_ligand_is_lopsided(self):
        # The centroid of an uneven ligand is not the middle of its bounding
        # box, so sizing from the bounding box would leave one face of the box
        # almost touching the ligand.
        atoms = [{'xyz': (x, 0.0, 0.0)} for x in (0.0, 1.0, 2.0, 3.0, 9.0)]
        center, size = redock.box_for(atoms)
        lo, hi = center[0] - size[0] / 2, center[0] + size[0] / 2
        self.assertLessEqual(lo, 0.0 - redock.BOX_PADDING / 2 + 1e-9)
        self.assertGreaterEqual(hi, 9.0 + redock.BOX_PADDING / 2 - 1e-9)
        self.assertTrue(redock.box_contains(atoms, center, size))

    def test_a_ligand_too_long_for_the_clamped_box_is_detected(self):
        # The 26 Å ceiling can cut off a very long, lopsided ligand. Reporting
        # an RMSD it was never possible to achieve would be worse than refusing.
        atoms = [{'xyz': (x, 0.0, 0.0)} for x in (0.0, 1.0, 2.0, 3.0, 40.0)]
        center, size = redock.box_for(atoms)
        self.assertEqual(size[0], redock.BOX_MAX)
        self.assertFalse(redock.box_contains(atoms, center, size))


class CacheKeyTests(unittest.TestCase):
    def test_settings_and_engine_are_part_of_the_key(self):
        base = redock.cache_key('3ptb', 'ben', 4, 9, 'AutoDock Vina v1.2.7')
        self.assertEqual(base, redock.cache_key('3PTB', 'BEN', 4, 9, 'AutoDock Vina v1.2.7'))
        self.assertNotEqual(base, redock.cache_key('3PTB', 'BEN', 8, 9, 'AutoDock Vina v1.2.7'))
        self.assertNotEqual(base, redock.cache_key('3PTB', 'BEN', 4, 20, 'AutoDock Vina v1.2.7'))
        self.assertNotEqual(base, redock.cache_key('3PTB', 'BEN', 4, 9, 'AutoDock Vina v1.1.2'))

    def test_memory_cache_roundtrip(self):
        key = redock.cache_key('TEST', 'LIG', 1, 1, 'v0')
        redock.cache_put(key, {'verdict': 'reliable'})
        hit = redock.cache_get(key)
        self.assertEqual(hit['verdict'], 'reliable')
        self.assertTrue(hit['computed_at'])


class ScoringTests(unittest.TestCase):
    """score() turns poses into the headline numbers, and must not flatter them."""

    def setUp(self):
        self.crystal = _embed('CCO')
        self.crystal_sdf = Chem.MolToMolBlock(self.crystal)

    def _pose(self, mode, offset, affinity):
        return {'mode': mode, 'affinity': affinity,
                'sdf': Chem.MolToMolBlock(_translate(self.crystal, (offset, 0, 0)))}

    def test_headline_is_the_top_pose_not_the_best_of_n(self):
        poses = [self._pose(1, 4.0, -6.1), self._pose(2, 2.5, -5.9), self._pose(3, 0.4, -5.5)]
        out = redock.score(self.crystal_sdf, poses)
        self.assertEqual(out['top_pose']['rank'], 1)
        self.assertAlmostEqual(out['top_pose']['rmsd'], 4.0, places=1)
        self.assertEqual(out['best_pose']['rank'], 3)
        self.assertAlmostEqual(out['best_pose']['rmsd'], 0.4, places=1)
        # The verdict follows the top pose — the pose a user would act on.
        self.assertEqual(out['verdict'], 'unreliable')

    def test_a_good_run_reports_reliable(self):
        out = redock.score(self.crystal_sdf, [self._pose(1, 0.9, -6.4), self._pose(2, 3.0, -5.1)])
        self.assertEqual(out['verdict'], 'reliable')
        self.assertEqual(out['message'], redock.VERDICTS['reliable'])
        self.assertEqual(len(out['poses']), 2)

    def test_unreadable_poses_are_reported_not_hidden(self):
        poses = [self._pose(1, 1.0, -6.0), {'mode': 2, 'affinity': -5.0, 'sdf': None}]
        out = redock.score(self.crystal_sdf, poses)
        self.assertEqual(out['unscored_poses'], [2])
        self.assertEqual(out['top_pose']['rank'], 1)

    def test_no_comparable_poses_is_an_error_not_a_verdict(self):
        out = redock.score(self.crystal_sdf, [{'mode': 1, 'affinity': -5.0, 'sdf': None}])
        self.assertIn('error', out)
        self.assertNotIn('verdict', out)


@unittest.skipUnless(INTEGRATION, 'set REDOCK_INTEGRATION=1 (needs network, Open Babel and Vina)')
class IntegrationTests(unittest.TestCase):
    """End-to-end runs against real structures. Minutes, not seconds."""

    @classmethod
    def setUpClass(cls):
        import docking
        if not docking.DOCKING_AVAILABLE:
            raise unittest.SkipTest('Vina or Open Babel is not available on this host')

    def _validate(self, pdb_id, exhaustiveness=8, poses=9):
        job_id, _cached = redock.start(pdb_id, exhaustiveness, poses, user_id='test')
        import time
        for _ in range(400):
            st = redock.status(job_id)
            if st['status'] in ('done', 'error'):
                return st
            time.sleep(2)
        self.fail(f'{pdb_id} validation did not finish')

    def test_prepare_rebuilds_the_ligand_independently(self):
        # The test is only real if the conformer handed to Vina is not the crystal
        # pose. ETKDG builds it in its own frame, so this is a large number — the
        # assertion is that it is never ~0.
        prep = redock.prepare('3PTB')
        self.assertEqual(prep['ligand']['comp_id'], 'BEN')
        self.assertGreater(prep['start_rmsd'], redock.MIN_START_RMSD)

    def test_3ptb_trypsin_benzamidine_is_reliable(self):
        st = self._validate('3PTB')
        self.assertEqual(st['status'], 'done', st.get('error'))
        self.assertEqual(st['verdict'], 'reliable',
                         f"top-pose RMSD was {st.get('top_pose', {}).get('rmsd')} Å")
        self.assertIsNotNone(st['docked_sdf'])
        self.assertIsNotNone(st['crystal_sdf'])

    def test_1hsg_hiv_protease_indinavir_validates(self):
        """1HSG is the larger-ligand check: indinavir, 45 heavy atoms, 12 torsions.

        It is deliberately NOT asserted to come back "reliable". Measured here
        with Vina 1.1.2 it does not: top-pose RMSD 9.9 Å at exhaustiveness 8 and
        6.2 Å at 16, with no pose closer than 4.4 Å. That is the feature working
        — HIV-1 protease is one of the pockets whose flap water (HOH 301) is
        stripped by receptor preparation, and indinavir's flexibility needs far
        more sampling than the page's defaults give it. Making this assertion
        pass by raising exhaustiveness until it did, or by widening the bands,
        is exactly what this feature exists to refuse.

        What must hold is that the run is real and self-consistent.
        """
        st = self._validate('1HSG')
        self.assertEqual(st['status'], 'done', st.get('error'))
        self.assertEqual(st['ligand']['comp_id'], 'MK1')
        self.assertIn(st['verdict'], ('reliable', 'marginal', 'unreliable'))
        self.assertEqual(st['verdict'], redock.verdict_for(st['top_pose']['rmsd']))
        self.assertLessEqual(st['best_pose']['rmsd'], st['top_pose']['rmsd'])
        self.assertIsNotNone(st['docked_sdf'])
        self.assertGreater(st['start_rmsd'], redock.MIN_START_RMSD)

    def test_second_run_is_served_from_cache(self):
        self._validate('3PTB')
        _job_id, cached = redock.start('3PTB', 8, 9, user_id='test')
        self.assertTrue(cached)

    def test_structure_without_a_ligand_is_refused(self):
        # 1UBQ (ubiquitin) was solved with nothing but waters bound.
        with self.assertRaises(redock.RedockError) as cm:
            redock.prepare('1UBQ')
        self.assertEqual(str(cm.exception), redock.NO_LIGAND)

    def test_covalent_ligand_structure_is_refused(self):
        # 6LU7: the SARS-CoV-2 Mpro inhibitor N3 is covalently bound to Cys145.
        with self.assertRaises(redock.RedockError) as cm:
            redock.prepare('6LU7')
        self.assertIn('covalently bound', str(cm.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
