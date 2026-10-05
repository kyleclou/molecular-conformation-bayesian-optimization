import itertools
import unittest

import numpy as np
from rdkit.Chem import rdMolTransforms

from conformation_search.molecule import CANDIDATES, TORSIONS, PentaneEnergy
from conformation_search.search import expected_improvement, periodic_features, run_search


def internal_coordinates(mol):
    conf = mol.GetConformer()
    bonds = [rdMolTransforms.GetBondLength(conf, b.GetBeginAtomIdx(), b.GetEndAtomIdx())
             for b in mol.GetBonds()]
    angles = []
    for atom in mol.GetAtoms():
        for left, right in itertools.combinations([n.GetIdx() for n in atom.GetNeighbors()], 2):
            angles.append(rdMolTransforms.GetAngleDeg(conf, left, atom.GetIdx(), right))
    return np.array(bonds), np.array(angles)


class MoleculeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = PentaneEnergy()

    def test_rotations_preserve_all_bonds_and_angles(self):
        original = internal_coordinates(self.oracle.template)
        for angles in [(-180, -180), (0, 0), (60, -60), (170, -170), (-90, 30)]:
            mol = self.oracle.geometry(angles)
            for before, after in zip(original, internal_coordinates(mol)):
                np.testing.assert_allclose(after, before, atol=1e-9, rtol=0)
            for indices, target in zip(TORSIONS, angles):
                observed = rdMolTransforms.GetDihedralDeg(mol.GetConformer(), *indices)
                error = (observed - target + 180) % 360 - 180
                self.assertAlmostEqual(error, 0, places=8)

    def test_unselected_dihedrals_are_fixed(self):
        template = self.oracle.template
        original = template.GetConformer()
        rotated = self.oracle.geometry((70, -50)).GetConformer()
        for bond in template.GetBonds():
            j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if {j, k} in ({1, 2}, {2, 3}):
                continue
            left = [a.GetIdx() for a in template.GetAtomWithIdx(j).GetNeighbors() if a.GetIdx() != k]
            right = [a.GetIdx() for a in template.GetAtomWithIdx(k).GetNeighbors() if a.GetIdx() != j]
            for i, l in itertools.product(left, right):
                before = rdMolTransforms.GetDihedralDeg(original, i, j, k, l)
                after = rdMolTransforms.GetDihedralDeg(rotated, i, j, k, l)
                self.assertAlmostEqual((after-before+180) % 360 - 180, 0, places=8)

    def test_energy_periodicity_and_order_independence(self):
        first = self.oracle((-60, 80))
        self.oracle((0, 0))
        self.assertAlmostEqual(first, self.oracle((-60, 80)), places=10)
        self.assertAlmostEqual(first, self.oracle((300, -280)), places=8)

    def test_reproducible_template(self):
        np.testing.assert_array_equal(self.oracle.template.GetConformer().GetPositions(),
                                      PentaneEnergy().template.GetConformer().GetPositions())


class SearchTests(unittest.TestCase):
    def test_grid_and_periodic_features(self):
        self.assertEqual(CANDIDATES.shape, (1296, 2))
        self.assertEqual(len(np.unique(CANDIDATES, axis=0)), 1296)
        np.testing.assert_allclose(periodic_features([[-180, 20]]),
                                   periodic_features([[180, 380]]), atol=1e-15)

    def test_expected_improvement(self):
        ei = expected_improvement(np.array([0., -1., 1.]), np.array([1., 0., 0.]), 0.)
        np.testing.assert_allclose(ei, [1/np.sqrt(2*np.pi), 1., 0.])

    def test_budget_uniqueness_shared_initialization_and_reproducibility(self):
        # An oracle that only exposes requested values: no full energy array.
        candidates = CANDIDATES[::43]
        initial = [0, 5, 10]
        results = {}
        for method in ['random', 'bayesian']:
            calls = []
            def oracle(angles):
                calls.append(tuple(angles))
                return float(np.cos(np.deg2rad(angles)).sum())
            result = run_search(oracle, candidates, initial, 8, method, seed=19)
            self.assertEqual(len(calls), 8)
            self.assertEqual(len(set(calls)), 8)
            np.testing.assert_array_equal(result.indices[:3], initial)
            repeated = run_search(oracle, candidates, initial, 8, method, seed=19)
            np.testing.assert_array_equal(result.indices, repeated.indices)
            self.assertTrue(np.all(np.diff(result.elapsed_seconds) >= 0))
            results[method] = result
        np.testing.assert_array_equal(results['random'].energies[:3], results['bayesian'].energies[:3])

    def test_invalid_budget_and_initialization(self):
        for initial, budget in [([0, 0], 10), ([0, 1], 1), ([1296], 2), ([], 10)]:
            with self.assertRaises(ValueError):
                run_search(lambda _: 0., CANDIDATES, initial, budget, 'random', 0)


if __name__ == '__main__':
    unittest.main()
