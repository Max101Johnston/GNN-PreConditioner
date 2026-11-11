"""Checks for overlapping modal transfers and the recorded performance gate."""
import unittest

import numpy as np

from gnn_amg.modal_basis import assemble, make_patch, patch_ranks, rigid_fields
from gnn_amg.modal_gate import summarize_gate


class ModalBasisTest(unittest.TestCase):
    def test_overlap_preserves_field_and_rank_budget(self):
        rigid = np.ones((5, 1))
        # An independent small global direction remains eligible after
        # normalization even though its original amplitude is much smaller.
        directions = np.array([[1., 0.], [-1., 0.], [0., 1e-7], [0., -1e-7], [0., 0.]])
        patches = [make_patch(np.array([0, 1, 2]), np.array([1., 1., .5]),
                              np.zeros(3), 0, rigid, directions, 2),
                   make_patch(np.array([2, 3, 4]), np.array([.5, 1., 1.]),
                              np.ones(3), 1, rigid, directions, 2)]
        ranks = patch_ranks(patches, 4)
        self.assertEqual(sum(ranks), 4)
        transfer, _, region, ordinal, coefficients = assemble(patches, ranks, rigid, (5, 4), 0)
        np.testing.assert_allclose(transfer @ coefficients, rigid, atol=1e-14)
        self.assertEqual(len(region), len(ordinal))
        self.assertEqual(np.linalg.matrix_rank(transfer.toarray()), 4)

    def test_constrained_rigid_fields(self):
        coords = np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])
        result = rigid_fields(coords, np.array([0, 1, 2]), np.array([False, True, False]),
                              np.zeros(3))
        np.testing.assert_array_equal(result[0], [1., 0., 0., 0., 3., -2.])
        np.testing.assert_array_equal(result[1], np.zeros(6))
        np.testing.assert_array_equal(result[2], [0., 0., 1., 2., -1., 0.])


class ModalGateTest(unittest.TestCase):
    def test_each_round_must_win_in_true_residual_and_wall_time(self):
        data = [dict(method=m, key="a", repeat=r, iterations=s, total_s=t,
                     passed=True, true_limit=1e-7)
                for r in (1, 2) for m, s, t in
                (("factorized", 100, 10.), ("student", 110, 9.), ("modal-oracle", 83, 7.))]
        cases = [dict(basis={"budget":{"passed":True}}, solves=data)]
        settings = dict(steps_ratio=.85, wall_ratio=.85, single_step_ratio=1.05)
        self.assertTrue(summarize_gate(cases, ["a"], settings)["passed"])
        data[-1]["total_s"] = 7.9
        self.assertFalse(summarize_gate(cases, ["a"], settings)["passed"])
