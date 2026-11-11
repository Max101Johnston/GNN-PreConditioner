"""Regression checks for signed support, block context and recursive coarse solves."""
import unittest
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from gnn_amg.direct_bank import block_context, edge_examples
from gnn_amg.direct_depth import DepthCycle, third_endpoints


class DirectBankTest(unittest.TestCase):
    def test_blocks_preserve_signed_cross_component_entries(self):
        a = sparse.csr_matrix([[2., -1., 0., -.5], [.25, 2., .5, 0.],
                               [0., -1., 2., 0.], [.5, 0., 0., 1.]])
        native = np.array([3, 4, 5, 6])
        features = block_context(a, native, native).reshape(4, 3, 3)
        np.testing.assert_array_equal(features[0], features[2])
        self.assertAlmostEqual(float(features[0, 0, 1]), -1/3.5)
        self.assertAlmostEqual(float(features[3, 0, 0]), 1/1.5, places=6)

    def test_sample_does_not_truncate_any_signed_teacher_entry(self):
        matrix = sparse.csr_matrix([[.6, -.2, .4, .2, 0.], [0., 0., 1., 0., 0.]])
        ids = np.array([0, 2, 4, 6, 8])
        coordinates = np.arange(9, dtype=float).reshape(3, 3)
        starts, cols, target = edge_examples(matrix, np.arange(2), np.array([False, True]),
                                              np.array([0, 1]), ids, coordinates, 1,
                                              np.random.default_rng(11))
        self.assertEqual(starts[1], starts[2])
        np.testing.assert_array_equal(cols[target != 0], np.arange(4))
        np.testing.assert_allclose(target[target != 0], [.6, -.2, .4, .2])


class DirectDepthTest(unittest.TestCase):
    def test_same_terminal_policy_across_three_recursions(self):
        transfer = [np.array([[1., 0.], [0., 1.], [.5, .5]]),
                    np.array([[1.], [1.]]), np.array([[1.]])]
        parent = np.diag([2., 3., 4.])
        levels, coarse = [], []
        for mat in transfer:
            parent = mat.T @ parent @ mat
            coarse.append(parent)
            levels.append(SimpleNamespace(
                P=lambda x, transpose=False, m=mat: (m.T if transpose else m) @ x,
                K=lambda x, k=parent: k @ x))
        vector = np.array([1., 2., -.5])
        cycle = DepthCycle.__new__(DepthCycle)
        for count in (1, 2, 3):
            cycle.levels = levels[:count]
            cycle.smoothers = [lambda x: 0*x] * (count-1)
            cycle.terminal = lambda x, c=count: np.linalg.solve(coarse[c-1], x)
            p = np.linalg.multi_dot(transfer[:count]) if count > 1 else transfer[0]
            np.testing.assert_allclose(cycle.gain(vector),
                                       p @ np.linalg.solve(coarse[count-1], p.T @ vector))

    def test_third_coarse_points_are_component_balanced(self):
        native = np.arange(24, dtype=np.int64)
        positions = np.column_stack((native // 3, np.zeros((len(native), 2))))
        mask = third_endpoints(native, positions, 6)
        np.testing.assert_array_equal(np.bincount(native[mask] % 3), [2, 2, 2])


class DirectTrainTest(unittest.TestCase):
    def test_hard_selected_weights_give_support_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch is installed in the training environment")
        from gnn_amg.direct_train import final_weights, support_gate

        score = torch.tensor([[2., 1., -1.]], requires_grad=True)
        raw = torch.tensor([[.4, -.25, .1]], requires_grad=True)
        hard, gate = support_gate(score, torch.tensor([[True, True, True]]), torch.tensor([2]))
        result = final_weights(raw, gate, hard, torch.tensor([[True, False, True]]), True)
        torch.testing.assert_close(result, torch.tensor([[1.25, -.25, 0.]]))
        result.square().sum().backward()
        self.assertGreater(abs(float(score.grad[0, 2])), 0.)
