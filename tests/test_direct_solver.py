"""Check the deployed budget and signed head on their executed sparse support."""
import unittest

import numpy as np
from scipy import sparse

from gnn_amg.direct_solver import DirectStencil, allocate, deduplicate, routed_candidates


class DirectSolverTest(unittest.TestCase):
    def test_bounded_allocation_redistributes_excess(self):
        quota = allocate([100., 2., 0., 1.], [1, 2, 4, 2], 8)
        np.testing.assert_array_equal(quota, [1, 2, 3, 2])

    def test_distinct_routed_regions_append_candidates(self):
        q = np.array([[1., 0.], [0., 1.]])
        keys = np.array([[1., 0.], [.8, .2], [0., 1.], [.2, .8]])
        route = np.array([[2., 1.], [1., 2.]])
        candidates = routed_candidates(q, keys, route, np.array([0, 0, 1, 1]), np.arange(2), 1)
        np.testing.assert_array_equal(candidates, [[0, 3], [2, 1]])
        unique = deduplicate(np.array([[0, 0, 2], [1, 3, 1]]))
        self.assertEqual(np.count_nonzero(unique >= 0), 4)

    def test_projected_sparse_head_pullback(self):
        pattern = sparse.csr_matrix((np.ones(5), [0, 1, 0, 1, 2], [0, 2, 3, 5]),
                                    shape=(3, 3))
        basis = np.array([[.5, .2], [-.1, .4], [0., 0.], [1.1, 0.], [.1, .6]])
        stencil = DirectStencil(pattern, np.array([0, 0, 1, 2, 2]), basis,
                               np.array([True, False, True, True, True]),
                               np.array([False, False, True, False, False]), True,
                               np.array([0]), np.array([0, 3]), np.array([0, 3]), {})
        weight = np.array([1., -.25, .1])
        shift = np.array([.3, .7, -.8])
        gradient = sparse.csr_matrix((np.array([.3, -.7, .2, .5, -.1]),
                                      pattern.indices, pattern.indptr), shape=pattern.shape)
        output = stencil.values(weight)
        np.testing.assert_allclose([sum(output[:2]), output[2], sum(output[3:])], [1., 1., 1.])
        eps = 1e-6
        finite = gradient.data @ (stencil.values(weight + eps*shift)
                                  - stencil.values(weight - eps*shift))/(2*eps)
        self.assertAlmostEqual(float(finite), float(stencil.pullback(gradient) @ shift), places=8)
