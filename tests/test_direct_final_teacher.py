"""Numerical checks for complete, signed transfer export."""
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from gnn_amg.direct_final_teacher import audit_transfer, real_residuals, sparse_cost


class FinalTeacherTest(unittest.TestCase):
    def test_signed_cross_rank_rows_and_constraints(self):
        transfer = csr_matrix([[0, 1, 0], [0, -.25, 1.25], [0, 0, 0], [1, 0, 0]])
        native = np.array([27, 28, 21, 22])
        fixed = np.array([False, False, True, False])
        errors, row_error = audit_transfer(transfer, native, np.array([27]), fixed, [1, 2], True)
        self.assertEqual(errors, (0, 0, 0, 0))
        self.assertEqual(row_error, 0.0)
        self.assertEqual(sparse_cost(transfer, [1, 2]), (4, 2, 2))

    def test_injection_and_empty_rows_are_distinguished(self):
        transfer = csr_matrix([[1, 0], [0, 0]])
        errors, row_error = audit_transfer(transfer, np.array([0, 1]), np.array([0]),
                                           np.array([False, False]), [1, 2], False)
        self.assertEqual(errors, (1, 0, 1, 0))
        self.assertIsNone(row_error)

    def test_residual_steps_use_distinct_actual_iterations(self):
        samples = {0: (0, 10), 8: (8, 7), 24: (24, 3), "late": (24, 3)}
        self.assertEqual(real_residuals(samples, 30), [("early", 0, 10), ("middle", 8, 7),
                                                       ("late", 24, 3)])
        self.assertEqual(real_residuals({0: (0, 10)}, 4), [("early", 0, 10)])


if __name__ == "__main__":
    unittest.main()
