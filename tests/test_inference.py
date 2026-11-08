import socket
import threading
import unittest

import numpy as np
from scipy import sparse

from gnn_amg.inference import receive_packet, send_packet, spatial_decode


class InferenceTests(unittest.TestCase):
    def test_numeric_wire_round_trip_preserves_dtypes(self):
        left, right = socket.socketpair()
        arrays = dict(native=np.arange(13, dtype=np.int64), selected=np.arange(13) % 2 == 0,
                      features=np.arange(39, dtype=np.float32).reshape(13, 3))
        worker = threading.Thread(target=send_packet, args=(left, dict(stage=1), arrays))
        worker.start()
        metadata, actual = receive_packet(right)
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(metadata, dict(stage=1))
        for key in arrays:
            self.assertEqual(actual[key].dtype, arrays[key].dtype)
            np.testing.assert_array_equal(actual[key], arrays[key])
        left.close()
        right.close()

    def test_spatial_decoder_is_order_independent_and_score_sensitive(self):
        xyz = np.stack(np.meshgrid(np.arange(7), np.arange(5), np.arange(3), indexing='ij'), axis=-1).reshape(-1, 3)
        xyz = np.repeat(xyz, 3, axis=0).astype(float)
        n = len(xyz)
        features = np.zeros((n, 27))
        features[:, 10:13] = xyz
        native = np.arange(n, dtype=np.int64)
        comp = native % 3
        distance = np.abs(xyz[:, None]-xyz[None, :]).sum(axis=-1)
        graph = sparse.csr_matrix((distance == 1) & (comp[:, None] == comp[None, :]))
        eligible = native % 11 != 0
        scores = np.sin(native*.13)
        chosen, meta = spatial_decode(scores, features, native, eligible, graph.indptr, graph.indices, .08)
        self.assertTrue(np.all(~chosen | eligible))
        self.assertLess(chosen.sum(), eligible.sum()/3)
        self.assertGreater(chosen.sum(), 3)
        swapped, _ = spatial_decode(-scores, features, native, eligible, graph.indptr, graph.indices, .08)
        self.assertTrue(np.any(chosen != swapped))
        order = np.random.default_rng(9).permutation(n)
        reordered = graph[order][:, order].tocsr()
        predicted, _ = spatial_decode(scores[order], features[order], native[order], eligible[order],
            reordered.indptr, reordered.indices, .08)
        np.testing.assert_array_equal(np.sort(native[chosen]), np.sort(native[order][predicted]))


if __name__ == '__main__':
    unittest.main()
