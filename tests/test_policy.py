import unittest

import torch

from gnn_amg.policy import TwoStagePolicy, sample_local, supervised_loss
from gnn_amg.objective import RewardBaseline, reinforce_loss


class PolicyTests(unittest.TestCase):
    def test_repair_does_not_change_saved_action(self):
        torch.manual_seed(3)
        logits = torch.zeros(6, requires_grad=True)
        eligible = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
        mask = torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.bool)
        frozen = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.bool)
        sample = sample_local(logits, frozen, mask, eligible)
        saved = sample.raw.clone()
        repaired = sample.raw.clone()
        repaired[:] = True
        self.assertTrue(torch.equal(saved, sample.raw))
        self.assertTrue(torch.equal(sample.raw[~mask & eligible], frozen[~mask & eligible]))
        self.assertFalse(sample.raw[-1])
        loss = reinforce_loss([dict(key="a", samples=[sample], reward=1, importance=1)], RewardBaseline())
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[~sample.explored], torch.zeros(3)))
        self.assertTrue(torch.all(logits.grad[sample.explored] != 0))

    def test_shared_model_accepts_actual_coarse_graph_size(self):
        policy = TwoStagePolicy(5)
        for stage, n in [(0, 11), (1, 3)]:
            edges = torch.eye(n).to_sparse()
            out = policy(torch.randn(n, 5), edges, torch.zeros(n), stage)
            self.assertEqual(tuple(out.shape), (n,))
            self.assertTrue(torch.isfinite(out).all())

    def test_csr_supervised_gradient_reaches_both_stage_heads(self):
        torch.manual_seed(17)
        policy = TwoStagePolicy(5)
        before = [head.weight.detach().clone() for head in policy.heads]
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        optimizer.zero_grad()
        for stage in (0, 1):
            # Unequal weights and an empty row exercise the actual CSR path.
            dense = torch.tensor([[0., .2, .8], [1., 0., 0.], [0., 0., 0.]])
            graph = dense.to_sparse_csr()
            logits = policy(torch.randn(3, 5), graph, torch.zeros(3), stage)
            supervised_loss(logits, torch.tensor([1, 0, 0]), torch.ones(3, dtype=torch.bool)).backward()
            self.assertTrue(torch.equal(graph.to_dense(), dense))
        optimizer.step()
        for old, head in zip(before, policy.heads):
            self.assertTrue(torch.isfinite(head.weight).all())
            self.assertFalse(torch.equal(old, head.weight))


if __name__ == "__main__":
    unittest.main()
