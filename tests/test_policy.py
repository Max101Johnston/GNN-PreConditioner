import unittest

import torch

from gnn_amg.policy import TwoStagePolicy, sample_local, supervised_loss
from gnn_amg.objective import RewardBaseline, reinforce_loss


class PolicyTests(unittest.TestCase):
    def test_context_action_backpropagates_through_hard_support_and_budget(self):
        from gnn_amg.train_student import context_support

        scores = torch.linspace(-.7,.8,14).reshape(1,14).requires_grad_()
        budget = torch.tensor([[0.,.3,0.,0.]],requires_grad=True)
        valid = torch.ones_like(scores,dtype=torch.bool)
        valid[0,-1] = False
        hard = context_support(scores,budget,valid)
        surrogate = context_support(scores,budget,valid,.2)
        torch.testing.assert_close(surrogate,hard.to(scores.dtype),atol=0,rtol=0)
        self.assertEqual(hard.sum().item(),8)
        action = (surrogate*torch.linspace(.1,1.5,14)).sum()
        action.backward()
        self.assertGreater(scores.grad[valid].abs().sum().item(),0)
        self.assertEqual(scores.grad[~valid].item(),0)
        self.assertGreater(budget.grad.abs().sum().item(),0)

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

    def test_streaming_on_policy_batch_and_checkpoint_resume(self):
        import copy
        import json
        import tempfile
        from pathlib import Path
        import numpy as np
        from scipy import sparse
        from gnn_amg.inference import PolicyLearner
        from gnn_amg.backend import digest
        # Artifacts stay outside the source tree. Do not recursively remove them.
        root = Path(tempfile.mkdtemp(prefix="gnn-rl-test-"))
        for name in ("records", "data", "checkpoints"):
            (root/name).mkdir()
        settings = dict(seed=81, learning_rate=1e-3, phase_updates=[20, 20],
            exploration_dofs=[4, 4], reference_probability=.8, sampling_temperature=1.,
            episode_timeout_seconds=10., gradient_clip=1e6, checkpoint_interval=20)
        config = dict(training=settings, output_directory=str(root))
        torch.manual_seed(4)
        model = TwoStagePolicy(5)
        saved = dict(feature_names=list("abcde"), feature_mean=np.zeros((2, 5)),
            feature_scale=np.ones((2, 5)), coarse_fraction=np.array([.1, .1]), dataset_job="test")
        learner = PolicyLearner(config, model, torch.device("cpu"), saved)
        learner.baseline.values = {"a": .3, "b": -.2}
        initial = {name: p.detach().clone() for name, p in model.named_parameters()}
        learner.begin(dict(keys=["a", "b"], common_scale=2., reference_times={"a": 5., "b": 4.}))
        n = 9
        graph = sparse.diags([np.ones(n-1), np.ones(n-1)], [-1, 1], shape=(n, n)).tocsr()
        features = np.arange(n*5, dtype=np.float32).reshape(n, 5)/17
        arrays = dict(native=np.arange(n), eligible=np.ones(n, bool), features=features,
            indptr=graph.indptr, indices=graph.indices)
        frozen = np.arange(n) % 3 == 0
        tensor_graph = torch.tensor(graph.toarray(), dtype=torch.float32).to_sparse_csr()
        trainable = [p for p in model.parameters() if p.requires_grad]
        expected = [torch.zeros_like(p) for p in trainable]
        for key, seconds in (("a", 2.), ("b", 3.)):
            logits = model(torch.tensor(features), tensor_graph, torch.zeros(n), 1)
            raw, detail = learner.sample(dict(key=key, stage=1), logits, frozen, arrays)
            sample = learner.pending[key][0]
            tag = key+"-stage1"
            probability = learner.action_arrays[tag+"-probability"]
            draws = learner.action_arrays[tag+"-draw"]
            actual_log = np.where(draws, np.log(probability), np.log1p(-probability)).sum()
            self.assertAlmostEqual(actual_log, detail["log_probability"], places=5)
            reward = ((5. if key == "a" else 4.)-seconds)/2.
            loss = -(reward-learner.baseline.get(key))*sample.log_probability/2
            derivative = torch.autograd.grad(loss, trainable, retain_graph=True)
            expected = [a+b for a, b in zip(expected, derivative)]
            execution = raw.copy()
            execution[:] = True  # numerical repair must never rewrite raw likelihood
            response = learner.feedback(dict(key=key, seconds=seconds, converged=True, training_pool_size=2))
            self.assertEqual(response["past_baseline"], .3 if key == "a" else -.2)
            for name, p in model.named_parameters():
                self.assertTrue(torch.equal(p, initial[name]))
        for parameter, derivative in zip(trainable, expected):
            torch.testing.assert_close(parameter.grad, derivative)
        # A common shift of all logits leaves the centered probabilities unchanged.
        bias_grad = model.heads[1].bias.grad
        torch.testing.assert_close(bias_grad, torch.zeros_like(bias_grad), rtol=0, atol=1e-7)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            result = learner.commit()
        self.assertGreater(result["parameter_delta_norm"], 0)
        self.assertEqual(result["phase"], "second")
        for name, delta in result["parameter_changes"].items():
            if not name.startswith("heads.1."):
                self.assertEqual(delta, 0)
        receipt = result["checkpoint"]
        self.assertEqual(digest(receipt["path"]), receipt["sha256"])
        checkpoint = torch.load(receipt["path"], map_location="cpu")
        restored_model = TwoStagePolicy(5)
        restored_model.load_state_dict(checkpoint["model"])
        restored = PolicyLearner(config, restored_model, torch.device("cpu"), checkpoint)
        self.assertEqual(restored.update, 1)
        self.assertEqual(restored.baseline.values, learner.baseline.values)
        self.assertEqual(json.dumps(restored.rng.bit_generator.state), json.dumps(learner.rng.bit_generator.state))
        for key, values in learner.optimizer.state_dict()["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(restored.optimizer.state_dict()["state"][key][name], value)
        self.assertTrue(torch.equal(torch.get_rng_state(), checkpoint["rng_torch"]))


    def test_parallel_replicas_apply_one_identical_weighted_update(self):
        import contextlib
        import copy
        import io
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import numpy as np
        from scipy import sparse
        from gnn_amg.inference import PolicyLearner
        root = Path(tempfile.mkdtemp(prefix="gnn-parallel-test-"))
        for name in ("records", "data", "checkpoints"):
            (root/name).mkdir()
        settings = dict(seed=91, learning_rate=1e-3, phase_updates=[20, 20],
            exploration_dofs=[4, 4], reference_probability=.8, sampling_temperature=1.,
            episode_timeout_seconds=10., gradient_clip=1., checkpoint_interval=20)
        config = dict(training=settings, output_directory=str(root), parallel_workers=3)
        saved = dict(feature_names=list("abcde"), feature_mean=np.zeros((2, 5)),
            feature_scale=np.ones((2, 5)), coarse_fraction=np.array([.1, .1]), dataset_job="test")
        torch.manual_seed(17)
        template = TwoStagePolicy(5)
        n = 9
        graph = sparse.diags([np.ones(n-1), np.ones(n-1)], [-1, 1], shape=(n, n)).tocsr()
        features = np.arange(n*5, dtype=np.float32).reshape(n, 5)/13
        arrays = dict(native=np.arange(n), eligible=np.ones(n, bool), features=features,
            indptr=graph.indptr, indices=graph.indices)
        tensor_graph = torch.tensor(graph.toarray(), dtype=torch.float32).to_sparse_csr()
        replicas, pieces = [], []
        for worker, key in enumerate(("a", "b", "c")):
            with patch.dict("os.environ", {"GNN_POLICY_WORKER": str(worker)}):
                learner = PolicyLearner(config, copy.deepcopy(template), torch.device("cpu"), saved)
            learner.begin(dict(keys=["a", "b", "c"], common_scale=2., reference_times=dict(a=6., b=6., c=6.)))
            logits = learner.model(torch.tensor(features), tensor_graph, torch.zeros(n), 1)
            learner.sample(dict(key=key, stage=1), logits, np.arange(n) % 3 == 0, arrays)
            lane_size = [19, 18, 18][worker]
            learner.feedback(dict(key=key, seconds=2.+worker, converged=True, training_pool_size=55,
                sampling_probability=1/(3*lane_size), importance=3*lane_size/55))
            self.assertEqual(learner.identity(), replicas[0].identity() if replicas else learner.identity())
            replicas.append(learner)
            pieces.append(json.loads(json.dumps(learner.partial_gradient())))
        merged = dict(names=pieces[0]["names"],
            gradient=np.asarray([p["gradient"] for p in pieces]).sum(axis=0).tolist(),
            parameter_sha256=pieces[0]["parameter_sha256"], update=0,
            observations=[row for p in pieces for row in p["observations"]])
        with self.assertRaises(ValueError):
            replicas[0].merge_gradient(dict(merged, parameter_sha256="stale"))
        results = []
        for replica in replicas:
            replica.merge_gradient(merged)
            with contextlib.redirect_stdout(io.StringIO()):
                results.append(replica.commit())
        self.assertEqual(len({r["parameter_sha256"] for r in results}), 1)
        self.assertEqual(len({r["actions"]["path"] for r in results}), 3)
        self.assertTrue(all(r["update"] == 1 and r["parameter_delta_norm"] > 0 for r in results))
        for replica in replicas[1:]:
            self.assertEqual(replica.baseline.values, replicas[0].baseline.values)
            for key, values in replicas[0].optimizer.state_dict()["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(replica.optimizer.state_dict()["state"][key][name], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
