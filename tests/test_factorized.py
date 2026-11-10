"""Focused regressions for learned factor decoding and actual-cycle feedback."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from gnn_amg.factorized import (FactorModel, REGIONS, STAGES, coefficients, groups, layer_sizes,
    load_model, pick_representatives, save_model, support_mask, support_order)
from gnn_amg.student_cycle import factor_reward


class FactorizedTest(unittest.TestCase):
    def test_group_budgets_exclude_fixed_preserve_components_and_cap(self):
        native = np.arange(30, dtype=np.int64)
        xyz = np.repeat(np.linspace(-.5, .5, 10), 3)[:, None] * np.ones((30, 3))
        eligible = native != 3
        labels = groups(native, xyz, eligible, 5)
        logits = np.column_stack((native.astype(float), np.ones(len(native))))
        maximum = pick_representatives(native, labels, logits)
        capped = pick_representatives(native, labels, logits, limit=7)
        self.assertFalse(maximum[3])
        self.assertEqual(int(capped.sum()), 7)
        self.assertTrue(np.all(maximum[capped]))
        np.testing.assert_array_equal(np.unique(labels[capped]), np.unique(labels[eligible]))
        np.testing.assert_array_equal(np.unique(native[capped] % 3), np.arange(3))
        for group in np.unique(labels[eligible]):
            self.assertTrue(capped[np.flatnonzero(labels == group)[-1]])

    def test_hard_selection_and_signed_weights(self):
        scores = np.array([[0., 3., 5., 2.], [0., 0., 0., 0.]])
        valid = np.array([[True, True, False, True], [False, False, False, False]])
        same = np.array([[True, False, False, True], [False, False, False, False]])
        ordered = support_order(scores, valid, same)
        np.testing.assert_array_equal(ordered[0, :2], [3, 1])
        keep = np.zeros_like(valid)
        np.put_along_axis(keep, ordered[:, :2], True, axis=1)
        keep &= valid
        raw = coefficients(np.zeros_like(scores), np.array([[0., -.5, 0., 0.], [0., 0., 0., 0.]]),
                           keep, same)
        self.assertEqual(raw[0, 3], 1.)
        self.assertLess(raw[0, 1], 0.)
        self.assertEqual(np.count_nonzero(raw), 2)

    def test_near_ties_share_support_and_budget(self):
        # The same scores can be rounded in opposite directions by the GPU
        # and CPU. The higher cross-component score still fits the budget.
        low = np.nextafter(np.float32(1), np.float32(0))
        high = np.nextafter(np.float32(1), np.float32(2))
        scores = np.array([[low, high, 2.], [high, low, 2.]], np.float32)
        quota = np.array([np.nextafter(np.float32(-1.5), np.float32(-2)),
                          np.nextafter(np.float32(-1.5), np.float32(0))], np.float32)
        valid = np.ones_like(scores, bool)
        same = np.array([[True, True, False]] * 2)
        expected = np.array([[True, False, True]] * 2)
        np.testing.assert_array_equal(support_mask(scores, quota, valid, same), expected)
        try:
            import torch
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
        else:
            from gnn_amg.train_student import factor_decode

            outputs = torch.zeros((2, 3, 3))
            outputs[:, :, 0] = torch.from_numpy(scores)
            chosen = factor_decode(outputs, torch.from_numpy(quota),
                                   torch.from_numpy(valid), torch.from_numpy(same))
            np.testing.assert_array_equal(chosen.numpy(), expected)

    def test_signed_factor_product_retains_active_rows(self):
        keep = np.array([[True, True], [False, False]])
        same = np.array([[True, False], [False, False]])
        logits = np.zeros(keep.shape)
        # Both entries of the first row are nonzero in the old decoder.
        # Their paths coalesce at the same representative and cancel.
        correction = np.array([[np.arctanh(-1/3), -50.], [0., 0.]])
        old = coefficients(logits, correction, keep, same)
        self.assertEqual(np.count_nonzero(old[0]), 2)
        next_factor = np.ones((2, 1))
        np.testing.assert_allclose(old @ next_factor, np.zeros((2, 1)), atol=1e-15)
        fixed = coefficients(logits, correction, keep, same, unit_row_sum=True)
        self.assertLess(fixed[0, 1], 0.)
        np.testing.assert_allclose(fixed @ next_factor, [[1.], [0.]], atol=1e-15)
        no_own_component = coefficients(logits, correction, keep, np.zeros_like(same),
                                         unit_row_sum=True)
        np.testing.assert_allclose(no_own_component.sum(1), [1., 0.], atol=1e-15)

    def test_compact_inference_keeps_the_original_row_context(self):
        from types import SimpleNamespace

        from scipy.sparse import csr_matrix

        from gnn_amg.factorized import edge_features

        rng = np.random.default_rng(21)
        xyz = rng.uniform(-.5, .5, size=(53, 3))
        native = np.arange(159)
        graph = SimpleNamespace(xyz=xyz, native=3*np.arange(len(xyz)))
        cols = rng.choice(len(native), size=(53, 20), replace=True)
        cols[:, -3:] = -1
        rows = np.arange(len(cols))
        values = csr_matrix(rng.normal(size=(len(rows), len(native))))
        spacing = np.ones(len(rows))
        edges, same = edge_features(graph, cols, rows, spacing, values, native, xyz)
        packed, packed_same = edge_features(graph, cols, rows, spacing, values, native, xyz, packed=True)
        np.testing.assert_array_equal(edges, packed)
        np.testing.assert_array_equal(same, packed_same)

        width = 12
        size = sum((a+1)*b for a,b in layer_sizes(width))
        model = FactorModel(width, rng.normal(0., .2, size).astype(np.float32),
                           rng.normal(0., .2, (STAGES, REGIONS, 6)).astype(np.float32))
        fine = rng.normal(size=(len(rows), width)).astype(np.float32)
        coarse = rng.normal(size=(*cols.shape, width)).astype(np.float32)
        valid = cols >= 0
        regions = np.zeros(len(rows), dtype=int)
        full, quotas = model.edge_outputs(fine, coarse, edges, valid, 0, regions)
        keep = support_mask(full[:, :, 0], quotas, valid, same)
        sparse, _ = model.edge_outputs(fine, coarse, packed, valid, 0, regions, keep=keep)
        a = coefficients(full[:, :, 1], full[:, :, 2], keep, same, unit_row_sum=True)
        b = coefficients(sparse[:, :, 1], sparse[:, :, 2], keep, same, unit_row_sum=True)
        # A compact BLAS batch can round the same float32 dot product
        # differently. The production experiment separately checks hashes
        # of the actual five factors before timing a path.
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=4*np.finfo(np.float32).eps)

    def test_paired_cost_requires_equal_accuracy_and_prefers_time(self):
        from gnn_amg.factor_replay import cost_pair

        cases = [dict(repeat=i, key="one", method=method, passed=True, fallback=None,
                      final_true_residual=1e-9, true_limit=1e-8, setup_s=.2, solve_s=seconds-.2,
                      total_s=seconds, iterations=steps)
                 for i in range(2) for method, seconds, steps in (("base", 2., 80), ("new", 1.8, 84))]
        pair = cost_pair(cases, ["one"], "base", "new", .02)
        self.assertTrue(pair["eligible"])
        np.testing.assert_array_equal(pair["ratios"], [.9, .9])
        cases[-1]["true_limit"] *= 2.
        self.assertFalse(cost_pair(cases, ["one"], "base", "new", .02)["eligible"])
        cases[-1]["true_limit"] = 1e-8
        self.assertFalse(cost_pair(cases[:-1], ["one"], "base", "new", .02)["eligible"])
        cases[-1]["total_s"] = 2.1
        self.assertFalse(cost_pair(cases, ["one"], "base", "new", .02)["eligible"])

    def test_explicit_product_drop_preserves_components_and_transposes(self):
        from types import SimpleNamespace

        from scipy.sparse import csr_matrix

        from gnn_amg.factor_compress import compress_product

        source = csr_matrix([[1., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0.],
                             [2., -1., -1., .4, .7, .1, .2]])
        fine, coarse = np.array([0, 3, 6]), np.array([0, 3, 6, 9, 1, 4, 2])
        xyz = np.arange(12, dtype=float).reshape(4, 3)/12.
        weights = np.zeros(16)
        weights[4] = 1.
        p = SimpleNamespace(maximum=float, reduce=np.asarray)
        compressed, tape, audit = compress_product(p, source, fine, coarse, xyz, weights)
        self.assertEqual(audit["restored_component_groups"], 1)
        self.assertEqual(audit["changed_rows"], 1)
        self.assertEqual(compressed.nnz, source.nnz-1)
        np.testing.assert_array_equal(compressed[:2].toarray(), source[:2].toarray())
        for component in range(3):
            columns = coarse % 3 == component
            np.testing.assert_allclose(compressed[:, columns].sum(axis=1),
                                       source[:, columns].sum(axis=1))
        np.testing.assert_array_equal(tape.apply(source).toarray(), compressed.toarray())
        rng = np.random.default_rng(8)
        perturbation = csr_matrix(rng.normal(size=source.shape))
        upstream = csr_matrix(rng.normal(size=source.shape))
        np.testing.assert_allclose(tape.apply(perturbation).multiply(upstream).sum(),
                                   perturbation.multiply(tape.pullback(upstream)).sum(), rtol=1e-14)

    def test_checkpoint_owns_regional_effects(self):
        size = sum((inputs+1)*outputs for inputs, outputs in layer_sizes(3))
        rng = np.random.default_rng(13)
        weights = rng.normal(0, .2, size).astype(np.float32)
        adapters = np.zeros((STAGES, REGIONS, 6), np.float32)
        adapters[0, 7] = np.ones(6)
        model = FactorModel(3, weights, adapters, unit_sum_depth=3, product_compression=np.ones(16))
        # The same neighbourhood in another region sees its own model quota.
        x = np.ones((2, layer_sizes(3)[0][0]), np.float32)
        regions = np.array([0, 7])
        hidden, rep = model.node_outputs(x, 0, regions)
        self.assertAlmostEqual(rep[1, 1] - rep[0, 1], 1.)
        edges = np.ones((2, 3, 8), np.float32)
        out, budget = model.edge_outputs(hidden, np.repeat(hidden[:, None, :], 3, axis=1), edges,
                                        np.ones((2, 3), bool), 0, regions)
        np.testing.assert_allclose(out[1, :, 0] - out[0, :, 0], 2., atol=1e-6)
        np.testing.assert_allclose(out[1, :, 2] - out[0, :, 2], 1.5, atol=1e-6)
        self.assertAlmostEqual(budget[1] - budget[0], 1.)
        root = Path(tempfile.mkdtemp(prefix="gnn-factor-test-"))
        receipt = save_model(root/"model.npz", model)
        restored = load_model(receipt)
        np.testing.assert_array_equal(restored.adapters, adapters)
        np.testing.assert_array_equal(restored.weights, weights)
        self.assertEqual(restored.unit_sum_depth, 3)
        np.testing.assert_array_equal(restored.product_compression, model.product_compression)

        from gnn_amg.backend import digest
        old = root/"legacy.npz"
        np.savez(old, width=model.width, weights=model.weights, adapters=model.adapters)
        self.assertEqual(load_model(dict(path=old, sha256=digest(old))).unit_sum_depth, 0)

    def test_full_seconds_rank_successes_and_own_cycle_ranks_failures(self):
        config = {"training": {"common_time_scale_s": 2., "episode_timeout_seconds": 10.}}
        result = dict(passed=False, total_s=9., final_true_residual=1., true_limit=1e-8,
                      cycle_probes=[{"residual_ratio": 2.}])
        improved = dict(result, cycle_probes=[{"residual_ratio": .2}])
        converged = dict(result, passed=True, total_s=10., final_true_residual=1e-9)
        self.assertLess(factor_reward(result, 3., config), factor_reward(improved, 3., config))
        self.assertLess(factor_reward(improved, 3., config), factor_reward(converged, 3., config))

    def test_short_krylov_adjoint_through_galerkin(self):
        from gnn_amg.krylov import short_gmres

        rng = np.random.default_rng(28)
        k = rng.normal(size=(12, 12)) + 4*np.eye(12)
        p = rng.normal(size=(12, 7))
        direction = rng.normal(size=p.shape)
        residual = rng.normal(size=7)

        def run(transfer, differentiate):
            galerkin = transfer.T @ k @ transfer
            gradient = np.zeros_like(galerkin)

            def apply(vector):
                return galerkin @ vector, vector.copy()

            def transpose(bar, vector):
                gradient[:] += np.outer(bar, vector)
                return galerkin.T @ bar

            result = short_gmres(apply, np.asarray, residual, 4, transpose if differentiate else None)
            adjoint = k @ transfer @ gradient.T + k.T @ transfer @ gradient
            return result["log_loss"], float(np.sum(adjoint*direction))

        epsilon = 1e-6
        _, analytic = run(p, True)
        finite = (run(p+epsilon*direction, False)[0]-run(p-epsilon*direction, False)[0])/(2*epsilon)
        np.testing.assert_allclose(analytic, finite, rtol=1e-7, atol=1e-8)

    def test_policy_score_matches_executed_choice_and_positive_update(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from gnn_amg.factorized import FactorPolicy
        from gnn_amg.factor_search import probabilities

        p = SimpleNamespace(rank=0, size=1, reduce=np.asarray,
                            gather_equal=lambda values: np.asarray(values)[None])
        width = 2
        size = sum((a+1)*b for a,b in layer_sizes(width))
        model = FactorModel(width, np.zeros(size), np.zeros((STAGES, REGIONS, 6)))
        temperature = .7
        features = np.array([[1., .1], [.9, .3], [.6, .8], [.3, .5]])
        fixed_draw = SimpleNamespace(choice=lambda count, p: count-1)
        settings = dict(stage=0, region=0, temperature=temperature, seed=1, subset_seed=2,
                        actions_per_rank=1)
        for role, layer in (("representatives", 1), ("support", 4), ("budget", 5)):
            with self.subTest(role=role):
                actor = FactorPolicy(p, model, dict(settings, role=role))
                actor.rng = fixed_draw
                if role == "representatives":
                    native = np.arange(4)*3
                    logits = np.column_stack((features[:, 0], np.zeros(4)))
                    with patch("gnn_amg.factorized.representative_domain",
                               return_value=(native, np.zeros(4, int), slice(None), None)):
                        selected = actor.representatives({}, None, features, logits, np.zeros(4))
                    np.testing.assert_array_equal(selected, [True, False, False, True])
                    legal_features, scores = features[1:], logits[1:, 0]
                elif role == "support":
                    valid = np.ones((1, 4), bool)
                    same = np.array([[True, False, False, False]])
                    keep = np.array([[True, True, False, False]])
                    outputs = np.zeros((1, 4, 3))
                    outputs[0, :, 0] = features[:, 0]
                    got = actor.edges(outputs, np.zeros(1), valid, same, keep, features[:1],
                                      features[1:2], features[None], np.zeros(1), np.zeros(1))
                    np.testing.assert_array_equal(got, [[True, False, False, True]])
                    legal_features, scores = features[1:], outputs[0, 1:, 0]
                else:
                    valid = np.ones((1, 4), bool)
                    keep = np.array([[True, True, True, False]])
                    got = actor.edges(np.zeros((1, 4, 3)), np.zeros(1), valid, valid, keep,
                                      features[:1], features[1:2], features[None], np.zeros(1), np.zeros(1))
                    self.assertEqual(got.sum(), 4)
                    options = np.arange(2, 5)
                    scores = -(options-3.)**2
                    legal_features = 2*(options[:, None]-3.)*np.r_[features[0], features[1], 1.]
                receipt = actor.receipt()
                gradient, bias = actor.terms[layer]
                delta = np.r_[gradient[0], bias[0]] if role == "budget" else gradient[0]
                before = probabilities(scores/temperature)[-1]
                epsilon = 1e-6
                shift = epsilon*(legal_features@delta)
                after = probabilities((scores+shift)/temperature)[-1]
                previous = probabilities((scores-shift)/temperature)[-1]
                self.assertEqual(receipt["changes"], 1)
                self.assertAlmostEqual(receipt["log_probability"], np.log(before))
                self.assertGreater(after, before)
                self.assertAlmostEqual((np.log(after)-np.log(previous))/(2*epsilon), delta@delta, places=7)

    def test_confirmation_matches_signatures_across_mpi_serialization(self):
        import json

        from gnn_amg.factor_search import confirmed_improvement

        base = dict(passed=True, total_s=8., iterations=50, pc_apply_s=3., setup_s=1.,
                    p1_nnz=100, k1_nnz=100, hierarchy_signature=("base-c", "base-p"))
        proposal = dict(base, total_s=7.5, iterations=46, hierarchy_signature=("new-c", "new-p"))
        screened_base, screened_proposal = json.loads(json.dumps([base, proposal]))
        settings = dict(absolute_margin_s=.18, relative_margin=.02)
        self.assertTrue(confirmed_improvement(screened_base, screened_proposal, base, proposal, settings))
        self.assertFalse(confirmed_improvement(screened_base, screened_proposal, base,
                                              dict(proposal, hierarchy_signature=("different",)), settings))
        self.assertFalse(confirmed_improvement(screened_base, screened_proposal, base,
                                              dict(proposal, hierarchy_signature=None), settings))

    def test_rebuilt_labels_are_only_read_by_solver_root(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from gnn_amg.factor_search import restore_labels
        from gnn_amg.factorized import hierarchy_signature

        numerical = {level: {"stages": [{"signatures": {
            "representatives": level+"-c", "support": level+"-p", "coefficients": level+"-w"}}]}
            for level in ("level1", "level2")}
        signature = hierarchy_signature(numerical)
        chosen = dict(key="state", method="proposal-0", accepted=True, candidate=dict(
            hierarchy_signature=list(signature), factor_policy=dict(signature="moves", examples=[])))
        frames = {"state": {"proposals": [dict(role="support")]}}
        example = dict(unchanged=False)
        for rank in (0, 1):
            with self.subTest(rank=rank):
                policy = dict(signature="moves")
                if rank == 0:
                    policy["examples"] = [example]
                solve = Mock(return_value=dict(numerical=numerical, failure=None, factor_policy=policy))
                rebuilt = restore_labels(SimpleNamespace(rank=rank, worker=0), ["state"], frames,
                                         [chosen], None, 1, str, lambda _: None, solve, list)
                solve.assert_called_once()
                if rank == 0:
                    self.assertEqual(rebuilt[0]["examples"], [example])
                    self.assertTrue(rebuilt[0]["matches"])
                else:
                    self.assertEqual(rebuilt, [])

    def test_short_prefixes_and_frozen_signed_head(self):
        from gnn_amg.krylov import short_gmres, signed_head

        rng = np.random.default_rng(5)
        matrix = rng.normal(size=(8, 8))+3*np.eye(8)
        residual = rng.normal(size=8)
        apply = lambda v:(matrix @ v, None)
        result = short_gmres(apply, np.asarray, residual, 6, prefixes=(2, 4, 6))
        for n in (2, 4, 6):
            self.assertAlmostEqual(result["prefix_ratios"][str(n)],
                                   short_gmres(apply, np.asarray, residual, n)["residual_ratio"])
        width = 3
        size = sum((a+1)*b for a,b in layer_sizes(width))
        model = FactorModel(width, np.ones(size), np.zeros((STAGES, REGIONS, 6)))
        masked = FactorModel(width, signed_head(model, model.weights), model.adapters)
        for layer, (weight, bias) in enumerate(masked.layers):
            expected = 2*(width+1) if layer == 4 else 0
            self.assertEqual(np.count_nonzero(weight)+np.count_nonzero(bias), expected)
        np.testing.assert_array_equal(masked.layers[4][0][0], 0.)

    def test_search_budget_preserves_anchors_and_pair_budget(self):
        from types import SimpleNamespace

        from gnn_amg.factor_search import SearchProposal

        p = SimpleNamespace(rank=0, size=1, reduce=np.asarray,
                            gather_equal=lambda values: np.asarray(values)[None],
                            maximum=float)
        width = 2
        size = sum((a+1)*b for a,b in layer_sizes(width))
        model = FactorModel(width, np.zeros(size), np.zeros((STAGES, REGIONS, 6)))
        policy = SearchProposal(p, model, dict(role="budget", stage=0, region=0, seed=4,
                                subset_seed=2, temperature=.7, actions_per_rank=3, examples=2))
        valid = np.ones((2, 4), bool)
        same = np.array([[True, False, False, False]]*2)
        keep = np.array([[True, True, True, False], [True, True, False, False]])
        moved = policy.edges(np.zeros((2, 4, 3)), np.array([0., -1.]), valid, same, keep,
                             np.zeros((2, width)), np.zeros((2, width)), np.zeros((2, 4, width)),
                             np.array([0, 3]), np.zeros(2), xyz=np.zeros((2, 3)))
        self.assertEqual(moved.sum(), 5)
        self.assertTrue(moved[:, 0].all())
        receipt = policy.receipt()
        self.assertEqual(receipt["changes"], 2)
        self.assertEqual(len(receipt["examples"]), 2)

    def test_split_generator_keeps_coefficient_encoders_independent(self):
        from gnn_amg.factor_operator import split_model
        from gnn_amg.backend import digest

        rng = np.random.default_rng(97)
        width = 3
        size = sum((i+1)*o for i,o in layer_sizes(width))
        origin = FactorModel(width,rng.normal(size=size).astype(np.float32),
                             np.zeros((STAGES,REGIONS,6),np.float32),3)
        structure = FactorModel(width,origin.weights+.1,origin.adapters,3)
        generated = split_model(structure,origin)
        np.testing.assert_array_equal(generated.coefficients_at(0).weights,origin.weights)
        np.testing.assert_array_equal(generated.coefficients_at(4).weights,origin.weights)
        generated.coefficient_weights[0,0] += 1.
        np.testing.assert_array_equal(generated.coefficients_at(4).weights,origin.weights)
        np.testing.assert_array_equal(generated.weights,structure.weights)
        path = Path(tempfile.mkdtemp(prefix="gnn-generator-check-"))/"split.npz"
        save_model(path,generated)
        restored = load_model(dict(path=path,sha256=digest(path)))
        np.testing.assert_array_equal(restored.coefficient_weights,generated.coefficient_weights)
        np.testing.assert_array_equal(restored.weights,structure.weights)

    def test_complete_structure_network_derivatives(self):
        from gnn_amg.factor_operator import structure_example

        rng = np.random.default_rng(20)
        width = 4
        layers = layer_sizes(width)
        size = sum((i+1)*o for i,o in layers)
        weights = rng.uniform(.01,.06,size)
        model = FactorModel(width,weights,np.full((STAGES,REGIONS,6),.015),3)
        dimension = layers[0][0]
        representative = dict(kind="representative",stage=1,temperature=.7,chosen=1,
                              unchanged=False,regions=np.array([1,2,4]),
                              fine=rng.uniform(.1,.6,(3,dimension)))
        row = dict(stage=1,temperature=.7,chosen=1,unchanged=False,regions=2,
                   fine=rng.uniform(.1,.6,dimension),coarse=rng.uniform(.1,.6,(3,dimension)),
                   edge=rng.uniform(.1,.6,(3,8)),valid=np.array([True,True,False]),
                   same=np.array([True,False,False]),keep=np.array([True,True,False]))
        support = dict(row,kind="support",options=np.array([0,1]),domain=np.array([2,7]))
        budget = dict(row,kind="budget",domain=np.arange(1,4))
        for example in (representative,support,budget):
            with self.subTest(kind=example["kind"]):
                _,_,_,gradient = structure_example(model,example,True)
                self.assertGreater(np.linalg.norm(gradient[:(dimension+1)*width]),0.)
                direction = gradient/np.linalg.norm(gradient)
                epsilon = 2e-4
                v = []
                for sign in (-1,1):
                    perturbed = FactorModel(width,weights+sign*epsilon*direction,model.adapters,3)
                    v.append(structure_example(perturbed,example,False)[0])
                finite = (v[1]-v[0])/(2*epsilon)
                np.testing.assert_allclose(finite,gradient@direction,atol=2e-5,rtol=3e-4)

    def test_replay_objective_balances_types_and_differentiates_regression_tail(self):
        from gnn_amg.factor_replay import balanced_action

        losses = {"initial/pcapply-24":.2, "added/pcapply-24":.4,
                  "initial/region-1-component-2":1.2, "added/region-4-component-0":.3}
        reference = dict.fromkeys(losses,.5)
        settings = dict(pc_weight=.6,tail_floor=.1,tail_fraction=.25,tail_weight=.2)
        score,weights,terms = balanced_action(losses,reference,settings)
        self.assertAlmostEqual(score,.6*.3+.4*.75+.2*1.4)
        self.assertEqual(terms["tail_names"],["initial/region-1-component-2"])
        for name in losses:
            epsilon=1e-6
            perturbed=dict(losses)
            perturbed[name]+=epsilon
            actual=(balanced_action(perturbed,reference,settings)[0]-score)/epsilon
            self.assertAlmostEqual(actual,weights[name],places=8)
        same = dict(losses,**{"refresh/pcapply-8":.3})
        no_tail=dict(settings,tail_weight=0.)
        self.assertAlmostEqual(balanced_action(losses,reference,no_tail)[0],
                               balanced_action(same,dict.fromkeys(same,.5),no_tail)[0])

    def test_replay_training_visits_training_split_across_cases(self):
        from gnn_amg.factor_replay import training_order

        frames = {str(i):dict(case=str(i%3),t=float(i),split="training" if i<55 else "validation")
                  for i in range(63)}
        order = training_order(frames,2,2026)
        self.assertEqual(order,training_order(frames,2,2026))
        self.assertEqual(sorted(int(k) for batch in order for k in batch),list(range(55)))
        self.assertTrue(all(frames[a]["case"]!=frames[b]["case"] for a,b in order[:-1]))

    def test_rebuilt_teacher_retains_only_source_gaps(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from scipy import sparse
        from gnn_amg.factorized import prune_rebuilt_teacher
        from gnn_amg.gate import checked_hierarchy

        p=SimpleNamespace(reduce=np.asarray)
        raw=sparse.csr_matrix([[1.,0.],[0.,0.],[.2,.8]])
        pruned=sparse.csr_matrix([[1.,0.],[0.,0.],[0.,0.]])
        legacy=SimpleNamespace(prune_transfer_connections=lambda *_:(pruned,{}))
        transfer,proof=prune_rebuilt_teacher(p,legacy,raw,np.array([0,3]),np.ones(3,bool),.05)
        np.testing.assert_array_equal(transfer.toarray(),raw.toarray())
        self.assertEqual(proof,dict(source_empty_rows=1,rejected_pruned_rows=1,applied_drop=0.))
        info=dict(empty_free_rows=1,uncovered_free_rows=1,verified_source_empty_rows=1,
                  repair_budget_exhausted=0,injection_error=0.)
        second=SimpleNamespace(stats=info,owners=np.array([0,2]))
        config=dict(terminal_max_dofs=4)
        with patch("gnn_amg.gate.Hierarchy",return_value="verified"):
            self.assertEqual(checked_hierarchy(config,p,legacy,None,second,offline_teacher=True),
                             ("verified",None))
            self.assertEqual(checked_hierarchy(config,p,legacy,None,second)[1],
                             "second_interpolation_coverage_or_injection")

    def test_fused_replay_uses_exact_signature_and_numerical_action_tolerance(self):
        from gnn_amg.factor_replay import equivalent_probe_actions

        row=dict(key="state",failure=None,signature=["same-transfer"],nonzeros=[4,3,2],
                 probes={"a":.15,"b":.3})
        rounded=dict(row,probes={"a":.15+1e-15,"b":.3-1e-15})
        self.assertTrue(equivalent_probe_actions([row],[rounded]))
        self.assertFalse(equivalent_probe_actions([row],[dict(rounded,signature=["different-transfer"])]))
        self.assertFalse(equivalent_probe_actions([row],[dict(rounded,probes={"a":.16,"b":.3})]))

    def test_sparse_ilu_numeric_adjoint(self):
        from scipy import sparse
        from gnn_amg.krylov import ILU0

        matrix = sparse.csr_matrix([[5., -1., 0., 1.], [2., 4., -.4, 0.],
                                    [0., .5, 3., .2], [-1., 0., .8, 4.]])
        rng = np.random.default_rng(8)
        direction = rng.normal(size=matrix.nnz)
        rhs, bar = rng.normal(size=(2, 4))
        tape = ILU0(matrix)
        solution, adjoint = tape.apply(rhs), tape.apply(bar, transpose=True)
        gradient = tape.reverse(tape.factors_adjoint(solution, adjoint))

        def objective(step):
            perturbed = matrix.copy()
            perturbed.data += step*direction
            return float(bar @ ILU0(perturbed).apply(rhs))

        epsilon = 1e-6
        np.testing.assert_allclose(float(gradient @ direction),
            (objective(epsilon)-objective(-epsilon))/(2*epsilon), rtol=1e-7, atol=1e-9)

    def test_torch_and_deployment_decode(self):
        try:
            from gnn_amg.train_student import HardFactorTrainingModel, factor_loss, factor_parity
            import torch
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            self.skipTest("Torch is available in the training environment")
        torch.manual_seed(23)
        rng = np.random.default_rng(23)
        n, k, features = 16, 12, layer_sizes(4)[0][0]
        online = rng.random((n, k)) > .2
        online[:, -2:] = False
        target = np.zeros((n, k), np.float32)
        target[:, :4] = rng.uniform(-1, 1, (n, 4))
        frame = dict(x=rng.normal(size=(n, features)).astype(np.float32),
            coarse=rng.normal(size=(n, k, features)).astype(np.float32),
            edge=rng.normal(size=(n, k, 8)).astype(np.float32),
            online=online, valid=np.ones_like(online), same=rng.random((n, k)) > .5,
            target=target, active=np.arange(n) % 2 == 0, selected=np.arange(n) % 2 == 1,
            quota=np.full(n, 2, np.float32), probes=rng.random((n, k, 5)).astype(np.float32),
            action=rng.random((n, 5)).astype(np.float32))
        model = HardFactorTrainingModel(4, unit_sum_depth=3)
        names = ("shared", "representatives", "support", "signed", "budget")
        params = [p for group in model.optimizer_groups(dict.fromkeys(names, .001)) for p in group["params"]]
        self.assertEqual({id(p) for p in params}, {id(p) for p in model.parameters()})
        self.assertEqual(len(params), len(list(model.parameters())))
        for stage in (0, 4):
            self.assertTrue(factor_parity(model, frame, stage)["exact_support"])
        batch = {key: torch.as_tensor(value) for key, value in frame.items()}
        for hard in (False, True):
            model.zero_grad()
            loss, _ = factor_loss(model, batch, hard)
            loss.backward()
            for parameter in model.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(model.signed.weight.grad[1].norm()), 0.)
