"""Independent two-decision policy evaluation in a persistent MPI allocation.

This path constructs fresh observable features and both actual numerical levels.
It never reads teacher hierarchy maps or supervised graph NPZs. The separately
named retained-base comparator reads its own cached interpolation as a baseline.
"""
import argparse
import ctypes as C
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import numpy as np

from .backend import (Action, Hierarchy, Level, digest, graph_features, load_bridge,
    load_dependencies, local_csr, ownership, pooled_attributes, read_json, record, solve_with_deadline,
    strength_graph)
from .dataset import gather_array, gather_graph, physical_attributes
from .inference import PolicyClient, connection_prior, graph_decode, load_selector


def broadcast_selection(p, selected, count):
    values = np.asarray(selected, dtype=np.uint8) if p.rank == 0 else np.empty(count, np.uint8)
    assert len(values) == count
    mpi_byte = p.P(C.addressof((C.c_char*1).in_dll(p.mpi, "ompi_mpi_byte")))
    p.mpi.MPI_Bcast.argtypes = [p.P, p.I, p.P, p.I, p.P]
    p.mpi.MPI_Bcast.restype = p.I
    assert p.mpi.MPI_Bcast(values.ctypes.data, count, mpi_byte, 0, p.world) == 0
    return values.astype(bool)


def choose(config, p, bridge, rpc, matrix, native, comp, attributes, fixed, mass, stage, neutral, context=None):
    start = time.perf_counter()
    fast = (context or {}).get("fast_prior", False)
    strength = config["interpolation"]["strength_threshold"]
    if fast:
        graph, owners = strength_graph(p, bridge, matrix, comp, strength)
    else:
        x, graph, owners = graph_features(p, bridge, matrix, comp, attributes, fixed, stage, mass, strength)
    local_features_end = time.perf_counter()
    buffers = gather_graph(p, graph, owners)
    arrays = dict(native=gather_array(p, native, owners), eligible=gather_array(p, ~fixed, owners))
    if not fast:
        arrays["features"] = gather_array(p, x, owners)
    gather_end = time.perf_counter()
    selected, reply = None, {}
    if p.rank == 0:
        arrays.update(buffers)
        if fast:
            begin = time.perf_counter()
            scores = connection_prior(arrays["indices"], arrays["weights"], len(arrays["native"]))
            selector = load_selector(config["selector"]) if "selector" in config else None
            radius = config["decoder"]["radii"][stage]
            if stage == 1 and context is not None:
                radius = max(radius, context.get("terminal_min_radius", radius))
            while True:
                selected, decoder = graph_decode(scores, arrays["native"], arrays["eligible"],
                    arrays["indptr"], arrays["indices"], radius, selector=selector)
                adaptive = config.get("student_cycle") and stage == 1
                if not adaptive or int(selected.sum()) <= config["terminal_max_dofs"] or radius == 9:
                    break
                radius += 1
            reply = dict(policy="prior_only", decoder=decoder, device="cpu",
                         network_with_copies_s=0.0, checkpoint_sha256=None,
                         service_compute_s=time.perf_counter()-begin)
        else:
            reply, response = rpc.request(dict(operation="select", stage=stage, neutral=neutral, **(context or {})), arrays)
            selected = response["selected"]
        assert selected.dtype == bool and selected.shape == (int(owners[-1]),)
        assert not np.any(selected & ~arrays["eligible"])
    global_selection = broadcast_selection(p, selected, int(owners[-1]))
    end = time.perf_counter()
    lo, hi = map(int, owners[p.rank:p.rank+2])
    timing = dict(feature_s=p.maximum(local_features_end-start), gather_s=p.maximum(gather_end-local_features_end),
                  rpc_broadcast_s=p.maximum(end-gather_end), wall_s=p.maximum(end-start))
    if p.rank == 0:
        timing["policy_reply"] = reply
    return global_selection[lo:hi].copy(), timing


@dataclass
class DisplacementPC:
    """Numerical resources shared by matrix replay and the live OGS child PC."""
    p: object
    fine: object
    local: object
    first: object
    second: object
    space: object
    first_seed: object
    second_seed: object
    proposed_first: np.ndarray
    proposed_second: object
    numerical: dict
    timings: dict
    failure: object

    def correction(self, residual, tape=None):
        answer = self.local.apply(residual)
        answer += self.space.gain(residual - self.fine(answer), tape)
        answer += .5 * self.local.apply(residual - self.fine(answer))
        return answer

    def apply(self, source, target):
        self.p.assign(target, self.correction(self.p.array(source)))

    def close(self):
        seen = set()
        for item in (self.space, self.second, self.second_seed, self.first_seed,
                     self.first, self.local, self.fine):
            if item is not None and id(item) not in seen:
                item.close()
                seen.add(id(item))


def finish_hierarchy(config, p, legacy, bridge, fine, first, second, numerical,
                     export_level=0):
    if any(second.stats.get(key) for key in ("uncovered_free_rows", "empty_free_rows",
                                            "injection_error", "repair_budget_exhausted")):
        return None, "second_interpolation_coverage_or_injection"
    if int(second.owners[-1]) > config["terminal_max_dofs"]:
        return None, "terminal_size_limit"
    try:
        space = Hierarchy(p, legacy, first, second, config["terminal_max_dofs"])
    except ValueError as error:
        if str(error) != "terminal_condition_limit":
            raise
        return None, "terminal_condition_limit"
    numerical["condition"] = space.condition
    numerical["galerkin_errors"] = space.parity(fine)
    if export_level:
        space.direction_samples = []
    if config.get("student_cycle"):
        bridge.gnn_local_nnz.argtypes = [p.P, C.POINTER(p.D)]
        bridge.gnn_local_nnz.restype = p.I
        nnz = p.D()
        code = bridge.gnn_local_nnz(first.coarse, C.byref(nnz))
        if code:
            raise RuntimeError("Cannot query K1 nonzeros: %d" % code)
        numerical["k1_nnz"] = int(p.reduce([nnz.value])[0])
    return space, "galerkin_identity" if max(numerical["galerkin_errors"]) > 1e-9 else None


def prepare_factorized(config, p, legacy, bridge, k, layout, fields, fine, local, model,
                       train_weights=False, policy=None):
    from .factorized import InvalidFactorization, first_graph, interval, next_graph

    first = second = space = None
    proposed_first, proposed_second = None, None
    numerical, timings, failure = {}, {}, None
    level_key = "level1"
    start = time.perf_counter()
    try:
        graph, attributes, coordinates = first_graph(p, bridge, k, layout, fields)
        timings["fine_graph_s"] = p.maximum(time.perf_counter()-start)
        first = interval(config, p, legacy, bridge, k, graph, model, 0, coordinates, train_weights, policy)
        numerical["level1"] = first.stats
        proposed_first = first.executed_markers == 1
        level_key = "level2"
        start = time.perf_counter()
        graph = next_graph(p, bridge, first, attributes, coordinates)
        timings["actual_k1_graph_s"] = p.maximum(time.perf_counter()-start)
        second = interval(config, p, legacy, bridge, first.coarse, graph, model, 3, coordinates,
                          train_weights, policy)
        numerical["level2"] = second.stats
        proposed_second = second.executed_markers == 1
        space, failure = finish_hierarchy(config, p, legacy, bridge, fine, first, second, numerical)
    except InvalidFactorization as error:
        failure = str(error)
        numerical[level_key] = error.stats
    return DisplacementPC(p, fine, local, first, second, space, None, None,
                          proposed_first, proposed_second, numerical, timings, failure)


def prepare_displacement(config, p, legacy, bridge, rpc, k, layout, fields, cache,
                         neutral=False, context=None):
    fine = Action(p, k)
    lo, hi = map(int, layout["ownership"][p.rank:p.rank+2])
    native = (3*layout["native_node"] + layout["component"])[lo:hi].astype(np.int64)
    comp = layout["component"][lo:hi].astype(np.int32)
    fixed = fields["constrained"][native]
    local = legacy.LocalPC(p, k, cache, lo, hi, "ras")
    if (context or {}).get("factor_model") is not None:
        return prepare_factorized(config, p, legacy, bridge, k, layout, fields, fine, local,
                                  context["factor_model"], context.get("factor_feedback") is not None,
                                  context.get("factor_policy"))
    fast = (context or {}).get("fast_prior", False)
    if fast and (neutral or context.get("policy") != "prior_only" or context.get("training")
                 or config["decoder"]["kind"] != "directed_graph"):
        raise ValueError("The fast path requires deterministic directed-graph prior selection")
    attributes = None if fast else physical_attributes(fields, layout["xyz"], native)
    first = second = space = None
    first_seed = second_seed = None
    export_level = (context or {}).get("export_teacher_level", 0)
    models = (context or {}).get("students", {})
    numerical, timings, failure = {}, {}, None
    proposed_first, timings["first_decision"] = choose(config, p, bridge, rpc, k, native, comp,
        attributes, fixed, None if fast else np.ones(len(native)), 0, neutral, context)
    sweeps = (context or {}).get("p1_refinement_sweeps", 0)
    initial_start = time.perf_counter()
    first = Level(p, bridge, k, native, comp, proposed_first, fixed, config["interpolation"],
                  form_coarse=not (fast and (sweeps or "p1" in models)))
    if export_level == 1:
        first_seed = first
    numerical["p1_initial_s"] = p.maximum(time.perf_counter()-initial_start)
    numerical["level1"] = first.stats
    if first.stats["uncovered_free_rows"] or first.stats["empty_free_rows"] or first.stats["injection_error"] or first.stats["repair_budget_exhausted"]:
        failure = "first_interpolation_coverage_or_injection"
    if (sweeps or "p1" in models) and failure is None:
        refine_start = time.perf_counter()
        try:
            if "p1" in models:
                from .student import student_level, DEFAULT_CONTROL

                refined = student_level(config, p, legacy, bridge, k, first, comp, models["p1"], 1,
                    controls=context.get("controls_p1", DEFAULT_CONTROL), global_components=layout["component"])
            else:
                from .interpolation import enrich_first

                refined = enrich_first(p, legacy, bridge, k, first, sweeps,
                                       config["p1_refinement_row_budget"], native_scaling=fast)
        except ValueError as error:
            failure = str(error)
        else:
            if first is not first_seed:
                first.close()
            first = refined
            numerical["level1"] = first.stats
        numerical["p1_refinement_s"] = p.maximum(time.perf_counter()-refine_start)
    proposed_second = None
    if failure is None:
        pooled, mass = (None, None) if fast else pooled_attributes(p, first, attributes, bridge)
        proposed_second, timings["second_decision"] = choose(config, p, bridge, rpc, first.coarse,
            first.native, first.components, pooled, np.zeros(len(first.native), bool), mass, 1, neutral, context)
        builder = "student" if "p2" in models else (context or {}).get("p2_builder", "direct")
        if builder == "staged":
            from .interpolation import staged_level

            second, numerical["level2"], failure = staged_level(
                dict(config, p2_intermediate_radius=context["p2_radius"]),
                p, bridge, first, proposed_second, broadcast_selection)
        elif builder == "harmonic":
            from .interpolation import harmonic_level

            second, numerical["level2"], failure = harmonic_level(
                config, p, legacy, bridge, first, proposed_second,
                context["p2_entries"], broadcast_json)
        elif builder in ("direct", "weights", "student"):
            setting = dict(config["interpolation"], max_coarse_dofs=config["terminal_max_dofs"])
            if builder != "direct":
                setting["coverage_repair_fraction"] = 0.0
            second = Level(p, bridge, first.coarse, first.native, first.components, proposed_second,
                np.zeros(len(first.native), bool), setting, form_coarse=builder == "direct")
            numerical["level2"] = second.stats
            if builder != "direct":
                original = second
                if export_level == 2:
                    second_seed = original
                second = None
                bad = any(original.stats[k] for k in ("uncovered_free_rows", "empty_free_rows",
                                                       "injection_error", "repair_budget_exhausted"))
                if bad or original.owners[-1] > config["terminal_max_dofs"]:
                    failure = "p2_seed_failed"
                elif builder == "weights":
                    from .interpolation import refine_second

                    second, numerical["level2"] = refine_second(p, legacy, bridge, first, original,
                        context["p2_expand_budget"], broadcast_json)
                    if second is None:
                        failure = "p2_weight_optimization_failed"
                else:
                    from .student import student_level, DEFAULT_CONTROL

                    try:
                        second = student_level(config, p, legacy, bridge, first.coarse, original,
                            first.components, models["p2"], 2, controls=context.get("controls_p2", DEFAULT_CONTROL))
                    except ValueError as error:
                        failure = str(error)
                    else:
                        numerical["level2"] = second.stats
                if original is not second_seed:
                    original.close()
        else:
            raise ValueError("Unknown P2 builder: " + builder)
        if second is None:
            assert failure is not None
        else:
            space, failure = finish_hierarchy(config, p, legacy, bridge, fine, first, second,
                                              numerical, export_level)
    return DisplacementPC(p, fine, local, first, second, space, first_seed, second_seed,
                          proposed_first, proposed_second, numerical, timings, failure)


def solve_policy(config, p, replay, legacy, bridge, rpc, row, layout, fields, data,
                 neutral=False, context=None, capture=None):
    md, rows, full, rhs, initial = data
    solution = p.vector(full)
    if md["configuration"]["initial_guess_nonzero"]:
        p.call("VecCopy", initial, solution)
    rhs_norm = p.norm(rhs)
    p.barrier()
    begin = time.perf_counter()
    outer, fs, options = replay.create_solver(p, full, md, config["max_iterations"],
        overrides={"fieldsplit_1_pc_type": "none"})
    assert options["ksp_type"] == "gmres" and options["ksp_pc_side"] == "right"
    assert options["pc_type"] == "fieldsplit" and options["pc_fieldsplit_type"] == "multiplicative"
    assert float(options["ksp_rtol"]) == config["rtol"] and float(options["ksp_atol"]) == config["atol"]
    p.call("PCSetUp", fs)
    child, original, operator, k = replay.pcapply_child(p, fs, rows, expected_pc=b"none")
    cache = Path(config["baseline_directory"]) / "cache" / row["case"]
    displacement = prepare_displacement(config, p, legacy, bridge, rpc, k, layout, fields,
                                        cache, neutral, context)
    context = context or {}
    probe_windows = context.get("factor_probe_windows")
    probe_inputs = context.get("factor_probe_inputs")
    probe_capture = context.get("factor_probe_capture")
    signatures = context.get("factor_signatures")
    signature = None
    duplicate = False
    if signatures is not None and displacement.failure is None:
        from .factorized import hierarchy_signature

        signature = hierarchy_signature(displacement.numerical)
        duplicate = signature in signatures
        signatures.add(signature)
    if duplicate or context.get("factor_policy_replay"):
        try:
            if duplicate:
                return dict(duplicate=True, hierarchy_signature=signature,
                            numerical=displacement.numerical)
            return dict(factor_policy=context["factor_policy"].receipt(),
                        failure=displacement.failure, numerical=displacement.numerical)
        finally:
            p.destroy("KSP", outer)
            displacement.close()
            p.destroy("Vec", solution)
    first, second, space = displacement.first, displacement.second, displacement.space
    first_seed, second_seed = displacement.first_seed, displacement.second_seed
    proposed_first, proposed_second = displacement.proposed_first, displacement.proposed_second
    numerical, timings, failure = displacement.numerical, displacement.timings, displacement.failure
    lo, hi = map(int, layout["ownership"][p.rank:p.rank+2])
    native = (3*layout["native_node"] + layout["component"])[lo:hi].astype(np.int64)
    comp = layout["component"][lo:hi].astype(np.int32)
    export_level = (context or {}).get("export_teacher_level", 0)
    models = (context or {}).get("students", {})
    shell = callback = None
    apply_times = []
    observed = (context or {}).get("capture_inputs")
    short_lengths = (context or {}).get("short_krylov")
    factor_feedback = (context or {}).get("factor_feedback")
    capture_probes = probe_capture is not None or (factor_feedback is not None and probe_inputs is None)
    residual_samples = {} if short_lengths or capture_probes else None
    if failure is None:
        action = displacement.apply
        if observed is not None:
            def action(source, target):
                call = len(apply_times)
                if call in (0, 8, 24, 48) or call % 32 == 0:
                    residual = p.array(source)
                    if call >= 64 and observed and observed[-1][0] >= 64:
                        observed[-1] = call, residual
                    else:
                        observed.append((call, residual))
                displacement.apply(source, target)
        shell, callback = replay.pcapply_shell(p, child, operator, k, action, apply_times)
        replay.setup_solver(p, outer, fs, options)
        setup_s = p.maximum(time.perf_counter()-begin)
        apply_times.clear()
        if observed is not None:
            observed.clear()
        if export_level:
            space.direction_samples.clear()
            space.direction_calls = 0
        if context.get("factor_probe_only"):
            from .krylov import probe_trajectories

            started = time.perf_counter()
            try:
                return dict(trajectories=probe_trajectories(p, full, fs, probe_inputs, probe_windows),
                            probe_s=p.maximum(time.perf_counter()-started), failure=None)
            finally:
                p.destroy("KSP", outer)
                p.destroy("PC", shell)
                displacement.close()
                p.destroy("Vec", solution)
        if residual_samples is not None or context.get("factor_policy") is not None or \
                config["stage"] == "rl" or config.get("student_cycle") in (
                "tune", "factor_tune", "factor_search"):
            result = solve_with_deadline(p, replay, outer, full, rhs, solution,
                config["max_iterations"], max(config["atol"], config["rtol"]*rhs_norm),
                begin+config["training"]["episode_timeout_seconds"], residual_samples)
        else:
            result = legacy.solve_with_true_residual(p, replay, outer, full, rhs, solution,
                config["max_iterations"], max(config["atol"], config["rtol"]*rhs_norm), True)
        result["total_wall_s"] = p.maximum(result.pop("_solve_end_time")-begin)
        failure = "episode_timeout" if result.get("timed_out") else (None if result["reached_true_target"] else "gmres_true_residual")
    else:
        setup_s = p.maximum(time.perf_counter()-begin)
        result = dict(reached_true_target=False, total_wall_s=setup_s)
    full_apply_count, full_apply_seconds = len(apply_times), sum(apply_times)
    if context.get("factor_probe_only"):
        p.destroy("KSP", outer)
        displacement.close()
        p.destroy("Vec", solution)
        return dict(trajectories={}, failure=failure, probe_s=0.)
    probes_short = probe_inputs
    if space is not None and residual_samples:
        from .krylov import verify_middle_ilu, verify_short_segments

        actual = {iteration:values for iteration, values in residual_samples.values()}
        probes_short = [("iteration-%d" % i, actual[i]) for i in sorted(actual)]
        # Component-local directions supplement the actual early and late
        # residuals. They have the full system layout and fixed physical IDs.
        parent_start = ownership(p, full)[p.rank]
        for component in range(3):
            direction = np.zeros_like(next(iter(actual.values())))
            chosen = (comp == component) & (fields["phi"][native//3] < .95)
            direction[rows[chosen]-parent_start] = np.sin(.037*(native[chosen]+1))
            probes_short.append(("displacement-%d" % component, direction))
        if short_lengths:
            result["short_krylov"] = verify_short_segments(p, full, fs, probes_short, short_lengths)
            result["middle_ilu_adjoint_parity"] = verify_middle_ilu(p, bridge, first, space.middle)
        if probe_capture is not None or (factor_feedback or {}).get("quality_targets") is not None:
            late = residual_samples.get("late")
            probes_short = [pair for pair in probes_short if pair[0] == "iteration-24"
                            or pair[0].startswith("displacement-")]
            if late is not None and late[0] > 24:
                probes_short.append(("late", late[1]))
            if probe_capture is not None:
                probe_capture.extend(probes_short)
    if probes_short and failure is None and (factor_feedback is not None or probe_windows):
        from .krylov import coefficient_feedback, probe_trajectories

        started = time.perf_counter()
        if factor_feedback is not None:
            result["coefficient_feedback"] = coefficient_feedback(
                p, legacy, bridge, full, fs, shell, callback, displacement, context["factor_model"],
                probes_short, factor_feedback)
        if probe_windows:
            result["factor_trajectories"] = probe_trajectories(p, full, fs, probes_short, probe_windows)
        result["factor_probe_s"] = p.maximum(time.perf_counter()-started)
    if signature is not None:
        result["hierarchy_signature"] = signature
    probes = (context or {}).get("probe_inputs")
    if probes and space is not None and failure != "galerkin_identity":
        ratios = []
        for call, residual in probes:
            difference = residual - displacement.fine(displacement.correction(residual))
            error, norm = p.reduce([difference @ difference, residual @ residual])
            if norm > 0 and np.isfinite(error):
                ratios.append(dict(call=call, residual_ratio=float(np.sqrt(error/norm))))
        result["student_cycle_probes"] = ratios
    decoder_name = "graph" if config.get("decoder", {}).get("kind") == "directed_graph" else "spatial"
    factorized = (context or {}).get("factor_model") is not None
    method = "gnn_factorized" if factorized else ("neutral_" if neutral else "gnn_")+decoder_name
    result.update(method=method, failure=failure,
        setup_wall_s=setup_s, numerical=numerical, timings=timings,
        displacement_pc_apply_s=p.maximum(full_apply_seconds), displacement_pc_apply_count=full_apply_count,
        gnn_inference=factorized or bool(models) or (not neutral and (context or {}).get("policy") != "prior_only"),
        interpolation_students={name:model["sha256"] for name,model in models.items()},
        checkpoint_sha256=(timings.get("first_decision", {}).get("policy_reply") or {}).get("checkpoint_sha256"),
        teacher_hierarchy_read=False,
        policy_update=(context or {}).get("update", 0),
        checkpoint_scope="Initial checkpoint; policy_update identifies subsequent on-policy updates"
            if (context or {}).get("training") else "Frozen inference checkpoint",
        timing_scope=("Fresh graph, all five selections and transfers, products, Galerkin, "
                      "RAS, audits, and full true-residual solve. Loaded weights and replay "
                      "input are outside timing." if factorized else
                      "Fresh features, policy decisions, repair, preconditioner construction "
                      "and complete true-residual solve. Algebra audits included. Replay input "
                      "loading and artifact logging excluded."))
    policy = (context or {}).get("factor_policy")
    if policy is not None:
        result["factor_policy"] = policy.receipt()
    if capture is not None and result["reached_true_target"]:
        for name, matrix in (("P1", first.transfer), ("K1", first.coarse), ("P2", second.transfer)):
            values, _ = local_csr(p, matrix, bridge)
            values.eliminate_zeros()
            values.sort_indices()
            capture[name] = values
    if export_level and result["reached_true_target"]:
        from .student import export_teacher

        q = [directions[export_level-1] for directions in space.direction_samples]
        if export_level == 1:
            receipt = export_teacher(config, p, legacy, bridge, row, k, first_seed, first, comp, q, 1,
                native, layout["component"], fields["phi"][native//3])
        else:
            receipt = export_teacher(config, p, legacy, bridge, row, first.coarse, second_seed, second,
                first.components, q, 2, first.native)
        if p.rank == 0:
            result["student_teacher"] = receipt
    # Archive raw/executed choices after the measured numerical work.
    arrays = {}
    if (context or {}).get("archive_choices", True):
        arrays = dict(native=gather_array(p, native, layout["ownership"]).astype(np.int64),
            raw_first=gather_array(p, proposed_first, layout["ownership"]).astype(bool),
            executed_first=gather_array(p, first.executed_markers, layout["ownership"]).astype(np.int8))
        if proposed_second is not None:
            arrays.update(coarse_native=gather_array(p, first.native, first.owners).astype(np.int64),
                raw_second=gather_array(p, proposed_second, first.owners).astype(bool))
            if second is not None:
                arrays["executed_second"] = gather_array(p, second.executed_markers, first.owners).astype(np.int8)
        if p.rank == 0:
            run = os.environ.get("GNN_RUN_ID", os.environ.get("SLURM_JOB_ID", "local"))
            path = Path(config["output_directory"]) / "data" / ("prediction-%s-%s-%s%s.npz" % (run, row["key"], result["method"], "-"+context["tag"] if context else ""))
            with path.open("xb") as f:
                np.savez_compressed(f, **arrays)
            result["raw_executed_choices"] = dict(path=str(path), sha256=digest(path))
    if result["reached_true_target"] and (context or {}).get("export_p2_sample"):
        from .interpolation import collect_csr

        _, ki, kj, kv = collect_csr(p, bridge, first.coarse)
        _, pi, pj, pv = collect_csr(p, bridge, second.transfer)
        if p.rank == 0:
            name = "interpolation-%s-%s-%s.npz" % (run, row["key"], context["tag"])
            sample = Path(config["output_directory"]) / "data" / name
            with sample.open("xb") as stream:
                np.savez_compressed(stream, k_indptr=ki, k_indices=kj, k_values=kv,
                                    p_indptr=pi, p_indices=pj, p_values=pv,
                                    native=arrays["coarse_native"], c2=arrays["raw_second"])
            result["interpolation_sample"] = dict(path=str(sample), sha256=digest(sample))
    p.destroy("KSP", outer)
    if shell is not None:
        p.destroy("PC", shell)
    displacement.close()
    p.destroy("Vec", solution)
    return result


def split_solver_world(config, p):
    """Independent 64-rank environments inside one Slurm/MPI allocation."""
    workers = int(config.get("parallel_workers", 1))
    p.global_world, p.global_rank, p.global_size = p.P(p.world.value), p.rank, p.size
    p.worker = p.rank // config["mpi_ranks"]
    assert p.size == workers*config["mpi_ranks"]
    if workers > 1:
        sub = p.P()
        p.mpi.MPI_Comm_split.argtypes = [p.P, p.I, p.I, C.POINTER(p.P)]
        p.mpi.MPI_Comm_split.restype = p.I
        assert p.mpi.MPI_Comm_split(p.global_world, p.worker, p.rank, C.byref(sub)) == 0
        p.world = sub
        rank, size = p.I(), p.I()
        assert p.mpi.MPI_Comm_rank(p.world, C.byref(rank)) == 0
        assert p.mpi.MPI_Comm_size(p.world, C.byref(size)) == 0
        p.rank, p.size = rank.value, size.value
        assert p.size == config["mpi_ranks"]
    return workers


def global_objects(p, value):
    """Exchange small root-only control/gradient payloads, never numerical matrices."""
    payload = json.dumps(value, allow_nan=False).encode() if p.rank == 0 else b""
    count = np.asarray([len(payload)], np.int32)
    counts = np.empty(p.global_size, np.int32)
    mpi_int = p.P(C.addressof((C.c_char*1).in_dll(p.mpi, "ompi_mpi_int")))
    mpi_byte = p.P(C.addressof((C.c_char*1).in_dll(p.mpi, "ompi_mpi_byte")))
    assert p.mpi.MPI_Allgather(count.ctypes.data, 1, mpi_int, counts.ctypes.data, 1, mpi_int, p.global_world) == 0
    offsets = np.r_[0, np.cumsum(counts[:-1])].astype(np.int32)
    source = np.frombuffer(payload, np.uint8)
    target = np.empty(int(counts.sum()), np.uint8)
    assert p.mpi.MPI_Allgatherv(source.ctypes.data, len(source), mpi_byte, target.ctypes.data,
        counts.ctypes.data, offsets.ctypes.data, mpi_byte, p.global_world) == 0
    return [json.loads(target[offset:offset+n].tobytes()) for offset, n in zip(offsets, counts) if n]


def broadcast_json(p, value):
    payload = json.dumps(value, allow_nan=False).encode() if p.rank == 0 else b""
    count = int(p.reduce([len(payload)])[0])
    buffer = np.frombuffer(payload, dtype=np.uint8).copy() if p.rank == 0 else np.empty(count, np.uint8)
    mpi_byte = p.P(C.addressof((C.c_char*1).in_dll(p.mpi, "ompi_mpi_byte")))
    p.mpi.MPI_Bcast.argtypes = [p.P, p.I, p.P, p.I, p.P]
    p.mpi.MPI_Bcast.restype = p.I
    assert p.mpi.MPI_Bcast(buffer.ctypes.data, count, mpi_byte, 0, p.world) == 0
    return json.loads(buffer.tobytes())


def load_observed_state(config, p, replay, row):
    old = Path(config["baseline_directory"])
    if p.rank == 0:
        manifest = read_json(Path(row["directory"]) / "manifest.json")
        identities = {f["path"]: f for f in manifest["files"]}
        for name in ("A.bin", "b.bin", "x0.bin"):
            assert digest(Path(row["directory"]) / name, "sha1") == identities[name]["sha1"]
    p.barrier()
    with np.load(old / "cache" / row["case"] / "layout.npz") as f:
        layout = {name: f[name] for name in f.files}
    receipt = read_json(old / "records" / ("fields-"+row["key"]+".json"))
    assert receipt["manifest"] == row["manifest_sha256"]
    assert digest(old / "data" / row["key"] / "fields.npz") == receipt["sha256"]
    with np.load(old / "data" / row["key"] / "fields.npz") as f:
        fields = {name: f[name] for name in f.files}
    return layout, fields, replay.pcapply_snapshot(p, row, layout)


def training_loop(config, p, replay, legacy, bridge, rpc, inventory):
    from .gate import run_method
    setting = config["training"]
    workers = int(config.get("parallel_workers", 1))
    if workers > 1:
        assert workers == setting["batch_states"] == 3
    old, output = Path(config["baseline_directory"]), Path(config["output_directory"])
    job = os.environ.get("GNN_RUN_ID", os.environ.get("SLURM_JOB_ID", "local"))
    receipt = read_json(config["prediction_acceptance"])
    assert receipt["passed"] and receipt["method_totals"]["gnn_graph"]["accurate"] == 3
    assert digest(config["prediction_acceptance_config"]) == receipt["config_sha256"]
    accepted = read_json(config["prediction_acceptance_config"])
    for key in ("decoder", "interpolation", "terminal_max_dofs", "max_iterations", "rtol", "atol"):
        assert config[key] == accepted[key], ("Numerical policy changed after acceptance", key)
    assert config["policy_checkpoint_sha256"] == accepted["policy_checkpoint_sha256"] or setting.get("resume_verified")
    training = [row for row in inventory["snapshots"] if row["split"] == "training"]
    validation = [row for row in inventory["snapshots"] if row["split"] == "validation"]
    smoke = setting.get("smoke", False)
    if smoke:
        training = [row for row in training if row["key"] in config["gate_keys"]]
        evaluation = training
    else:
        evaluation = [next(row for row in validation if row["key"] == key) for key in setting["validation_keys"]]
    assert 2 <= setting["batch_states"] <= 4 and len(training) >= setting["batch_states"]
    assert len({row["key"] for row in evaluation}) == len(evaluation)
    assignment = {row["key"]: i % workers for i, row in enumerate(training)}
    for i, row in enumerate(evaluation):
        assignment.setdefault(row["key"], i % workers)
    lanes = [[row for row in training if assignment[row["key"]] == i] for i in range(workers)]
    assert all(lanes)
    started = time.perf_counter()
    deadline = started + setting["hours"]*3600
    references, reference_details, updates, checks = {}, {}, [], []
    def save_result(row, tag, result):
        if p.rank == 0:
            result["worker"] = p.worker
            result["solver_ranks"] = p.size
            path = output / "records" / ("rl-%s-%s-%s.json" % (job, row["key"], tag))
            with path.open("x") as f:
                json.dump(result, f, allow_nan=False, indent=2)
    def close_data(data):
        p.destroy("Vec", data[3], data[4])
        p.destroy("Mat", data[2])
    def command(request):
        answer = rpc.request(request)[0] if p.rank == 0 else None
        return broadcast_json(p, answer)
    # Reference is standard AMG, available for every training state. Retained
    # base is additionally measured on validation wherever its accepted export
    # exists. References are constants; no training gradient enters them.
    reference_rows = {row["key"]: row for row in training+evaluation}
    for row in reference_rows.values():
        if assignment[row["key"]] != p.worker:
            continue
        layout, fields, data = load_observed_state(config, p, replay, row)
        result = run_method(config, p, replay, legacy, bridge, row, layout, data, "amg", None)
        assert result["reached_true_target"] and not result.get("timed_out"), "Reference AMG failed"
        references[row["key"]] = result["total_wall_s"]
        reference_details[row["key"]] = dict(amg_seconds=result["total_wall_s"])
        save_result(row, "reference-amg", result)
        if row in evaluation:
            prefix = config.get("retained_export_prefixes", {}).get(row["split"], "retained-solver")
            export_path = old / "records" / ("solver-export-"+prefix+"-"+row["key"]+".json")
            if row["split"] == "validation" and not export_path.exists():
                raise FileNotFoundError("Required retained validation comparator: "+str(export_path))
            if export_path.exists():
                export = read_json(export_path)
                teacher = dict(retained_entry=next(e for e in export["spaces"] if e["label"] == config["baseline_label"]))
                retained = run_method(config, p, replay, legacy, bridge, row, layout, data, "retained_base", teacher)
                assert retained["reached_true_target"] and not retained.get("timed_out")
                reference_details[row["key"]]["retained_seconds"] = retained["total_wall_s"]
                save_result(row, "reference-retained", retained)
        close_data(data)
        command(dict(operation="ping"))
        if p.rank == 0:
            record(config, "rl_reference_complete", key=row["key"], **reference_details[row["key"]])
    if workers > 1:
        collected = global_objects(p, dict(references=references, details=reference_details))
        references = {key: value for part in collected for key, value in part["references"].items()}
        reference_details = {key: value for part in collected for key, value in part["details"].items()}
        assert len(references) == len(reference_rows)
    scale = setting.get("common_time_scale_s") or float(np.mean([references[row["key"]] for row in training]))
    if p.rank == 0:
        record(config, "rl_references_ready", job=job, common_scale=scale, count=len(references),
            training_count=len(training), evaluation_split="training-smoke" if smoke else "validation",
            references=reference_details, budget_seconds=setting["hours"]*3600)
    def evaluate(update, tag):
        cases = []
        for row in evaluation:
            if assignment[row["key"]] != p.worker:
                continue
            layout, fields, data = load_observed_state(config, p, replay, row)
            result = solve_policy(config, p, replay, legacy, bridge, rpc, row, layout, fields, data,
                context=dict(key=row["key"], training=False, tag=tag, update=update))
            save_result(row, tag, result)
            cases.append(dict(key=row["key"], split=row["split"], seconds=result["total_wall_s"],
                accurate=result["reached_true_target"] and result["failure"] is None,
                failure=result["failure"], iterations=result.get("iterations"),
                amg_seconds=references[row["key"]], retained_seconds=reference_details[row["key"]].get("retained_seconds")))
            close_data(data)
        if workers > 1:
            pieces = global_objects(p, cases)
            by_key = {case["key"]: case for part in pieces for case in part}
            assert len(by_key) == len(evaluation)
            cases = [by_key[row["key"]] for row in evaluation]
        check = command(dict(operation="validation", cases=cases))
        checks.append(check)
        return check
    if not smoke:
        evaluate(setting.get("initial_updates", 0), "initial-evaluation")
    stop_reason = "update_budget"
    while True:
        remaining = p.maximum(float(time.perf_counter() + (len(evaluation)+setting["batch_states"])*setting["episode_timeout_seconds"]+60 > deadline))
        if workers > 1:
            remaining = any(global_objects(p, bool(remaining)))
        if remaining:
            stop_reason = "wall_clock_budget"
            break
        index = setting.get("initial_updates", 0) + len(updates)
        # Uniform without-replacement batches. Seed+global update makes exact
        # state sampling resumable without persisting a second hidden RNG.
        rng = np.random.default_rng(setting["seed"]+index)
        if workers > 1:
            batch = [lane[int(rng.integers(len(lane)))] for lane in lanes]
        else:
            batch = [training[int(i)] for i in rng.choice(len(training), setting["batch_states"], replace=False)]
        start = command(dict(operation="begin_batch", keys=[row["key"] for row in batch],
            common_scale=scale, reference_times=references))
        for row in batch:
            if workers > 1 and assignment[row["key"]] != p.worker:
                continue
            layout, fields, data = load_observed_state(config, p, replay, row)
            tag = "update%06d" % start["update"]
            result = solve_policy(config, p, replay, legacy, bridge, rpc, row, layout, fields, data,
                context=dict(key=row["key"], training=True, tag=tag, update=start["policy_version"]))
            save_result(row, tag, result)
            command(dict(operation="feedback", key=row["key"], seconds=result["total_wall_s"],
                converged=result["reached_true_target"] and result["failure"] is None,
                failure=result["failure"], training_pool_size=len(training),
                sampling_probability=1/(workers*len(lanes[p.worker])) if workers > 1 else 1/len(training),
                importance=workers*len(lanes[p.worker])/len(training) if workers > 1 else 1.0))
            close_data(data)
        if workers > 1:
            partial = command(dict(operation="partial_gradient"))
            pieces = global_objects(p, partial)
            assert len(pieces) == workers
            assert len({piece["parameter_sha256"] for piece in pieces}) == 1
            assert all(piece["names"] == pieces[0]["names"] and piece["update"] == pieces[0]["update"] for piece in pieces)
            gradient = np.sum(np.asarray([piece["gradient"] for piece in pieces], np.float64), axis=0)
            observations = [ob for piece in pieces for ob in piece["observations"]]
            command(dict(operation="merge_gradient", names=pieces[0]["names"], gradient=gradient.tolist(),
                parameter_sha256=pieces[0]["parameter_sha256"], update=pieces[0]["update"], observations=observations))
        update = command(dict(operation="commit"))
        if workers > 1:
            replicas = global_objects(p, dict(worker=p.worker, parameter_sha256=update["parameter_sha256"], actions=update["actions"]))
            assert len({replica["parameter_sha256"] for replica in replicas}) == 1, "Worker policies diverged"
            update["worker_actions"] = replicas
        updates.append(update)
        if len(updates) == 1:
            assert update["parameter_delta_norm"] > 0 and update["gradient_norm"] > 0, "No actual policy update"
        if not smoke and update["update"] % setting["validation_interval"] == 0:
            evaluate(update["update"], "validation%06d" % update["update"])
        if setting.get("max_updates", 0) and len(updates) >= setting["max_updates"]:
            break
    if updates:
        final_check = evaluate(updates[-1]["update"], "final-evaluation")
    else:
        final_check = checks[-1] if checks else None
    final = rpc.close() if p.rank == 0 else None
    final = broadcast_json(p, final)
    worker_finals = global_objects(p, dict(worker=p.worker, checkpoint=final["checkpoint"])) if workers > 1 else [dict(worker=0, checkpoint=final["checkpoint"])]
    summary = dict(job=job, parallel_workers=workers, total_cpu_ranks=p.global_size, worker_checkpoints=worker_finals, state="complete", smoke=smoke, stop_reason=stop_reason,
        elapsed_seconds=p.maximum(time.perf_counter()-started), updates=len(updates),
        total_updates=updates[-1]["update"] if updates else setting.get("initial_updates", 0),
        common_time_scale_s=scale, training_states=len(training), evaluation_states=len(evaluation),
        evaluation_split="training-smoke" if smoke else "validation", final_checkpoint=final["checkpoint"],
        final_check=final_check, validation_checks=checks,
        real_updates=[dict(update=u["update"], phase=u["phase"], gradient_norm=u["gradient_norm"],
            parameter_delta_norm=u["parameter_delta_norm"], observations=u["observations"], actions=u["actions"],
            worker_actions=u.get("worker_actions")) for u in updates],
        next="Inspect exact checkpoint resume and residuals, then start24hRL" if smoke else "Assess cumulative validation cost and convergence before autonomous continuation")
    if p.global_rank == 0:
        (output / "records" / ("rl-"+job+".json")).write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        record(config, "rl_complete", job=job, smoke=smoke, updates=len(updates),
            seconds=summary["elapsed_seconds"], checkpoint=final["checkpoint"], final_check=final_check)
    # A failed training candidate is valid RL feedback. A smoke is accepted
    # only if the changed deterministic policy still solves all three states.
    return 0 if updates and (not smoke or final_check["failures"] == 0) else 2


def review(config, p, replay, legacy, bridge, rpc, inventory):
    """Compare retained, fixed teacher, and deployed policies on training frames."""
    from .gate import load_teacher, run_method

    workers = config["parallel_workers"]
    validation = config.get("interpolation_validation", False)
    keys = config["training"]["validation_keys"] if validation else config["gate_keys"]
    if not validation:
        assert len(keys) == workers
    split = "validation" if validation else "training"
    rows = {row["key"]: row for row in inventory["snapshots"] if row["split"] == split}
    assert len(keys) == len(set(keys)) and all(key in rows for key in keys)
    output = Path(config["output_directory"])
    old = Path(config["baseline_directory"])
    job = os.environ["GNN_RUN_ID"]

    def save(result, row, label):
        result.update(key=row["key"], label=label, worker=p.worker, solver_ranks=p.size)
        if p.rank == 0:
            suffix = row["key"]+"-"+label if validation else label
            path = output / "records" / ("review-%s-w%d-%s.json" % (job, p.worker, suffix))
            with path.open("x") as stream:
                json.dump(result, stream, allow_nan=False, indent=2)
            record(config, "review_solve", key=row["key"], method=label,
                   worker=p.worker, accurate=result["reached_true_target"],
                   iterations=result.get("iterations"), seconds=result.get("total_wall_s"),
                   failure=result["failure"])
        return result

    def destroy(data):
        p.destroy("Vec", data[3], data[4])
        p.destroy("Mat", data[2])

    if validation:
        fast_comparison = config.get("interpolation_fast_validation", False)
        results = []
        for index, key in enumerate(keys):
            if index % workers != p.worker:
                continue
            row = rows[key]
            layout, fields, data = load_observed_state(config, p, replay, row)
            # The frozen parameters come from training-only comparisons.
            # Alternate order within a frame to reduce timing-order effects.
            paired = ([("p1_reference", 2), ("p1_fast", 2)] if fast_comparison
                      else [("prior_only", 0), ("prior_p1_twice", 2)])
            if index % 2:
                paired.reverse()
            matrices, pair = {}, {}
            for label, sweeps in paired:
                capture = {} if fast_comparison else None
                pair[label] = solve_policy(config, p, replay, legacy, bridge, rpc, row, layout, fields, data,
                    context=dict(policy="prior_only", tag=label, p1_refinement_sweeps=sweeps,
                                 fast_prior=label == "p1_fast"), capture=capture)
                matrices[label] = capture
            if fast_comparison:
                from .interpolation import compare_transfers

                pair["p1_fast"]["matrix_comparison"] = compare_transfers(
                    p, matrices["p1_reference"], matrices["p1_fast"])
            for label, result in pair.items():
                results.append(save(result, row, label))
            if not fast_comparison:
                prefix = config["retained_export_prefixes"]["validation"]
                export = read_json(old / "records" / ("solver-export-"+prefix+"-"+key+".json"))
                teacher = dict(retained_entry=next(e for e in export["spaces"] if e["label"] == config["baseline_label"]))
                for label in ("retained_base", "amg"):
                    result = run_method(config, p, replay, legacy, bridge, row, layout, data, label, teacher)
                    results.append(save(result, row, label))
            destroy(data)
        if p.rank == 0:
            rpc.close()
        pieces = global_objects(p, dict(worker=p.worker, results=results))
        if p.global_rank == 0:
            path = output / "records" / ("review-%s.json" % job)
            with path.open("x") as stream:
                json.dump(dict(job=job, split=split, workers=pieces), stream, allow_nan=False, indent=2)
        labels = ("p1_reference", "p1_fast") if fast_comparison else ("prior_only", "retained_base", "amg")
        controls = [r for item in pieces for r in item["results"] if r["label"] in labels]
        return 0 if len(controls) == len(labels)*len(keys) and all(r["reached_true_target"] for r in controls) else 2

    parity = None
    if not config.get("interpolation_review"):
        # One identical graph, score vector and frame per group. Deliberately keep
        # the expensive Python equality check out of timed method comparisons.
        probe = rows[keys[0]]
        layout, fields, data = load_observed_state(config, p, replay, probe)
        parity = solve_policy(config, p, replay, legacy, bridge, rpc, probe, layout, fields, data,
            context=dict(policy="pretrained", check_decoder=True, tag="parity-w%d" % p.worker))
        save(parity, probe, "decoder_parity")
        destroy(data)

    row = rows[keys[(p.worker + config.get("review_frame_offset", 0)) % workers]]
    layout, fields, data = load_observed_state(config, p, replay, row)
    teacher = load_teacher(config, p, row, layout)
    entry = teacher["retained_entry"]
    with np.load(old / "cache" / row["key"] / (entry["shard_pattern"] % p.rank)) as shard:
        teacher["baseline_empty"] = np.diff(shard["T0_indptr"]) == 0

    if not config.get("interpolation_review"):
        # Use the previously accepted executed endpoints at both levels so that a
        # changed strength topology at K1 cannot silently change the comparison.
        fixed_job = config["teacher_endpoints_job"]
        suffix = "%s-%s-r%03d.npz" % (fixed_job, row["key"], p.rank)
        lo, hi = layout["ownership"][p.rank:p.rank+2]
        with np.load(output / "data" / ("repair-" + suffix)) as first:
            np.testing.assert_array_equal(first["native"], teacher["native"][lo:hi])
            np.testing.assert_array_equal(first["raw_selected"], teacher["depth"][lo:hi] >= 3)
            teacher["first_mask"] = first["executed_markers"] == 1
        with np.load(output / "data" / ("repair-second-" + suffix)) as second:
            np.testing.assert_array_equal(second["native"], teacher["native"][lo:hi][teacher["first_mask"]])
            np.testing.assert_array_equal(second["raw_selected"], np.isin(second["native"], teacher["terminal_native"]))
            local_terminal = second["native"][second["executed_markers"] == 1].tolist()
        terminal = global_teacher_ids(p, local_terminal)
        teacher["terminal_native"] = np.asarray(terminal, dtype=np.int64)
        assert len(terminal) <= config["terminal_max_dofs"]

    methods = list(config["review_methods"])
    rotate = (p.worker * 3) % len(methods)
    results = []
    for name in methods[rotate:] + methods[:rotate]:
        if name in ("amg", "retained_base", "teacher_cached_first", "teacher_cached_first_repaired",
                    "teacher_cross", "teacher_cached_coverage"):
            result = run_method(config, p, replay, legacy, bridge, row, layout, data, name, teacher)
        elif name in config["teacher_variants"]:
            setting = dict(config["interpolation"], coverage_repair_fraction=0.0)
            setting.update(config["teacher_variants"][name])
            result = run_method(dict(config, interpolation=setting), p, replay, legacy, bridge,
                                row, layout, data, "teacher_multipass_repaired", teacher)
            result["interpolation"] = setting
            result["fixed_endpoints_job"] = fixed_job
        else:
            context = dict(policy=name, tag=name,
                           export_p2_sample=name in config.get("export_p2_samples", []))
            if name in config.get("p2_weight_variants", {}):
                context.update(policy="prior_only", p2_builder="weights",
                               p2_expand_budget=config["p2_weight_variants"][name])
            if name in config.get("fast_prior_methods", []):
                context.update(policy="prior_only", fast_prior=True)
            if name in config.get("p2_staged_variants", {}):
                context.update(policy="prior_only", p2_builder="staged",
                               p2_radius=config["p2_staged_variants"][name])
            if name in config.get("p2_harmonic_variants", {}):
                context.update(policy="prior_only", p2_builder="harmonic",
                               p2_entries=config["p2_harmonic_variants"][name])
            if name in config.get("p1_refinement_variants", {}):
                context.update(policy="prior_only",
                               p1_refinement_sweeps=config["p1_refinement_variants"][name])
            result = solve_policy(config, p, replay, legacy, bridge, rpc, row, layout, fields, data,
                                  context=context)
        results.append(save(result, row, name))
    destroy(data)
    if p.rank == 0:
        rpc.close()
    packet = dict(worker=p.worker, binding=rpc.binding if p.rank == 0 else None,
                  parity=parity, results=results)
    pieces = global_objects(p, packet)
    if p.global_rank == 0:
        with (output / "records" / ("review-%s.json" % job)).open("x") as stream:
            json.dump(dict(job=job, workers=pieces), stream, allow_nan=False, indent=2)
    # Policy or weight variants can fail numerically. Preserve and report those
    # outcomes without losing the diagnostic job and its successful controls.
    labels = ("retained_base", "prior_only") if config.get("interpolation_review") else (
        "amg", "retained_base", "teacher_current")
    controls = [r for item in pieces for r in item["results"] if r["label"] in labels]
    return 0 if len(controls) == len(labels)*workers and all(r["reached_true_target"] for r in controls) else 2


def global_teacher_ids(p, local_ids):
    """Share the tiny fixed teacher terminal set inside a solver group."""
    data = np.asarray(local_ids, dtype=np.int64)
    counts = p.gather_equal([len(data)]).ravel().astype(np.int32)
    offsets = np.r_[0, np.cumsum(counts[:-1])].astype(np.int32)
    collected = np.empty(int(counts.sum()), dtype=np.int64)
    mpi_int64 = p.P(C.addressof((C.c_char*1).in_dll(p.mpi, "ompi_mpi_int64_t")))
    assert p.mpi.MPI_Allgatherv(data.ctypes.data, len(data), mpi_int64, collected.ctypes.data,
                              counts.ctypes.data, offsets.ctypes.data, mpi_int64, p.world) == 0
    return collected.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("endpoint", type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    replay, legacy = load_dependencies(config)
    p = replay.Petsc()
    workers = split_solver_world(config, p)
    if workers > 1:
        config["progress_name"] = "progress-%s-worker%d.jsonl" % (os.environ.get("SLURM_JOB_ID", "local"), p.worker)
    replay.pcapply_bind(p)
    assert p.size == config["mpi_ranks"]
    if config["stage"] != "parallel_smoke":
        assert p.size == 64
    bridge = load_bridge(p, config["bridge"])
    old, output = Path(config["baseline_directory"]), Path(config["output_directory"])
    job = os.environ.get("GNN_RUN_ID", os.environ.get("SLURM_JOB_ID", "local"))
    rpc = None
    try:
        from .gate import numerical_smoke, run_method
        numerical_smoke(config, p, legacy, bridge)
        if config["stage"] == "parallel_smoke":
            receipts = global_objects(p, dict(worker=p.worker, ranks=p.size, value=p.maximum(p.worker+1)))
            assert len(receipts) == workers and [r["value"] for r in receipts] == list(range(1, workers+1))
            if p.global_rank == 0:
                record(config, "parallel_communicators_passed", total_ranks=p.global_size, workers=receipts)
            p.call("PetscFinalize")
            return 0
        if p.rank == 0 and not config.get("student_cycle"):
            endpoint = Path(str(args.endpoint)+"-worker%d" % p.worker) if workers > 1 else args.endpoint
            rpc = PolicyClient(endpoint, colocated=config.get("launch_mode") == "colocated_cpu")
            if rpc.binding:
                record(config, "solver_policy_binding_verified", worker=p.worker,
                       global_rank=p.global_rank, **rpc.binding)
        assert digest(old / "data/inventory.json") == config["inventory_sha256"]
        inventory = read_json(old / "data/inventory.json")
        if config.get("student_cycle"):
            from .student_cycle import run_cycle

            status = run_cycle(config, p, replay, legacy, bridge, inventory)
            p.call("PetscFinalize")
            return status
        if config["stage"] == "review":
            status = review(config, p, replay, legacy, bridge, rpc, inventory)
            p.call("PetscFinalize")
            return status
        if config["stage"] == "rl":
            status = training_loop(config, p, replay, legacy, bridge, rpc, inventory)
            p.call("PetscFinalize")
            return status
        results = []
        for key in config["gate_keys"]:
            row = next(r for r in inventory["snapshots"] if r["key"] == key)
            assert row["split"] == "training"
            if p.rank == 0:
                manifest = read_json(Path(row["directory"]) / "manifest.json")
                identity = {f["path"]: f for f in manifest["files"]}
                for name in ("A.bin", "b.bin", "x0.bin"):
                    assert digest(Path(row["directory"]) / name, "sha1") == identity[name]["sha1"]
                record(config, "prediction_state_begin", key=key, job=job)
            p.barrier()
            with np.load(old / "cache" / row["case"] / "layout.npz") as f:
                layout = {name: f[name] for name in f.files}
            receipt = read_json(old / "records" / ("fields-"+key+".json"))
            assert receipt["manifest"] == row["manifest_sha256"]
            assert digest(old / "data" / key / "fields.npz") == receipt["sha256"]
            with np.load(old / "data" / key / "fields.npz") as f:
                fields = {name: f[name] for name in f.files}
            data = replay.pcapply_snapshot(p, row, layout)
            for method in config["prediction_methods"]:
                if method in ("amg", "retained_base"):
                    teacher = None
                    if method == "retained_base":
                        export = read_json(old / "records" / ("solver-export-retained-solver-"+key+".json"))
                        teacher = dict(retained_entry=next(e for e in export["spaces"] if e["label"] == config["baseline_label"]))
                    result = run_method(config, p, replay, legacy, bridge, row, layout, data, method, teacher)
                else:
                    assert method in ("gnn_spatial", "neutral_spatial", "gnn_graph", "neutral_graph")
                    result = solve_policy(config, p, replay, legacy, bridge, rpc, row, layout, fields, data,
                        neutral=method.startswith("neutral_"))
                result["key"] = key
                results.append(result)
                if p.rank == 0:
                    path = output / "records" / ("prediction-%s-%s-%s.json" % (job, key, method))
                    with path.open("x") as f:
                        json.dump(result, f, indent=2, allow_nan=False)
                    record(config, "prediction_method_complete", job=job, key=key, method=method,
                        accurate=result["reached_true_target"], failure=result["failure"],
                        iterations=result.get("iterations"), total_wall_s=result.get("total_wall_s"),
                        numerical=result["numerical"], timings=result.get("timings"))
            p.destroy("Vec", data[3], data[4])
            p.destroy("Mat", data[2])
        passed = all(r["reached_true_target"] for r in results if not r["method"].startswith("neutral_"))
        if p.rank == 0:
            summary = dict(passed=passed, job=job, config_sha256=digest(args.config),
                failures=[dict(key=r["key"], method=r["method"], reason=r["failure"]) for r in results if not r["reached_true_target"]],
                method_totals={method:dict(accurate=sum(r["reached_true_target"] for r in results if r["method"] == method),
                    seconds=sum(r.get("total_wall_s", r["setup_wall_s"]) for r in results if r["method"] == method))
                    for method in config["prediction_methods"]},
                next="Only after independent predictions pass, test a true multi-state on-policy update before24hRL.")
            with (output / "records" / ("prediction-"+job+".json")).open("x") as f:
                json.dump(summary, f, indent=2)
            record(config, "prediction_gate_complete", **summary)
            rpc.close()
        p.call("PetscFinalize")
        return 0 if passed else 2
    except BaseException:
        import traceback
        traceback.print_exc()
        p.mpi.MPI_Abort(getattr(p, "global_world", p.world), 1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
