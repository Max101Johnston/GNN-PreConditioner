"""Validate existing teacher endpoints with direct numerical interpolation.

This command does not train a policy. Failure leaves the complete evidence in
the configured analysis directory and prevents the expensive RL stage.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import time

import numpy as np

from .backend import (Action, Hierarchy, Level, digest, load_bridge,
                      load_dependencies, read_json, record, graph_features, pooled_attributes)


def numerical_smoke(config, p, legacy, bridge):
    import scipy.sparse as sp
    n = 3*p.size
    owners = 3*np.arange(p.size+1)
    lo, hi = owners[p.rank:p.rank+2]
    matrix = sp.diags([np.full(n-3, -.5), np.full(n, 2.), np.full(n-3, -.5)],
                      [-3, 0, 3], format="csr")
    if p.size > 2:
        matrix = matrix.tolil()
        for row in range(3, 6):
            matrix.rows[row] = [row]
            matrix.data[row] = [2.]
        matrix = matrix.tocsr()
    distributed = legacy.DistributedSparse(p, matrix[lo:hi], owners, owners)
    native = np.arange(lo, hi)
    selected = ((native//3) % 3 == 0) | (native//3 == p.size-1)
    level = Level(p, bridge, distributed.matrix, native, (native % 3).astype(np.int32),
                  selected, np.zeros(3, dtype=bool), config["interpolation"])
    assert level.stats["uncovered_free_rows"] == level.stats["empty_free_rows"] == 0
    assert level.stats["injection_error"] == 0
    assert level.stats["special_fine_rows"] == (3 if p.size > 2 else 0)
    start, stop = level.owners[p.rank:p.rank+2]
    v = np.sin(.19*(np.arange(start, stop)+1))
    truth = level.P(distributed.apply(level.P(v)), transpose=True)
    actual = level.K(v)
    errors = p.reduce([np.sum((actual-truth)**2), truth@truth])
    relative = float(np.sqrt(errors[0]/max(errors[1], 1e-300)))
    assert relative < 1e-11
    if p.rank == 0:
        record(config, "native_mpi_smoke_passed", ranks=p.size,
            empty_coarse_ranks=int(np.sum(np.diff(level.owners) == 0)),
            galerkin_relative=relative, interpolation=level.stats)
    attributes = np.column_stack([np.ones(3), np.arange(lo, hi), np.arange(lo, hi)**2])
    means, mass = pooled_attributes(p, level, attributes)
    assert np.allclose(means[:, 0], 1) and np.all(mass > 0)
    features, graph, graph_owners = graph_features(p, bridge, distributed.matrix,
        (native % 3).astype(np.int32), attributes, np.zeros(3, bool), 0, np.ones(3))
    expected = matrix.copy()
    expected.setdiag(0)
    expected.eliminate_zeros()
    expected.data = np.abs(expected.data)
    row_sum = np.asarray(expected.sum(axis=1)).ravel()
    expected = sp.diags(1/np.maximum(row_sum, 1e-30)) @ expected
    difference = graph - expected[lo:hi]
    assert not difference.nnz or np.max(np.abs(difference.data)) < 1e-14
    assert np.array_equal(graph_owners, owners)
    np.testing.assert_allclose(features[:, attributes.shape[1]], np.log10(2), atol=1e-7)
    if p.rank == 0:
        record(config, "actual_strength_graph_and_support_pooling_smoke_passed", ranks=p.size)
    level.close()
    distributed.close()
    if p.size >= 20:
        # Three disconnected component chains with no proposed C points in
        # their last six nodes. Connectivity crosses rank ownership borders.
        cut = n - 18
        matrix = matrix.tolil()
        for component in range(3):
            matrix[cut-3+component, cut+component] = 0
            matrix[cut+component, cut-3+component] = 0
        matrix = matrix.tocsr()
        matrix.eliminate_zeros()
        distributed = legacy.DistributedSparse(p, matrix[lo:hi], owners, owners)
        proposal = (native < cut) & (((native//3) % 3 == 0) | (native//3 == p.size-7))
        for fraction in (0.0, 0.2):
            setting = dict(config["interpolation"], coverage_repair_fraction=fraction)
            repaired = Level(p, bridge, distributed.matrix, native,
                (native % 3).astype(np.int32), proposal, np.zeros(3, bool), setting)
            assert repaired.stats["raw_uncovered_9_hops"] == 18
            assert repaired.stats["raw_uncovered_99_hops"] == 18
            assert repaired.stats["repair_added_coarse"] == (3 if fraction else 0)
            assert repaired.stats["uncovered_free_rows"] == (0 if fraction else 18)
            assert repaired.stats["repair_budget_exhausted"] == (0 if fraction else 1)
            assert np.all(repaired.executed_markers[proposal] == 1)
            assert repaired.stats["injection_error"] == 0
            if p.rank == 0:
                record(config, "coverage_repair_smoke_passed", budget_fraction=fraction,
                    interpolation=repaired.stats)
            repaired.close()
        distributed.close()


class RetainedFirst:
    """Diagnostic P1 and its matching K1, with verified L3 column identity."""
    def __init__(self, config, p, legacy, row, teacher, assemble=True):
        from scipy.sparse import csr_matrix

        entry = teacher["retained_entry"]
        path = (Path(config["baseline_directory"]) / "cache" / row["key"] /
                (entry["shard_pattern"] % p.rank))
        assert digest(path) == entry["shards"][p.rank]
        with np.load(path) as archive:
            fine_owners, self.owners = archive["ownership0"], archive["ownership1"]
            self.terminal_owners = archive["ownership2"]
            self.cached_second = csr_matrix(
                (archive["T1_data"], archive["T1_indices"], archive["T1_indptr"]),
                shape=archive["T1_shape"])
            transfer, coarse = (
                csr_matrix((archive[name+"_data"], archive[name+"_indices"], archive[name+"_indptr"]),
                           shape=archive[name+"_shape"])
                for name in ("T0", "A1")
            )
            lo, hi = fine_owners[p.rank:p.rank+2]
            np.testing.assert_array_equal(archive["native"], teacher["native"][lo:hi])
        rows = np.flatnonzero(teacher["depth"][lo:hi] >= 3)
        positions = transfer.indptr[rows]
        np.testing.assert_array_equal(np.diff(transfer.indptr)[rows], 1)
        start, stop = self.owners[p.rank:p.rank+2]
        columns = transfer.indices[positions]
        # Retained transfers have energy-scaled columns, so singleton
        # injection rows identify columns without requiring a unit value.
        injection = transfer.data[positions]
        assert np.all(np.isfinite(injection) & (injection != 0))
        # Cached coarse columns have their own balanced ownership. An injection
        # row can identify a column owned by another rank.
        mapping = np.zeros((3, self.owners[-1]), dtype=np.float64)
        mapping[0, columns] = teacher["native"][lo:hi][rows]
        np.add.at(mapping[1], columns, 1)
        mapping[2, columns] = injection
        mapping = p.reduce(mapping)
        assert np.all(mapping[1] == 1), "L3 injection is not a bijection"
        self.native = mapping[0, start:stop].astype(np.int64)
        self.column_scale = mapping[2]
        self.components = (self.native % 3).astype(np.int32)
        if not assemble:
            self.receipt = dict(path=str(teacher["retained_export"]),
                                sha256=digest(teacher["retained_export"]))
            self.fine_owners = fine_owners
            self.fine_native = teacher["native"][lo:hi]
            self.raw_transfer = transfer
            self.raw_coarse = coarse
            self.injection_rows = rows
            self.injection_columns = columns
            self.markers = np.full(hi-lo, -1, np.int32)
            empty = np.diff(transfer.indptr) == 0
            self.markers[empty] = -3
            self.markers[rows] = 1
            assert np.all(empty[teacher["fixed"][lo:hi]])
            assert np.isfinite(transfer.data).all() and np.isfinite(self.cached_second.data).all()
            terminal_rows = np.flatnonzero(np.isin(self.native, teacher["terminal_native"]))
            assert np.all(np.diff(self.cached_second.indptr)[terminal_rows] == 1)
            locations = self.cached_second.indptr[terminal_rows]
            assert np.all(self.cached_second.data[locations] != 0)
            terminal_columns = self.cached_second.indices[locations]
            counts = np.bincount(terminal_columns, minlength=int(self.terminal_owners[-1]))
            assert np.all(p.reduce(counts) == 1), "L5 injection is not a bijection"
            assert p.reduce([len(terminal_rows)])[0] == len(teacher["terminal_native"])
            return
        self.transfer_action = legacy.DistributedSparse(p, transfer, fine_owners, self.owners)
        self.coarse_action = legacy.DistributedSparse(p, coarse, self.owners, self.owners)
        self.coarse = self.coarse_action.matrix
        self.P, self.K = self.transfer_action.apply, self.coarse_action.apply
        self.stats = dict(coarse_dofs=int(self.owners[-1]), nnz=int(p.reduce([transfer.nnz])[0]),
                          source="retained_P1")

    def close(self):
        self.coarse_action.close()
        self.transfer_action.close()


def fixed_first_cross(config, p, legacy, bridge, k, cached, teacher, lo, hi):
    from scipy.sparse import csr_matrix
    from .backend import local_csr
    from .interpolation import imported_transfer

    native = teacher["native"][lo:hi]
    selected = teacher["depth"][lo:hi] >= 3
    settings = dict(config["interpolation"], coverage_repair_fraction=0.0)
    direct = Level(p, bridge, k, native, (native % 3).astype(np.int32), selected,
                   teacher["fixed"][lo:hi], settings)
    try:
        retained_ids = p.all_rows(cached.native, cached.owners).astype(np.int64)
        direct_ids = p.all_rows(direct.native, direct.owners).astype(np.int64)
        order = np.argsort(retained_ids)
        column_map = order[np.searchsorted(retained_ids[order], direct_ids)]
        np.testing.assert_array_equal(retained_ids[column_map], direct_ids)
        assert len(direct_ids) == len(retained_ids) and direct.stats["injection_error"] == 0
        csr, owners = local_csr(p, direct.transfer, bridge)
        cols = column_map[csr.indices]
        # The cached P2 acts in the cached P1 column coordinates, including
        # their energy scale. Permutation alone would change the experiment.
        replacement = csr_matrix((csr.data*cached.column_scale[cols], cols, csr.indptr),
                                 shape=csr.shape)
        replacement.eliminate_zeros()
        original = cached.transfer_action.cpu
        fill = (np.diff(replacement.indptr) == 0) & (np.diff(original.indptr) > 0)
        assert not np.any(fill & selected)
        replacement = (replacement + original.multiply(fill[:, None])).tocsr()
        replacement.sort_indices()
        nnz, fill_count, missing = map(int, p.reduce([replacement.nnz, fill.sum(),
            np.count_nonzero((np.diff(replacement.indptr) == 0) & (np.diff(original.indptr) > 0))]))
        assert missing == 0 and np.isfinite(replacement.data).all()
        stats = dict(source="fixed_direct_P1_with_cached_empty_row_completion",
                     coarse_dofs=len(retained_ids), nnz=nnz, cached_rows=fill_count,
                     original_direct=direct.stats, coarse_ids_preserved=True)
        return imported_transfer(p, legacy, k, replacement, owners, cached.owners, stats)
    finally:
        direct.close()


def checked_hierarchy(config, p, legacy, first, second):
    stats = second.stats
    if any(stats[k] for k in ("repair_budget_exhausted", "uncovered_free_rows",
                              "empty_free_rows", "injection_error")):
        return None, "second_interpolation_coverage_or_injection"
    if second.owners[-1] > config["terminal_max_dofs"]:
        return None, "terminal_size_limit"
    try:
        return Hierarchy(p, legacy, first, second, config["terminal_max_dofs"]), None
    except ValueError as error:
        if str(error) != "terminal_condition_limit":
            raise
        return None, str(error)


def retained_pair(config, p, legacy, bridge, k, pair, components, model, control=None, geometry=None):
    """Rebuild both Galerkin operators with the original coarse partitions."""
    from .backend import local_csr
    from .interpolation import imported_transfer
    from .student import direct_level

    missing = dict(uncovered_free_rows=0, empty_free_rows=0, injection_error=0.0,
                   repair_budget_exhausted=0)
    stats = {}
    if model is None:
        stats["level1"] = dict(source="retained_P1", fine_dofs=int(pair.fine_owners[-1]),
            coarse_dofs=int(pair.owners[-1]), nnz=int(p.reduce([pair.raw_transfer.nnz])[0]), **missing)
        first = imported_transfer(p, legacy, k, pair.raw_transfer, pair.fine_owners,
                                  pair.owners, stats["level1"])
        transfer = pair.cached_second
    else:
        first, stats["level1"] = direct_level(config, p, legacy, bridge, k, pair, components, model,
                                              control, geometry)
        if first is None:
            return None, None, stats, "direct_p1_uncovered_rows"
        # Unit injection changes the C1 coordinates. Rebase P2's rows while
        # retaining its terminal column scale and identity for this control.
        lo, hi = pair.owners[p.rank:p.rank+2]
        transfer = pair.cached_second.multiply(pair.column_scale[lo:hi, None]).tocsr()
        transfer.sort_indices()
    fresh, _ = local_csr(p, first.coarse, bridge)
    stats["k1_nnz"] = int(p.reduce([fresh.nnz])[0])
    if model is None:
        delta = fresh - pair.raw_coarse
        error, norm = p.reduce([delta.data @ delta.data, pair.raw_coarse.data @ pair.raw_coarse.data])
        stats["retained_k1_relative_error"] = float(np.sqrt(error/max(norm, 1e-300)))
    stats["level2"] = dict(source="retained_P2", coarse_dofs=int(pair.terminal_owners[-1]),
        nnz=int(p.reduce([transfer.nnz])[0]), **missing)
    second = imported_transfer(p, legacy, first.coarse, transfer, pair.owners, pair.terminal_owners,
                               stats["level2"])
    return first, second, stats, None


def run_method(config, p, replay, legacy, bridge, row, layout, data, method, teacher,
               probe_inputs=None, probe_windows=None):
    old = Path(config["baseline_directory"])
    md, rows, full, rhs, initial = data
    solution = p.vector(full)
    if md["configuration"]["initial_guess_nonzero"]:
        p.call("VecCopy", initial, solution)
    rhs_norm = p.norm(rhs)
    p.barrier()
    start = time.perf_counter()
    outer, fs, options = replay.create_solver(p, full, md, config["max_iterations"],
        overrides=None if method == "amg" else {"fieldsplit_1_pc_type": "none"})
    assert options["ksp_type"] == "gmres" and options["ksp_pc_side"] == "right"
    assert float(options["ksp_rtol"]) == 1e-8 and float(options["ksp_atol"]) == 1e-12
    p.call("PCSetUp", fs)
    child, original, operator, k = replay.pcapply_child(p, fs, rows,
        expected_pc=b"hypre" if method == "amg" else b"none")
    fine = Action(p, k)
    local = space = first = second = cached = None
    shell = callback = None
    stats = {}
    apply_times = []
    capture = teacher.get("capture_apply") if teacher else None
    capture_calls = set(config["direct_feedback_calls"]) if capture is not None else set()
    calls = 0
    failure = None
    if method != "amg":
        lo, hi = map(int, layout["ownership"][p.rank:p.rank+2])
        local = legacy.LocalPC(p, k, old / "cache" / row["case"], lo, hi, "ras")
        if method == "retained_base":
            entry = teacher["retained_entry"]
            assert digest(old / "cache" / row["key"] / (entry["shard_pattern"] % p.rank)) == entry["shards"][p.rank]
            space = legacy.SolverRetainedHierarchy(p, old, row, entry, layout)
            stats["level1"] = dict(coarse_dofs=space.structure["level_dofs"][1],
                                   nnz=int(p.reduce([space.t0.cpu.nnz])[0]))
            stats["level2"] = dict(coarse_dofs=space.structure["level_dofs"][2],
                                   nnz=int(p.reduce([space.t1.cpu.nnz])[0]))
        elif method in ("teacher_roundtrip", "direct_p1"):
            first, second, stats, failure = retained_pair(config, p, legacy, bridge, k,
                teacher["pair"], layout["component"], teacher.get("direct_model"), teacher.get("direct_control"),
                teacher.get("geometry"))
            if failure is None:
                space, failure = checked_hierarchy(config, p, legacy, first, second)
                if stats.get("retained_k1_relative_error", 0) > 1e-9:
                    failure = "retained_k1_roundtrip"
        elif method == "teacher_cross":
            from .interpolation import imported_transfer

            cached = RetainedFirst(config, p, legacy, row, teacher)
            first = fixed_first_cross(config, p, legacy, bridge, k, cached, teacher, lo, hi)
            stats["level1"] = first.stats
            cached_p2 = cached.cached_second
            second_stats = dict(source="retained_P2", coarse_dofs=int(cached.terminal_owners[-1]),
                                nnz=int(p.reduce([cached_p2.nnz])[0]),
                                uncovered_free_rows=0, empty_free_rows=0, injection_error=0,
                                repair_budget_exhausted=0)
            second = imported_transfer(p, legacy, first.coarse, cached_p2,
                                       cached.owners, cached.terminal_owners, second_stats)
            stats["level2"] = second.stats
            space, failure = checked_hierarchy(config, p, legacy, first, second)
        elif method in ("teacher_cached_first", "teacher_cached_first_repaired", "teacher_cached_coverage"):
            first = RetainedFirst(config, p, legacy, row, teacher)
            stats["level1"] = first.stats
            terminal = teacher["native"][teacher["depth"] >= 5]
            fraction = config["interpolation"]["coverage_repair_fraction"] if method.endswith("_repaired") else 0.0
            second = Level(p, bridge, first.coarse, first.native, first.components,
                np.isin(first.native, terminal), np.zeros(len(first.native), bool),
                dict(config["interpolation"], max_coarse_dofs=config["terminal_max_dofs"],
                     coverage_repair_fraction=fraction))
            stats["level2"] = second.stats
            if method == "teacher_cached_coverage":
                from .interpolation import coverage_audit
                from .online import broadcast_json

                stats["reachability"] = coverage_audit(config, p, bridge, first,
                    np.isin(first.native, terminal), first.cached_second, broadcast_json)
                assert stats["reachability"]["uncovered_9"] == second.stats["raw_uncovered_9_hops"]
            space, failure = checked_hierarchy(config, p, legacy, first, second)
        else:
            native = teacher["native"][lo:hi]
            components = np.asarray(layout["component"][lo:hi], dtype=np.int32)
            assert np.array_equal(native % 3, components)
            first_mask = teacher.get("first_mask")
            if first_mask is None:
                first_mask = teacher["depth"][lo:hi] >= 3
            first = Level(p, bridge, k, native, components, first_mask,
                          teacher["fixed"][lo:hi], config["interpolation"])
            stats["level1"] = first.stats
            # Rank shards preserve original versus executed selections; these
            # arrays are outputs, never policy labels or on-policy actions.
            if config["stage"] != "review":
                if config["stage"] != "review":
                    np.savez_compressed(Path(config["output_directory"]) / "data" /
                        ("repair-%s-%s-r%03d.npz" % (os.environ.get("SLURM_JOB_ID", "local"), row["key"], p.rank)),
                        native=native, raw_selected=teacher["depth"][lo:hi] >= 3,
                        executed_markers=first.executed_markers, raw_uncovered=first.raw_uncovered)
            stats["retained_reference_available"] = teacher["retained_entry"] is not None
            if teacher["retained_entry"] is not None:
                baseline_empty = teacher.get("baseline_empty")
                if baseline_empty is None:
                    with np.load(old / "cache" / row["key"] / (teacher["retained_entry"]["shard_pattern"] % p.rank)) as f:
                        baseline_empty = np.diff(f["T0_indptr"]) == 0
                stats["raw_uncovered_nonempty_in_base"] = int(p.reduce([np.count_nonzero(first.raw_uncovered & ~baseline_empty)])[0])
                mismatches = int(np.count_nonzero(first.special_fine & ~baseline_empty))
                stats["special_fine_nonempty_in_base"] = int(p.reduce([mismatches])[0])
                if stats["special_fine_nonempty_in_base"]:
                    failure = "special_fine_rule_disagrees_with_effective_base"
            else:
                # Most training labels have no pre-exported old retained P.
                # No special-F row may be accepted without that reference.
                stats["raw_uncovered_nonempty_in_base"] = None
                stats["special_fine_nonempty_in_base"] = None
                if first.stats["special_fine_rows"]:
                    failure = "special_fine_requires_retained_reference"
            if first.stats["repair_budget_exhausted"] or first.stats["uncovered_free_rows"] or first.stats["empty_free_rows"] or first.stats["injection_error"]:
                failure = "first_interpolation_coverage_or_injection"
            if failure is None:
                second_mask = np.isin(first.native, teacher["terminal_native"])
                setting = dict(config["interpolation"], max_coarse_dofs=config["terminal_max_dofs"])
                if "second_repair_fraction" in setting:
                    setting["coverage_repair_fraction"] = setting["second_repair_fraction"]
                second = Level(p, bridge, first.coarse, first.native, first.components,
                    second_mask, np.zeros(len(first.native), dtype=bool), setting)
                stats["level2"] = second.stats
                np.savez_compressed(Path(config["output_directory"]) / "data" /
                    ("repair-second-%s-%s-r%03d.npz" % (os.environ.get("SLURM_JOB_ID", "local"), row["key"], p.rank)),
                    native=first.native, raw_selected=second_mask,
                    executed_markers=second.executed_markers, raw_uncovered=second.raw_uncovered)
                space, failure = checked_hierarchy(config, p, legacy, first, second)
        if space:
            if isinstance(space, Hierarchy):
                stats["coarse_condition"] = space.condition
            stats["galerkin_errors"] = space.parity(fine)
            if max(stats["galerkin_errors"]) > 1e-9:
                failure = "galerkin_identity"
            if config.get("export_supervised") and first is not None and failure is None:
                from .dataset import export_state
                stats["supervised_files"] = export_state(config, p, bridge, row, layout, teacher, k, first, second)

            def action(v, z):
                nonlocal calls
                residual = p.array(v)
                if capture is not None:
                    if calls in capture_calls:
                        capture.append(residual.copy())
                    calls += 1
                answer = local.apply(residual)
                answer += space.gain(residual - fine(answer))
                answer += 0.5 * local.apply(residual - fine(answer))
                p.assign(z, answer)

            shell, callback = replay.pcapply_shell(p, child, operator, k, action, apply_times)
    replay.setup_solver(p, outer, fs, options)
    setup = p.maximum(time.perf_counter()-start)
    apply_times.clear()
    if capture is not None:
        capture.clear()
        calls = 0
    export_direct = method == "teacher_roundtrip" and teacher.get("export_direct", False)
    if export_direct and not failure:
        space.direction_samples = []
        if config.get("direct_teacher_kind") in ("E", "F"):
            space.direction_sample_calls = tuple(config["direct_sample_calls"])
    if probe_inputs is not None:
        from .krylov import probe_trajectories

        began = time.perf_counter()
        result = dict(trajectories=probe_trajectories(p, full, fs, probe_inputs, probe_windows)
                      if not failure else {}, failure=failure,
                      probe_s=p.maximum(time.perf_counter()-began))
    elif failure:
        result = dict(reached_true_target=False, failure=failure)
    else:
        if config["stage"] == "rl" or config.get("student_cycle") == "factor_tune":
            from .backend import solve_with_deadline
            result = solve_with_deadline(p, replay, outer, full, rhs, solution,
                config["max_iterations"], max(1e-12, 1e-8*rhs_norm),
                start+config["training"]["episode_timeout_seconds"])
        else:
            result = legacy.solve_with_true_residual(p, replay, outer, full, rhs, solution,
                config["max_iterations"], max(1e-12, 1e-8*rhs_norm), True)
        result["total_wall_s"] = p.maximum(result.pop("_solve_end_time")-start)
        result["failure"] = None if result["reached_true_target"] else "gmres_true_residual"
    result.update(method=method, setup_wall_s=setup, numerical=stats,
        displacement_pc_apply_s=p.maximum(sum(apply_times)) if method != "amg" else None,
        displacement_pc_apply_count=len(apply_times) if method != "amg" else None,
        includes_teacher_loading=False, gnn_inference=method == "direct_p1",
        timing_scope="Gate diagnostic: construction, algebra checks and true-residual monitored solve; not deployment speedup.")
    if export_direct and result["reached_true_target"]:
        from .student import export_direct_teacher

        directions = [v[0] for v in space.direction_samples]
        feedback = teacher.get("feedback_inputs", [])
        if feedback:
            space.direction_samples = []
            space.direction_calls = 0
            space.direction_sample_calls = tuple(range(len(feedback)))
            for residual in feedback:
                space.gain(residual-fine(local.apply(residual)))
            directions.extend(v[0] for v in space.direction_samples)
        result["student_teacher"] = export_direct_teacher(config, p, legacy, bridge, row, k,
            teacher["pair"], layout["component"], directions, teacher["phase"], teacher.get("geometry"),
            len(feedback))
    if capture is not None:
        result["feedback_pc_apply_calls"] = [i for i in sorted(capture_calls) if i < calls]
    p.destroy("KSP", outer)
    if shell is not None:
        p.destroy("PC", shell)
    if space:
        space.close()
    if second:
        second.close()
    if first:
        first.close()
    if cached:
        cached.close()
    if local:
        local.close()
    fine.close()
    p.destroy("Vec", solution)
    return result


def load_teacher(config, p, row, layout):
    old = Path(config["baseline_directory"])
    key = row["key"]
    hrecord = read_json(old / "records" / ("hierarchy-" + key + ".json"))
    hmap = old / "data" / key / hrecord["map_file"]
    assert digest(hmap) == hrecord["map_sha256"]
    with np.load(hmap) as f:
        native = f["native_for_petsc"].copy()
        depth = f["depth_petsc"].copy()
    assert np.array_equal(native, 3*layout["native_node"]+layout["component"])
    field_record = read_json(old / "records" / ("fields-"+key+".json"))
    assert field_record["manifest"] == row["manifest_sha256"]
    assert digest(old / "data" / key / "fields.npz") == field_record["sha256"]
    with np.load(old / "data" / key / "fields.npz") as f:
        fixed = f["constrained"][native]
    export_path = old / "records" / ("solver-export-retained-solver-" + key + ".json")
    entry = None
    if export_path.exists():
        export = read_json(export_path)
        entry = next(e for e in export["spaces"] if e["label"] == config["baseline_label"])
    elif not config.get("export_supervised"):
        raise FileNotFoundError(export_path)
    teacher = dict(native=native, depth=depth, terminal_native=native[depth >= 5], fixed=fixed,
                   retained_entry=entry, retained_export=export_path if entry is not None else None)
    return teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    replay, legacy = load_dependencies(config)
    p = replay.Petsc()
    replay.pcapply_bind(p)
    assert p.size == config["mpi_ranks"]
    bridge = load_bridge(p, config["bridge"])
    old = Path(config["baseline_directory"])
    target = Path(config["output_directory"])
    job = os.environ.get("SLURM_JOB_ID", "local")
    try:
        numerical_smoke(config, p, legacy, bridge)
        assert digest(old / "data/inventory.json") == config["inventory_sha256"]
        inventory = read_json(old / "data/inventory.json")
        results = []
        keys = ([r["key"] for r in inventory["snapshots"] if r["split"] == "training"]
                if config.get("export_supervised") else config["gate_keys"])
        for key in keys:
            row = next(r for r in inventory["snapshots"] if r["key"] == key)
            assert row["split"] == "training", "Gate may not tune on validation/test states"
            if p.rank == 0:
                manifest = read_json(Path(row["directory"]) / "manifest.json")
                files = {f["path"]: f for f in manifest["files"]}
                for name in ("A.bin", "b.bin", "x0.bin"):
                    assert digest(Path(row["directory"]) / name, "sha1") == files[name]["sha1"]
                record(config, "gate_state_begin", key=key)
            p.barrier()
            with np.load(old / "cache" / row["case"] / "layout.npz") as f:
                layout = {n: f[n] for n in f.files}
            teacher = load_teacher(config, p, row, layout)
            data = replay.pcapply_snapshot(p, row, layout)
            methods = ("teacher_multipass_repaired",) if config.get("export_supervised") else ("amg", "retained_base", "teacher_multipass_repaired")
            for method in methods:
                result = run_method(config, p, replay, legacy, bridge, row, layout, data, method, teacher)
                result["key"] = key
                results.append(result)
                if p.rank == 0:
                    record(config, "gate_method_complete", key=key, method=method,
                        accurate=result["reached_true_target"], failure=result["failure"],
                        numerical=result["numerical"], iterations=result.get("iterations"),
                        total_wall_s=result.get("total_wall_s"))
                    with (target / "records" / ("gate-"+job+"-"+key+"-"+method+".json")).open("x") as stream:
                        json.dump(result, stream, indent=2, allow_nan=False)
            p.destroy("Vec", data[3], data[4])
            p.destroy("Mat", data[2])
        passed = all(r["reached_true_target"] for r in results)
        if p.rank == 0:
            summary = dict(passed=passed, job=job, methods=len(results),
                config_sha256=digest(args.config),
                failures=[dict(key=r["key"], method=r["method"], reason=r["failure"]) for r in results if not r["reached_true_target"]],
                next="Build the shared multi-frame policy only after the numerical gate passes.")
            with (target / "records" / ("gate-"+job+".json")).open("x") as stream:
                json.dump(summary, stream, indent=2)
            record(config, "gate_complete", **summary)
        p.call("PetscFinalize")
        return 0 if passed else 2
    except BaseException:
        import traceback
        traceback.print_exc()
        p.mpi.MPI_Abort(p.world, 1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
