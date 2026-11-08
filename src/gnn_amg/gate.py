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
                      load_dependencies, read_json, record)


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


def run_method(config, p, replay, legacy, bridge, row, layout, data, method, teacher):
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
    local = space = first = second = None
    shell = callback = None
    stats = {}
    apply_times = []
    failure = None
    if method != "amg":
        lo, hi = map(int, layout["ownership"][p.rank:p.rank+2])
        local = legacy.LocalPC(p, k, old / "cache" / row["case"], lo, hi, "ras")
        if method == "retained_base":
            entry = teacher["retained_entry"]
            assert digest(old / "cache" / row["key"] / (entry["shard_pattern"] % p.rank)) == entry["shards"][p.rank]
            space = legacy.SolverRetainedHierarchy(p, old, row, entry, layout)
        else:
            native = teacher["native"][lo:hi]
            components = np.asarray(layout["component"][lo:hi], dtype=np.int32)
            assert np.array_equal(native % 3, components)
            first = Level(p, bridge, k, native, components,
                teacher["depth"][lo:hi] >= 3, teacher["fixed"][lo:hi], config["interpolation"])
            stats["level1"] = first.stats
            # Rank shards preserve original versus executed selections; these
            # arrays are outputs, never policy labels or on-policy actions.
            np.savez_compressed(Path(config["output_directory"]) / "data" /
                ("repair-%s-%s-r%03d.npz" % (os.environ.get("SLURM_JOB_ID", "local"), row["key"], p.rank)),
                native=native, raw_selected=teacher["depth"][lo:hi] >= 3,
                executed_markers=first.executed_markers, raw_uncovered=first.raw_uncovered)
            with np.load(old / "cache" / row["key"] / (teacher["retained_entry"]["shard_pattern"] % p.rank)) as f:
                baseline_empty = np.diff(f["T0_indptr"]) == 0
            stats["raw_uncovered_nonempty_in_base"] = int(p.reduce([np.count_nonzero(first.raw_uncovered & ~baseline_empty)])[0])
            mismatches = int(np.count_nonzero(first.special_fine & ~baseline_empty))
            stats["special_fine_nonempty_in_base"] = int(p.reduce([mismatches])[0])
            if stats["special_fine_nonempty_in_base"]:
                failure = "special_fine_rule_disagrees_with_effective_base"
            if first.stats["repair_budget_exhausted"] or first.stats["uncovered_free_rows"] or first.stats["empty_free_rows"] or first.stats["injection_error"]:
                failure = "first_interpolation_coverage_or_injection"
            if failure is None:
                second_mask = np.isin(first.native, teacher["terminal_native"])
                second = Level(p, bridge, first.coarse, first.native, first.components,
                    second_mask, np.zeros(len(first.native), dtype=bool),
                    dict(config["interpolation"], max_coarse_dofs=config["terminal_max_dofs"]))
                stats["level2"] = second.stats
                np.savez_compressed(Path(config["output_directory"]) / "data" /
                    ("repair-second-%s-%s-r%03d.npz" % (os.environ.get("SLURM_JOB_ID", "local"), row["key"], p.rank)),
                    native=first.native, raw_selected=second_mask,
                    executed_markers=second.executed_markers, raw_uncovered=second.raw_uncovered)
                if second.stats["repair_budget_exhausted"] or second.stats["uncovered_free_rows"] or second.stats["empty_free_rows"] or second.stats["injection_error"]:
                    failure = "second_interpolation_coverage_or_injection"
                else:
                    space = Hierarchy(p, legacy, first, second, config["terminal_max_dofs"])
                    stats["coarse_condition"] = space.condition
        if space:
            stats["galerkin_errors"] = space.parity(fine)
            if max(stats["galerkin_errors"]) > 1e-9:
                failure = "galerkin_identity"

            def action(v, z):
                residual = p.array(v)
                answer = local.apply(residual)
                answer += space.gain(residual - fine(answer))
                answer += 0.5 * local.apply(residual - fine(answer))
                p.assign(z, answer)

            shell, callback = replay.pcapply_shell(p, child, operator, k, action, apply_times)
    replay.setup_solver(p, outer, fs, options)
    setup = p.maximum(time.perf_counter()-start)
    if failure:
        result = dict(reached_true_target=False, failure=failure)
    else:
        result = legacy.solve_with_true_residual(p, replay, outer, full, rhs, solution,
            config["max_iterations"], max(1e-12, 1e-8*rhs_norm), True)
        result["total_wall_s"] = p.maximum(result.pop("_solve_end_time")-start)
        result["failure"] = None if result["reached_true_target"] else "gmres_true_residual"
    result.update(method=method, setup_wall_s=setup, numerical=stats,
        includes_teacher_loading=False, gnn_inference=False,
        timing_scope="Gate diagnostic: construction, algebra checks and true-residual monitored solve; not deployment speedup.")
    p.destroy("KSP", outer)
    if shell is not None:
        p.destroy("PC", shell)
    if space:
        space.close()
    if second:
        second.close()
    if first:
        first.close()
    if local:
        local.close()
    fine.close()
    p.destroy("Vec", solution)
    return result


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
        for key in config["gate_keys"]:
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
            export = read_json(old / "records" / ("solver-export-retained-solver-" + key + ".json"))
            entry = next(e for e in export["spaces"] if e["label"] == config["baseline_label"])
            teacher = dict(native=native, depth=depth, terminal_native=native[depth >= 5],
                           fixed=fixed, retained_entry=entry)
            data = replay.pcapply_snapshot(p, row, layout)
            for method in ("amg", "retained_base", "teacher_multipass_repaired"):
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
