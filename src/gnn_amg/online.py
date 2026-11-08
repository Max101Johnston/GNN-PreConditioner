"""Independent two-decision policy evaluation in a persistent MPI allocation.

This path constructs fresh observable features and both actual numerical levels.
It never reads teacher hierarchy maps or supervised graph NPZs. The separately
named retained-base comparator reads its own cached interpolation as a baseline.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import time

import numpy as np

from .backend import (Action, Hierarchy, Level, digest, graph_features, load_bridge,
    load_dependencies, pooled_attributes, read_json, record)
from .dataset import gather_array, gather_graph, physical_attributes
from .inference import PolicyClient


def broadcast_selection(p, selected, count):
    values = np.asarray(selected, dtype=np.uint8) if p.rank == 0 else np.empty(count, np.uint8)
    assert len(values) == count
    mpi_byte = p.P(C.addressof((C.c_char*1).in_dll(p.mpi, "ompi_mpi_byte")))
    p.mpi.MPI_Bcast.argtypes = [p.P, p.I, p.P, p.I, p.P]
    p.mpi.MPI_Bcast.restype = p.I
    assert p.mpi.MPI_Bcast(values.ctypes.data, count, mpi_byte, 0, p.world) == 0
    return values.astype(bool)


def choose(config, p, bridge, rpc, matrix, native, comp, attributes, fixed, mass, stage, neutral):
    start = time.perf_counter()
    x, graph, owners = graph_features(p, bridge, matrix, comp, attributes, fixed, stage, mass,
        config["interpolation"]["strength_threshold"])
    local_features_end = time.perf_counter()
    buffers = gather_graph(p, graph, owners)
    arrays = dict(features=gather_array(p, x, owners).astype(np.float32),
        native=gather_array(p, native, owners).astype(np.int64),
        eligible=gather_array(p, ~fixed, owners).astype(bool))
    gather_end = time.perf_counter()
    selected, reply = None, {}
    if p.rank == 0:
        arrays.update(buffers)
        reply, response = rpc.request(dict(operation="select", stage=stage, neutral=neutral), arrays)
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


def solve_policy(config, p, replay, legacy, bridge, rpc, row, layout, fields, data, neutral=False):
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
    fine = Action(p, k)
    lo, hi = map(int, layout["ownership"][p.rank:p.rank+2])
    native = (3*layout["native_node"] + layout["component"])[lo:hi].astype(np.int64)
    comp = layout["component"][lo:hi].astype(np.int32)
    fixed = fields["constrained"][native]
    local = legacy.LocalPC(p, k, Path(config["baseline_directory"]) / "cache" / row["case"], lo, hi, "ras")
    attributes = physical_attributes(fields, layout["xyz"], native)
    first = second = space = shell = callback = None
    numerical, timings, failure = {}, {}, None
    proposed_first, timings["first_decision"] = choose(config, p, bridge, rpc, k, native, comp,
        attributes, fixed, np.ones(len(native)), 0, neutral)
    first = Level(p, bridge, k, native, comp, proposed_first, fixed, config["interpolation"])
    numerical["level1"] = first.stats
    if first.stats["uncovered_free_rows"] or first.stats["empty_free_rows"] or first.stats["injection_error"] or first.stats["repair_budget_exhausted"]:
        failure = "first_interpolation_coverage_or_injection"
    proposed_second = None
    if failure is None:
        pooled, mass = pooled_attributes(p, first, attributes)
        proposed_second, timings["second_decision"] = choose(config, p, bridge, rpc, first.coarse,
            first.native, first.components, pooled, np.zeros(len(first.native), bool), mass, 1, neutral)
        second = Level(p, bridge, first.coarse, first.native, first.components, proposed_second,
            np.zeros(len(first.native), bool), dict(config["interpolation"], max_coarse_dofs=config["terminal_max_dofs"]))
        numerical["level2"] = second.stats
        if second.stats["uncovered_free_rows"] or second.stats["empty_free_rows"] or second.stats["injection_error"] or second.stats["repair_budget_exhausted"]:
            failure = "second_interpolation_coverage_or_injection"
        elif int(second.owners[-1]) > config["terminal_max_dofs"]:
            failure = "terminal_size_limit"
        else:
            space = Hierarchy(p, legacy, first, second, config["terminal_max_dofs"])
            numerical["condition"] = space.condition
            numerical["galerkin_errors"] = space.parity(fine)
            if max(numerical["galerkin_errors"]) > 1e-9:
                failure = "galerkin_identity"
    apply_times = []
    if failure is None:
        def apply(v, z):
            residual = p.array(v)
            answer = local.apply(residual)
            answer += space.gain(residual-fine(answer))
            answer += .5*local.apply(residual-fine(answer))
            p.assign(z, answer)
        shell, callback = replay.pcapply_shell(p, child, operator, k, apply, apply_times)
        replay.setup_solver(p, outer, fs, options)
        setup_s = p.maximum(time.perf_counter()-begin)
        result = legacy.solve_with_true_residual(p, replay, outer, full, rhs, solution,
            config["max_iterations"], max(config["atol"], config["rtol"]*rhs_norm), True)
        result["total_wall_s"] = p.maximum(result.pop("_solve_end_time")-begin)
        failure = None if result["reached_true_target"] else "gmres_true_residual"
    else:
        setup_s = p.maximum(time.perf_counter()-begin)
        result = dict(reached_true_target=False, total_wall_s=setup_s)
    result.update(method="neutral_spatial" if neutral else "gnn_spatial", failure=failure,
        setup_wall_s=setup_s, numerical=numerical, timings=timings, gnn_inference=not neutral,
        checkpoint_sha256=config["policy_checkpoint_sha256"], teacher_hierarchy_read=False,
        timing_scope="Fresh physical/algebraic features,2decisionpolicy/RPC path,repair,preconditioner construction and complete true-residual-monitored solve. Algebra audits included. Replay input loading and artifact logging excluded for all methods.")
    # Archive raw/executed choices after the measured numerical work.
    arrays = dict(native=gather_array(p, native, layout["ownership"]).astype(np.int64),
        raw_first=gather_array(p, proposed_first, layout["ownership"]).astype(bool),
        executed_first=gather_array(p, first.executed_markers, layout["ownership"]).astype(np.int8))
    if second is not None:
        arrays.update(coarse_native=gather_array(p, first.native, first.owners).astype(np.int64),
            raw_second=gather_array(p, proposed_second, first.owners).astype(bool),
            executed_second=gather_array(p, second.executed_markers, first.owners).astype(np.int8))
    if p.rank == 0:
        run = os.environ.get("GNN_RUN_ID", os.environ.get("SLURM_JOB_ID", "local"))
        path = Path(config["output_directory"]) / "data" / ("prediction-%s-%s-%s.npz" % (run, row["key"], result["method"]))
        with path.open("xb") as f:
            np.savez_compressed(f, **arrays)
        result["raw_executed_choices"] = dict(path=str(path), sha256=digest(path))
    p.destroy("KSP", outer)
    if shell is not None:
        p.destroy("PC", shell)
    if space is not None:
        space.close()
    if second is not None:
        second.close()
    first.close()
    local.close()
    fine.close()
    p.destroy("Vec", solution)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("endpoint", type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    replay, legacy = load_dependencies(config)
    p = replay.Petsc()
    replay.pcapply_bind(p)
    assert p.size == config["mpi_ranks"] == 64
    bridge = load_bridge(p, config["bridge"])
    old, output = Path(config["baseline_directory"]), Path(config["output_directory"])
    job = os.environ.get("GNN_RUN_ID", os.environ.get("SLURM_JOB_ID", "local"))
    rpc = None
    try:
        from .gate import numerical_smoke, run_method
        numerical_smoke(config, p, legacy, bridge)
        if p.rank == 0:
            rpc = PolicyClient(args.endpoint)
        assert digest(old / "data/inventory.json") == config["inventory_sha256"]
        inventory = read_json(old / "data/inventory.json")
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
                    assert method in ("gnn_spatial", "neutral_spatial")
                    result = solve_policy(config, p, replay, legacy, bridge, rpc, row, layout, fields, data,
                        neutral=method == "neutral_spatial")
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
        passed = all(r["reached_true_target"] for r in results if r["method"] != "neutral_spatial")
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
        p.mpi.MPI_Abort(p.world, 1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
