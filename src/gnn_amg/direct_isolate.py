"""Isolate the two directly generated levels on identical matrix snapshots."""
import os
from pathlib import Path
import time

import numpy as np

from .backend import digest, local_csr, read_json, record
from .dataset import ATTRIBUTE_NAMES
from .direct_depth import DepthCycle, InvalidDepth, parity
from .direct_solver import DirectWeights, InvalidDirect, generate_level
from .factorized import (InvalidFactorization, first_graph, interval, load_model,
                         next_graph, normalized_coordinates)
from .online import (DisplacementPC, finish_hierarchy, global_objects,
                     load_observed_state, solve_policy)
from .student_cycle import compact, save_once


VARIANTS = (("factorized", "factorized"), ("direct", "direct"),
            ("factorized", "direct"), ("direct", "factorized"),
            ("factorized", "exact"), ("direct", "exact"))


def hierarchy(config, p, legacy, bridge, layout, fields, fine, ras, teacher, direct,
              first_kind, next_kind):
    first = second = space = None
    numerical, timings, failure = {}, {}, None
    lo, hi = layout["ownership"][p.rank:p.rank+2]
    native = 3*layout["native_node"][lo:hi] + layout["component"][lo:hi]
    fixed = fields["constrained"][native]
    coordinates = normalized_coordinates(layout["xyz"])
    try:
        start = time.perf_counter()
        if first_kind == "factorized":
            graph, attributes, coordinates = first_graph(
                p, bridge, fine.matrix, layout, fields, matrix_only=teacher.matrix_only)
            first = interval(config, p, legacy, bridge, fine.matrix, graph, teacher, 0, coordinates)
        else:
            first = generate_level(config, p, legacy, bridge, fine.matrix, native, coordinates,
                                   fixed, direct, 1, False)
            if next_kind == "factorized":
                attributes = np.zeros((len(native), len(ATTRIBUTE_NAMES)))
                attributes[:, -3:] = coordinates[native//3]
        numerical["level1"] = first.stats
        timings["first_level_s"] = p.maximum(time.perf_counter()-start)
        start = time.perf_counter()
        if next_kind == "exact":
            numerical["galerkin_errors"], _ = parity(p, [first], fine, [])
            if max(numerical["galerkin_errors"]) > 1e-9:
                raise InvalidDepth("first_galerkin")
            space = DepthCycle(p, legacy, bridge, [first],
                               config["direct_final"]["depth"]["sparse_terminal_limit"])
            numerical["terminal"] = space.terminal.stats
            numerical["k1_nnz"] = int(p.reduce([local_csr(p, first.coarse, bridge)[0].nnz])[0])
            timings["terminal_setup_s"] = space.terminal_setup_s
        else:
            if next_kind == "factorized":
                graph = next_graph(p, bridge, first, attributes, coordinates)
                second = interval(config, p, legacy, bridge, first.coarse, graph, teacher, 3, coordinates)
            else:
                second = generate_level(config, p, legacy, bridge, first.coarse, first.native,
                                        coordinates, np.zeros(len(first.native), bool), direct, 2, False)
            numerical["level2"] = second.stats
            timings["second_level_s"] = p.maximum(time.perf_counter()-start)
            space, failure = finish_hierarchy(config, p, legacy, bridge, fine, first, second, numerical)
    except (InvalidDepth, InvalidDirect, InvalidFactorization) as error:
        failure = str(error)
    return DisplacementPC(p, fine, ras, first, second, space, None, None,
                          None, None, numerical, timings, failure)


def first_parity(p, bridge, displacement, kind, reference):
    """Check that only the second construction changes within a first-level group."""
    first = displacement.first
    transfer, _ = local_csr(p, first.transfer, bridge)
    transfer.eliminate_zeros()
    transfer.sort_indices()
    if kind not in reference:
        reference[kind] = (first.native.copy(), transfer)
        return dict(same_support=True, coefficient_error=0.)
    native, saved = reference[kind]
    identical = (np.array_equal(first.native, native)
                 and np.array_equal(transfer.indptr, saved.indptr)
                 and np.array_equal(transfer.indices, saved.indices))
    difference = float(np.max(np.abs(transfer.data-saved.data), initial=0.)) if identical else 0.
    return dict(same_support=not bool(p.maximum(not identical)),
                coefficient_error=p.maximum(difference))


def run_isolation(config, p, replay, legacy, bridge, inventory):
    settings = config["direct_isolate"]
    parent = settings["parent"]
    if digest(parent["path"]) != parent["sha256"]:
        raise ValueError("Changed direct cold-start baseline receipt")
    prior = read_json(parent["path"])
    teacher = load_model(prior["teacher"])
    seed = DirectWeights.load(settings["distilled"])
    if not teacher.matrix_only:
        raise ValueError("The crossed control requires a matrix-only factorized generator")
    config = dict(config, factorized=dict(config["factorized"], shared_anchor_exchange=True))
    job = os.environ["GNN_RUN_ID"]
    output = Path(config["output_directory"])
    keys = prior["diagnostic"]
    frames = {row["key"]:row for row in inventory["snapshots"]}
    baseline = {(key, method): sorted(r["iterations"] for r in prior["diagnostic_rows"]
                                     if r["key"]==key and r["method"]==method)
                for key in keys for method in ("factorized-teacher", "direct-distilled")}
    local = []
    for index, key in enumerate(keys):
        if index % config["parallel_workers"] != p.worker:
            continue
        state = load_observed_state(config, p, replay, frames[key])
        layout, fields, data = state
        reference = {}
        order = VARIANTS[index:] + VARIANTS[:index]
        try:
            for first_kind, next_kind in order:
                checks = []

                def factory(fine, ras):
                    return hierarchy(config, p, legacy, bridge, layout, fields, fine, ras,
                                     teacher, seed, first_kind, next_kind)

                def observe(displacement, result):
                    checks.append(first_parity(p, bridge, displacement, first_kind, reference))

                result = solve_policy(config, p, replay, legacy, bridge, None, frames[key], layout,
                    {"constrained":fields["constrained"]}, data,
                    context=dict(displacement_factory=factory, archive_choices=False,
                                 displacement_observer=observe))
                row = compact(result, frames[key], first_kind + "/" + next_kind)
                row.update(first_generator=first_kind, second=next_kind,
                           p1_parity=checks[0] if checks else None,
                           level_stats=[result["numerical"].get("level1"),
                                        result["numerical"].get("level2")],
                           setup_parts=result["timings"], terminal=result["numerical"].get("terminal"))
                if first_kind == next_kind and result.get("iterations") is not None:
                    original = baseline[key, "factorized-teacher" if first_kind == "factorized"
                                        else "direct-distilled"]
                    row["baseline_iteration_delta"] = min(abs(row["iterations"]-i) for i in original)
                if len(reference) == 2:
                    common = np.intersect1d(reference["direct"][0], reference["factorized"][0])
                    row["shared_c1"] = int(p.reduce([len(common)])[0])
                if p.rank == 0:
                    local.append(row)
                    record(config, "direct_isolation_solve", job=job, **row)
        finally:
            p.destroy("Vec", data[3], data[4])
            p.destroy("Mat", data[2])
    rows = [row for part in global_objects(p, local) for row in part]
    expected = {(key,a,b) for key in keys for a,b in VARIANTS}
    if len(rows) != len(expected) or {(r["key"],r["first_generator"],r["second"]) for r in rows} != expected:
        raise ValueError("Crossed diagnostic solves are incomplete")
    verified = all(row["passed"] and row["p1_parity"]["same_support"]
                   and row["p1_parity"]["coefficient_error"]<=5e-12
                   and row.get("baseline_iteration_delta",0)<=settings["reproduction_steps"]
                   for row in rows)
    if p.global_rank == 0:
        receipt = dict(job=job, parent=parent, distilled=settings["distilled"], teacher=prior["teacher"],
                       keys=keys, rows=rows, verified=verified,
                       scope="Crossed two-level generators and exact one-level controls on identical matrices.")
        save_once(output / "records" / ("direct-isolate-" + job + ".json"), receipt)
        record(config, "direct_isolation_complete", job=job, verified=verified,
               solves=len(rows), accurate=sum(row["passed"] for row in rows))
    return 0
