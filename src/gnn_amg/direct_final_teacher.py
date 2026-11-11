"""Export the executed two-transfer matrix-only teacher for direct generation.

Each state contains rank-local CSR shards, persistent native IDs for every
level, and raw replay probes. The original snapshot supplies the fine matrix.
"""
import os
from pathlib import Path

import numpy as np

from .backend import digest, local_csr, ownership, read_json, record
from .factorized import load_model, normalized_coordinates, region_ids
from .online import global_objects, load_observed_state, solve_policy
from .student_cycle import compact, save_once


def sparse_cost(matrix, columns):
    start, stop = columns
    remote = matrix.indices[(matrix.indices < start) | (matrix.indices >= stop)]
    return matrix.nnz, len(remote), np.unique(remote).size


def audit_transfer(matrix, native, coarse_native, fixed, columns, unit_sum):
    """Check exact nested injection, coverage and optional constant reproduction."""
    degree = np.diff(matrix.indptr)
    selected = np.isin(native, coarse_native)
    positions = matrix.indptr[:-1][selected]
    injection = (
        selected.sum() == len(coarse_native)
        and not np.any(fixed[selected])
        and np.all(degree[selected] == 1)
        and np.array_equal(native[selected], coarse_native)
        and np.array_equal(matrix.indices[positions], columns[0] + np.arange(len(coarse_native)))
        and np.all(matrix.data[positions] == 1)
    )
    errors = (
        np.count_nonzero((degree == 0) & ~fixed),
        np.count_nonzero((degree > 0) & fixed),
        int(not injection),
        int(not np.isfinite(matrix.data).all()),
    )
    row_error = None
    if unit_sum:
        row_error = float(np.max(np.abs(np.asarray(matrix.sum(axis=1)).ravel() - ~fixed),
                                 initial=0.0))
    return errors, row_error


def real_residuals(samples, iterations):
    """Label distinct true-residual samples, preserving their actual iteration IDs."""
    values = {iteration: residual for iteration, residual in samples.values()}
    steps = sorted(values)
    if not steps:
        return []
    chosen = [("early", steps[0])]
    if len(steps) >= 3:
        chosen.append(("middle", min(steps[1:-1], key=lambda i: abs(i - iterations / 2))))
    if len(steps) >= 2:
        chosen.append(("late", steps[-1]))
    return [(phase, i, values[i]) for phase, i in chosen]


def supplemental_probes(p, native, fixed, coordinates, index):
    """Cover small component/region directions without relying on phase fields."""
    region = region_ids(coordinates[native // 3])
    groups = 3 * region + native % 3
    counts = p.reduce(np.bincount(groups[~fixed], minlength=24)).reshape((8, 3))
    rng = np.random.default_rng(np.random.SeedSequence([20261110, index, p.rank]))
    values = rng.standard_normal(len(native))
    for component in range(3):
        regions = np.flatnonzero(counts[:, component])
        if not len(regions):
            continue
        selected = int(regions[index % len(regions)])
        mask = ~fixed & (groups == 3 * selected + component)
        residual = np.where(mask, values, 0.0)
        length = np.sqrt(p.reduce([residual @ residual])[0])
        if length > 0:
            yield "region-%d-component-%d" % (selected, component), residual / length
    residual = np.where(fixed, 0.0, values)
    length = np.sqrt(p.reduce([residual @ residual])[0])
    if length > 0:
        yield "random-all-free", residual / length


def add_csr(arrays, name, matrix):
    arrays.update({name + "_indptr": matrix.indptr, name + "_indices": matrix.indices,
                   name + "_values": matrix.data, name + "_columns": matrix.shape[1]})


def archive_state(config, p, bridge, model, row, index, state, displacement, result,
                  calls, samples, directory, role):
    layout, fields, (_, full_rows, full, _, _) = state
    fine_owners = layout["ownership"]
    start, stop = map(int, fine_owners[p.rank:p.rank + 2])
    native = 3 * layout["native_node"][start:stop] + layout["component"][start:stop]
    fixed = np.asarray(fields["constrained"][native], dtype=bool)
    full_owners = ownership(p, full)
    arrays = dict(c0_native=native.astype(np.int64), c0_owners=fine_owners,
                  constrained=fixed, full_owners=full_owners, fine_full_rows=full_rows,
                  fine_coordinates=layout["xyz"][native // 3])
    parent_native, parent_owners, parent_fixed = native, fine_owners, fixed
    reports = []
    for depth, level in enumerate((displacement.first, displacement.second), 1):
        transfer, row_owners = local_csr(p, level.transfer, bridge)
        matrix, coarse_owners = local_csr(p, level.coarse, bridge)
        if not (np.array_equal(row_owners, parent_owners)
                and np.array_equal(coarse_owners, level.owners)
                and np.array_equal(ownership(p, level.transfer, columns=True), level.owners)
                and len(level.native) == np.diff(level.owners)[p.rank]
                and transfer.shape[1] == matrix.shape[1] == level.owners[-1]):
            raise ValueError("Inconsistent PETSc ownership on level %d" % depth)
        transfer.sort_indices()
        matrix.sort_indices()
        transfer.eliminate_zeros()
        columns = level.owners[p.rank:p.rank + 2]
        errors, row_error = audit_transfer(transfer, parent_native, level.native, parent_fixed,
                                          columns, model.unit_sum_depth >= (3 if depth == 1 else 5))
        errors = p.reduce([*errors, int(not np.isfinite(matrix.data).all())]).astype(int)
        row_error = p.maximum(row_error) if row_error is not None else None
        if np.any(errors) or (row_error is not None and row_error > 3e-5):
            raise ValueError("Teacher CSR audit failed on %s level %d: %s, row sum %s" %
                             (row["key"], depth, errors.tolist(), row_error))
        metrics = ("nnz", "off_rank_entries", "remote_columns_rank_sum")
        p_cost = dict(zip(metrics, p.reduce(sparse_cost(transfer, columns)).astype(int).tolist()))
        k_cost = dict(zip(metrics, p.reduce(sparse_cost(matrix, columns)).astype(int).tolist()))
        if p_cost["nnz"] != level.stats["nnz"]:
            raise ValueError("Teacher nonzero count changed during export")
        reports.append(dict(depth=depth, fine_dofs=int(parent_owners[-1]),
                            coarse_dofs=int(level.owners[-1]),
                            shrinkage=float(level.owners[-1] / parent_owners[-1]),
                            p=p_cost, k=k_cost, row_sum_error=row_error,
                            audit_errors=errors.tolist()))
        arrays["c%d_native" % depth] = level.native
        arrays["c%d_owners" % depth] = level.owners
        add_csr(arrays, "p%d" % depth, transfer)
        add_csr(arrays, "k%d" % depth, matrix)
        parent_native, parent_owners = level.native, level.owners
        parent_fixed = np.zeros(len(parent_native), dtype=bool)

    windows = real_residuals(samples, result["iterations"])
    if result["iterations"] > 0 and (not calls or not windows):
        raise ValueError("Missing actual replay probes on " + row["key"])
    arrays["gmres_phase"] = np.asarray([phase for phase, _, _ in windows], dtype="U6")
    arrays["gmres_iteration"] = np.asarray([step for _, step, _ in windows], dtype=np.int32)
    arrays["gmres_inputs"] = (np.stack([x for _, _, x in windows]) if windows else
                             np.empty((0, int(np.diff(full_owners)[p.rank]))))
    actual = [("pcapply-%d" % call, residual) for call, residual in calls]
    supplementary = supplemental_probes(p, native, fixed, normalized_coordinates(layout["xyz"]),
                                         index)
    inputs, targets, defects, probe_rows = [], [], [], []
    for name, residual in [*actual, *supplementary]:
        defect = residual - displacement.fine(displacement.local.apply(residual))
        target = displacement.correction(residual)
        norms = p.reduce([residual @ residual, defect @ defect, target @ target])
        if not np.isfinite(norms).all():
            raise ValueError("Non-finite teacher action on " + row["key"] + "/" + name)
        inputs.append(residual)
        defects.append(defect)
        targets.append(target)
        probe_rows.append(dict(name=name, input_l2=float(np.sqrt(norms[0])),
                               post_ras_l2=float(np.sqrt(norms[1])),
                               correction_l2=float(np.sqrt(norms[2]))))
    arrays.update(pcapply_names=np.asarray([item["name"] for item in probe_rows]),
                  pcapply_inputs=np.stack(inputs), pcapply_teacher=np.stack(targets),
                  post_ras_residuals=np.stack(defects))

    folder = directory / row["key"]
    if p.rank == 0:
        folder.mkdir()
    p.barrier()
    path = folder / ("rank-%03d.npz" % p.rank)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    checksums = p.gather_equal(np.frombuffer(bytes.fromhex(digest(path)), np.uint8)).astype(np.uint8)
    sizes = p.gather_equal([path.stat().st_size]).ravel().astype(np.int64)
    receipt = dict(key=row["key"], role=role, snapshot=row["directory"],
                   snapshot_sha256=row["manifest_sha256"], model=config["factor_model"],
                   directory=str(folder), shard_pattern="rank-%03d.npz",
                   shards=[dict(rank=rank, bytes=int(size), sha256=sha.tobytes().hex())
                           for rank, (sha, size) in enumerate(zip(checksums, sizes))],
                   levels=reports, actual_pcapply=len(actual), probes=probe_rows,
                   gmres_samples=[dict(phase=phase, iteration=i) for phase, i, _ in windows],
                   solve=compact(result, row, "matrix-only-factorized"))
    if p.rank == 0:
        save_once(folder / "receipt.json", receipt)
    return receipt


def run_teacher(config, p, replay, legacy, bridge, inventory):
    """Use the exact hierarchy from each real PETSc solve for the export."""
    settings = config["direct_final_teacher"]
    job = os.environ["GNN_RUN_ID"]
    keys = settings["keys"]
    diagnostic = set(settings["diagnostic_keys"])
    frames = {row["key"]: row for row in inventory["snapshots"]}
    if (len(set(keys)) != len(keys) or not diagnostic.issubset(keys)
            or any(frames[key]["split"] != "training" for key in keys)):
        raise ValueError("Final transfer export requires distinct frozen training states")
    teacher_receipt = settings["checkpoint_receipt"]
    if digest(teacher_receipt["path"]) != teacher_receipt["sha256"]:
        raise ValueError("Matrix-only teacher receipt changed")
    source = read_json(teacher_receipt["path"])
    if source["winner"] != config["factor_model"] or not source["live_eligible"]:
        raise ValueError("Wrong frozen teacher")
    model = load_model(config["factor_model"])
    if not model.matrix_only:
        raise ValueError("The final-transfer teacher requires a matrix-only checkpoint")
    output = Path(config["output_directory"])
    directory = output / "data" / ("final-teacher-" + job)
    if p.global_rank == 0:
        directory.mkdir()
        record(config, "final_teacher_begin", job=job, frames=len(keys),
               diagnostic=sorted(diagnostic), workers=config["parallel_workers"], ranks=p.size)
    global_objects(p, [str(directory)] if p.global_rank == 0 else [])
    local = []
    for index, key in enumerate(keys):
        if index % config["parallel_workers"] != p.worker:
            continue
        row = frames[key]
        role = "diagnostic" if key in diagnostic else "fit"
        if p.rank == 0:
            record(config, "final_teacher_state_begin", job=job, key=key, role=role)
        state = load_observed_state(config, p, replay, row)
        layout, fields, data = state
        calls, samples, exported = [], {}, []

        def observe(displacement, result):
            exported.append(archive_state(config, p, bridge, model, row, index, state,
                                          displacement, result, calls, samples, directory, role))

        try:
            result = solve_policy(config, p, replay, legacy, bridge, None, row, layout,
                                  {"constrained": fields["constrained"]}, data, context=dict(
                                      factor_model=model, factor_fusion=True, archive_choices=False,
                                      capture_inputs=calls, factor_residual_samples=samples,
                                      displacement_observer=observe))
            if p.rank == 0:
                receipt = exported[0] if exported else dict(key=key, role=role, shards=[],
                    solve=compact(result, row, "matrix-only-factorized"))
                local.append(receipt)
                record(config, "final_teacher_state_end", job=job, key=key, role=role,
                       passed=bool(receipt["shards"]), solve=receipt["solve"],
                       actual_pcapply=receipt.get("actual_pcapply", 0),
                       gmres_samples=receipt.get("gmres_samples", []))
        finally:
            p.destroy("Vec", data[3], data[4])
            p.destroy("Mat", data[2])
    all_rows = [receipt for group in global_objects(p, local) for receipt in group]
    indexed = {receipt["key"]: receipt for receipt in all_rows}
    if len(indexed) != len(all_rows) or set(indexed) != set(keys):
        raise ValueError("Incomplete teacher replay accounting")
    passed = all(len(row["shards"]) == p.size for row in all_rows)
    if p.global_rank == 0:
        report = dict(schema=1, job=job, passed=passed, source=settings["live_provenance"],
                      model=config["factor_model"], image_sha256=config["image_sha256"],
                      inventory_sha256=config["inventory_sha256"], fit=[k for k in keys if k not in diagnostic],
                      diagnostic=sorted(diagnostic), validation_keys=config["training"]["validation_keys"],
                      captured=sum(bool(row["shards"]) for row in all_rows),
                      bytes=sum(shard["bytes"] for row in all_rows for shard in row["shards"]),
                      rows=[indexed[key] for key in keys])
        target = output / "records" / ("direct-final-teacher-" + job + ".json")
        save_once(target, report)
        record(config, "final_teacher_complete", job=job, passed=passed,
               captured=report["captured"], bytes=report["bytes"], receipt=str(target))
    return 0 if passed else 2
