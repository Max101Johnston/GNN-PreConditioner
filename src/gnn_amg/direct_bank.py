"""Prepare graph contexts and complete sampled edge targets for direct distillation."""
from dataclasses import replace
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .backend import digest, local_csr, read_json, record
from .direct_depth import depth_control, shard_csr
from .factorized import (STAGES, encoded_inputs, import_matrix, matrix_graph,
                         normalized_coordinates, region_ids)
from .online import global_objects, load_observed_state
from .student_cycle import save_once


def block_context(matrix, native, column_native):
    """The signed, row-scaled 3x3 diagonal stiffness block at each physical node."""
    component = native % 3
    physical, mapping = np.unique(native // 3, return_inverse=True)
    rows = np.repeat(np.arange(len(native)), np.diff(matrix.indptr))
    columns = column_native[matrix.indices]
    same = native[rows] // 3 == columns // 3
    i = rows[same]
    scale = np.maximum(np.asarray(abs(matrix).sum(axis=1)).ravel(), 1e-300)
    blocks = np.zeros((len(physical), 3, 3), dtype=np.float32)
    blocks[mapping[i], component[i], columns[same] % 3] = matrix.data[same] / scale[i]
    return blocks[mapping].reshape((-1, 9))


def graph_context(p, bridge, operator, native, coordinates, constrained, depth):
    """Use the same local stiffness neighbourhood and halo at both real levels."""
    from .dataset import ATTRIBUTE_NAMES

    attributes = np.zeros((len(native), len(ATTRIBUTE_NAMES)))
    attributes[:, -3:] = coordinates[native // 3]
    graph = matrix_graph(p, bridge, operator, native, attributes, constrained, np.ones(len(native)))
    owned, owners = local_csr(p, operator, bridge)
    global_native = p.all_rows(native, owners).astype(np.int64)
    if not np.array_equal(owners, graph.owners):
        raise ValueError("Direct graph and matrix ownership differ")
    compact = np.column_stack((graph.nodes[:, len(ATTRIBUTE_NAMES)-3:],
                               block_context(owned, native, global_native)))
    inputs = encoded_inputs(p, replace(graph, nodes=compact), 0 if depth == 1 else 3)
    # Graph messages see the actual block features. A level flag and the
    # current operator scale are shared inputs, independent of factor depth.
    features = np.column_stack((inputs[:, :-STAGES], np.full(len(native), np.log1p(owners[-1])/16),
                                np.full(len(native), depth == 1)))
    if not np.isfinite(features).all():
        raise ValueError("Non-finite direct graph inputs")
    return graph, features.astype(np.float32)


def region_context(p, features, region, valid, selected):
    slots = 8
    count = np.bincount(region[valid], minlength=slots)
    sums = np.zeros((slots, features.shape[1]), dtype=np.float64)
    np.add.at(sums, region[valid], features[valid])
    sums = p.reduce(sums)
    count = p.reduce(count).astype(np.int64)
    coarse = p.reduce(np.bincount(region[selected], minlength=slots)).astype(np.int64)
    return (sums/np.maximum(count[:, None], 1)).astype(np.float32), count, coarse


def sampled_rows(selected, fixed, difficulty, limit, rng):
    if limit is None:
        return np.flatnonzero(~fixed)
    coarse = np.flatnonzero(selected)
    fine = np.flatnonzero(~(selected | fixed))
    chosen_coarse = rng.choice(coarse, min(len(coarse), limit//4), replace=False)
    remaining = min(limit-len(chosen_coarse), len(fine))
    random = rng.choice(fine, remaining//2, replace=False)
    pool = np.setdiff1d(fine, random, assume_unique=True)
    if len(pool):
        hard = pool[np.argsort(-difficulty[pool], kind="stable")[:remaining-len(random)]]
    else:
        hard = pool
    return np.sort(np.r_[chosen_coarse, random, hard])


def edge_examples(matrix, sample, selected, native, endpoints, xyz, negative_count, rng):
    """Keep every signed teacher entry on sampled non-injection rows."""
    if not len(sample):
        return np.zeros(1, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)
    positions = xyz[endpoints//3]
    fine_positions = xyz[native[sample]//3]
    tree = cKDTree(positions)
    _, nearby = tree.query(fine_positions, k=min(negative_count*3, len(endpoints)))
    nearby = nearby.reshape(len(sample), -1)
    candidates, targets, pointers = [], [], [0]
    for i, row in enumerate(sample):
        if selected[row]:
            pointers.append(pointers[-1])
            continue
        a, b = matrix.indptr[row:row+2]
        positive = matrix.indices[a:b]
        if not len(positive):
            raise ValueError("A sampled fine row has no teacher support")
        random = rng.integers(0, len(endpoints), size=3*negative_count)
        local = nearby[i]
        local = local[np.argsort((endpoints[local] % 3 != native[row] % 3), kind="stable")]
        present = set(positive)
        negative = [int(c) for c in dict.fromkeys(np.r_[local[:negative_count], random])
                    if c not in present][:2*negative_count]
        edge = np.r_[positive, np.asarray(negative, np.int32)].astype(np.int32)
        value = np.r_[matrix.data[a:b], np.zeros(len(negative))].astype(np.float32)
        candidates.append(edge)
        targets.append(value)
        pointers.append(pointers[-1]+len(edge))
    columns = np.concatenate(candidates) if candidates else np.empty(0, np.int32)
    values = np.concatenate(targets) if targets else np.empty(0, np.float32)
    return np.asarray(pointers, np.int32), columns, values


def archive_bank_level(config, p, bridge, operator, shard, depth, xyz, folder, defects, rng):
    settings = config["direct_final"]
    native = shard["c%d_native" % (depth-1)]
    c_native = shard["c%d_native" % depth]
    owners = shard["c%d_owners" % depth]
    constrained = shard["constrained"] if depth == 1 else np.zeros(len(native), dtype=bool)
    selected = np.isin(native, c_native)
    if not np.array_equal(native[selected], c_native):
        raise ValueError("Teacher coarse order differs from executed row order")
    graph, features = graph_context(p, bridge, operator, native, xyz, constrained, depth)
    np.testing.assert_array_equal(graph.owners, shard["c%d_owners" % (depth-1)])
    width = features.shape[1]
    global_native = p.all_rows(c_native, owners).astype(np.int64)
    endpoints = p.all_rows(features[selected].ravel(), owners*width).reshape((-1, width))
    endpoint_region = region_ids(xyz[global_native // 3])
    regions = region_ids(xyz[native // 3])
    context, counts, coarse_count = region_context(p, features, regions, ~constrained, selected)
    matrix = shard_csr(shard, "p%d" % depth)
    norms = np.sqrt(p.reduce(np.sum(defects**2, axis=1)))
    difficulty = np.sqrt(np.mean((defects / np.maximum(norms[:, None], 1e-300))**2, axis=0))
    difficulty /= max(p.maximum(difficulty.max(initial=0.)), 1e-300)
    sample = sampled_rows(selected, constrained, difficulty,
                          settings["fine_rows_per_rank"] if depth == 1 else None, rng)
    offsets, edges, values = edge_examples(matrix, sample, selected, native, global_native,
                                            xyz, settings["negative_count"], rng)
    with (folder / ("l%d-r%03d.npz" % (depth, p.rank))).open("xb") as stream:
        np.savez_compressed(stream, features=features[sample], region=regions[sample],
            native=native[sample], selected=selected[sample], rank=np.full(len(sample), p.rank),
            xyz=xyz[native[sample]//3].astype(np.float32), difficulty=difficulty[sample].astype(np.float32),
            indptr=offsets, columns=edges, values=values)
    if p.rank == 0:
        with (folder / ("l%d-context.npz" % depth)).open("xb") as stream:
            np.savez_compressed(stream, regions=context, region_counts=counts,
                region_coarse_count=coarse_count, endpoint_features=endpoints.astype(np.float32),
                endpoint_region=endpoint_region, endpoint_xyz=xyz[global_native//3].astype(np.float32),
                endpoint_native=global_native,
                endpoint_rank=np.repeat(np.arange(p.size), np.diff(owners)))
    totals = p.reduce([len(sample), np.count_nonzero(selected[sample]),
                       np.count_nonzero(values), len(values)]).astype(np.int64)
    return dict(depth=depth, rows=int(totals[0]), coarse_labels=int(totals[1]),
                positive_edges=int(totals[2]), candidate_edges=int(totals[3]), features=width)


def run_prepare(config, p, replay, legacy, bridge, inventory):
    settings = config["direct_final"]
    source = settings["teacher"]
    if digest(source["path"]) != source["sha256"]:
        raise ValueError("Changed final-teacher receipt")
    teacher = read_json(source["path"])
    if not teacher["passed"] or teacher["captured"] != len(teacher["rows"]):
        raise ValueError("The direct training input has not passed")
    frames = {row["key"]: row for row in inventory["snapshots"]}
    job = os.environ["GNN_RUN_ID"]
    out = Path(config["output_directory"])
    bank = out / "data" / ("direct-bank-" + job)
    if p.global_rank == 0:
        bank.mkdir()
    global_objects(p, [str(bank)] if p.global_rank == 0 else [])
    replay.operator_bind(p)
    local_rows = []
    for index, receipt in enumerate(teacher["rows"]):
        if index % config["parallel_workers"] != p.worker:
            continue
        key = receipt["key"]
        row = frames[key]
        assert row["manifest_sha256"] == receipt["snapshot_sha256"]
        part = receipt["shards"][p.rank]
        file = Path(receipt["directory"]) / (receipt["shard_pattern"] % p.rank)
        if digest(file) != part["sha256"]:
            raise ValueError("Teacher CSR or probe shard changed on " + key)
        state = load_observed_state(config, p, replay, row)
        layout, _, data = state
        c0 = replay.operator_submatrix(p, data[2], data[1])
        c1 = None
        folder = bank / key
        if p.rank == 0:
            folder.mkdir()
            record(config, "direct_bank_begin", job=job, key=key, role=receipt["role"])
        p.barrier()
        checks, comparisons = [], []
        try:
            with np.load(file) as shard:
                xyz = normalized_coordinates(layout["xyz"])
                rng = np.random.default_rng(np.random.SeedSequence([settings["seed"], index, p.rank]))
                c1 = import_matrix(p, legacy, shard_csr(shard, "k1"),
                                   shard["c1_owners"], shard["c1_owners"])
                fine_defect = shard["post_ras_residuals"][:receipt["actual_pcapply"]]
                if not len(fine_defect):
                    raise ValueError("No actual post-RAS directions on " + key)
                begin, end = shard["c1_owners"][p.rank:p.rank+2]
                coarse_defect = p.reduce(shard_csr(shard, "p1").T @ fine_defect.T).T[:, begin:end]
                for depth, (operator, defects) in enumerate(
                        ((c0, fine_defect), (c1, coarse_defect)), 1):
                    checks.append(archive_bank_level(config, p, bridge, operator, shard, depth,
                                                     xyz, folder, defects, rng))
                if key in settings["depth"]["keys"]:
                    for repeat, order in enumerate(((1, 2, 3), (3, 2, 1)), 1):
                        for depth in order:
                            result = depth_control(config, p, replay, legacy, bridge, row, state,
                                                   shard, receipt, depth)
                            result["repeat"] = repeat
                            if p.rank == 0:
                                comparisons.append(result)
                                record(config, "direct_depth_solve", job=job, key=key, result=result)
        finally:
            p.destroy("Mat", c0)
            if c1 is not None:
                p.destroy("Mat", c1)
            p.destroy("Vec", data[3], data[4])
            p.destroy("Mat", data[2])
        if p.rank == 0:
            local_rows.append(dict(key=key, role=receipt["role"], levels=checks,
                                   depths=comparisons))
            record(config, "direct_bank_end", job=job, key=key, levels=checks)
    rows = [row for group in global_objects(p, local_rows) for row in group]
    if {row["key"] for row in rows} != set(teacher["fit"] + teacher["diagnostic"]):
        raise ValueError("Direct distillation coverage does not match teacher manifest")
    comparisons = [r for row in rows for r in row["depths"]]
    if len(comparisons) != 6*len(settings["depth"]["keys"]):
        raise ValueError("Incomplete coarse depth experiment")
    if p.global_rank == 0:
        summary = dict(job=job, source=source, directory=str(bank), shards=p.size, rows=rows,
                       fit=teacher["fit"], diagnostic=teacher["diagnostic"],
                       comparison_count=len(comparisons))
        save_once(out / "records" / ("direct-bank-" + job + ".json"), summary)
        record(config, "direct_bank_complete", job=job, fit=len(summary["fit"]),
               diagnostic=len(summary["diagnostic"]), depth_comparisons=len(comparisons))
    return 0
