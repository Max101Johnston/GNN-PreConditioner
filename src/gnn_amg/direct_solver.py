"""Matrix-only prediction of the actual coarse endpoints and transfers."""
from dataclasses import dataclass
import time

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.special import expit, softmax

from .backend import digest
from .direct_bank import graph_context, region_context
from .direct_final_teacher import audit_transfer, sparse_cost
from .factorized import (InvalidFactorization, candidates, coarse_owners, groups,
                         normalized_coordinates, pick_representatives, region_ids)
from .interpolation import imported_transfer
from .online import DisplacementPC, finish_hierarchy


def normalize_rows(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


class DirectWeights:
    """NumPy execution of the frozen supervised network in the existing SIF."""
    def __init__(self, arrays, coefficient=None):
        self.arrays = arrays
        self.width = int(arrays["width"])
        self.input_features = int(arrays["input_features"])
        self.coefficient = (np.broadcast_to(np.r_[arrays["edge_heads.weight"][1],
                                                arrays["edge_heads.bias"][1]],
                                          (2, self.width+1)).astype(np.float64).copy()
                            if coefficient is None else np.asarray(coefficient, dtype=np.float64))
        if self.coefficient.shape != (2, self.width+1) or not np.isfinite(self.coefficient).all():
            raise ValueError("Invalid per-level signed heads")

    @classmethod
    def load(cls, receipt):
        if digest(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Direct checkpoint identity changed")
        with np.load(receipt["path"]) as arrays:
            if int(arrays["schema"]) != 1:
                raise ValueError("Unsupported direct checkpoint schema")
            saved = {name: arrays[name] for name in arrays.files}
        return cls(saved, saved.get("signed_coefficients"))

    def adjusted(self, coefficient):
        return DirectWeights(self.arrays, coefficient)

    def layer(self, name, x):
        return x @ self.arrays[name + ".weight"].T + self.arrays[name + ".bias"]

    def context(self, features, available):
        h = np.tanh(self.layer("encoder", features))
        score = self.layer("region_query", h) @ self.layer("region_key", h).T / np.sqrt(self.width)
        attention = softmax(np.where(available[None, :], score, -1e9), axis=1)
        return h + attention @ self.layer("region_value", h)

    def nodes(self, features, ids, context):
        h = np.tanh(self.layer("encoder", features))
        return h + np.tanh(self.layer("fuse", np.concatenate([h, context[ids]], axis=-1)))

    def routing(self, hidden, context, available):
        route = self.layer("route_query", hidden) @ self.layer("route_key", context).T / np.sqrt(self.width)
        return np.where(available[None, :], route, -1e9)

    def edges(self, fine, coarse, descriptors, query, key, region, route):
        fine = np.broadcast_to(fine[:, None, :], coarse.shape)
        inputs = np.concatenate([fine, coarse, descriptors], axis=-1)
        h = np.tanh(self.layer("edge", inputs))
        address = 4. * np.sum(query[:, None, :] * key, axis=-1)
        score = (h @ self.arrays["edge_heads.weight"][0] + self.arrays["edge_heads.bias"][0]
                 + address + np.take_along_axis(route, region, axis=1))
        return score, h


def allocate(weights, capacity, total):
    """Bounded largest remainders, with deterministic exact integer totals."""
    capacity = np.asarray(capacity, dtype=np.int64)
    result = np.zeros_like(capacity)
    if not 0 <= total <= capacity.sum():
        raise ValueError("Coarse budget exceeds available rows")
    while total:
        room = capacity - result
        available = room > 0
        weight = np.where(available, weights, 0.)
        if not np.any(weight > 0):
            weight = room.astype(float)
        shares = total * (weight / weight.sum())
        base = np.minimum(np.floor(shares).astype(np.int64), room)
        result += base
        total -= int(base.sum())
        if total:
            eligible = np.flatnonzero(result < capacity)
            order = eligible[np.argsort(-(shares-base)[eligible], kind="stable")]
            take = order[:total]
            result[take] += 1
            total -= len(take)
    return result


def select_points(p, graph, region, logits, allocation, counts, total, learned_fraction):
    groups_id = 3*region + graph.native % 3
    local = np.bincount(groups_id[~graph.fixed], minlength=24)
    capacity = p.gather_equal(local).astype(np.int64).reshape((-1, 8, 3))
    probability = softmax(np.where(counts > 0, allocation, -1e9))
    weight = learned_fraction*probability + (1-learned_fraction)*counts/counts.sum()
    local_weight = weight[None, :, None]*capacity/np.maximum(counts[None, :, None], 1)
    quota = allocate(local_weight.ravel(), capacity.ravel(), total).reshape((-1, 24))[p.rank]
    selected = np.zeros(len(region), dtype=bool)
    scores = np.column_stack([logits, np.full(len(logits), -2.)])
    for group in np.flatnonzero(quota):
        eligible = (groups_id == group) & ~graph.fixed
        labels = groups(graph.native, graph.xyz, eligible, local[group]/quota[group])
        selected |= pick_representatives(graph.native, labels, scores)
    if np.count_nonzero(selected) != quota.sum():
        raise ValueError("Distributed coarse allocation did not use its exact budget")
    return selected


def routed_candidates(queries, keys, route, endpoint_region, rows, per_region):
    result = np.full((len(queries), 2*per_region), -1, dtype=np.int32)
    if not len(rows) or not per_region:
        return result
    pools = [np.flatnonzero(endpoint_region == region) for region in range(route.shape[1])]
    available = np.array([len(pool) > 0 for pool in pools])
    ranks = np.argsort(-np.where(available[None, :], route[rows], -np.inf), axis=1, kind="stable")
    for position in range(min(2, int(available.sum()))):
        chosen = ranks[:, position]
        for region, pool in enumerate(pools):
            owned = rows[chosen == region]
            if not len(owned):
                continue
            count = min(per_region, len(pool))
            _, nearest = cKDTree(keys[pool]).query(queries[owned], k=count)
            result[owned, position*per_region:position*per_region+count] = \
                pool[nearest.reshape((len(owned), count))]
    return result


def deduplicate(columns):
    ordered = np.sort(columns, axis=1)
    ordered[:, 1:] = np.where(ordered[:, 1:] == ordered[:, :-1], -1, ordered[:, 1:])
    return ordered


def project_rows(raw, rows, preferred, fixed, unit_sum, n):
    values = raw.copy()
    values[fixed] = 1.
    if unit_sum:
        count = np.bincount(rows[preferred], minlength=n)
        sums = np.bincount(rows, weights=values, minlength=n)
        values[preferred] += ((1.-sums)/np.maximum(count, 1))[rows[preferred]]
    return values


@dataclass
class DirectStencil:
    pattern: object
    rows: np.ndarray
    basis: np.ndarray
    preferred: np.ndarray
    injection: np.ndarray
    unit_sum: bool
    native: np.ndarray
    row_owners: np.ndarray
    column_owners: np.ndarray
    stats: dict

    def values(self, coefficient):
        raw = self.basis @ coefficient[:-1] + coefficient[-1]
        return project_rows(raw, self.rows, self.preferred, self.injection,
                            self.unit_sum, self.pattern.shape[0])

    def rebuild(self, p, legacy, parent, coefficient):
        transfer = sparse.csr_matrix((self.values(coefficient), self.pattern.indices,
                                      self.pattern.indptr), shape=self.pattern.shape)
        level = imported_transfer(p, legacy, parent, transfer, self.row_owners,
                                  self.column_owners, self.stats)
        level.native = self.native
        level.product_pattern = self.pattern
        level.direct_stencil = self
        return level

    def pullback(self, gradient):
        if not (np.array_equal(self.pattern.indptr, gradient.indptr)
                and np.array_equal(self.pattern.indices, gradient.indices)):
            raise ValueError("Direct adjoint sparsity changed")
        upstream = gradient.data.copy()
        if self.unit_sum:
            count = np.bincount(self.rows[self.preferred], minlength=self.pattern.shape[0])
            weighted = np.bincount(self.rows[self.preferred], weights=upstream[self.preferred],
                                   minlength=self.pattern.shape[0])
            upstream -= (weighted/np.maximum(count, 1))[self.rows]
        upstream[self.injection] = 0.
        return np.r_[upstream @ self.basis, upstream.sum()]


class InvalidDirect(ValueError):
    pass


def generate_level(config, p, legacy, bridge, parent, native, coordinates, fixed, model, depth,
                   retain_gradient):
    settings = config["direct_solve"]
    started = time.perf_counter()
    graph, inputs = graph_context(p, bridge, parent, native, coordinates, fixed, depth)
    if inputs.shape[1] != model.input_features:
        raise ValueError("Direct checkpoint input width differs from the matrix encoder")
    regions = region_ids(graph.xyz)
    pool, count, _ = region_context(p, inputs, regions, ~fixed, np.zeros(len(fixed), bool))
    context = model.context(pool, count > 0)
    hidden = model.nodes(inputs, regions, context)
    stage = depth-1
    selected = select_points(p, graph, regions, model.layer("coarse", hidden).ravel(),
                             model.layer("region_budget", context).ravel(), count,
                             settings["coarse_dofs"][stage], settings["learned_coarse_fraction"])
    owners = coarse_owners(p, selected)
    width = model.width
    endpoint_h = p.all_rows(hidden[selected].ravel(), width*owners).reshape((-1, width)).astype(np.float32)
    graph_s = p.maximum(time.perf_counter()-started)
    started = time.perf_counter()
    options = dict(config["factorized"], geometry_candidates=settings["geometry"][stage],
                   graph_candidates=settings["algebraic"][stage])
    base, _, _, _, endpoint_native = candidates(dict(config, factorized=options), p, graph, selected,
                                                 owners, coordinates, 0 if depth == 1 else 3)
    endpoint_xyz = coordinates[endpoint_native//3]
    endpoint_region = region_ids(endpoint_xyz)
    endpoint_rank = np.repeat(np.arange(p.size), np.diff(owners))
    route = model.routing(hidden, context, count > 0)
    queries = normalize_rows(model.layer("query", hidden))
    keys = normalize_rows(model.layer("key", endpoint_h))
    active = np.flatnonzero(~(selected | fixed))
    regional = routed_candidates(queries, keys, route, endpoint_region, active,
                                 settings["route_per_region"][stage])
    global_geo = np.full((len(fixed), settings["global_geometry"][stage]), -1, dtype=np.int32)
    if len(active) and global_geo.shape[1]:
        _, nearest = cKDTree(endpoint_xyz).query(graph.xyz[active], k=global_geo.shape[1])
        global_geo[active] = nearest.reshape((len(active), global_geo.shape[1]))
    columns = deduplicate(np.column_stack([base, global_geo, regional]))
    candidate_s = p.maximum(time.perf_counter()-started)
    started = time.perf_counter()
    cap = config["direct_final"]["distillation"]["caps"][stage]
    predicted = np.expm1(expit(model.layer("row_budget", hidden).ravel())*np.log1p(cap))
    quota = np.clip(np.rint(settings["quota_scale"][stage]*predicted).astype(np.int32),
                    settings["minimum_support"][stage], cap)
    rows, kept, weights, bases = [], [], [], []
    for start in range(0, len(active), settings["row_chunk"]):
        r = active[start:start+settings["row_chunk"]]
        col = columns[r]
        valid = col >= 0
        ids = np.maximum(col, 0)
        same = (endpoint_native[ids] % 3) == (native[r, None] % 3)
        delta = (endpoint_xyz[ids] - graph.xyz[r, None, :]).astype(np.float32)
        remote = endpoint_rank[ids] != p.rank
        edge = np.concatenate([delta, np.linalg.norm(delta, axis=-1)[..., None],
                               same[..., None], remote[..., None]], axis=-1).astype(np.float32)
        score, h = model.edges(hidden[r], endpoint_h[ids], edge, queries[r], keys[ids],
                               endpoint_region[ids], route[r])
        order = np.argsort(-np.where(valid, score, -np.inf), axis=1, kind="stable")
        take = np.zeros_like(valid)
        np.put_along_axis(take, order, np.arange(col.shape[1]) < quota[r, None], axis=1)
        i, j = np.where(take & valid)
        rows.append(r[i])
        kept.append(col[i, j])
        basis = h[i, j]
        coeff = model.coefficient[stage]
        weights.append(basis @ coeff[:-1] + coeff[-1])
        if retain_gradient:
            bases.append(basis)
    inject = np.flatnonzero(selected)
    rows.append(inject)
    kept.append(np.arange(owners[p.rank], owners[p.rank+1], dtype=np.int32))
    weights.append(np.ones(len(inject)))
    if retain_gradient:
        bases.append(np.zeros((len(inject), width), dtype=np.float32))
    rr, cc = np.concatenate(rows), np.concatenate(kept)
    order = np.lexsort((cc, rr))
    rr, cc = rr[order], cc[order]
    injection = selected[rr]
    same = (endpoint_native[cc] % 3) == (native[rr] % 3)
    has_same = np.bincount(rr[same], minlength=len(fixed)) > 0
    preferred = same | ~has_same[rr]
    values = project_rows(np.concatenate(weights)[order], rr, preferred, injection,
                          depth == 1, len(fixed))
    offset = np.r_[0, np.cumsum(np.bincount(rr, minlength=len(fixed)))]
    transfer = sparse.csr_matrix((values, cc, offset), shape=(len(fixed), int(owners[-1])))
    bound = owners[p.rank:p.rank+2]
    errors, sum_error = audit_transfer(transfer, native, native[selected], fixed, bound, depth == 1)
    zeros = int(np.count_nonzero((np.bincount(rr, weights=np.abs(values), minlength=len(fixed))==0) & ~fixed))
    faults = p.reduce([*errors, zeros]).astype(int)
    row_error = p.maximum(sum_error) if sum_error is not None else None
    if np.any(faults) or (row_error is not None and row_error > 1e-10):
        raise InvalidDirect("direct_transfer_audit_l%d: %s" % (depth, faults.tolist()))
    nnz, off_rank, remote_cols = p.reduce(sparse_cost(transfer, bound)).astype(int)
    stats = dict(fine_dofs=int(graph.owners[-1]), coarse_dofs=int(owners[-1]), nnz=int(nnz),
                 off_rank_entries=int(off_rank), remote_columns_rank_sum=int(remote_cols),
                 row_sum_error=row_error, empty_free_rows=0, injection_error=0,
                 graph_context_s=graph_s, candidate_s=candidate_s,
                 native_inference_s=p.maximum(time.perf_counter()-started),
                 region_coarse_counts=p.reduce(np.bincount(regions[selected], minlength=8)).astype(int).tolist())
    started = time.perf_counter()
    level = imported_transfer(p, legacy, parent, transfer, graph.owners, owners, stats)
    level.native = native[selected]
    if retain_gradient:
        level.product_pattern = transfer
        level.direct_stencil = DirectStencil(transfer, rr, np.concatenate(bases)[order], preferred,
                                             injection, depth == 1, level.native, graph.owners, owners, stats)
    stats["galerkin_s"] = p.maximum(time.perf_counter()-started)
    return level


def prepare_direct(config, p, legacy, bridge, layout, fixed, fine, ras, model, retain_gradient=False):
    first = second = space = None
    numerical, timings, failure = {}, {}, None
    lo, hi = layout["ownership"][p.rank:p.rank+2]
    native = 3*layout["native_node"][lo:hi] + layout["component"][lo:hi]
    coordinates = normalized_coordinates(layout["xyz"])
    try:
        first = generate_level(config, p, legacy, bridge, fine.matrix, native, coordinates,
                               fixed[native], model, 1, retain_gradient)
        numerical["level1"] = first.stats
        second = generate_level(config, p, legacy, bridge, first.coarse, first.native,
                                coordinates, np.zeros(len(first.native), dtype=bool),
                                model, 2, retain_gradient)
        numerical["level2"] = second.stats
        space, failure = finish_hierarchy(config, p, legacy, bridge, fine, first, second, numerical)
    except (InvalidDirect, InvalidFactorization) as error:
        failure = str(error)
    return DisplacementPC(p, fine, ras, first, second, space, None, None, None, None,
                          numerical, timings, failure)
