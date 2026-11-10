"""Learn five local transfers and compose the two Galerkin levels."""
import ctypes as C
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from .backend import Action, bind, digest, local_csr, neighbour_blocks, ownership, pooled_attributes, read_json
from .dataset import ATTRIBUTE_NAMES, physical_attributes
from .interpolation import imported_transfer
from .student import (CONTEXT_SCORE_SCALE, geometry_candidates, match_targets, pack, row_csr,
                      stiffness_features)

DEPTHS = (3, 2)
NODE_FEATURES = len(ATTRIBUTE_NAMES) + 5 + 3 + 2
EDGE_FEATURES = 8
STAGES = sum(DEPTHS)
REGIONS = 8
MAX_SUPPORT = 6


class InvalidFactorization(ValueError):
    def __init__(self, reason, stats=None):
        super().__init__(reason)
        self.stats = stats or {}


def layer_sizes(width):
    return ((3 * NODE_FEATURES + STAGES, width), (width, 2),
            (2 * width + EDGE_FEATURES, width), (2 * width, width),
            (width, 3), (2 * width, 1))


def coarse_owners(p, selected):
    counts = p.gather_equal([np.count_nonzero(selected)]).ravel().astype(np.int64)
    return np.r_[0, np.cumsum(counts)]


def region_ids(xyz):
    return (xyz >= 0) @ np.array([1, 2, 4], dtype=np.int32)


def groups(native, xyz, eligible, group_size):
    """Spatial neighbourhoods limit the size of each joint selection."""
    ids = np.flatnonzero(eligible)
    coords = np.clip((xyz[ids] + .5) * 1023, 0, 1023).astype(np.int32)
    morton = np.zeros(len(ids), np.int32)
    for bit in range(10):
        morton |= (((coords >> bit) & 1) @ np.array([1, 2, 4], np.int32)) << (3 * bit)
    component = native[ids] % 3
    order = np.lexsort((native[ids], morton, component))
    ids, component = ids[order], component[order]
    starts = np.r_[0, np.flatnonzero(np.diff(component)) + 1, len(ids)]
    labels = np.full(len(native), -1, np.int32)
    offset = 0
    for begin, end in zip(starts[:-1], starts[1:]):
        length = end - begin
        if length:
            count = max(1, int(np.floor(length / group_size + .5)))
            labels[ids[begin:end]] = offset + np.arange(length) * count // length
            offset += count
    return labels


def pick_representatives(native, labels, logits, limit=None):
    eligible = np.flatnonzero(labels >= 0)
    if not len(eligible):
        return labels >= 0
    label = labels[eligible]
    quota = np.bincount(label, weights=logits[eligible, 1]) / np.bincount(label)
    quota = np.clip(np.floor(2 + quota + .5), 1, 3)
    ordered = eligible[np.lexsort((native[eligible], -logits[eligible, 0], label))]
    counts = np.bincount(labels[ordered])
    ranks = np.arange(len(ordered)) - np.repeat(np.r_[0, np.cumsum(counts[:-1])], counts)
    selected = np.zeros(len(native), bool)
    mandatory = ordered[ranks == 0]
    optional = ordered[(ranks > 0) & (ranks < quota[labels[ordered]])]
    if limit is not None:
        priority = np.lexsort((native[optional], -logits[optional, 0]))
        optional = optional[priority[:max(0, limit - len(mandatory))]]
    selected[mandatory] = True
    selected[optional] = True
    return selected


def support_order(scores, valid, same):
    """Stable hard top-k with one same-component connection where available."""
    # The fixed ranking grid makes near ties stable across the training
    # accelerator and rank-local inference arithmetic.
    scores = np.where(valid, np.rint(CONTEXT_SCORE_SCALE * scores), -np.inf)
    anchor = np.argmax(np.where(same & valid, scores, -np.inf), axis=1)
    scores[np.arange(len(scores)), anchor] = np.where(
        (same & valid).any(1), np.inf, scores[np.arange(len(scores)), anchor])
    return np.argsort(-scores, axis=1, kind="stable")


def support_mask(scores, quota, valid, same):
    order = support_order(scores, valid, same)
    quota = np.rint(CONTEXT_SCORE_SCALE * quota) / CONTEXT_SCORE_SCALE
    budget = np.clip(np.floor(3.5 + quota), 1, MAX_SUPPORT)
    keep = np.zeros(valid.shape, bool)
    np.put_along_axis(keep, order, np.arange(order.shape[1]) < budget[:, None], axis=1)
    return keep & valid


def coefficients(logit, correction, keep, same, unit_row_sum=False):
    """Decode signed weights on the learned support."""
    primary = keep & same
    if unit_row_sum:
        primary = np.where(primary.any(axis=1, keepdims=True), primary, keep)
    scores = np.where(primary, np.clip(logit, -12, 12), -np.inf)
    maximum = np.max(scores, axis=1, keepdims=True)
    exp = np.exp(scores - np.where(np.isfinite(maximum), maximum, 0))
    base = exp / np.maximum(exp.sum(1, keepdims=True), 1e-30)
    signed = .75 * np.tanh(correction)
    if unit_row_sum:
        # Zero-sum signed corrections preserve the simplex's row sum. The
        # exact product then has coverage even when individual paths cancel.
        signed *= keep
        center = signed.sum(1, keepdims=True) / np.maximum(keep.sum(1, keepdims=True), 1)
        signed = signed - center
    return keep * (base + signed)


@dataclass
class FactorModel:
    width: int
    weights: np.ndarray
    adapters: np.ndarray
    unit_sum_depth: int = 0

    def __post_init__(self):
        self.layers = []
        offset = 0
        for inputs, outputs in layer_sizes(self.width):
            end = offset + inputs * outputs
            self.layers.append((self.weights[offset:end].reshape(outputs, inputs),
                                self.weights[end:end + outputs]))
            offset = end + outputs
        if offset != self.weights.size or self.adapters.shape != (STAGES, REGIONS, 6):
            raise ValueError("Invalid factorized model dimensions")
        if not 0 <= self.unit_sum_depth <= STAGES:
            raise ValueError("Invalid row-sum constraint depth")
        if not np.isfinite(self.weights).all() or not np.isfinite(self.adapters).all():
            raise ValueError("Non-finite factorized model")

    def linear(self, index, features):
        weight, bias = self.layers[index]
        return features @ weight.T + bias

    def node_outputs(self, x, stage, regions):
        hidden = np.maximum(0, self.linear(0, x))
        logits = self.linear(1, hidden)
        adapter = self.adapters[stage, regions]
        logits[:, 0] += adapter[:, 0] * (hidden.mean(1) + .25*x[:, len(ATTRIBUTE_NAMES)+1])
        logits[:, 1] += adapter[:, 1]
        return hidden, logits

    def edge_outputs(self, fine, coarse, edges, valid, stage, regions, tape=None):
        weight, bias = self.layers[2]
        w = self.width
        hidden = np.maximum(0, (fine @ weight[:, :w].T)[:, None, :] +
            coarse @ weight[:, w:2*w].T + edges @ weight[:, 2*w:].T + bias)
        summary = (hidden * valid[:, :, None]).sum(1) / np.maximum(valid.sum(1)[:, None], 1)
        summary = summary.astype(np.float32)
        weight, bias = self.layers[3]
        refined = np.maximum(0, hidden @ weight[:, :w].T + (summary @ weight[:, w:].T)[:, None, :] + bias)
        if tape is not None:
            tape.extend((hidden, summary, refined))
        out = self.linear(4, refined)
        quota = self.linear(5, np.column_stack((fine, summary))).ravel()
        adapter = self.adapters[stage, regions]
        out[:, :, 0] += adapter[:, 2, None] * edges[:, :, 0] + adapter[:, 3, None] * edges[:, :, 1]
        out[:, :, 2] += adapter[:, 4, None] * (edges[:, :, 1] + edges[:, :, 2] - .5)
        quota += adapter[:, 5]
        return out, quota


def save_model(path, model):
    with Path(path).open("xb") as stream:
        np.savez(stream, width=model.width, weights=model.weights, adapters=model.adapters,
                 unit_sum_depth=model.unit_sum_depth)
    return dict(path=str(path), sha256=digest(path))


def load_model(receipt):
    if digest(receipt["path"]) != receipt["sha256"]:
        raise ValueError("Factorized checkpoint changed")
    with np.load(receipt["path"]) as f:
        depth = int(f["unit_sum_depth"]) if "unit_sum_depth" in f.files else 0
        return FactorModel(int(f["width"]), f["weights"], f["adapters"], depth)


@dataclass
class Graph:
    adjacency: sparse.csr_matrix
    owners: np.ndarray
    native: np.ndarray
    nodes: np.ndarray
    fixed: np.ndarray

    @property
    def xyz(self):
        return self.nodes[:, len(ATTRIBUTE_NAMES)-3:len(ATTRIBUTE_NAMES)]


def node_features(attributes, algebra, native, fixed, mass):
    attributes = np.column_stack((np.tanh(.25 * attributes[:, :-3]), attributes[:, -3:]))
    return np.column_stack((attributes, algebra[:, :5], np.eye(3)[native % 3],
                            fixed, .1 * np.log1p(mass))).astype(np.float32)


def matrix_graph(p, bridge, matrix, native, attributes, fixed, mass):
    owners = ownership(p, matrix)
    components = p.all_rows(native % 3, owners).astype(np.int32)
    _, algebra, cols, weights = stiffness_features(p, bridge, matrix, components, len(native), 8)
    adjacency = row_csr(cols, weights, (cols >= 0) & (weights != 0), owners[-1])
    return Graph(adjacency, owners, native, node_features(attributes, algebra, native, fixed, mass), fixed)


def encoded_inputs(p, graph, stage):
    columns, inverse = np.unique(graph.adjacency.indices, return_inverse=True)
    halo = neighbour_blocks(p, graph.owners, graph.nodes, columns)
    w = graph.adjacency
    signed = sparse.csr_matrix((w.data.astype(np.float32), inverse, w.indptr),
                               shape=(len(graph.native), len(columns)))
    positive = abs(signed)
    norm = np.maximum(np.asarray(positive.sum(axis=1)), 1e-30)
    stage_id = np.broadcast_to(np.eye(STAGES, dtype=np.float32)[stage], (len(graph.native), STAGES))
    return np.column_stack((graph.nodes, (positive @ halo) / norm,
                            (signed @ halo) / norm, stage_id))


def representative_domain(config, p, graph, stage):
    size = config["factorized"]["group_sizes"][stage]
    limit = config["terminal_max_dofs"] if stage == STAGES - 1 else None
    if stage < DEPTHS[0]:
        return graph.native, groups(graph.native, graph.xyz, ~graph.fixed, size), slice(None), limit
    begin, end = graph.owners[p.rank:p.rank+2]
    native = p.all_rows(graph.native, graph.owners).astype(np.int64)
    xyz = p.all_rows(graph.xyz.ravel(), graph.owners * 3).reshape((-1, 3))
    if limit is not None:
        size = max(size, int(np.ceil(2 * len(native) / limit)))
    return native, groups(native, xyz, np.ones(len(native), bool), size), slice(begin, end), limit


def selections(config, p, graph, stage, logits):
    native, labels, owned, limit = representative_domain(config, p, graph, stage)
    if stage >= DEPTHS[0]:
        logits = p.all_rows(logits.ravel(), graph.owners * 2).reshape((-1, 2))
    return pick_representatives(native, labels, logits, limit)[owned]


def import_matrix(p, legacy, csr, rows, cols):
    result = legacy.DistributedSparse(p, csr, rows, cols)
    p.destroy("Vec", result.right, result.left)
    return result.matrix


def multiply(p, left, right, left_transpose=False):
    product = p.P()
    name = "MatTransposeMatMult" if left_transpose else "MatMatMult"
    bind(p, name, [p.P, p.P, p.I, p.D, C.POINTER(p.P)])
    p.call(name, left, right, 0, -2.0, C.byref(product))
    return product


def compose(p, factors):
    product = multiply(p, factors[0], factors[1])
    if len(factors) == 3:
        try:
            return multiply(p, product, factors[2])
        finally:
            p.destroy("Mat", product)
    return product


def signed_contacts(graph, assignment):
    w = graph.adjacency
    # The neighbour assignments are already in global coarse-column order.
    valid = assignment >= 0
    offsets = np.r_[0, np.cumsum(np.bincount(
        np.repeat(np.arange(w.shape[0]), np.diff(w.indptr))[valid],
        minlength=w.shape[0]))]
    return offsets, assignment[valid], w.data[valid]


def candidates(config, p, graph, selected, owners, coordinates, stage):
    """Use geometry and stiffness edges between the current local groups."""
    start, stop = owners[p.rank:p.rank+2]
    local_map = np.full(len(selected), -1, np.int32)
    local_map[selected] = np.arange(start, stop)
    native = p.all_rows(graph.native[selected], owners).astype(np.int64)
    if not len(native):
        raise InvalidFactorization("factor_empty_representatives")
    active = ~graph.fixed & ~selected
    neighbours = graph.adjacency.indices
    halo = neighbour_blocks(p, graph.owners, local_map[:, None], neighbours).ravel()
    nearby = np.unique(np.r_[np.arange(start, stop), halo[halo >= 0]]).astype(np.int32)
    if stage >= DEPTHS[0]:
        nearby = np.arange(len(native), dtype=np.int32)
    else:
        missing = ~np.isin(np.arange(3), native[nearby] % 3)
        nearby = np.unique(np.r_[nearby, np.flatnonzero(missing[native % 3])]).astype(np.int32)
    count = config["factorized"]["geometry_candidates"]
    geo, spacing = geometry_candidates(graph.xyz, coordinates[native[nearby] // 3],
        graph.native % 3, native[nearby] % 3, active, count)
    geo = np.where(geo >= 0, nearby[np.maximum(geo, 0)], -1)
    absent = active & (geo[:, 0] < 0)
    if np.any(absent):
        # A component may be absent on a teacher-forced input. Keep these
        # rows visible in the coverage diagnostic using the same local pool.
        from scipy.spatial import cKDTree

        nearby = nearby if len(nearby) else np.arange(len(native), dtype=np.int32)
        distance, closest = cKDTree(coordinates[native[nearby] // 3]).query(graph.xyz[absent])
        geo[absent, 0] = nearby[closest]
        spacing[absent] = np.maximum(distance, 1e-6)
    assigned = geo[:, 0].copy()
    if stage < DEPTHS[0] and stop > start:
        # Keep fine-grid pooling local. Edges still use the MPI halo.
        owned = (geo >= start) & (geo < stop)
        first = np.argmax(owned, axis=1)
        assigned = geo[np.arange(len(geo)), first]
        missing = active & ~owned.any(1)
        if np.any(missing):
            from scipy.spatial import cKDTree

            _, closest = cKDTree(coordinates[native[start:stop] // 3]).query(graph.xyz[missing])
            assigned[missing] = start + closest
    assignment = np.where(selected, local_map, assigned)
    mapped = neighbour_blocks(p, graph.owners, assignment[:, None], neighbours).ravel()
    offsets, indices, values = signed_contacts(graph, mapped)
    shape = (len(selected), len(native))
    contacts = sparse.csr_matrix((values, indices, offsets), shape=shape)
    contacts.sum_duplicates()
    contacts.eliminate_zeros()
    # Native packing orders the signed contact evidence by magnitude.
    cap = config["factorized"]["graph_candidates"]
    empty = sparse.csr_matrix(shape)
    markers = np.where(active, -1, -3).astype(np.int32)
    packed = pack(dict(config, student_candidates=max(12, cap)),
        markers, np.ones(len(selected)), empty, contacts, empty,
        np.zeros((len(selected), 10), np.float32))
    algebra = packed["cols"]
    novel = (algebra >= 0) & ~np.any(algebra[:, :, None] == geo[:, None, :], axis=2)
    rank = np.cumsum(novel, axis=1) - 1
    rr, cc = np.where(novel & (rank < cap))
    extra = np.full((len(selected), cap), -1, np.int32)
    extra[rr, rank[rr, cc]] = algebra[rr, cc]
    return np.column_stack((geo, extra)), assignment, spacing, contacts, native


def edge_features(graph, columns, rows, spacing, contacts, native, coordinates):
    valid = columns >= 0
    ids = native[np.maximum(columns, 0)]
    distance = np.linalg.norm(coordinates[ids // 3] - graph.xyz[rows, None, :], axis=2)
    distance = np.minimum(distance / spacing[rows, None], 6)
    components = ids % 3
    same = (components == (graph.native[rows, None] % 3)) & valid
    signed = np.zeros(columns.shape)
    rr, cc = np.where(valid)
    if len(rr):
        signed[rr, cc] = np.asarray(contacts[rows[rr], columns[rr, cc]]).ravel()
    features = np.concatenate((distance[:, :, None], np.arcsinh(12 * signed)[:, :, None],
        same[:, :, None], (1 / (.25 + distance**2))[:, :, None],
        np.eye(3)[components], np.abs(signed)[:, :, None]), axis=2)
    return (features * valid[:, :, None]).astype(np.float32), same


def contract(config, p, legacy, bridge, graph, selected, owners, assignment):
    """Sparse quotient graph and pooled observations, with no intermediate KSP."""
    n, width = len(selected), int(owners[-1])
    membership = row_csr(assignment[:, None], np.ones((n, 1)), assignment[:, None] >= 0, width)
    source = import_matrix(p, legacy, graph.adjacency, graph.owners, graph.owners)
    grouping = import_matrix(p, legacy, membership, graph.owners, owners)
    reduced = p.P()
    bind(p, "MatPtAP", [p.P, p.P, p.I, p.D, C.POINTER(p.P)])
    try:
        p.call("MatPtAP", source, grouping, 0, -2.0, C.byref(reduced))
        adjacency, actual = local_csr(p, reduced, bridge)
        np.testing.assert_array_equal(actual, owners)
    finally:
        p.destroy("Mat", reduced, source, grouping)
    lo, hi = owners[p.rank:p.rank+2]
    assigned = assignment >= 0
    remote = assigned & ((assignment < lo) | (assignment >= hi))
    observations = np.column_stack((graph.nodes, np.ones(n)))
    if p.maximum(remote.any()):
        pooled = p.reduce(membership.T @ observations)[lo:hi]
    else:
        pooled = membership[:, lo:hi].T @ observations
    nodes = (pooled[:, :-1] / np.maximum(pooled[:, -1:], 1)).astype(np.float32)
    nodes[:, len(ATTRIBUTE_NAMES)-3:len(ATTRIBUTE_NAMES)] = graph.xyz[selected]
    nodes[:, -5:] = graph.nodes[selected, -5:]
    nodes[:, -1] = .1 * np.log1p(pooled[:, -1])
    rows = np.repeat(np.arange(lo, hi), np.diff(adjacency.indptr))
    adjacency.data[adjacency.indices == rows] = 0
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    limit = config["factorized"]["quotient_degree"]
    count = np.diff(adjacency.indptr)
    if len(count) and np.max(count) > limit:
        # Native packing is also used for the bounded signed quotient stencil.
        empty = sparse.csr_matrix(adjacency.shape)
        compact = pack(dict(config, student_candidates=limit), np.full(len(count), -1, np.int32),
            np.ones(len(count)), empty, adjacency, empty, np.zeros((len(count), 10), np.float32))
        cols, values = compact["cols"], -compact["hints"][:, :, 1]
        adjacency = row_csr(cols, values.astype(np.float64), cols >= 0, width)
    return Graph(adjacency, owners, graph.native[selected], nodes, np.zeros(hi-lo, bool))



@dataclass
class FixedFactor:
    """Rank-local feature and support tape for one coefficient update."""

    stage: int
    rows: np.ndarray
    owners: np.ndarray
    x: np.ndarray
    selected: np.ndarray
    active: np.ndarray
    regions: np.ndarray
    columns: np.ndarray
    edges: np.ndarray
    valid: np.ndarray
    same: np.ndarray
    keep: np.ndarray
    csr: object

    def network(self, p, model, tape=None):
        cols, inverse = np.unique(self.columns[self.valid], return_inverse=True)
        x_coarse = neighbour_blocks(p, self.owners, self.x[self.selected], cols)
        x_fine = self.x[self.active]
        fine = np.maximum(0, model.linear(0, x_fine))
        anchors = np.maximum(0, model.linear(0, x_coarse))
        coarse = np.zeros(self.columns.shape + (model.width,), fine.dtype)
        coarse[self.valid] = anchors[inverse]
        out, quota = model.edge_outputs(fine, coarse, self.edges, self.valid, self.stage,
                                         self.regions, tape)
        return out, fine, coarse, x_fine, x_coarse, anchors, inverse

    def fixed_matrix(self, p, model):
        out = self.network(p, model)[0]
        weights = coefficients(out[:, :, 1], out[:, :, 2], self.keep, self.same,
                               self.stage < model.unit_sum_depth)
        rr, cc = np.nonzero(self.keep)
        local, cols = self.active[rr], self.columns[rr, cc]
        csr = self.csr.copy()
        csr[local, cols] = weights[rr, cc]
        return csr

    def backward(self, p, model, grad):
        tape = []
        out, fine, coarse, x, x_coarse, anchors, inverse = self.network(p, model, tape)
        hidden, summary, refined = tape
        rr, cc = np.nonzero(self.keep)
        grad_csr = sparse.csr_matrix((grad, self.csr.indices, self.csr.indptr), shape=self.csr.shape)
        values = np.zeros(self.keep.shape)
        values[rr, cc] = sparse_entries(grad_csr, self.active[rr], self.columns[rr, cc])
        unit = self.stage < model.unit_sum_depth
        base = coefficients(out[:, :, 1], np.zeros(self.keep.shape), self.keep, self.same, unit)
        logit = base * (values - (base*values).sum(1, keepdims=True))
        logit *= np.abs(out[:, :, 1]) < 12.
        if unit:
            values -= (values.sum(1, keepdims=True) / np.maximum(self.keep.sum(1, keepdims=True), 1))
        signed = self.keep * .75 * (1 - np.tanh(out[:, :, 2])**2) * values
        derivative = np.stack((np.zeros_like(logit), logit, signed), axis=2)
        gradients = [(np.zeros(w.shape), np.zeros(b.shape)) for w, b in model.layers]
        width = model.width

        output = derivative.reshape((-1, 3))
        gradients[4] = output.T @ refined.reshape((-1, width)), output.sum(0)
        d_refined = (derivative @ model.layers[4][0]) * (refined > 0)
        sums = d_refined.sum(1)
        gradients[3] = (np.column_stack((d_refined.reshape((-1, width)).T @ hidden.reshape((-1, width)),
                                        sums.T @ summary)), sums.sum(0))
        left, right = np.hsplit(model.layers[3][0], 2)
        d_hidden = d_refined @ left
        d_summary = (sums @ right) / np.maximum(self.valid.sum(1, keepdims=True), 1)
        d_hidden = (d_hidden + self.valid[:, :, None]*d_summary[:, None, :]) * (hidden > 0)
        flat = d_hidden.reshape((-1, width))
        grad_fine = d_hidden.sum(1)
        gradients[2] = (np.column_stack((grad_fine.T @ fine,
                         flat.T @ coarse.reshape((-1, width)),
                         flat.T @ self.edges.reshape((-1, EDGE_FEATURES)))), grad_fine.sum(0))
        left, middle, _ = np.split(model.layers[2][0], [width, 2*width], axis=1)
        d_fine = (grad_fine @ left) * (fine > 0)
        d_anchors = np.zeros(anchors.shape)
        np.add.at(d_anchors, inverse, d_hidden[self.valid] @ middle)
        d_anchors *= anchors > 0
        gradients[0] = d_fine.T @ x + d_anchors.T @ x_coarse, d_fine.sum(0) + d_anchors.sum(0)
        return np.concatenate([part.ravel() for group in gradients for part in group])


def sparse_entries(matrix, rows, columns):
    return np.asarray(matrix[rows, columns]).ravel() if len(rows) else np.empty(0)


def projected_factor(p, upstream, raw, suffix, suffix_rows):
    """Sample upstream @ suffix.T only at executed factor entries."""
    rows = np.repeat(np.arange(raw.shape[0]), np.diff(raw.indptr))
    if suffix is None:
        return sparse_entries(upstream, rows, raw.indices)
    counts = np.diff(suffix.indptr)
    width = int(p.maximum(np.max(counts, initial=0)))
    if not width:
        return np.zeros(len(rows))
    packed = np.zeros((suffix.shape[0], 2*width))
    ids = np.repeat(np.arange(suffix.shape[0]), counts)
    slots = np.arange(suffix.nnz) - np.repeat(suffix.indptr[:-1], counts)
    packed[ids, slots] = suffix.indices + 1
    packed[ids, slots+width] = suffix.data
    requested, inverse = np.unique(raw.indices, return_inverse=True)
    neighbours = neighbour_blocks(p, suffix_rows, packed, requested)
    column, values = np.hsplit(neighbours, 2)
    result = np.zeros(len(rows))
    for j in range(width):
        col = column[inverse, j].astype(np.int64)-1
        valid = col >= 0
        result[valid] += values[inverse[valid], j] * sparse_entries(upstream, rows[valid], col[valid])
    return result


def factor_cotangent(p, legacy, bridge, level, transfer_gradient, model):
    """Backpropagate through the products without building a dense factor gradient."""
    fixed = level.fixed_factors
    factors = []
    upstream = suffix_product = None
    try:
        for tape in fixed:
            factors.append(import_matrix(p, legacy, tape.csr, tape.rows, tape.owners))
        suffixes = [fixed[-1].csr, None]
        if len(fixed) == 3:
            suffix_product = multiply(p, factors[1], factors[2])
            suffixes.insert(0, local_csr(p, suffix_product, bridge)[0])
        gradient = np.zeros(model.weights.shape, np.float64)
        local = transfer_gradient
        for j, (tape, suffix) in enumerate(zip(fixed, suffixes)):
            entries = projected_factor(p, local, tape.csr, suffix, tape.owners)
            gradient += tape.backward(p, model, entries)
            if j+1 < len(fixed):
                if upstream is None:
                    upstream = import_matrix(p, legacy, local, tape.rows, level.owners)
                next_upstream = multiply(p, factors[j], upstream, left_transpose=True)
                p.destroy("Mat", upstream)
                upstream = next_upstream
                local = local_csr(p, upstream, bridge)[0]
        return gradient
    finally:
        if upstream is not None:
            p.destroy("Mat", upstream)
        if suffix_product is not None:
            p.destroy("Mat", suffix_product)
        p.destroy("Mat", *factors)


def frozen_transfer(p, legacy, bridge, level, parent, model):
    """Rebuild Galerkin numerics while holding the batch's decode and features."""
    factors = []
    product = None
    try:
        for tape in level.fixed_factors:
            factors.append(import_matrix(p, legacy, tape.fixed_matrix(p, model), tape.rows, tape.owners))
        product = compose(p, factors)
        csr = local_csr(p, product, bridge)[0]
        csr.eliminate_zeros()
        csr.sort_indices()
        return imported_transfer(p, legacy, parent, csr, level.fixed_factors[0].rows,
                                 level.owners, {})
    finally:
        if product is not None:
            p.destroy("Mat", product)
        p.destroy("Mat", *factors)


def hierarchy_signature(numerical, fields=("representatives", "support", "coefficients")):
    return tuple(stage["signatures"][name]
                 for level in ("level1", "level2") for stage in numerical[level]["stages"]
                 for name in fields)


class FactorPolicy:
    """Conditional draws from the hard decoder's local choice sets.

    A rollout releases one ranking slot per sampled group or row, or the
    budget of a sampled row. Its exact categorical log likelihood is used
    to update the corresponding structural output head.
    """

    def __init__(self, p, model, settings):
        self.p, self.settings = p, settings
        self.stage, self.role = int(settings["stage"]), settings["role"]
        self.region, self.temperature = int(settings["region"]), float(settings["temperature"])
        if self.role not in ("representatives", "support", "budget") or self.temperature <= 0:
            raise ValueError("Invalid structural policy role or temperature")
        self.terms = [(np.zeros(w.shape), np.zeros(b.shape)) for w,b in model.layers]
        self.log_probability, self.actions, self.changes = 0., 0, 0
        self.trace = hashlib.sha256()
        self.modify_edges = self.role != "representatives"
        rank = 0 if self.role == "representatives" and self.stage >= DEPTHS[0] else p.rank
        self.subset = np.random.default_rng(np.random.SeedSequence([settings["subset_seed"], rank]))
        self.rng = np.random.default_rng(np.random.SeedSequence([settings["seed"], rank]))

    def choose(self, scores):
        logits = np.asarray(scores, np.float64)/self.temperature
        prob = np.exp(logits-logits.max())
        prob /= prob.sum()
        chosen = self.rng.choice(len(prob), p=prob)
        return int(chosen), prob

    def record(self, entity, chosen, original, probability, record):
        if record:
            self.actions += 1
            self.changes += chosen != original
            self.log_probability += np.log(probability)
            self.trace.update(np.asarray([entity, chosen, original], dtype=np.int64).tobytes())

    def representative_choice(self, features, scores, chosen, original, native, record):
        pass

    def representatives(self, config, graph, hidden, logits, regions):
        p = self.p
        native, labels, owned, limit = representative_domain(config, p, graph, self.stage)
        global_group = self.stage >= DEPTHS[0]
        if global_group:
            logits = p.all_rows(logits.ravel(), graph.owners*2).reshape((-1,2))
            width = hidden.shape[1]
            hidden = p.all_rows(hidden.ravel(), graph.owners*width).reshape((-1,width))
            regions = p.all_rows(regions, graph.owners).astype(np.int32)
        selected = pick_representatives(native, labels, logits, limit)
        eligible = np.flatnonzero(labels >= 0)
        group = labels[eligible]
        ordered = eligible[np.argsort(group, kind="stable")]
        offsets = np.r_[0,np.cumsum(np.bincount(group))]
        possible = np.unique(labels[eligible[regions[eligible] == self.region]])
        available = [i for i in possible if np.any(~selected[ordered[offsets[i]:offsets[i+1]]])]
        rows = self.subset.choice(available, min(len(available), self.settings["actions_per_rank"]),
                                  replace=False).astype(np.int64)
        weight, _ = self.terms[1]
        record = not global_group or p.rank == 0
        background = np.setdiff1d(available, rows)[:self.example_limit] if getattr(
            self, "example_limit", 0) else rows[:0]
        for i in np.r_[rows, background]:
            members = ordered[offsets[i]:offsets[i+1]]
            retained = members[selected[members]]
            if not len(retained):
                raise ValueError("Structural proposal would remove required group coverage")
            # Keep the budget and mandatory group coverage, and release only
            # the lowest-priority currently selected representative.
            sorted_members = retained[np.lexsort((native[retained], -logits[retained,0]))]
            released = sorted_members[-1]
            options = np.r_[released, members[~selected[members]]]
            if i in background:
                self.representative_choice(hidden[options], logits[options, 0], 0, 0, native[options], record)
                continue
            j, prob = self.choose(logits[options,0])
            winner = options[j]
            selected[released], selected[winner] = False, True
            if record:
                weight[0] += (hidden[winner] - prob @ hidden[options])/self.temperature
            self.record(native[released], native[winner], native[released], prob[j], record)
            self.representative_choice(hidden[options], logits[options, 0], j, 0, native[options], record)
        return selected[owned]

    def edges(self, outputs, quota, valid, same, keep, fine, summary, refined, native, regions,
              candidate_ids=None, xyz=None):
        count = keep.sum(1)
        capacity = np.minimum(MAX_SUPPORT, valid.sum(1))
        if self.role == "support":
            eligible = (count >= 2) & (valid.sum(1) > count)
        else:
            eligible = capacity > 1
        candidates = np.flatnonzero(eligible & (regions == self.region))
        rows = self.subset.choice(candidates, min(len(candidates), self.settings["actions_per_rank"]),
                                  replace=False)
        order = support_order(outputs[:,:,0], valid, same)
        for i in rows:
            if self.role == "support":
                previous = order[i,count[i]-1]
                options = np.r_[previous, np.flatnonzero(valid[i] & ~keep[i])]
                j, prob = self.choose(outputs[i,options,0])
                chosen = options[j]
                keep[i,previous], keep[i,chosen] = False, True
                weight, _ = self.terms[4]
                weight[0] += (refined[i,chosen] - prob @ refined[i,options])/self.temperature
            else:
                previous = count[i]
                options = np.arange(max(1,previous-1), min(capacity[i],previous+1)+1)
                j, prob = self.choose(-(options - 3. - quota[i])**2)
                chosen = options[j]
                keep[i] = False
                keep[i,order[i,:chosen]] = True
                weight, bias = self.terms[5]
                derivative = 2*(chosen - prob @ options)/self.temperature
                weight[0] += derivative*np.r_[fine[i],summary[i]]
                bias[0] += derivative
            self.record(native[i], chosen, previous, prob[j], True)
        return keep

    def receipt(self):
        p = self.p
        gradient = p.reduce(np.concatenate([part.ravel() for group in self.terms for part in group]))
        logp, actions, changes = p.reduce([self.log_probability, self.actions, self.changes])
        ranks = p.gather_equal(np.frombuffer(self.trace.digest(), np.uint32)).astype(np.uint32)
        return dict(stage=self.stage+1, region=self.region, role=self.role, seed=self.settings["seed"],
                    subset_seed=self.settings["subset_seed"], log_probability=float(logp),
                    actions=int(actions), changes=int(changes),
                    signature=hashlib.sha256(ranks.tobytes()).hexdigest(),
                    gradient=gradient.tolist())

def predicted_factor(config, p, graph, model, stage, coordinates, training=None, policy=None):
    start = time.perf_counter()
    regions = region_ids(graph.xyz)
    x = encoded_inputs(p, graph, stage)
    hidden, logits = model.node_outputs(x, stage, regions)
    actor = policy if policy is not None and policy.stage == stage else None
    if actor is not None and actor.role == "representatives":
        selected = actor.representatives(config, graph, hidden, logits, regions)
    else:
        selected = selections(config, p, graph, stage, logits)
    owners = coarse_owners(p, selected)
    if owners[-1] == 0 or (stage == STAGES - 1 and owners[-1] > config["terminal_max_dofs"]):
        raise InvalidFactorization("factor_representative_budget",
                                   dict(stage=stage+1, columns=int(owners[-1])))
    encoded = time.perf_counter()
    cols, assignment, spacing, contacts, native = candidates(
        config, p, graph, selected, owners, coordinates, stage)
    active = np.flatnonzero(~graph.fixed & ~selected)
    edges, same = edge_features(graph, cols[active], active, spacing, contacts, native, coordinates)
    anchor_hidden = neighbour_blocks(p, owners, hidden[selected], cols[active])
    prepared = time.perf_counter()
    valid = cols[active] >= 0
    tape = [] if actor is not None and actor.modify_edges else None
    outputs, quota = model.edge_outputs(hidden[active], anchor_hidden, edges, valid,
                                        stage, regions[active], tape)
    keep = support_mask(outputs[:, :, 0], quota, valid, same)
    if tape is not None:
        keep = actor.edges(outputs, quota, valid, same, keep, hidden[active], tape[1], tape[2],
                           graph.native[active], regions[active],
                           np.where(valid, native[np.maximum(cols[active], 0)], -1)
                           if actor.settings.get("search") else None, graph.xyz[active])
    weights = coefficients(outputs[:, :, 1], outputs[:, :, 2], keep, same,
                           unit_row_sum=stage < model.unit_sum_depth)
    raw = np.zeros(cols.shape, dtype=np.float64)
    raw[active] = weights
    cols[selected, 0] = assignment[selected]
    raw[selected, 0] = 1.
    csr = row_csr(cols, raw, (cols >= 0) & (raw != 0), int(owners[-1]))
    empty = (~graph.fixed) & (np.diff(csr.indptr) == 0)
    counts = p.reduce([csr.nnz, empty.sum(), int(not np.isfinite(csr.data).all())])
    if counts[1] or counts[2]:
        raise InvalidFactorization("factor_coverage_or_finite",
                                   dict(stage=stage+1, empty_free_rows=int(counts[1])))

    fingerprints = []
    for array in (graph.native[selected], np.r_[csr.indptr, csr.indices], csr.data):
        digest_bytes = hashlib.sha256(np.ascontiguousarray(array).tobytes()).digest()
        fingerprints.append(np.frombuffer(digest_bytes, np.uint32)[:2])
    all_ranks = p.gather_equal(np.stack(fingerprints)).astype(np.uint32)
    signatures = {name: hashlib.sha256(all_ranks[:, i].tobytes()).hexdigest()[:16]
                  for i, name in enumerate(("representatives", "support", "coefficients"))}
    stats = dict(stage=stage + 1, rows=int(graph.owners[-1]), columns=int(owners[-1]),
                 nnz=int(counts[0]), empty_free_rows=int(counts[1]),
                 selection_s=p.maximum(encoded - start), candidate_s=p.maximum(prepared - encoded),
                 inference_s=p.maximum(time.perf_counter() - prepared),
                 signatures=signatures)
    if training is not None:
        training.append(FixedFactor(stage, graph.owners, owners, x, selected, active,
            regions[active], cols[active], edges, valid, same, keep & (weights != 0), csr))
    return csr, selected, owners, assignment, stats


def interval(config, p, legacy, bridge, matrix, graph, model, stage, coordinates,
             train_weights=False, policy=None):
    """Assemble only the final product and its actual Galerkin matrix."""
    product, factors, stages = None, [], []
    training = [] if train_weights else None
    initial, parent_owners = graph, graph.owners
    depth = DEPTHS[0] if stage == 0 else DEPTHS[1]
    try:
        for number in range(stage, stage + depth):
            csr, selected, owners, assignment, info = predicted_factor(
                config, p, graph, model, number, coordinates, training, policy)
            factors.append(import_matrix(p, legacy, csr, graph.owners, owners))
            stages.append(info)
            if number + 1 < stage + depth:
                start = time.perf_counter()
                graph = contract(config, p, legacy, bridge, graph, selected, owners, assignment)
                info["quotient_s"] = p.maximum(time.perf_counter() - start)
        start = time.perf_counter()
        product = compose(p, factors)
        raw, _ = local_csr(p, product, bridge)
        if training is not None:
            pattern = raw.copy()
            pattern.sort_indices()
        raw.eliminate_zeros()
        native = graph.native[selected]
        # Nested injection stays exact, with columns owned by their anchor rank.
        local_selected = np.isin(initial.native, native)
        degree = np.diff(raw.indptr)
        positions = raw.indptr[:-1][local_selected]
        offset = owners[p.rank]
        injection = (np.all(degree[local_selected] == 1) and
            np.array_equal(raw.indices[positions], offset + np.arange(len(native))) and
            np.array_equal(raw.data[positions], np.ones(len(native))))
        stats = p.reduce([raw.nnz, np.count_nonzero((degree == 0) & ~initial.fixed),
            int(not injection or not np.isfinite(raw.data).all() or np.any(degree[initial.fixed]))])
        maximum = config["factorized"]["product_average_limit"][stage != 0]
        info = dict(fine_dofs=int(parent_owners[-1]), coarse_dofs=int(owners[-1]),
                    nnz=int(stats[0]), empty_free_rows=int(stats[1]), injection_error=float(stats[2] > 0),
                    stages=stages, composition_s=p.maximum(time.perf_counter() - start))
        if stats[1] or stats[2] or stats[0] > maximum * parent_owners[-1]:
            raise InvalidFactorization("factor_product_coverage_injection_or_density", info)
        if stage + depth <= model.unit_sum_depth:
            sums = np.asarray(raw.sum(axis=1)).ravel()
            info["max_product_row_sum_error"] = p.maximum(
                np.max(np.abs(sums - (~initial.fixed).astype(float)), initial=0.0))
            if info["max_product_row_sum_error"] > 3e-5:
                raise InvalidFactorization("factor_product_row_sum", info)
        start = time.perf_counter()
        # Reimport the audited product after removing exact structural zeros.
        raw.sort_indices()
        level = imported_transfer(p, legacy, matrix, raw, parent_owners, owners, info,
                                  markers=np.where(local_selected, 1, np.where(initial.fixed, -3, -1)))
        level.native = native
        level.components = (native % 3).astype(np.int32)
        if training is not None:
            level.fixed_factors = training
            level.product_pattern = pattern
        info["galerkin_s"] = p.maximum(time.perf_counter() - start)
        return level
    except InvalidFactorization as error:
        if "stages" not in error.stats:
            error.stats["completed_stages"] = stages
        raise
    finally:
        if product is not None:
            p.destroy("Mat", product)
        p.destroy("Mat", *factors)


def normalized_coordinates(xyz):
    return (xyz - (xyz.max(0) + xyz.min(0))/2) / np.maximum(np.ptp(xyz, axis=0), 1e-30)


def first_graph(p, bridge, matrix, layout, fields):
    lo, hi = layout["ownership"][p.rank:p.rank+2]
    native = (3*layout["native_node"] + layout["component"])[lo:hi].astype(np.int64)
    attributes = physical_attributes(fields, layout["xyz"], native)
    fixed = fields["constrained"][native]
    graph = matrix_graph(p, bridge, matrix, native, attributes, fixed, np.ones(len(native)))
    return graph, attributes, normalized_coordinates(layout["xyz"])


def next_graph(p, bridge, level, attributes, coordinates):
    values, mass = pooled_attributes(p, level, attributes, bridge)
    values[:, -3:] = coordinates[level.native // 3]
    return matrix_graph(p, bridge, level.coarse, level.native, values,
                        np.zeros(len(level.native), bool), mass)


def read_factors(config, p, row, graph):
    """Load the original signed HMIS factors and verify their native identities."""
    base = Path(config["baseline_directory"])
    receipt_path = base / "records" / ("hierarchy-" + row["key"] + ".json")
    receipt = read_json(receipt_path)
    if (row["split"] != "training" or receipt["split"] != row["split"]
            or receipt["manifest"] != row["manifest_sha256"]
            or receipt["image_sha256"] != config["image_sha256"]
            or receipt["levels"] <= STAGES or len(receipt["shards"]) != p.size):
        raise ValueError("Teacher factor provenance mismatch")
    directory = base / "data" / row["key"]
    mapping = directory / receipt["map_file"]
    shard = directory / (receipt["shard_pattern"] % p.rank)
    if digest(mapping) != receipt["map_sha256"] or digest(shard) != receipt["shards"][p.rank]:
        raise ValueError("Teacher factor files changed")
    with np.load(mapping) as f:
        native, depth = f["native_for_petsc"], f["depth_petsc"]
        maps = [native] + [f["level_%d_native_dofs" % level] for level in range(1, STAGES + 1)]
    lo, hi = graph.owners[p.rank:p.rank+2]
    np.testing.assert_array_equal(graph.native, native[lo:hi])
    if np.any(graph.fixed & (depth[lo:hi] > 0)):
        raise ValueError("A constrained row was retained by the teacher")
    owners = [coarse_owners(p, depth[lo:hi] >= level) for level in range(STAGES + 1)]
    factors = []
    with np.load(shard) as f:
        np.testing.assert_array_equal(f["native"], native[lo:hi])
        np.testing.assert_array_equal(f["depth"], depth[lo:hi])
        for stage in range(STAGES):
            prefix = "P_%d_" % stage
            begin, end = owners[stage][p.rank:p.rank+2]
            first, last = owners[stage+1][p.rank:p.rank+2]
            np.testing.assert_array_equal(f[prefix+"ownership"], [begin, end])
            np.testing.assert_array_equal(f[prefix+"shape"], [owners[stage][-1], owners[stage+1][-1]])
            target = sparse.csr_matrix((f[prefix+"data"], f[prefix+"indices"], f[prefix+"indptr"]),
                                       shape=(end-begin, owners[stage+1][-1]))
            target.sum_duplicates()
            target.eliminate_zeros()
            selected = np.isin(maps[stage][begin:end], maps[stage+1][first:last])
            positions = target.indptr[:-1][selected]
            if (not np.all(np.diff(target.indptr)[selected] == 1)
                    or not np.array_equal(target.indices[positions], np.arange(first, last))
                    or not np.allclose(target.data[positions], 1., rtol=0, atol=1e-12)
                    or not np.isfinite(target.data).all()
                    or np.max(np.diff(target.indptr), initial=0) > MAX_SUPPORT):
                raise ValueError("Invalid teacher factor injection, coefficients or support")
            if stage == 0:
                target.data[np.repeat(graph.fixed, np.diff(target.indptr))] = 0.
                target.eliminate_zeros()
            factors.append(target)
    return maps, owners, factors, dict(path=str(receipt_path), sha256=digest(receipt_path),
                                       raw_hierarchy_job=receipt["job"], map_sha256=receipt["map_sha256"])


def ordered_columns(native, ordered):
    permutation = np.argsort(ordered)
    mapped = permutation[np.searchsorted(ordered[permutation], native)]
    np.testing.assert_array_equal(ordered[mapped], native)
    return mapped


def relative_error(p, actual, expected):
    difference = actual - expected
    if sparse.issparse(actual):
        difference, expected = difference.data, expected.data
    error, norm = p.reduce([np.sum(difference**2), np.sum(expected**2)])
    return float(np.sqrt(error / max(norm, 1e-300)))


def cached_composition(config, p, legacy, row, maps, owners, products, graph):
    """Reconstruct the accepted cache including signed dropping and scaling."""
    from .gate import RetainedFirst
    from .student import normalized_teacher

    base = Path(config["baseline_directory"])
    receipt = base / "records" / ("solver-export-retained-solver-" + row["key"] + ".json")
    if not receipt.exists():
        return None
    export = read_json(receipt)
    assert export["manifest"] == row["manifest_sha256"]
    entry = next(s for s in export["spaces"] if s["label"] == config["baseline_label"])
    lo, hi = graph.owners[p.rank:p.rank+2]
    depth = np.zeros(len(maps[0]), np.int32)
    depth[np.isin(maps[0], maps[3])] = 3
    fixed = np.zeros(len(depth), bool)
    fixed[lo:hi] = graph.fixed
    teacher = dict(retained_entry=entry, retained_export=receipt, native=maps[0],
                   depth=depth, fixed=fixed, terminal_native=maps[5])
    cached = RetainedFirst(config, p, legacy, row, teacher, assemble=False)
    middle = p.all_rows(cached.native, cached.owners).astype(np.int64)
    column_map = ordered_columns(middle, maps[3])
    old = normalized_teacher(cached)
    first = sparse.csr_matrix((old.data, column_map[old.indices], old.indptr), shape=old.shape)
    first.sort_indices()
    drop = config["factorized"]["cached_drop"]
    pruned, _ = legacy.prune_transfer_connections(products[0], maps[3], drop)
    error1 = relative_error(p, first, pruned)

    # Cached P2 owns its rows independently of raw Q4. Compare in native order
    # using the small L3/L5 operator, with both column scalings undone.
    lo, hi = cached.owners[p.rank:p.rank+2]
    old = cached.cached_second
    rows = np.flatnonzero(np.isin(cached.native, maps[5]))
    offsets = old.indptr[rows]
    columns = old.indices[offsets]
    scale = np.zeros((2, cached.terminal_owners[-1]))
    scale[0, columns] = cached.native[rows]
    scale[1, columns] = old.data[offsets] * cached.column_scale[lo:hi][rows]
    terminal, terminal_scale = p.reduce(scale)
    target_cols = ordered_columns(terminal.astype(np.int64), maps[5])
    unscaled = old.multiply(cached.column_scale[lo:hi, None]).multiply(1 / terminal_scale).tocsr()
    lengths = p.all_rows(np.diff(unscaled.indptr), cached.owners).astype(np.int64)
    nonzeros = np.r_[0, np.cumsum(p.gather_equal([unscaled.nnz]).ravel().astype(np.int64))]
    data = p.all_rows(unscaled.data, nonzeros)
    indices = p.all_rows(target_cols[unscaled.indices], nonzeros).astype(np.int32)
    second = sparse.csr_matrix((data, indices, np.r_[0, np.cumsum(lengths)]),
                                shape=(len(middle), len(terminal)))
    begin, end = owners[3][p.rank:p.rank+2]
    aligned = second[ordered_columns(maps[3][begin:end], middle)].tocsr()
    aligned.sort_indices()
    pruned, _ = legacy.prune_transfer_connections(products[1], maps[5], drop)
    error2 = relative_error(p, aligned, pruned)
    if max(error1, error2) > 1e-10:
        raise ValueError("Cached retained composition mismatch: %s" % ([error1, error2],))
    return dict(cache_sha256=digest(receipt), normalized_relative_errors=[error1, error2],
                relative_drop=drop, column_scaling="undo cached L3/L5 injection scales")


def teacher_probes(p, handles, owners, maps, start, end):
    """Probe the signed product in its own endpoint coordinates."""
    lo, hi = owners[end][p.rank:p.rank+2]
    ids = maps[end][lo:hi]
    probes = np.column_stack((np.eye(3)[ids % 3], np.sin(.137 * ids), np.cos(.317 * ids)))
    result = {}
    for stage in range(end - 1, start - 1, -1):
        action = Action(p, handles[stage])
        try:
            target = np.column_stack([action(values) for values in probes.T])
        finally:
            action.close()
        result[stage] = probes, target
        probes = target
    return result


def export_samples(config, p, graph, stage, selected, owners, target, probes,
                   coordinates, rng):
    x = encoded_inputs(p, graph, stage)
    native, labels, owned, _ = representative_domain(config, p, graph, stage)
    all_selected = selected if stage < DEPTHS[0] else p.all_rows(selected, graph.owners).astype(bool)
    eligible = labels >= 0
    quota = np.bincount(labels[eligible], weights=all_selected[eligible])
    quota = np.clip(quota, 1, 3)
    eligible_rows = np.flatnonzero(~graph.fixed)
    count = min(len(eligible_rows), config["factorized"]["samples_per_rank"])
    # Rotate the uniform sample and dedicate one third to the most damaged
    # rows. Both strata keep their real neighbours and representative context.
    focus = eligible_rows[graph.nodes[eligible_rows, 0] <= .15]
    focused = rng.choice(focus, min(len(focus), count//3), replace=False)
    others = np.setdiff1d(eligible_rows, focused, assume_unique=True)
    rows = np.sort(np.r_[focused, rng.choice(others, count-len(focused), replace=False)])
    columns, assignment, spacing, contacts, coarse_native = candidates(
        config, p, graph, selected, owners, coordinates, stage)
    initial = columns[rows]
    augment = np.full((count, MAX_SUPPORT), -1, np.int32)
    for j, row in enumerate(rows):
        if not selected[row]:
            teacher = target.indices[target.indptr[row]:target.indptr[row+1]]
            extra = teacher[~np.isin(teacher, initial[j])]
            augment[j, :len(extra)] = extra
    augmented = np.column_stack((initial, augment))
    edges, same = edge_features(graph, augmented, rows, spacing, contacts, coarse_native, coordinates)
    target_weights, mass, _ = match_targets({"cols": augmented}, target[rows], np.arange(count))
    anchor = neighbour_blocks(p, owners, np.column_stack((x[selected], probes[0])), augmented)
    budget = quota[labels[owned][rows]]
    online = augmented >= 0
    online[:, initial.shape[1]:] = False
    active = (~selected[rows]) & (mass > 0)
    covered = np.sum((target_weights * online)**2, axis=1)
    parity = p.reduce([np.sum(covered[active]), np.sum(mass[active]**2), int(active.sum())])
    data = dict(x=x[rows], coarse=anchor[:, :, :x.shape[1]].astype(np.float32), edge=edges,
        same=same, valid=augmented >= 0, online=online, target=target_weights,
        active=active, selected=selected[rows], quota=budget.astype(np.float32),
        probes=anchor[:, :, x.shape[1]:].astype(np.float32), action=probes[1][rows].astype(np.float32))
    return data, assignment, dict(rows=int(parity[2]), online_candidate_energy=float(parity[0]/max(parity[1], 1e-300)))


def export_state(config, p, replay, legacy, bridge, row, layout, fields, data, job):
    """Export aligned training neighbourhoods using the deployment graph builder."""
    md, rows, full, _, _ = data
    outer, fs, _ = replay.create_solver(p, full, md, config["max_iterations"],
                                       overrides={"fieldsplit_1_pc_type": "none"})
    handles, products = [], []
    first = None
    try:
        p.call("PCSetUp", fs)
        _, _, _, matrix = replay.pcapply_child(p, fs, rows, expected_pc=b"none")
        graph, attributes, coordinates = first_graph(p, bridge, matrix, layout, fields)
        maps, owners, factors, source = read_factors(config, p, row, graph)
        for stage in range(STAGES):
            handles.append(import_matrix(p, legacy, factors[stage], owners[stage], owners[stage+1]))
        first_pair = multiply(p, handles[0], handles[1])
        try:
            products.append(multiply(p, first_pair, handles[2]))
        finally:
            p.destroy("Mat", first_pair)
        products.append(multiply(p, handles[3], handles[4]))
        product_csrs = [local_csr(p, product, bridge)[0] for product in products]
        for transfer in product_csrs:
            transfer.sum_duplicates()
            transfer.eliminate_zeros()
        probes = teacher_probes(p, handles, owners, maps, 0, 3)
        probes.update(teacher_probes(p, handles, owners, maps, 3, 5))
        errors = []
        for stage, end, product in zip((0, 3), (3, 5), products):
            action = Action(p, product)
            try:
                original = probes[end-1][0]
                actual = np.column_stack([action(values) for values in original.T])
                errors.append(relative_error(p, actual, probes[stage][1]))
            finally:
                action.close()
        if max(errors) > 1e-12:
            raise ValueError("Teacher factor product action mismatch")
        cache = cached_composition(config, p, legacy, row, maps, owners, product_csrs, graph)
        first = imported_transfer(p, legacy, matrix, product_csrs[0], owners[0], owners[3], {})
        start, stop = owners[3][p.rank:p.rank+2]
        first.native = maps[3][start:stop]
        sample, stats = {}, []
        seed = int.from_bytes(hashlib.sha256(row["key"].encode()).digest()[:4], "little")
        rng = np.random.default_rng(seed + p.rank)
        for stage in range(STAGES):
            if stage == 3:
                graph = next_graph(p, bridge, first, attributes, coordinates)
            start, stop = owners[stage+1][p.rank:p.rank+2]
            selected = np.isin(graph.native, maps[stage+1][start:stop])
            np.testing.assert_array_equal(coarse_owners(p, selected), owners[stage+1])
            arrays, assignment, stat = export_samples(config, p, graph, stage, selected,
                owners[stage+1], factors[stage], probes[stage], coordinates, rng)
            sample.update({"q%d_" % (stage+1)+name: value for name, value in arrays.items()})
            stats.append(stat)
            if stage not in (2, 4):
                graph = contract(config, p, legacy, bridge, graph, selected, owners[stage+1], assignment)
        path = Path(config["output_directory"]) / "data" / ("factors-%s-%s-%03d.npz" % (job, row["key"], p.rank))
        with path.open("xb") as stream:
            np.savez_compressed(stream, **sample)
        from .dataset import gather_array

        hashes = gather_array(p, np.frombuffer(bytes.fromhex(digest(path)), np.uint8)[None, :],
                              np.arange(p.size+1))
        if p.rank == 0:
            return dict(key=row["key"], split=row["split"], source=source, raw_product_errors=errors,
                        cache=cache, samples=stats, shard_pattern=str(path.with_name(
                        "factors-%s-%s-%%03d.npz" % (job, row["key"]))),
                        shards=[line.tobytes().hex() for line in hashes])
    finally:
        if first is not None:
            first.close()
        p.destroy("Mat", *handles, *products)
        p.destroy("KSP", outer)
