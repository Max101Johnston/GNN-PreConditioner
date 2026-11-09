"""Rank-local interpolation candidates, native deployment, and teacher export."""
import ctypes as C
from functools import lru_cache
import os
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from .backend import bind, digest, local_csr, neighbour_blocks, ownership
from .dataset import gather_array
from .interpolation import collect_csr, imported_transfer

ROW_FEATURES = 10
EDGE_FEATURES = 7
LOCAL_EDGE_FEATURES = 18
CONTEXT_EDGE_FEATURES = 22
CONTEXT_NODE_FEATURES = 13
CONTEXT_BUDGETS = (6, 8, 10, 12)
CONTEXT_SCORE_SCALE = 4096
SCHEMA = 1
DEFAULT_CONTROL = (0.0, 0.0, 1.0, 1.0)
STUDENT_KINDS = {"A": 0, "B": 1, "D": 2, "E": 3, "F": 4}


@lru_cache(maxsize=4)
def native_student(path):
    lib = C.CDLL(str(path))
    ptr = C.c_void_p
    lib.gnn_pack_candidates.argtypes = [C.c_int, C.c_int] + [ptr]*15
    lib.gnn_pack_candidates.restype = C.c_int
    lib.gnn_student_predict.argtypes = [C.c_int]*4 + [ptr]*10
    lib.gnn_student_predict.restype = C.c_int
    return lib


def ptr(array):
    return array.ctypes.data


def csr_buffers(matrix):
    return (np.ascontiguousarray(matrix.indptr, np.int32),
            np.ascontiguousarray(matrix.indices, np.int32),
            np.ascontiguousarray(matrix.data, np.float64))


def product_rows(p, bridge, a, b):
    product = p.P()
    bind(p, "MatMatMult", [p.P, p.P, p.I, p.D, C.POINTER(p.P)])
    p.call("MatMatMult", a, b, 0, -2.0, C.byref(product))
    try:
        result, _ = local_csr(p, product, bridge)
        result.sort_indices()
        return result
    finally:
        p.destroy("Mat", product)


def row_csr(columns, values, valid, width):
    offsets = np.r_[0, np.cumsum(np.count_nonzero(valid, axis=1))].astype(np.int32)
    a = sparse.csr_matrix((values[valid], columns[valid], offsets), shape=(len(columns), width))
    a.sort_indices()
    return a


def pack(config, markers, sums, p0, kp0, bounded, x):
    cap = config["student_candidates"]
    rows = len(markers)
    cols = np.empty((rows, cap), dtype=np.int32)
    hints = np.empty((rows, cap, 3), dtype=np.float32)
    norm = np.empty(rows, dtype=np.float32)
    buffers = (*csr_buffers(p0), *csr_buffers(kp0), *csr_buffers(bounded))
    lib = native_student(config["student_library"])
    code = lib.gnn_pack_candidates(rows, cap, ptr(markers), ptr(sums),
        *(ptr(a) for a in buffers), ptr(x), ptr(cols), ptr(hints), ptr(norm))
    if code:
        raise RuntimeError("Native student candidate packing failed: %d" % code)
    return dict(cols=cols, hints=hints, norm=norm, x=x)


def stiffness_features(p, bridge, matrix, all_comp, n, paths):
    sums = np.empty(n, np.float64)
    x = np.zeros((n, ROW_FEATURES), np.float32)
    near_cols = np.empty((n, paths), np.int32)
    near_values = np.empty((n, paths), np.float64)
    bridge.gnn_student_stiffness.argtypes = [p.P, p.P, p.I, p.P, p.P, p.P, p.P]
    bridge.gnn_student_stiffness.restype = p.I
    status = bridge.gnn_student_stiffness(matrix, ptr(all_comp), paths, ptr(sums), ptr(x),
                                          ptr(near_cols), ptr(near_values))
    if status:
        raise RuntimeError("Student stiffness input failed: %d" % status)
    return sums, x, near_cols, near_values


def local_inputs(config, p, legacy, bridge, matrix, initial, components, kind, level,
                 global_components=None, both=False):
    """Build the deployment inputs before sampling; PETSc exchanges real halos."""
    begin = time.perf_counter()
    rows = ownership(p, matrix)
    n = len(components)
    all_comp = (p.all_rows(components, rows) if global_components is None else global_components)
    all_comp = np.ascontiguousarray(all_comp, dtype=np.int32)
    assert len(all_comp) == rows[-1]
    paths = config["student_paths"] if level == 1 and (kind == 1 or both) else 0
    sums, x, near_cols, near_values = stiffness_features(p, bridge, matrix, all_comp, n, paths)
    p0, _ = local_csr(p, initial.transfer, bridge)
    p0.sort_indices()
    kp0 = product_rows(p, bridge, matrix, initial.transfer)
    empty = sparse.csr_matrix(p0.shape)
    first = pack(config, initial.executed_markers, sums, p0, kp0, empty, x.copy())
    inputs = {"A": first}
    if paths:
        width = config["student_seed_entries"]
        cols = first["cols"][:, :width]
        hint = first["hints"][:, :width]
        v = hint[:, :, 0].astype(np.float64) + hint[:, :, 1]
        seed = row_csr(cols, v, (cols >= 0) & (v != 0), p0.shape[1])
        kernel = row_csr(near_cols, near_values, near_cols >= 0, int(rows[-1]))
        s = legacy.DistributedSparse(p, seed, rows, initial.owners)
        k = legacy.DistributedSparse(p, kernel, rows, rows)
        try:
            bounded = product_rows(p, bridge, k.matrix, s.matrix)
        finally:
            k.close()
            s.close()
        inputs["B"] = pack(config, initial.executed_markers, sums, p0, kp0, bounded, x)
    elif kind == 1 or both:
        inputs["B"] = first
    return inputs, p0, rows, p.maximum(time.perf_counter()-begin)


def injection(pair):
    return sparse.csr_matrix((np.ones(len(pair.injection_rows)),
        (pair.injection_rows, pair.injection_columns)), shape=(len(pair.markers), int(pair.owners[-1])))


def direct_inputs(config, p, legacy, bridge, matrix, pair, components):
    """Propagate coarse column IDs on the actual stiffness graph, with owned rows."""
    start = time.perf_counter()
    settings = dict(config, student_candidates=config["direct_candidates"])
    steps = config["direct_hops"]
    assert 1 <= steps <= 9
    rows, columns = pair.fine_owners, pair.owners
    markers = pair.markers
    assert np.array_equal(rows, ownership(p, matrix))
    all_comp = np.ascontiguousarray(components, dtype=np.int32)
    n = len(markers)
    sums, features, _, _ = stiffness_features(p, bridge, matrix, all_comp, n, 0)
    seeds = injection(pair)
    stiffness, _ = local_csr(p, matrix, bridge)
    local_rows = np.repeat(np.arange(n), np.diff(stiffness.indptr))
    diagonal = stiffness.indices == local_rows + rows[p.rank]
    stiffness.data[diagonal | (markers[local_rows] != -1)] = 0
    stiffness.eliminate_zeros()
    stiffness = stiffness.multiply(-1/np.maximum(sums, 1e-30)[:, None]).tocsr()
    stiffness.sort_indices()
    positive = legacy.DistributedSparse(p, abs(stiffness), rows, rows)
    signed = legacy.DistributedSparse(p, stiffness, rows, rows)
    empty = sparse.csr_matrix(seeds.shape)
    state = seeds
    ones = np.ones(n)
    try:
        for hop in range(steps):
            distributed = legacy.DistributedSparse(p, state, rows, columns)
            try:
                reach = product_rows(p, bridge, positive.matrix, distributed.matrix) + seeds
                reach.sort_indices()
                if hop == 0:
                    adjacent = -product_rows(p, bridge, signed.matrix, distributed.matrix)
                signed_path = (-product_rows(p, bridge, signed.matrix, distributed.matrix)
                               if hop == steps-1 else empty)
            finally:
                distributed.close()
            candidate = pack(settings, markers, ones, reach, signed_path,
                             adjacent if hop == steps-1 else empty, features)
            if hop < steps-1:
                cols = candidate["cols"]
                state = row_csr(cols, candidate["hints"][:, :, 0].astype(np.float64), cols >= 0,
                                seeds.shape[1])
    finally:
        signed.close()
        positive.close()
    return candidate, seeds, p.maximum(time.perf_counter()-start)


def geometry_candidates(fine_xyz, coarse_xyz, fine_comp, coarse_comp, free, count):
    """Query only owned free rows and use component-local coarse lengths."""
    from scipy.spatial import cKDTree

    result = np.full((len(fine_xyz), count), -1, dtype=np.int32)
    spacing = np.ones(3)
    for component in range(3):
        rows = np.flatnonzero(free & (fine_comp == component))
        cols = np.flatnonzero(coarse_comp == component)
        if not len(cols):
            continue
        points = coarse_xyz[cols]
        tree = cKDTree(points)
        if len(cols) > 1:
            nearest, _ = tree.query(points, k=[2])
            distances = nearest[nearest > 0]
            if len(distances):
                spacing[component] = np.median(distances)
        k = min(count, len(cols))
        if len(rows):
            _, indices = tree.query(fine_xyz[rows], k=list(range(1, k+1)))
            result[rows, :k] = cols[indices]
    return result, spacing[fine_comp]


def extended_reach(config, p, legacy, bridge, pair, neighbors, strengths, reach, x):
    """Keep a small label frontier while extending actual stiffness paths."""
    hops = config["context_path_steps"]
    if not hops:
        return reach
    rows, cols = pair.fine_owners, pair.owners
    graph = row_csr(neighbors, np.abs(strengths), (neighbors >= 0) & (strengths != 0), rows[-1])
    left = legacy.DistributedSparse(p, graph, rows, rows)
    empty = sparse.csr_matrix(reach.shape)
    settings = dict(student_candidates=16, student_library=config["student_library"])
    base = reach
    try:
        for _ in range(hops):
            front = pack(settings, pair.markers, np.ones(len(x)), reach, empty, empty, x)
            ids = front["cols"]
            weights = front["hints"][:,:,0].astype(np.float64)
            state = row_csr(ids, weights, (ids >= 0) & (weights != 0), cols[-1])
            right = legacy.DistributedSparse(p, state, rows, cols)
            try:
                reach = base+product_rows(p, bridge, left.matrix, right.matrix)
                reach.sort_indices()
            finally:
                right.close()
    finally:
        left.close()
    return reach


def local_direct_inputs(config, p, legacy, bridge, matrix, pair, components, geometry, context=False):
    """Combine geometric candidates with direct and two-hop stiffness contacts."""
    start = time.perf_counter()
    setting = config["direct_local"]
    geo_count, graph_count = setting["geometry_neighbors"], setting["stiffness_neighbors"]
    assert 1 <= geo_count and 1 <= graph_count <= 16 and 12 <= geo_count+graph_count <= 48
    rows, cols = pair.fine_owners, pair.owners
    assert np.array_equal(rows, ownership(p, matrix))
    lo, hi = rows[p.rank:p.rank+2]
    n = hi-lo
    free = pair.markers == -1
    all_comp = np.ascontiguousarray(components, dtype=np.int32)
    comp = all_comp[lo:hi]
    xyz, phi = geometry
    coarse_native = p.all_rows(pair.native, cols).astype(np.int64)
    fine_nodes, coarse_nodes = pair.fine_native//3, coarse_native//3
    fine_xyz, coarse_xyz = xyz[fine_nodes], xyz[coarse_nodes]
    geom, spacing = geometry_candidates(fine_xyz, coarse_xyz, comp, coarse_native%3, free, geo_count)

    # Assemble one algebraic product from K and the actual C columns of K.
    # Column IDs can be owned independently from their fine injection rows.
    anchors = np.zeros(cols[-1], dtype=np.float64)
    anchors[pair.injection_columns] = lo+pair.injection_rows
    anchors = p.reduce(anchors).astype(np.int64)
    sums, x, neighbors, strengths = stiffness_features(p, bridge, matrix, all_comp, n, 8 if context else 0)
    stiffness, _ = local_csr(p, matrix, bridge)
    local_rows = np.repeat(np.arange(n), np.diff(stiffness.indptr))
    diagonal = stiffness.indices == local_rows+lo
    stiffness.data[diagonal | ~free[local_rows]] = 0
    stiffness.eliminate_zeros()
    stiffness = stiffness.multiply(-1/np.maximum(sums, 1e-30)[:, None]).tocsr()
    stiffness.sort_indices()
    contacts = stiffness[:, anchors].tocsr()
    contacts.sort_indices()
    graph = legacy.DistributedSparse(p, stiffness, rows, rows)
    coarse_contacts = legacy.DistributedSparse(p, contacts, rows, cols)
    try:
        paths = product_rows(p, bridge, graph.matrix, coarse_contacts.matrix)
    finally:
        coarse_contacts.close()
        graph.close()
    paths.eliminate_zeros()
    reach = abs(contacts)+abs(paths)
    reach.sort_indices()
    if context:
        reach = extended_reach(config, p, legacy, bridge, pair, neighbors, strengths, reach, x)
    algebra = pack(dict(student_candidates=16, student_library=config["student_library"]),
        pair.markers, np.ones(n), reach, -contacts, -paths, x)["cols"]
    novel = (algebra >= 0) & ~np.any(algebra[:, :, None] == geom[:, None, :], axis=2)
    rank = np.cumsum(novel, axis=1)-1
    r, c = np.where(novel & (rank < graph_count))
    extra = np.full((n, graph_count), -1, dtype=np.int32)
    extra[r, rank[r,c]] = algebra[r,c]
    columns = np.concatenate((geom, extra), axis=1)
    safe = np.maximum(columns, 0)
    r, c = np.where(columns >= 0)
    picked = columns[r,c]
    delta = (coarse_xyz[safe]-fine_xyz[:,None,:])/spacing[:,None,None]
    distance = np.linalg.norm(delta, axis=2)
    same = (coarse_native[safe]%3 == comp[:,None]) & (columns >= 0)
    radial = same/(.25+distance**2)
    radial /= np.maximum(radial.sum(axis=1, keepdims=True), 1e-30)
    edges = np.zeros(columns.shape+(CONTEXT_EDGE_FEATURES if context else LOCAL_EDGE_FEATURES,), dtype=np.float32)
    edges[:,:,0:3] = np.clip(delta,-6,6)
    edges[:,:,3:6] = np.minimum(np.abs(delta),6)
    edges[:,:,6] = np.minimum(distance,6)
    edges[:,:,7] = radial
    edges[:,:,8] = same
    if len(r):
        edges[r,c,9] = np.arcsinh(12*np.asarray(contacts[r,picked]).ravel())
        edges[r,c,10] = np.arcsinh(120*np.asarray(paths[r,picked]).ravel())
    edges[:,:,11] = phi[fine_nodes,None]
    edges[:,:,12] = phi[coarse_nodes[safe]]
    edges[:,:,13:17] = x[:,None,[0,1,4,5]]
    nearest = np.min(np.where(same, distance, 6), axis=1)
    edges[:,:,17] = np.minimum(nearest[:,None],6)
    candidate = dict(cols=columns, edges=edges, x=x)
    if context:
        if len(r):
            edges[r,c,18] = np.arcsinh(20*np.asarray(reach[r,picked]).ravel())
        edges[:,:,19:22] = np.eye(3, dtype=np.float32)[coarse_native[safe]%3]
        scale = np.maximum(np.ptp(xyz,axis=0), 1e-30)
        nodes = np.column_stack((x[:,:5], phi[fine_nodes], (fine_xyz-xyz.min(axis=0))/scale,
                                 np.eye(3)[comp], pair.markers == -3)).astype(np.float32)
        strengths /= np.maximum(np.abs(strengths).sum(axis=1,keepdims=True), 1e-30)
        membership = .02+radial+.1*np.abs(edges[:,:,9:11]).sum(axis=2)+.1*edges[:,:,18]
        pooling = row_csr(columns, membership, columns >= 0, cols[-1])
        packed = p.reduce(pooling.T @ np.column_stack((nodes,np.ones(n))))
        pooled = packed[:,:-1]/np.maximum(packed[:,-1:], 1e-30)
        candidate.update(nodes=nodes,neighbors=neighbors,strengths=strengths.astype(np.float32),
                         pooled=pooled.astype(np.float32))
    return candidate, injection(pair), p.maximum(time.perf_counter()-start)


def context_layers(width):
    return ((CONTEXT_NODE_FEATURES,width), (3*width,width), (CONTEXT_NODE_FEATURES,width),
            (CONTEXT_EDGE_FEATURES+3*width,width), (3*width,width), (width,2), (3*width,4))


def coarse_rows(p, pair, local):
    values = np.zeros((pair.owners[-1], local.shape[1]), dtype=np.float64)
    values[pair.injection_columns] = local[pair.injection_rows]
    return p.reduce(values).astype(np.float32)


def context_numpy_decode(raw, scores, budget_logits, valid):
    rank_scores = np.where(valid, np.rint(CONTEXT_SCORE_SCALE*scores), -np.inf)
    order = np.argsort(-rank_scores,axis=1,kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(order.shape[1])[None,:],axis=1)
    budget = np.asarray(CONTEXT_BUDGETS)[np.argmax(np.rint(CONTEXT_SCORE_SCALE*budget_logits),axis=1)]
    selected = (rank < budget[:,None]) & valid
    return np.where(selected,raw,0), selected.sum(axis=1).astype(np.uint8)


def context_parameters(width, parameters):
    offset = 0
    layers = []
    for inputs, outputs in context_layers(width):
        size = inputs*outputs
        matrix = parameters[offset:offset+size].reshape(outputs,inputs)
        layers.append((matrix,parameters[offset+size:offset+size+outputs]))
        offset += size+outputs
    assert offset == len(parameters)
    return layers


def context_edge_output(layers, fine, coarse, pooled, edges, columns):
    """Rank-local row and coarse-context decoding before sparsification."""
    valid = columns >= 0
    safe = np.maximum(columns,0)
    valid3 = valid[:,:,None]
    width, edge_width = fine.shape[1], edges.shape[2]
    w, bias = layers[3]
    anchor = (coarse @ w[:,edge_width+width:edge_width+2*width].T+
              pooled @ w[:,edge_width+2*width:].T)
    local = fine @ w[:,edge_width:edge_width+width].T
    base = (edges.reshape((-1,edge_width)) @ w[:,:edge_width].T).reshape(columns.shape+(width,))
    hidden = np.maximum(base+local[:,None,:]+anchor[safe]+bias,0)
    mean = (hidden*valid3).sum(axis=1)/np.maximum(valid.sum(axis=1,keepdims=True),1).astype(np.float32)
    maximum = (hidden*valid3).max(axis=1)
    c = np.concatenate((mean,maximum),axis=1)
    w, bias = layers[4]
    base = (hidden.reshape((-1,width)) @ w[:,:width].T).reshape(hidden.shape)
    refined = np.maximum(base+(c @ w[:,width:].T)[:,None,:]+bias,0)
    raw = (refined.reshape((-1,width)) @ layers[5][0].T+layers[5][1]).reshape(columns.shape+(2,))
    budget = np.concatenate((fine,c),axis=1) @ layers[6][0].T+layers[6][1]
    return raw[:,:,0],raw[:,:,1],budget


def context_node_output(layers, center, neighbors, strengths):
    message = np.concatenate((center,(neighbors*np.abs(strengths)[:,:,None]).sum(axis=1),
                              (neighbors*strengths[:,:,None]).sum(axis=1)),axis=1)
    return np.maximum(message @ layers[1][0].T+layers[1][1],0)


def predict_context(p, candidate, pair, width, params):
    layers = context_parameters(width, params)
    center = np.maximum(candidate["nodes"] @ layers[0][0].T+layers[0][1],0)
    neighbors = neighbour_blocks(p,pair.fine_owners,center,candidate["neighbors"])
    fine = context_node_output(layers,center,neighbors,candidate["strengths"])
    coarse = coarse_rows(p,pair,fine)
    pooled = np.maximum(candidate["pooled"] @ layers[2][0].T+layers[2][1],0)
    columns, edges = candidate["cols"], candidate["edges"]
    values = np.zeros(columns.shape,dtype=np.float32)
    budgets = np.zeros(len(columns),dtype=np.uint8)
    for start in range(0,len(columns),1024):
        part = slice(start,start+1024)
        raw,scores,logits = context_edge_output(layers,fine[part],coarse,pooled,edges[part],columns[part])
        if not np.isfinite(raw).all() or not np.isfinite(scores).all() or not np.isfinite(logits).all():
            raise ValueError("Nonfinite graph interpolation output")
        values[part],budgets[part] = context_numpy_decode(raw,scores,logits,columns[part]>=0)
    return values,budgets


def predict_local(config, candidate, markers, width, params):
    lib = native_student(config["student_library"])
    function = lib.gnn_direct_predict
    function.argtypes = [C.c_int]*5+[C.c_float]+[C.c_void_p]*6
    function.restype = C.c_int
    columns, edges = candidate["cols"], candidate["edges"]
    n, cap = columns.shape
    assert edges.shape == (n, cap, LOCAL_EDGE_FEATURES)
    assert edges.dtype == np.float32 and edges.flags.c_contiguous
    params = np.ascontiguousarray(params, dtype=np.float32)
    assert len(params) == (LOCAL_EDGE_FEATURES+3)*width+2
    values = np.zeros(columns.shape, dtype=np.float32)
    budgets = np.zeros(n, dtype=np.uint8)
    options = config["direct_local"]
    assert 1 <= options["min_entries"] <= options["max_entries"] <= 12
    code = function(n, cap, width, options["min_entries"], options["max_entries"], options["score_cutoff"],
        *(ptr(v) for v in (markers, columns, edges, params, values, budgets)))
    if code:
        raise ValueError("Native local direct inference failed: %d" % code)
    return values, budgets


def predict(config, candidate, markers, coarse_components, width, kind, params, controls):
    lib = native_student(config["student_library"])
    n, cap = candidate["cols"].shape
    values = np.zeros((n, cap), np.float32)
    budgets = np.zeros(n, np.uint8)
    model = np.ascontiguousarray(params, np.float32)
    control = np.ascontiguousarray(controls, np.float32)
    assert len(model) == (ROW_FEATURES+5)*width+4 + ((EDGE_FEATURES+3)*width+2 if kind else 0)
    assert control.shape == (4,) and np.isfinite(control).all()
    coarse = np.ascontiguousarray(coarse_components, np.int32)
    code = lib.gnn_student_predict(n, cap, width, kind, ptr(model), ptr(control),
        ptr(markers), ptr(candidate["cols"]), ptr(coarse), ptr(candidate["x"]),
        ptr(candidate["hints"]), ptr(candidate["norm"]), ptr(values), ptr(budgets))
    if code:
        raise ValueError("Native student inference failed: %d" % code)
    return values, budgets


@lru_cache(maxsize=32)
def load_student(path, sha256, level):
    assert digest(path) == sha256
    with np.load(path, allow_pickle=False) as f:
        assert int(f["schema"]) == SCHEMA and int(f["level"]) == level
        width, kind, cap = int(f["width"]), int(f["kind"]), int(f["candidates"])
        params = np.ascontiguousarray(f["params"], np.float32)
    assert kind in STUDENT_KINDS.values() and np.isfinite(params).all()
    assert 1 <= width <= (32 if kind >= STUDENT_KINDS["E"] else 16)
    return width, kind, cap, params


def student_level(config, p, legacy, bridge, matrix, initial, components, model, level,
                  controls=DEFAULT_CONTROL, global_components=None):
    start = time.perf_counter()
    width, kind, cap, params = load_student(model["path"], model["sha256"], level)
    assert kind in (STUDENT_KINDS["A"], STUDENT_KINDS["B"]) and cap == config["student_candidates"]
    inputs, p0, rows, input_s = local_inputs(config, p, legacy, bridge, matrix, initial,
        components, kind, level, global_components)
    candidate = inputs["B" if kind else "A"]
    coarse_components = p.all_rows(initial.components, initial.owners).astype(np.int32)
    markers = initial.executed_markers
    infer = time.perf_counter()
    try:
        values, budgets = predict(config, candidate, markers, coarse_components, width, kind, params, controls)
        invalid = 0
    except ValueError:
        invalid = 1
    if p.maximum(invalid):
        raise ValueError("Student native inference failed on an owned row")
    inference_s = p.maximum(time.perf_counter()-infer)
    csr = row_csr(candidate["cols"], values.astype(np.float64), values != 0, p0.shape[1])
    csr = (csr + p0.multiply((markers != -1)[:, None])).tocsr()
    csr.eliminate_zeros()
    csr.sort_indices()
    degree = np.diff(csr.indptr)
    protected = csr[markers != -1] - p0[markers != -1]
    checks = p.reduce([protected.nnz, np.count_nonzero((degree == 0) & (markers == -1)),
                      int(not np.isfinite(csr.data).all()), csr.nnz,
                      *[np.count_nonzero(budgets == x) for x in (6, 8, 12)]])
    if any(checks[:3]) or p.maximum(int(degree.max(initial=0))) > 12:
        raise ValueError("Student violated protected rows, finite values, coverage or row budget")
    stats = dict(initial.stats, nnz=int(checks[3]), interpolation_student=model["sha256"],
        student_kind="B" if kind else "A", budget_counts=list(map(int, checks[4:])),
        candidate_s=input_s, native_inference_s=inference_s)
    result = imported_transfer(p, legacy, matrix, csr, rows, initial.owners, stats)
    result.native, result.components = initial.native, initial.components
    result.executed_markers = markers
    stats["student_construction_s"] = p.maximum(time.perf_counter()-start)
    return result


def sample_rows(rng, markers, x, count, phase=None):
    eligible = np.flatnonzero(markers == -1)
    if len(eligible) <= count:
        return eligible
    # Allocate part of each frame to high cross-component coupling and to
    # the damage transition. Those strata use training-frame features only.
    cross = x[eligible, 4]
    focused = eligible[cross >= np.quantile(cross, .8)]
    if phase is not None:
        damage = (phase[eligible] > .05) & (phase[eligible] < .95)
        focused = np.union1d(focused, eligible[damage])
    chosen = rng.choice(focused, min(count//3, len(focused)), replace=False)
    other = np.setdiff1d(eligible, chosen, assume_unique=True)
    remainder = rng.choice(other, count-len(chosen), replace=False)
    return np.sort(np.r_[chosen, remainder])


def match_targets(candidate, target, indexes):
    coeff = np.zeros(candidate["cols"][indexes].shape, np.float32)
    mass = np.zeros(len(indexes), np.float32)
    covered = np.zeros(len(indexes), np.float32)
    for sample, i in enumerate(indexes):
        lo, hi = target.indptr[i:i+2]
        cols, values = target.indices[lo:hi], target.data[lo:hi]
        wanted = candidate["cols"][i]
        pos = np.searchsorted(cols, wanted)
        if len(cols):
            inrange = np.minimum(pos, len(cols)-1)
            found = (pos < len(cols)) & (cols[inrange] == wanted)
            coeff[sample, found] = values[inrange[found]]
        mass[sample] = np.linalg.norm(values)
        covered[sample] = np.linalg.norm(coeff[sample])
    return coeff, mass, covered


def normalized_teacher(pair):
    target = pair.raw_transfer.multiply(1/pair.column_scale[None, :]).tocsr()
    target.eliminate_zeros()
    target.sort_indices()
    return target


def teacher_on_support(candidate, pair, control, model_support):
    """Keep original teacher coefficients in the unit-injection coordinates."""
    target = normalized_teacher(pair)
    free = np.flatnonzero(pair.markers == -1)
    assert control in ("candidates", "top6", "top12", "model_support")
    top = {"top6": 6, "top12": 12}.get(control)
    columns = candidate["cols"].copy()
    values = np.zeros(columns.shape, dtype=np.float64)
    # Use full precision values for the controlled solves.
    for row in free:
        start, stop = target.indptr[row:row+2]
        source = target.indices[start:stop]
        selected = np.flatnonzero(columns[row] >= 0)
        if control == "model_support":
            selected = selected[model_support[row, selected]]
        indices = np.searchsorted(source, columns[row, selected])
        indices = np.minimum(indices, len(source)-1)
        matched = source[indices] == columns[row, selected]
        selected, indices = selected[matched], indices[matched]
        weights = target.data[start+indices]
        if top is not None:
            order = np.argsort(-np.abs(weights), kind="stable")[:top]
            selected, weights = selected[order], weights[order]
        if len(selected):
            values[row, selected] = weights
    missing = free[~np.any(values[free], axis=1)]
    for row in missing:
        start, stop = target.indptr[row:row+2]
        largest = start+int(np.argmax(np.abs(target.data[start:stop])))
        columns[row, 0] = target.indices[largest]
        values[row, 0] = target.data[largest]
    return columns, values, len(missing)


def direct_graph(config, kind):
    if kind == "F":
        assert config["direct_local"]["min_entries"] == CONTEXT_BUDGETS[0]
        assert config["direct_local"]["max_entries"] == CONTEXT_BUDGETS[-1]
        return dict(kind=kind, path_steps=config["context_path_steps"], **config["direct_local"])
    if kind == "E":
        return dict(kind=kind, **config["direct_local"])
    return dict(hops=config["direct_hops"], candidates=config["direct_candidates"])


def direct_level(config, p, legacy, bridge, matrix, pair, components, model, control=None, geometry=None):
    width, kind, cap, params = load_student(model["path"], model["sha256"], 1)
    context = kind == STUDENT_KINDS["F"]
    local = kind >= STUDENT_KINDS["E"]
    assert local or kind == STUDENT_KINDS["D"]
    if local:
        assert model["graph"] == direct_graph(config, "F" if context else "E")
        candidate, seeds, candidate_s = local_direct_inputs(config, p, legacy, bridge, matrix, pair,
            components, geometry, context=context or control == "context_candidates")
    else:
        candidate, seeds, candidate_s = direct_inputs(config, p, legacy, bridge, matrix, pair, components)
    assert cap == candidate["cols"].shape[1]
    inference_s = 0.0
    budgets = np.zeros(len(pair.markers), np.uint8)
    values = None
    if control is None or control == "model_support":
        begin = time.perf_counter()
        invalid = False
        try:
            if context:
                values, budgets = predict_context(p, candidate, pair, width, params)
            elif local:
                values, budgets = predict_local(config, candidate, pair.markers, width, params)
            else:
                comp = p.all_rows(pair.components, pair.owners).astype(np.int32)
                values, budgets = predict(config, candidate, pair.markers, comp, width, kind, params, DEFAULT_CONTROL)
        except ValueError:
            invalid = True
        if p.maximum(invalid):
            raise ValueError("direct_p1_nonfinite_prediction")
        inference_s = p.maximum(time.perf_counter()-begin)
    free = pair.markers == -1
    columns = candidate["cols"]
    guard_rows = 0
    if control is not None:
        columns, values, guard_rows = teacher_on_support(candidate, pair,
            "candidates" if control == "context_candidates" else control,
            values != 0 if values is not None else None)
    learned = row_csr(columns, values.astype(np.float64), values != 0, seeds.shape[1])
    assert not learned[~free].nnz
    csr = (learned + seeds).tocsr()
    missing = (np.diff(csr.indptr) == 0) & free
    budget_options = range(1, 13) if local else (6, 8, 12)
    nnz, empty, active, guard, *budget_counts = map(int, p.reduce([csr.nnz, missing.sum(), free.sum(), guard_rows,
        *(np.count_nonzero(budgets == k) for k in budget_options)]))
    stats = dict(source="direct_p1", fine_dofs=int(pair.fine_owners[-1]), coarse_dofs=int(pair.owners[-1]),
        nnz=nnz, active_rows=active, uncovered_free_rows=empty, candidate_s=candidate_s,
        native_inference_s=inference_s, budget_counts=budget_counts)
    if local:
        stats["budget_options"] = list(budget_options)
    if control is not None:
        stats.update(teacher_support=control, teacher_guard_rows=guard)
    if empty:
        return None, stats
    return imported_transfer(p, legacy, matrix, csr, pair.fine_owners, pair.owners, stats), stats


def export_direct_teacher(config, p, legacy, bridge, row, matrix, pair, components, directions, phase,
                          geometry=None, feedback_count=0):
    """Export signed unit-injection labels; keep full hierarchy shard provenance."""
    assert row["split"] == "training" and directions
    kind = config.get("direct_teacher_kind", "D")
    if kind in ("E", "F"):
        candidate, _, candidate_s = local_direct_inputs(config, p, legacy, bridge, matrix, pair,
                                                        components, geometry, context=kind == "F")
    else:
        candidate, _, candidate_s = direct_inputs(config, p, legacy, bridge, matrix, pair, components)
    target = normalized_teacher(pair)
    candidate_mask = row_csr(candidate["cols"], np.ones(candidate["cols"].shape, dtype=bool),
                             candidate["cols"] >= 0, target.shape[1])
    intersection = target.multiply(candidate_mask).tocsr()
    row_weight = np.asarray(target.power(2).sum(axis=1)).ravel()
    covered_weight = np.asarray(intersection.power(2).sum(axis=1)).ravel()
    free = pair.markers == -1
    crack = free & (phase > .05) & (phase < .95)
    row_fraction = covered_weight[free]/row_weight[free]
    totals = p.reduce([target[free].nnz, intersection[free].nnz, row_weight[free].sum(),
        covered_weight[free].sum(), row_weight[crack].sum(), covered_weight[crack].sum(),
        np.count_nonzero(row_fraction < .9), np.count_nonzero(row_fraction == 0), free.sum(), crack.sum()])
    minimum = p.reduce([np.min(row_fraction, initial=1.0)], "min")[0]
    coverage = dict(entries=int(totals[0]), matched_entries=int(totals[1]),
        weight_squared=float(totals[3]/totals[2]),
        transition_weight_squared=float(totals[5]/totals[4]) if totals[4] else None,
        rows_under_90_percent=int(totals[6]), uncovered_rows=int(totals[7]),
        free_rows=int(totals[8]), transition_rows=int(totals[9]), min_row_weight_squared=float(minimum))
    coarse_native = p.all_rows(pair.native, pair.owners).astype(np.int64)
    q = p.all_rows(np.column_stack(directions).ravel(), pair.owners*len(directions))
    q = q.reshape((-1, len(directions))) * pair.column_scale[:, None]
    q /= np.maximum(np.sqrt(np.mean(q*q, axis=0)), 1e-30)
    rng = np.random.default_rng(20261109+1009*p.rank)
    indices = sample_rows(rng, pair.markers, candidate["x"], config["student_rows_per_rank"], phase)
    counts = p.gather_equal([len(indices)]).ravel().astype(np.int64)
    owners = np.r_[0, np.cumsum(counts)]
    coeff, mass, covered = match_targets(candidate, target, indices)
    row_arrays = ("cols","edges","x","nodes","strengths") if kind == "F" else tuple(candidate)
    sampled = dict(**{name: candidate[name][indices] for name in row_arrays},
                   target=coeff, norm_teacher=mass, norm_covered=covered)
    coarse_features = {}
    if kind == "F":
        neighbors = neighbour_blocks(p,pair.fine_owners,candidate["nodes"],candidate["neighbors"])
        sampled.update(neighbor_nodes=neighbors[indices], region=np.full(len(indices),p.rank//4,dtype=np.int32))
        n = len(candidate["nodes"])
        bundle = np.column_stack((candidate["nodes"],neighbors.reshape((n,8*CONTEXT_NODE_FEATURES)),
                                  candidate["strengths"]))
        coarse = coarse_rows(p,pair,bundle)
        coarse_features = dict(coarse_nodes=coarse[:,:CONTEXT_NODE_FEATURES],
            coarse_neighbor_nodes=coarse[:,CONTEXT_NODE_FEATURES:-8].reshape((-1,8,CONTEXT_NODE_FEATURES)),
            coarse_strengths=coarse[:,-8:],pooled=candidate["pooled"])
    if kind in ("E", "F"):
        sampled["row_sum"] = np.asarray(target[indices].sum(axis=1)).ravel().astype(np.float32)
    values = {kind+"_"+name: gather_array(p, v, owners) for name, v in sampled.items()}
    action = gather_array(p, (target[indices] @ q).astype(np.float32), owners)
    if p.rank != 0:
        return None
    job = os.environ["GNN_RUN_ID"]
    file = Path(config["output_directory"])/"data"/("direct-%s-%s-samples.npz" % (job, row["key"]))
    with file.open("xb") as stream:
        np.savez(stream, **values, **coarse_features, q=q.astype(np.float32), action=action,
                 coarse_native=coarse_native)
    return dict(key=row["key"], split=row["split"], level=1, rows=int(owners[-1]),
        real_directions=len(directions), student_feedback_directions=feedback_count,
        candidate_s=candidate_s, coverage=coverage,
        graph=direct_graph(config, kind),
        manifest_sha256=row["manifest_sha256"], transfers=pair.receipt,
        samples=dict(path=str(file), sha256=digest(file)))


def export_teacher(config, p, legacy, bridge, row, matrix, initial, teacher, components, directions,
                   level, native, global_components=None, phase=None):
    """Export full endpoints and bounded samples from genuine solved frames."""
    assert row["split"] == "training" and directions
    inputs, _, rows, input_s = local_inputs(config, p, legacy, bridge, matrix, initial,
        components, 1, level, global_components, both=True)
    target, _ = local_csr(p, teacher.transfer, bridge)
    target.sort_indices()
    assert np.array_equal(initial.owners, teacher.owners)
    q = p.all_rows(np.column_stack(directions).ravel(), teacher.owners*len(directions))
    q = q.reshape((-1, len(directions)))
    q /= np.maximum(np.sqrt(np.mean(q*q, axis=0)), 1e-30)
    rng = np.random.default_rng(20261109 + 1009*p.rank + level)
    indexes = sample_rows(rng, teacher.executed_markers, inputs["A"]["x"],
                          config["student_rows_per_rank"] if level == 1 else 256, phase)
    counts = p.gather_equal([len(indexes)]).ravel().astype(np.int64)
    samples = np.r_[0, np.cumsum(counts)]
    actions = target[indexes] @ q
    export = dict(native=gather_array(p, np.asarray(native)[indexes], samples),
                  q=q.astype(np.float32),
                  action=gather_array(p, actions.astype(np.float32), samples),
                  coarse_native=p.all_rows(initial.native, initial.owners).astype(np.int64))
    for kind, candidate in inputs.items():
        coeff, mass, covered = match_targets(candidate, target, indexes)
        arrays = dict(**{name: value[indexes] for name, value in candidate.items()},
                      target=coeff, norm_teacher=mass, norm_covered=covered)
        for name, value in arrays.items():
            export[kind+"_"+name] = gather_array(p, value, samples)
    _, p0i, p0j, p0v = collect_csr(p, bridge, initial.transfer)
    _, pti, ptj, ptv = collect_csr(p, bridge, teacher.transfer)
    full_native = gather_array(p, native, rows)
    markers = gather_array(p, initial.executed_markers, rows)
    if p.rank != 0:
        return None
    run = os.environ["GNN_RUN_ID"]
    root = Path(config["output_directory"]) / "data"
    name = "student-%s-L%d-%s" % (run, level, row["key"])
    data_file, sparse_file = root/(name+"-samples.npz"), root/(name+"-transfers.npz")
    with data_file.open("xb") as f:
        np.savez(f, **export)
    with sparse_file.open("xb") as f:
        np.savez(f, p0_indptr=p0i, p0_indices=p0j, p0_values=p0v,
                 teacher_indptr=pti, teacher_indices=ptj, teacher_values=ptv,
                 native=full_native, coarse_native=export["coarse_native"], markers=markers)
    coverage = {kind: float(np.sum(export[kind+"_norm_covered"]**2) /
                                max(np.sum(export[kind+"_norm_teacher"]**2), 1e-30))
                for kind in inputs}
    return dict(key=row["key"], split=row["split"], level=level, rows=int(samples[-1]),
                real_directions=len(directions), graph_input_s=input_s,
                manifest_sha256=row["manifest_sha256"], coverage_weight_squared=coverage,
                samples=dict(path=str(data_file), sha256=digest(data_file)),
                transfers=dict(path=str(sparse_file), sha256=digest(sparse_file)))
