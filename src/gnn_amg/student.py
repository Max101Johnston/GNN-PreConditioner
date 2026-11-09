"""Rank-local interpolation candidates, native deployment, and teacher export."""
import ctypes as C
from functools import lru_cache
import os
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from .backend import bind, digest, local_csr, ownership
from .dataset import gather_array
from .interpolation import collect_csr, imported_transfer

ROW_FEATURES = 10
EDGE_FEATURES = 7
SCHEMA = 1
DEFAULT_CONTROL = (0.0, 0.0, 1.0, 1.0)


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
    assert 1 <= width <= 16 and kind in (0, 1) and np.isfinite(params).all()
    return width, kind, cap, params


def student_level(config, p, legacy, bridge, matrix, initial, components, model, level,
                  controls=DEFAULT_CONTROL, global_components=None):
    start = time.perf_counter()
    width, kind, cap, params = load_student(model["path"], model["sha256"], level)
    assert cap == config["student_candidates"]
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
