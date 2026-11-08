"""In-memory numerical bridge; no online teacher construction.

The original replay reader and RAS implementation are loaded from an explicit,
hashed dependency snapshot. Dataset paths and machine settings live outside
the source repository.
"""
import ctypes as C
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_dependencies(config):
    source = Path(config["dependency_directory"])
    for name, expected in config["dependency_hashes"].items():
        if digest(source / name) != expected:
            raise ValueError("Dependency changed: " + name)
    sys.path.insert(0, str(source))
    return importlib.import_module("linear_transfer"), importlib.import_module("adaptive_coarse")


def record(config, event, **values):
    row = dict(event=event, unix_time=time.time(), **values)
    with (Path(config["output_directory"]) / "records" / config.get("progress_name", "progress.jsonl")).open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
    print("GNN", json.dumps(row, allow_nan=False), flush=True)


def bind(p, name, args):
    fn = getattr(p.lib, name)
    fn.argtypes, fn.restype = args, p.I
    p.functions[name] = fn


def ownership(p, matrix):
    lo, hi = p.I(), p.I()
    p.call("MatGetOwnershipRange", matrix, C.byref(lo), C.byref(hi))
    ranges = p.gather_equal([lo.value, hi.value]).astype(np.int64)
    assert np.array_equal(ranges[:-1, 1], ranges[1:, 0])
    return np.r_[ranges[:, 0], ranges[-1, 1]]


class Action:
    """Rectangular PETSc actions, including ranks owning no coarse columns."""
    def __init__(self, p, matrix):
        self.p, self.matrix = p, matrix
        bind(p, "MatMultTranspose", [p.P, p.P, p.P])
        self.right, self.left = p.P(), p.P()
        p.call("MatCreateVecs", matrix, C.byref(self.right), C.byref(self.left))
        self.sizes = []
        for v in (self.right, self.left):
            n = p.I()
            p.call("VecGetLocalSize", v, C.byref(n))
            self.sizes.append(n.value)

    def __call__(self, values, transpose=False):
        p = self.p
        source, target = (self.left, self.right) if transpose else (self.right, self.left)
        ns, nt = self.sizes[::-1] if transpose else self.sizes
        assert len(values) == ns
        if ns:
            p.assign(source, values)
        p.call("MatMultTranspose" if transpose else "MatMult", self.matrix, source, target)
        return p.array(target) if nt else np.empty(0, dtype=np.float64)

    def close(self):
        self.p.destroy("Vec", self.right, self.left)


class Level:
    def __init__(self, p, bridge, matrix, native, components, selected, constrained, setting):
        self.p = p
        native = np.asarray(native, dtype=np.int64)
        assert native.shape == components.shape == selected.shape == constrained.shape
        cf = np.where(constrained, -3, np.where(selected, 1, -1)).astype(np.int32)
        original_cf = cf.copy()
        functions = np.ascontiguousarray(components, dtype=np.int32)
        assert np.isin(functions, [0, 1, 2]).all()
        self.transfer, self.coarse = p.P(), p.P()
        stats = np.zeros(11, np.int64)
        flags = np.zeros(len(native), np.int32)
        proposed = int(p.reduce([np.count_nonzero(cf == 1)])[0])
        if not proposed:
            raise ValueError("An empty C proposal cannot be repaired")
        budget = int(proposed * setting.get("coverage_repair_fraction", 0.0))
        if "max_coarse_dofs" in setting:
            budget = min(budget, max(0, setting["max_coarse_dofs"] - proposed))
        code = bridge.gnn_multipass(matrix, cf.ctypes.data, functions.ctypes.data,
            setting["strength_threshold"], setting["truncation"], setting["max_entries"],
            native.ctypes.data, budget, flags.ctypes.data,
            C.byref(self.transfer), C.byref(self.coarse), stats.ctypes.data)
        if code:
            raise RuntimeError("multipass native error %d" % code)
        self.stats = dict(zip(["fine_dofs", "coarse_dofs", "uncovered_free_rows", "empty_free_rows", "nnz", "special_fine_rows", "raw_uncovered_9_hops", "raw_uncovered_99_hops", "repair_added_coarse", "repair_rounds", "repair_budget_exhausted"], map(int, stats)))
        self.raw_uncovered = (flags & 1) != 0
        self.repair_added = (flags & 4) != 0
        self.special_fine = (original_cf == -1) & (cf == -3)
        self.executed_markers = cf.copy()
        self.native = native[cf == 1].copy()
        self.components = functions[cf == 1].copy()
        self.owners = ownership(p, self.coarse)
        assert len(self.native) == self.owners[p.rank+1] - self.owners[p.rank]
        self.P, self.K = Action(p, self.transfer), Action(p, self.coarse)
        # Every coarse point must inject to exactly its own distributed column.
        lo, hi = ownership(p, matrix)[p.rank:p.rank+2]
        offset = int(self.owners[p.rank])
        largest = 0.0
        for index in np.flatnonzero(cf == 1):
            n, columns, values = p.I(), p.P(), p.P()
            p.call("MatGetRow", self.transfer, int(lo + index), C.byref(n), C.byref(columns), C.byref(values))
            if n.value != 1:
                largest = 1.0
            else:
                col = C.cast(columns, C.POINTER(p.I))[0]
                val = C.cast(values, C.POINTER(p.D))[0]
                largest = max(largest, abs(val-1), float(col != offset))
            p.call("MatRestoreRow", self.transfer, int(lo + index), C.byref(n), C.byref(columns), C.byref(values))
            offset += 1
        self.stats["injection_error"] = p.maximum(largest)

    def close(self):
        self.P.close()
        self.K.close()
        self.p.destroy("Mat", self.transfer, self.coarse)


class Hierarchy:
    """Three levels: RAS is supplied externally; one fixed middle ILU cycle."""
    def __init__(self, p, legacy, first, second, terminal_limit):
        import scipy.linalg as la
        self.p, self.first, self.second = p, first, second
        assert int(second.owners[-1]) <= terminal_limit
        lo, hi = map(int, first.owners[p.rank:p.rank+2])
        self.middle = legacy.LocalPC(p, first.coarse, None, lo, hi, "bjacobi")
        n = int(second.owners[-1])
        dense = np.zeros((n, n), dtype=np.float64)
        start, stop = map(int, second.owners[p.rank:p.rank+2])
        for row in range(start, stop):
            count, columns, values = p.I(), p.P(), p.P()
            p.call("MatGetRow", second.coarse, row, C.byref(count), C.byref(columns), C.byref(values))
            if count.value:
                ix = np.ctypeslib.as_array(C.cast(columns, C.POINTER(p.I)), shape=(count.value,))
                v = np.ctypeslib.as_array(C.cast(values, C.POINTER(p.D)), shape=(count.value,))
                dense[row, ix] = v
            p.call("MatRestoreRow", second.coarse, row, C.byref(count), C.byref(columns), C.byref(values))
        self.energy = p.reduce(dense)
        self.condition = float(np.linalg.cond(self.energy))
        if not np.isfinite(self.condition) or self.condition >= 1e12:
            self.middle.close()
            raise ValueError("terminal_condition_limit")
        self.factor = la.lu_factor(self.energy)
        self.solve = la.lu_solve

    def gain(self, residual):
        a, b = self.first, self.second
        rhs = a.P(residual, transpose=True)
        answer = self.middle.apply(rhs)
        small = b.P(rhs - a.K(answer), transpose=True)
        global_rhs = self.p.all_rows(small, b.owners)
        correction = self.solve(self.factor, global_rhs)
        lo, hi = map(int, b.owners[self.p.rank:self.p.rank+2])
        answer += b.P(correction[lo:hi])
        answer += self.middle.apply(rhs - a.K(answer))
        return a.P(answer)

    def parity(self, fine):
        errors = []
        for frequency in (.137, .379):
            for level, parent in ((self.first, fine), (self.second, self.first.K)):
                lo, hi = level.owners[self.p.rank:self.p.rank+2]
                v = np.sin(frequency * (np.arange(lo, hi) + 1))
                truth = level.P(parent(level.P(v)), transpose=True)
                actual = level.K(v)
                pair = self.p.reduce([np.sum((truth-actual)**2), truth@truth])
                errors.append(float(np.sqrt(pair[0]/max(pair[1], 1e-300))))
        return errors

    def close(self):
        self.middle.close()


def load_bridge(p, path):
    bridge = C.CDLL(str(path))
    bridge.gnn_multipass.argtypes = [p.P, p.P, p.P, p.D, p.D, p.I, p.P, p.I, p.P,
                                    C.POINTER(p.P), C.POINTER(p.P), p.P]
    bridge.gnn_multipass.restype = p.I
    return bridge


def local_csr(p, matrix):
    """Copy owned rows only; global column numbers preserve MPI-boundary edges."""
    from scipy.sparse import csr_matrix
    owners = ownership(p, matrix)
    lo, hi = map(int, owners[p.rank:p.rank+2])
    n, width = p.I(), p.I()
    p.call("MatGetSize", matrix, C.byref(n), C.byref(width))
    indptr, indices, data = [0], [], []
    for row in range(lo, hi):
        count, columns, values = p.I(), p.P(), p.P()
        p.call("MatGetRow", matrix, row, C.byref(count), C.byref(columns), C.byref(values))
        if count.value:
            indices.append(np.ctypeslib.as_array(C.cast(columns, C.POINTER(p.I)), shape=(count.value,)).copy())
            data.append(np.ctypeslib.as_array(C.cast(values, C.POINTER(p.D)), shape=(count.value,)).copy())
        indptr.append(indptr[-1] + count.value)
        p.call("MatRestoreRow", matrix, row, C.byref(count), C.byref(columns), C.byref(values))
    return csr_matrix((np.concatenate(data) if data else np.empty(0),
                       np.concatenate(indices) if indices else np.empty(0, np.int32),
                       np.asarray(indptr, dtype=np.int32)), shape=(hi-lo, width.value)), owners


def graph_features(p, bridge, matrix, components, attributes, constrained, stage, support, strength=0.25):
    """Observable physical and numerical inputs; no teacher information."""
    graph = p.P()
    functions = np.ascontiguousarray(components, dtype=np.int32)
    algebra = np.empty((len(functions), 8), dtype=np.float64)
    bridge.gnn_graph.argtypes = [p.P, p.P, p.D, C.POINTER(p.P), p.P]
    bridge.gnn_graph.restype = p.I
    code = bridge.gnn_graph(matrix, functions.ctypes.data, strength, C.byref(graph), algebra.ctypes.data)
    if code:
        raise RuntimeError("Graph export error %d" % code)
    adjacency, owners = local_csr(p, graph)
    p.destroy("Mat", graph)
    features = np.column_stack([attributes, algebra, np.eye(3)[components],
        constrained.astype(float), np.full(len(functions), stage), np.log1p(support)]).astype(np.float32)
    assert np.isfinite(features).all() and np.isfinite(adjacency.data).all()
    sums = np.asarray(adjacency.sum(axis=1)).ravel()
    assert np.allclose(sums[np.diff(adjacency.indptr) > 0], 1)
    assert np.all(adjacency.data >= 0)
    return features, adjacency, owners


def pooled_attributes(p, level, attributes):
    """Fixed |P| support means of fine attributes and support mass.

    The caller includes phi and phi squared, so their means also encode the
    support variance. These statistics are not a coarse physical-field solve.
    """
    transfer, _ = local_csr(p, level.transfer)
    transfer.data = np.abs(transfer.data)
    mass = p.reduce(np.asarray(transfer.sum(axis=0)).ravel())
    weighted = p.reduce(transfer.T @ attributes)
    assert np.all(mass > 0)
    means = weighted / mass[:, None]
    lo, hi = map(int, level.owners[p.rank:p.rank+2])
    return means[lo:hi].copy(), mass[lo:hi].copy()

def solve_with_deadline(p, replay, outer, matrix, rhs, solution, cap, true_limit, deadline):
    """Same full-system true-residual stop, with a collective wall-clock limit."""
    build, work = p.vector(matrix), p.vector(matrix)
    history, errors = [], []
    def converge(ksp, iteration, internal, reason, context):
        try:
            current = p.P()
            p.call("KSPBuildSolution", ksp, build, C.byref(current))
            p.call("MatMult", matrix, current, work)
            p.call("VecAYPX", work, -1., rhs)
            residual = p.norm(work)
            overdue = p.maximum(float(time.perf_counter() > deadline)) > 0
            history.append(dict(iteration=int(iteration),
                internal_norm=float(internal) if np.isfinite(internal) else None,
                true_residual=residual if np.isfinite(residual) else None))
            reason[0] = -5 if overdue else (2 if residual <= true_limit else (-9 if not np.isfinite(residual) else (-3 if iteration >= cap else 0)))
            return 0
        except BaseException as error:
            errors.append(str(error))
            return 76
    callback = C.CFUNCTYPE(p.I, p.P, p.I, p.D, C.POINTER(p.I), p.P)(converge)
    p.call("KSPSetConvergenceTest", outer, C.cast(callback, p.P), None, None)
    p.barrier()
    begin = time.perf_counter()
    p.call("KSPSolve", outer, rhs, solution)
    end = time.perf_counter()
    assert not errors, errors
    iteration, reason = p.I(), p.I()
    p.call("KSPGetIterationNumber", outer, C.byref(iteration))
    p.call("KSPGetConvergedReason", outer, C.byref(reason))
    p.call("MatMult", matrix, solution, work)
    p.call("VecAYPX", work, -1., rhs)
    residual = p.norm(work)
    expired = p.maximum(float(end > deadline)) > 0
    p.destroy("Vec", build, work)
    return dict(iterations=iteration.value, reason=reason.value, cap=cap, true_limit=true_limit,
        final_true_residual=residual if np.isfinite(residual) else None,
        reached_true_target=bool(np.isfinite(residual) and residual <= true_limit),
        timed_out=expired, solve_s=p.maximum(end-begin), _solve_end_time=end,
        residual_curve=history, diagnostic_trace=True,
        stopping="Full true residual every iteration plus collective elapsed-time limit")

