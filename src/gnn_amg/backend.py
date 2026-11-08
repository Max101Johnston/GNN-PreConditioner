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
    with (Path(config["output_directory"]) / "records/progress.jsonl").open("a") as stream:
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
        functions = np.ascontiguousarray(components, dtype=np.int32)
        assert np.isin(functions, [0, 1, 2]).all()
        self.transfer, self.coarse = p.P(), p.P()
        stats = np.zeros(5, np.int64)
        code = bridge.gnn_multipass(matrix, cf.ctypes.data, functions.ctypes.data,
            setting["strength_threshold"], setting["truncation"], setting["max_entries"],
            C.byref(self.transfer), C.byref(self.coarse), stats.ctypes.data)
        if code:
            raise RuntimeError("multipass native error %d" % code)
        self.stats = dict(zip(["fine_dofs", "coarse_dofs", "uncovered_free_rows", "empty_free_rows", "nnz"], map(int, stats)))
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
        assert np.isfinite(self.condition) and self.condition < 1e12
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
    bridge.gnn_multipass.argtypes = [p.P, p.P, p.P, p.D, p.D, p.I,
                                    C.POINTER(p.P), C.POINTER(p.P), p.P]
    bridge.gnn_multipass.restype = p.I
    return bridge
