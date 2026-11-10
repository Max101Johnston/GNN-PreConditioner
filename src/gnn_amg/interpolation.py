"""Fixed-endpoint interpolation experiments on the actual Galerkin matrices."""
import ctypes as C

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from .backend import Action, Level, bind, graph_features, local_csr, ownership
from .dataset import gather_array, gather_graph
from .inference import graph_decode


class GalerkinTransfer:
    """An owned transfer and its freshly computed Galerkin operator."""
    def __init__(self, p, parent, transfer, owners, stats, markers=None):
        self.p = p
        self.transfer = transfer
        self.owners = owners
        self.stats = stats
        self.executed_markers = markers
        self.coarse = p.P()
        bind(p, "MatPtAP", [p.P, p.P, p.I, p.D, C.POINTER(p.P)])
        p.call("MatPtAP", parent, transfer, 0, -2.0, C.byref(self.coarse))
        self.P, self.K = Action(p, transfer), Action(p, self.coarse)

    def close(self):
        self.P.close()
        self.K.close()
        self.p.destroy("Mat", self.transfer, self.coarse)


def imported_transfer(p, legacy, parent, csr, rows, columns, stats, markers=None):
    # DistributedSparse creates a PETSc matrix independently of the input CSR.
    # Its two temporary work vectors are unnecessary in GalerkinTransfer.
    imported = legacy.DistributedSparse(p, csr, rows, columns)
    p.destroy("Vec", imported.right, imported.left)
    return GalerkinTransfer(p, parent, imported.matrix, columns, stats, markers)


def coarse_graph(config, p, bridge, first):
    n = len(first.native)
    _, graph, owners = graph_features(p, bridge, first.coarse, first.components,
        np.empty((n, 0)), np.zeros(n, bool), 1, np.ones(n),
        config["interpolation"]["strength_threshold"])
    arrays = gather_graph(p, graph, owners)
    return graph, owners, arrays


def coverage_audit(config, p, bridge, first, selected, old_p2, broadcast_json):
    graph, owners, arrays = coarse_graph(config, p, bridge, first)
    mask = gather_array(p, selected, owners)
    native = gather_array(p, first.native, owners)
    old_counts = gather_array(p, np.diff(old_p2.indptr), owners)
    result = None
    if p.rank == 0:
        n = len(mask)
        strength = sparse.csr_matrix((arrays["weights"], arrays["indices"], arrays["indptr"]),
                                     shape=(n, n))
        distance = dijkstra(strength.T.tocsr(), directed=True, indices=np.flatnonzero(mask),
                            unweighted=True, min_only=True)
        active = ~mask & (np.diff(strength.indptr) > 0)
        lost = active & ~np.isfinite(distance)
        result = dict(uncovered_9=int(np.count_nonzero(active & (distance > 9))),
                      uncovered_99=int(np.count_nonzero(active & (distance > 99))),
                      unreachable=int(lost.sum()),
                      reachable_beyond_9=int(np.count_nonzero(active & np.isfinite(distance) & (distance > 9))),
                      unreachable_with_cached_support=int(np.count_nonzero(lost & (old_counts > 0))),
                      cached_support_nnz_for_unreachable=int(old_counts[lost].sum()),
                      unreachable_native_ids=native[lost].tolist())
    return broadcast_json(p, result)


def staged_level(config, p, bridge, first, selected, broadcast_selection):
    """Build shorter Q and R transfers, then apply only their product."""
    graph, owners, arrays = coarse_graph(config, p, bridge, first)
    native = gather_array(p, first.native, owners)
    terminal = gather_array(p, selected, owners)
    intermediate = None
    radius = config["p2_intermediate_radius"]
    if p.rank == 0:
        n = len(native)
        prior = np.log1p(np.bincount(arrays["indices"], weights=arrays["weights"], minlength=n))
        prior[terminal] = prior.max()+1
        intermediate, _ = graph_decode(prior, native, np.ones(n, bool), arrays["indptr"],
                                       arrays["indices"], radius)
        intermediate |= terminal
    intermediate = broadcast_selection(p, intermediate, int(owners[-1]))
    lo, hi = owners[p.rank:p.rank+2]
    settings = dict(config["interpolation"], coverage_repair_fraction=0.0)
    q = Level(p, bridge, first.coarse, first.native, first.components,
              intermediate[lo:hi], np.zeros(len(first.native), bool), settings)
    stages = dict(intermediate_radius=radius, Q=q.stats)
    r = None
    try:
        if any(q.stats[k] for k in ("uncovered_free_rows", "empty_free_rows", "injection_error")):
            return None, stages, "staged_Q_coverage_or_injection"
        terminal_native = p.all_rows(first.native, first.owners)[
            broadcast_selection(p, terminal, int(owners[-1]))].astype(np.int64)
        r = Level(p, bridge, q.coarse, q.native, q.components, np.isin(q.native, terminal_native),
                  np.zeros(len(q.native), bool),
                  dict(settings, max_coarse_dofs=config["terminal_max_dofs"]))
        stages["R"] = r.stats
        if any(r.stats[k] for k in ("uncovered_free_rows", "empty_free_rows", "injection_error")):
            return None, stages, "staged_R_coverage_or_injection"
        transfer = p.P()
        bind(p, "MatMatMult", [p.P, p.P, p.I, p.D, C.POINTER(p.P)])
        p.call("MatMatMult", q.transfer, r.transfer, 0, -2.0, C.byref(transfer))
        csr, _ = local_csr(p, transfer, bridge)
        csr.eliminate_zeros()
        degree = np.diff(csr.indptr)
        missing = (degree == 0) & ((np.diff(graph.indptr) > 0) | selected)
        # C2 columns in the product inherit the intermediate ownership of the
        # same native rows. Check injection by global ID, not by local offset.
        end_ids = p.all_rows(r.native, r.owners).astype(np.int64)
        positions = np.flatnonzero(selected)
        injection = np.all(degree[positions] == 1)
        if injection:
            entries = csr.indptr[positions]
            injection = (np.array_equal(end_ids[csr.indices[entries]], first.native[positions])
                         and np.allclose(csr.data[entries], 1., atol=1e-14, rtol=0))
        invalid = p.reduce([missing.sum(), csr.nnz, int(not injection or not np.isfinite(csr.data).all())])
        nnz, failures = int(invalid[1]), int(invalid[2])
        stats = dict(coarse_dofs=int(r.owners[-1]), nnz=nnz,
                     uncovered_free_rows=int(invalid[0]), empty_free_rows=int(invalid[0]),
                     injection_error=float(failures > 0), repair_budget_exhausted=0,
                     repair_added_coarse=0, staged=stages)
        if invalid[0] or failures or nnz > config["p2_max_average_entries"]*int(owners[-1]):
            p.destroy("Mat", transfer)
            return None, stats, "staged_product_coverage_injection_or_density"
        markers = np.where(selected, 1, np.where(degree == 0, -3, -1)).astype(np.int8)
        return GalerkinTransfer(p, first.coarse, transfer, r.owners, stats, markers), stats, None
    finally:
        if r is not None:
            r.close()
        q.close()


def sparse_harmonic(matrix, selected, entries):
    """Offline target: bounded support and a row-scaled nonsymmetric residual."""
    from scipy.sparse.linalg import splu

    fine = np.flatnonzero(~selected)
    coarse = np.flatnonzero(selected)
    if not (0 < entries <= len(coarse) and len(fine)):
        raise ValueError("Invalid harmonic support budget")
    a = matrix[fine][:, fine].tocsr()
    b = matrix[fine][:, coarse].toarray()
    row_max = np.maximum(np.asarray(abs(a).max(axis=1).toarray()).ravel(),
                         np.max(np.abs(b), axis=1))
    if not np.all(np.isfinite(row_max) & (row_max > 0)):
        raise ValueError("Invalid fine rows for harmonic interpolation")
    scale = 1.0/row_max
    a = (sparse.diags(scale) @ a).tocsr()
    b *= scale[:, None]
    ideal = splu(a.tocsc()).solve(-b)
    if not np.isfinite(ideal).all():
        raise ValueError("Non-finite ideal interpolation")
    dense_residual = float(np.linalg.norm(a @ ideal + b))
    b_norm = max(float(np.linalg.norm(b)), 1e-30)
    if dense_residual > 1e-7*b_norm:
        raise ValueError("Inaccurate sparse factorization of K_FF")

    # Choose a fixed support from the ideal coefficients, then reduce the
    # residual on that support. No SPD or row-sum assumption is required.
    keep = np.argpartition(np.abs(ideal), -entries, axis=1)[:, -entries:]
    support = np.zeros(ideal.shape, dtype=bool)
    support[np.arange(len(fine))[:, None], keep] = True
    weights = np.where(support, ideal, 0.0)
    residual = a @ weights + b
    initial = float(np.linalg.norm(residual))/b_norm
    for _ in range(4):
        gradient = np.where(support, a.T @ residual, 0.0)
        direction = a @ gradient
        square = np.sum(direction*direction, axis=0)
        steps = np.divide(np.sum(gradient*gradient, axis=0), square,
                          out=np.zeros(len(coarse)), where=square > 0)
        weights -= gradient*steps
        residual -= direction*steps
    final = float(np.linalg.norm(a @ weights + b))/b_norm
    if not np.isfinite(final) or final > initial + 1e-12:
        raise ValueError("Restricted residual minimization failed")
    i, j = np.nonzero(weights)
    transfer = sparse.csr_matrix((np.r_[weights[i, j], np.ones(len(coarse))],
        (np.r_[fine[i], coarse], np.r_[j, np.arange(len(coarse))])),
        shape=(matrix.shape[0], len(coarse)))
    return transfer, dict(ideal_relative_residual=dense_residual/b_norm,
                          truncated_relative_residual=initial, refined_relative_residual=final,
                          maximum_entries_per_fine_row=entries,
                          negative_entries=int(np.count_nonzero(weights < 0)))


def collect_csr(p, bridge, matrix):
    """Gather an actual small PETSc matrix without narrowing its values."""
    local, rows = local_csr(p, matrix, bridge)
    counts = p.gather_equal([local.nnz]).ravel().astype(np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    lengths = gather_array(p, np.diff(local.indptr), rows)
    columns = gather_array(p, local.indices, offsets)
    values = gather_array(p, local.data, offsets)
    return rows, np.r_[0, np.cumsum(lengths)], columns, values


def harmonic_level(config, p, legacy, bridge, first, selected, entries, broadcast_json):
    rows, indptr, columns, values = collect_csr(p, bridge, first.coarse)
    terminal = gather_array(p, selected, rows)
    selected_counts = p.gather_equal([int(selected.sum())]).ravel().astype(np.int64)
    owners = np.r_[0, np.cumsum(selected_counts)]
    n, nc = int(rows[-1]), int(owners[-1])
    if not 0 < nc <= config["terminal_max_dofs"]:
        return None, dict(coarse_dofs=nc), "terminal_size_limit"
    transfer, meta = None, None
    if p.rank == 0:
        try:
            matrix = sparse.csr_matrix((values, columns, indptr), shape=(n, n))
            transfer, meta = sparse_harmonic(matrix, terminal, min(entries, nc))
            empty = int(np.count_nonzero(np.diff(transfer.indptr) == 0))
            meta.update(coarse_dofs=nc, nnz=int(transfer.nnz), empty_free_rows=empty,
                        uncovered_free_rows=empty, injection_error=0, repair_budget_exhausted=0,
                        repair_added_coarse=0)
        except (RuntimeError, ValueError) as error:
            meta = dict(error=str(error), coarse_dofs=nc)
    meta = broadcast_json(p, meta)
    if "error" in meta or meta["empty_free_rows"]:
        return None, meta, "harmonic_factorization_or_coverage"

    level = distribute_p2(p, legacy, first.coarse, transfer, rows, owners, meta)
    level.executed_markers = np.where(selected, 1, -1).astype(np.int8)
    return level, meta, None


def distribute_p2(p, legacy, matrix, transfer, rows, owners, meta):
    n, nc = int(rows[-1]), int(owners[-1])
    # The coarse matrix is small and sparse. Broadcast typed CSR buffers.
    mpi_byte = p.P(C.addressof((C.c_char*1).in_dll(p.mpi, "ompi_mpi_byte")))
    buffers = []
    for name, size, dtype in (("indptr", n+1, np.int32), ("indices", meta["nnz"], np.int32),
                              ("data", meta["nnz"], np.float64)):
        buf = (np.ascontiguousarray(getattr(transfer, name), dtype=dtype) if p.rank == 0
               else np.empty(size, dtype=dtype))
        assert p.mpi.MPI_Bcast(buf.ctypes.data, buf.nbytes, mpi_byte, 0, p.world) == 0
        buffers.append(buf)
    transfer = sparse.csr_matrix((buffers[2], buffers[1], buffers[0]), shape=(n, nc))
    lo, hi = rows[p.rank:p.rank+2]
    return imported_transfer(p, legacy, matrix, transfer[lo:hi], rows, owners, meta)


def enrich_first(p, legacy, bridge, matrix, first, sweeps, row_budget, native_scaling=False):
    """Expand support by actual stiffness coupling, with bounded signed weights."""
    if native_scaling:
        rows = ownership(p, matrix)
        sums = np.empty(rows[p.rank+1]-rows[p.rank], dtype=np.float64)
        code = bridge.gnn_abs_row_sums(matrix, sums.ctypes.data)
        if code:
            raise RuntimeError("Native absolute row sums failed: %d" % code)
    else:
        stiffness, rows = local_csr(p, matrix, bridge)
        sums = np.asarray(abs(stiffness).sum(axis=1)).ravel()
    scale = 1/np.maximum(sums, 1e-30)
    original, _ = local_csr(p, first.transfer, bridge)
    protected = first.executed_markers != -1
    scale[protected] = 0.0
    current = first.transfer
    temporary = None
    bind(p, "MatMatMult", [p.P, p.P, p.I, p.D, C.POINTER(p.P)])
    for sweep in range(sweeps):
        action = p.P()
        p.call("MatMatMult", matrix, current, 0, -2.0, C.byref(action))
        residual, _ = local_csr(p, action, bridge)
        p.destroy("Mat", action)
        transfer, _ = local_csr(p, current, bridge)
        if temporary is not None:
            temporary.close()
            temporary = None
        # Row equilibration gives a bounded local Jacobi update even when the
        # matrix is nonsymmetric. Keep injection and special-F rows exact.
        transfer = (transfer - residual.multiply(scale[:, None])).tocsr()
        degree = np.diff(transfer.indptr)
        for row in np.flatnonzero(degree > row_budget):
            lo, hi = transfer.indptr[row:row+2]
            small = np.argpartition(np.abs(transfer.data[lo:hi]), -row_budget)[:-row_budget]
            transfer.data[lo+small] = 0
        transfer.eliminate_zeros()
        transfer.sort_indices()
        if sweep+1 < sweeps:
            temporary = legacy.DistributedSparse(p, transfer, rows, first.owners)
            current = temporary.matrix
    difference = transfer[protected] - original[protected]
    degree = np.diff(transfer.indptr)
    invalid = p.reduce([difference.nnz, int(not np.isfinite(transfer.data).all()),
                        np.count_nonzero((degree == 0) & ~protected), transfer.nnz])
    if any(invalid[:3]):
        raise ValueError("P1 refinement failed protected rows, coverage or finite weights")
    stats = dict(first.stats, nnz=int(invalid[3]), refinement_sweeps=sweeps,
                 refinement_row_budget=row_budget, original_nnz=first.stats["nnz"])
    level = imported_transfer(p, legacy, matrix, transfer, rows, first.owners, stats)
    level.native, level.components = first.native, first.components
    level.executed_markers = first.executed_markers
    return level


def compare_transfers(p, reference, candidate):
    """Compare owned sparse rows outside the timed construction and solve."""
    result = {}
    for name in ("P1", "K1", "P2"):
        a, b = reference.get(name), candidate.get(name)
        if p.maximum(int(a is None or b is None or a.shape != b.shape)):
            result[name] = dict(comparable=False)
            continue
        pattern = np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
        diff = a-b
        diff2, ref2 = p.reduce([diff.data@diff.data, a.data@a.data])
        result[name] = dict(comparable=True, same_pattern=not bool(p.maximum(int(not pattern))),
            relative_frobenius=float(np.sqrt(diff2/max(ref2, 1e-300))))
    return result


def optimize_supported_weights(matrix, transfer, markers, expansion_budget=0):
    """Two signed least-squares steps with fixed injection and chosen support."""
    coarse = np.flatnonzero(markers == 1)
    fine = np.flatnonzero(markers != 1)
    a = matrix[fine][:, fine].tocsr()
    rhs = matrix[fine][:, coarse].tocsr()
    scale = 1/np.maximum(np.asarray(abs(matrix[fine]).sum(axis=1)).ravel(), 1e-30)
    a = a.multiply(scale[:, None]).tocsr()
    rhs = rhs.multiply(scale[:, None]).tocsr()
    weights = transfer[fine].copy()
    weights.eliminate_zeros()
    support = weights.astype(bool)
    residual = a @ weights + rhs
    reference = max(float(np.linalg.norm(rhs.data)), 1e-30)
    initial = float(np.linalg.norm(residual.data))/reference
    if expansion_budget:
        # Add the largest missing entries from one actual stiffness residual.
        # Starting weights and the subsequent optimizer are the same in both runs.
        missing = (residual - residual.multiply(support)).tocsr()
        missing.eliminate_zeros()
        new_rows, new_cols = [], []
        for row in np.flatnonzero(markers[fine] == -1):
            slots = expansion_budget-(support.indptr[row+1]-support.indptr[row])
            if slots <= 0:
                continue
            lo, hi = missing.indptr[row:row+2]
            choose = np.argsort(-np.abs(missing.data[lo:hi]), kind="stable")[:slots]
            new_rows.extend([row]*len(choose))
            new_cols.extend(missing.indices[lo+choose])
        extra = sparse.csr_matrix((np.ones(len(new_rows), bool), (new_rows, new_cols)), shape=weights.shape)
        support = support.maximum(extra)
    for _ in range(2):
        gradient = (a.T @ residual).multiply(support).tocsr()
        direction = a @ gradient
        numer = np.asarray(gradient.power(2).sum(axis=0)).ravel()
        denom = np.asarray(direction.power(2).sum(axis=0)).ravel()
        step = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
        weights = (weights - gradient.multiply(step)).tocsr()
        residual = a @ weights + rhs
    refined = float(np.linalg.norm(residual.data))/reference
    if not np.isfinite(refined) or refined > initial + 1e-12:
        raise ValueError("P2 weight optimization increased its residual")
    weights.eliminate_zeros()
    w = weights.tocoo()
    answer = sparse.csr_matrix((np.r_[w.data, np.ones(len(coarse))],
        (np.r_[fine[w.row], coarse], np.r_[w.col, np.arange(len(coarse))])),
        shape=transfer.shape)
    degree = np.diff(answer.indptr)
    if np.any((markers == -1) & (degree == 0)) or np.any((markers == -3) & (degree != 0)):
        raise ValueError("P2 weight optimization changed protected rows or coverage")
    if not np.isfinite(answer.data).all():
        raise ValueError("P2 weight optimization produced non-finite entries")
    return answer, dict(support_budget=expansion_budget, weight_sweeps=2,
                        original_nnz=int(transfer.nnz), candidate_nnz=int(support.nnz)+len(coarse),
                        maximum_fine_row_entries=int(degree[fine].max()),
                        initial_relative_residual=initial, refined_relative_residual=refined)


def refine_second(p, legacy, bridge, first, initial, expansion_budget, broadcast_json):
    rows, ki, kj, kv = collect_csr(p, bridge, first.coarse)
    _, pi, pj, pv = collect_csr(p, bridge, initial.transfer)
    markers = gather_array(p, initial.executed_markers, rows)
    transfer, meta = None, None
    if p.rank == 0:
        try:
            n, nc = int(rows[-1]), int(initial.owners[-1])
            k1 = sparse.csr_matrix((kv, kj, ki), shape=(n, n))
            p2 = sparse.csr_matrix((pv, pj, pi), shape=(n, nc))
            transfer, meta = optimize_supported_weights(k1, p2, markers, expansion_budget)
            meta = dict(initial.stats, **meta)
            meta["nnz"] = int(transfer.nnz)
        except ValueError as error:
            meta = dict(initial.stats, error=str(error))
    meta = broadcast_json(p, meta)
    if "error" in meta:
        return None, meta
    result = distribute_p2(p, legacy, first.coarse, transfer, rows, initial.owners, meta)
    result.executed_markers = initial.executed_markers
    return result, meta
