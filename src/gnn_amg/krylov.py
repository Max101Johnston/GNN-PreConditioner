"""Matrix-free short right GMRES and its Arnoldi reverse pass."""

import ctypes as C

import numpy as np

from .backend import Action, bind


def short_gmres(apply, reduce, residual, steps, transpose=None, epsilon=1e-12,
                prefixes=()):
    """Differentiate the minimized residual through the executed Arnoldi steps.

    apply returns the right-preconditioned matrix action and an optional
    context for its transpose callback. The callback accumulates parameter
    derivatives and returns the derivative with respect to the action input.
    Only the small Hessenberg problem is dense.
    """
    if steps < 1:
        raise ValueError("A Krylov segment needs at least one step")
    beta = float(np.sqrt(reduce([residual @ residual])[0]))
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("The starting residual must have a finite positive norm")
    basis = np.empty((len(residual), steps+1), np.float64)
    basis[:, 0] = residual / beta
    hessenberg = np.zeros((steps+1, steps))
    tape = []
    for col in range(steps):
        v = basis[:, :col+1]
        work, context = apply(basis[:, col])
        h = reduce(v.T @ work)
        remainder = work - v @ h
        norm = float(np.sqrt(reduce([remainder @ remainder])[0]))
        if not np.isfinite(norm):
            raise ValueError("Non-finite Arnoldi norm")
        hessenberg[:col+1, col] = h
        hessenberg[col+1, col] = norm
        breakdown = norm <= 1e-14 * max(float(np.linalg.norm(h)), 1e-300)
        basis[:, col+1] = 0. if breakdown else remainder/norm
        if transpose is not None:
            tape.append((work, context))
        if breakdown:
            break
    executed = col+1
    rhs = np.zeros(executed+1)
    rhs[0] = beta
    h = hessenberg[:executed+1, :executed]
    y = np.linalg.lstsq(h, rhs, rcond=None)[0]
    difference = rhs - h @ y
    norm = float(np.linalg.norm(difference))
    ratio = norm/beta
    if transpose is not None:
        if breakdown or norm <= 1e-15*beta:
            raise ValueError("The short Krylov gradient requires a nondegenerate Arnoldi segment")
        # Envelope derivative at the least-squares optimum. Reverse the
        # normalization and classical Gram-Schmidt used in each column.
        grad_h = -np.outer(difference, y)/(norm*(epsilon*beta+norm))
        grad_v = np.zeros_like(basis)
        for col in range(executed-1, -1, -1):
            v, tail = basis[:, :col+1], basis[:, col+1]
            g = grad_v[:, col+1]
            component = reduce([tail @ g])[0]
            grad_remainder = (g - component*tail)/h[col+1, col] + grad_h[col+1, col]*tail
            adjoint_h = grad_h[:col+1, col] - reduce(v.T @ grad_remainder)
            work, context = tape[col]
            grad_v[:, :col+1] += np.outer(work, adjoint_h) - np.outer(grad_remainder, h[:col+1, col])
            grad_v[:, col] += transpose(grad_remainder + v @ adjoint_h, context)
    result = dict(steps=executed, residual_ratio=ratio, log_loss=float(np.log(epsilon+ratio)))
    if prefixes:
        result["prefix_ratios"] = {}
        for requested in prefixes:
            n = min(requested, executed)
            small = h[:n+1, :n]
            residual = rhs[:n+1] - small @ np.linalg.lstsq(small, rhs[:n+1], rcond=None)[0]
            result["prefix_ratios"][str(requested)] = float(np.linalg.norm(residual)/beta)
    return result


def probe_trajectories(p, matrix, pc, probes, windows):
    """Measure requested prefixes on identical full-system inputs and the live PC."""
    source, target = p.vector(matrix), p.vector(matrix)
    operator = Action(p, matrix)

    def apply(vector):
        p.assign(source, vector)
        p.call("PCApply", pc, source, target)
        return operator(p.array(target)), None

    try:
        return {name: short_gmres(apply, p.reduce, residual, max(windows[name]),
                                  prefixes=windows[name])
                for name, residual in probes
                if name in windows and p.reduce([residual @ residual])[0] > 0}
    finally:
        p.destroy("Vec", source, target)
        operator.close()


def true_petsc_ratio(p, matrix, pc, residual, steps):
    """Reference with the exact same FieldSplit and original matrix."""
    solver = p.P()
    rhs = p.vector(matrix, residual)
    solution = p.vector(matrix, np.zeros_like(residual))
    work = p.vector(matrix)
    for name, args in {
        "KSPSetType": [p.P, C.c_char_p],
        "KSPSetPCSide": [p.P, p.I],
        "KSPSetNormType": [p.P, p.I],
        "KSPSetTolerances": [p.P, p.D, p.D, p.D, p.I],
        "KSPGMRESSetRestart": [p.P, p.I],
    }.items():
        bind(p, name, args)
    try:
        p.call("KSPCreate", p.world, C.byref(solver))
        p.call("KSPSetType", solver, b"gmres")
        p.call("KSPSetOperators", solver, matrix, matrix)
        p.call("KSPSetPC", solver, pc)
        p.call("KSPSetPCSide", solver, 1)
        p.call("KSPSetNormType", solver, 2)
        p.call("KSPSetInitialGuessNonzero", solver, 0)
        p.call("KSPGMRESSetRestart", solver, max(steps+1, 20))
        p.call("KSPSetTolerances", solver, 0., 0., 1e20, steps)
        p.call("KSPSolve", solver, rhs, solution)
        p.call("MatMult", matrix, solution, work)
        p.call("VecAYPX", work, -1., rhs)
        return p.norm(work)/p.norm(rhs)
    finally:
        p.destroy("KSP", solver)
        p.destroy("Vec", rhs, solution, work)


def verify_short_segments(p, matrix, pc, probes, lengths):
    """Compare actual residual ratios on saved full-system inputs."""
    vin, vout = p.vector(matrix), p.vector(matrix)
    fine = Action(p, matrix)

    def apply(vector):
        p.assign(vin, vector)
        p.call("PCApply", pc, vin, vout)
        return fine(p.array(vout)), None

    records = []
    try:
        for identity, residual in probes:
            norm = float(np.sqrt(p.reduce([residual @ residual])[0]))
            if not np.isfinite(norm) or norm <= 0:
                continue
            normalized = residual / norm
            for steps in lengths:
                result = short_gmres(apply, p.reduce, normalized, steps)
                reference = true_petsc_ratio(p, matrix, pc, normalized, steps)
                error = abs(result["residual_ratio"]-reference)
                if not np.isfinite(reference) or error > 2e-9 + 1e-6*reference:
                    raise ValueError("Short GMRES differs from PETSc: %s %d %.8g %.8g" %
                        (identity, steps, result["residual_ratio"], reference))
                records.append(dict(result, probe=identity, requested_steps=steps,
                                    initial_residual_norm=norm, petsc_ratio=reference, parity_error=error))
        if not records:
            raise ValueError("No usable short Krylov probes")
        return records
    finally:
        fine.close()
        p.destroy("Vec", vin, vout)


class ILU0:
    """Natural-order sparse ILU(0) with a reverse pass on the fixed pattern."""

    def __init__(self, matrix):
        from scipy import sparse

        pattern = matrix.tocsr(copy=True)
        pattern.sum_duplicates()
        pattern.sort_indices()
        self.matrix = pattern
        n = pattern.shape[0]
        if pattern.shape[1] != n:
            raise ValueError("ILU(0) block must be square")
        rows = [dict(zip(pattern.indices[a:b], range(a, b)))
                for a, b in zip(pattern.indptr[:-1], pattern.indptr[1:])]
        if any(i not in row for i, row in enumerate(rows)):
            raise ValueError("ILU(0) requires all structural diagonal entries")
        diagonal = np.array([row[i] for i, row in enumerate(rows)], np.int64)
        self.eliminations = []
        factor = pattern.data.astype(np.float64, copy=True)
        for k, row in enumerate(rows):
            for i, at in row.items():
                if i >= k:
                    break
                pivot = diagonal[i]
                multiplier = factor[at]/factor[pivot]
                factor[at] = multiplier
                common = [j for j in rows[i] if j > i and j in row]
                dest = np.array([row[j] for j in common], np.int64)
                source = np.array([rows[i][j] for j in common], np.int64)
                factor[dest] -= multiplier*factor[source]
                self.eliminations.append((at, pivot, dest, source))
            if not np.isfinite(factor[diagonal[k]]) or factor[diagonal[k]] == 0:
                raise ValueError("Unshifted ILU(0) has an invalid pivot")
        self.factors = factor
        combined = sparse.csr_matrix((factor, pattern.indices, pattern.indptr), shape=pattern.shape)
        self.lower = sparse.tril(combined, -1, format="csr") + sparse.eye(n, format="csr")
        self.upper = sparse.triu(combined, format="csr")
        self.rows = np.repeat(np.arange(n), np.diff(pattern.indptr))

    def apply(self, rhs, transpose=False):
        from scipy.sparse.linalg import spsolve_triangular

        if not len(rhs):
            return rhs.copy()
        if transpose:
            work = spsolve_triangular(self.upper.T.tocsr(), rhs, lower=True)
            return spsolve_triangular(self.lower.T.tocsr(), work, lower=False, unit_diagonal=True)
        work = spsolve_triangular(self.lower, rhs, lower=True, unit_diagonal=True)
        return spsolve_triangular(self.upper, work, lower=False)

    def factors_adjoint(self, solution, adjoint):
        """Adjoint of the triangular factors, given Hx=b and H.T z=bar_x."""
        rows, cols = self.rows, self.matrix.indices
        return -np.where(cols < rows, adjoint[rows]*(self.upper @ solution)[cols],
                         (self.lower.T @ adjoint)[rows]*solution[cols])

    def reverse(self, factors_gradient):
        """Pull back a factor adjoint to the original sparse block entries."""
        gradient = factors_gradient.copy()
        for at, pivot, dest, source in reversed(self.eliminations):
            multiplier = self.factors[at]
            grad_multiplier = gradient[at] - gradient[dest] @ self.factors[source]
            gradient[source] -= multiplier*gradient[dest]
            gradient[at] = grad_multiplier/self.factors[pivot]
            gradient[pivot] -= gradient[at]*multiplier
        return gradient


def pc_transpose(p, smoother, vector):
    bind(p, "PCApplyTranspose", [p.P, p.P, p.P])
    vin, vout = smoother.vectors
    p.assign(vin, vector)
    p.call("PCApplyTranspose", smoother.pc, vin, vout)
    return p.array(vout)


def middle_factor(p, bridge, level):
    from .backend import local_csr

    csr, owners = local_csr(p, level.coarse, bridge)
    lo, hi = map(int, owners[p.rank:p.rank+2])
    try:
        factor = ILU0(csr[:, lo:hi])
    except ValueError:
        factor = None
    return None if p.maximum(factor is None) else factor


def verify_middle_ilu(p, bridge, level, middle, factor=None):
    """Measure parity of the differentiable factorization with the actual PC."""
    factor = middle_factor(p, bridge, level) if factor is None else factor
    if factor is None:
        return dict(verified=False, failure="invalid_unshifted_ilu_pivot")
    lo, hi = level.owners[p.rank:p.rank+2]
    vector = np.sin(.19*(np.arange(lo, hi)+1))
    forward = middle.apply(vector)
    backward = pc_transpose(p, middle, vector)
    errors = []
    for transpose, reference in ((False, forward), (True, backward)):
        difference = factor.apply(vector, transpose)-reference
        error, norm = p.reduce([difference @ difference, reference @ reference])
        errors.append(float(np.sqrt(error/max(norm, 1e-300))))
    return dict(verified=max(errors)<1e-9, forward_relative=errors[0], transpose_relative=errors[1])


class RASTranspose:
    """Reverse the restriction, local ILU solve, and owned-row injection."""

    def __init__(self, p, local, matrix):
        from .backend import ownership

        self.p = p
        count, kind = p.I(), p.I()
        exterior, interior = C.POINTER(p.P)(), C.POINTER(p.P)()
        p.call("PCASMGetLocalSubdomains", local.pc, C.byref(count), C.byref(exterior),
               C.byref(interior))
        p.call("PCASMGetType", local.pc, C.byref(kind))
        if p.maximum(count.value != 1 or kind.value != 1 or not exterior or not interior):
            raise ValueError("Expected one actual restricted ASM block per rank")
        outer = self.indices(exterior[0])
        inner = self.indices(interior[0])
        self.lo, self.hi = ownership(p, matrix)[p.rank:p.rank+2]
        self.owned = np.flatnonzero((outer >= self.lo) & (outer < self.hi))
        if p.maximum(not np.array_equal(np.sort(inner), np.arange(self.lo, self.hi))
                     or not np.array_equal(np.sort(outer[self.owned]), np.sort(inner))
                     or len(np.unique(outer)) != len(outer)):
            raise ValueError("RAS inner/outer indices differ from the process ownership")
        self.local_ids = outer[self.owned]-self.lo
        self.size = len(outer)
        first, solvers = p.I(), C.POINTER(p.P)()
        p.call("PCASMGetSubKSP", local.pc, C.byref(count), C.byref(first), C.byref(solvers))
        solver = solvers[0]
        kind, pc_type = C.c_char_p(), C.c_char_p()
        self.pc, submatrix, pre = p.P(), p.P(), p.P()
        p.call("KSPGetType", solver, C.byref(kind))
        p.call("KSPGetPC", solver, C.byref(self.pc))
        p.call("PCGetType", self.pc, C.byref(pc_type))
        p.call("KSPGetOperators", solver, C.byref(submatrix), C.byref(pre))
        if p.maximum(count.value != 1 or (kind.value, pc_type.value) != (b"preonly", b"ilu")):
            raise ValueError("Expected the actual one-block ILU RAS solver")

        for name, args in (
            ("VecScatterCreate", [p.P, p.P, p.P, p.P, C.POINTER(p.P)]),
            ("VecScatterBegin", [p.P, p.P, p.P, p.I, p.I]),
            ("VecScatterEnd", [p.P, p.P, p.P, p.I, p.I]),
            ("VecScatterDestroy", [C.POINTER(p.P)]),
            ("PCApplyTranspose", [p.P, p.P, p.P]),
        ):
            bind(p, name, args)
        self.source, self.solution = p.vector(submatrix), p.vector(submatrix)
        self.target = p.vector(matrix)
        self.scatter = p.P()
        try:
            p.call("VecScatterCreate", self.target, exterior[0], self.solution, None,
                   C.byref(self.scatter))
        except BaseException:
            self.close()
            raise

    def indices(self, handle):
        p = self.p
        n, ptr = p.I(), p.P()
        p.call("ISGetLocalSize", handle, C.byref(n))
        p.call("ISGetIndices", handle, C.byref(ptr))
        try:
            return np.ctypeslib.as_array(C.cast(ptr, C.POINTER(p.I)), shape=(n.value,)).copy()
        finally:
            p.call("ISRestoreIndices", handle, C.byref(ptr))

    def __call__(self, values):
        p = self.p
        rhs = np.zeros(self.size)
        rhs[self.owned] = values[self.local_ids]
        p.assign(self.source, rhs)
        p.call("PCApplyTranspose", self.pc, self.source, self.solution)
        p.call("VecSet", self.target, 0.)
        # Reverse the original exterior gather: all overlapping outputs
        # must be summed at their owners. The inner mask was applied to rhs.
        p.call("VecScatterBegin", self.scatter, self.solution, self.target, 2, 1)
        p.call("VecScatterEnd", self.scatter, self.solution, self.target, 2, 1)
        return p.array(self.target)

    def audit(self, local):
        index = np.arange(self.lo, self.hi)+1
        u, v = np.sin(.213*index), np.cos(.389*index)
        applied, transpose = local.apply(u), self(v)
        left, right, unorm, ynorm, vnorm, tnorm = self.p.reduce(
            [v @ applied, u @ transpose, u @ u, applied @ applied, v @ v, transpose @ transpose])
        return float(abs(left-right)/max(abs(left), abs(right), 1e-300,
                                        1e-6*np.sqrt(unorm*tnorm+vnorm*ynorm)))

    def close(self):
        p = self.p
        p.destroy("VecScatter", self.scatter)
        p.destroy("Vec", self.source, self.solution, self.target)


class TransferAdjoint:
    """Owned cotangents on a rectangular transfer's sparse support."""

    def __init__(self, p, pattern, owners):
        self.p, self.pattern, self.owners = p, pattern, owners
        self.rows = np.repeat(np.arange(pattern.shape[0]), np.diff(pattern.indptr))
        self.values = np.zeros(pattern.nnz)
        self.weight = 1.

    def outer(self, fine, coarse):
        coarse = self.p.all_rows(coarse, self.owners)
        self.values += self.weight * fine[self.rows] * coarse[self.pattern.indices]

    def add(self, matrix):
        if len(self.rows):
            self.values += np.asarray(matrix[self.rows, self.pattern.indices]).ravel()

    def csr(self):
        from scipy.sparse import csr_matrix

        return csr_matrix((self.values, self.pattern.indices, self.pattern.indptr),
                          shape=self.pattern.shape)


class HierarchyAdjoint:
    """Reverse the executed RAS and two Galerkin levels on fixed supports."""

    def __init__(self, p, bridge, displacement):
        self.p, self.bridge, self.displacement = p, bridge, displacement
        a, b = displacement.first, displacement.second
        self.first, self.second = a, b
        self.grad_first = TransferAdjoint(p, a.product_pattern, a.owners)
        self.grad_second = TransferAdjoint(p, b.product_pattern, b.owners)
        self.ilu = middle_factor(p, bridge, a)
        if self.ilu is None:
            raise ValueError("Invalid unshifted middle ILU pivot")
        self.ilu_parity = verify_middle_ilu(p, bridge, a, displacement.space.middle, self.ilu)
        if not self.ilu_parity["verified"]:
            raise ValueError("Middle ILU differs from the actual PETSc factors")
        self.ilu_gradient = np.zeros_like(self.ilu.factors)
        self.ras = RASTranspose(p, displacement.local, displacement.fine.matrix)
        self.ras_dot_error = self.ras.audit(displacement.local)
        if not np.isfinite(self.ras_dot_error) or self.ras_dot_error > 1e-9:
            self.ras.close()
            raise ValueError("Actual RAS transpose dot relative error: %g" % self.ras_dot_error)

    def close(self):
        self.ras.close()

    def middle_transpose(self, solution, adjoint):
        derivative = self.ilu.apply(adjoint, transpose=True)
        self.ilu_gradient += self.grad_first.weight * self.ilu.factors_adjoint(solution, derivative)
        return derivative

    def first_galerkin(self, left, right):
        fine, transfer = self.displacement.fine, self.first.P
        self.grad_first.outer(fine(transfer(right)), left)
        self.grad_first.outer(fine(transfer(left), transpose=True), right)

    def gain_transpose(self, gradient, tape):
        r, pre, defect, coarse, middle, post, answer = tape
        a, b = self.first, self.second
        self.grad_first.outer(gradient, answer)
        bar_middle = a.P(gradient, transpose=True)
        bar_rhs = self.middle_transpose(post, bar_middle)
        self.first_galerkin(-bar_rhs, middle)
        bar_middle -= a.K(bar_rhs, transpose=True)

        self.grad_second.outer(bar_middle, coarse)
        lo, hi = b.owners[self.p.rank:self.p.rank+2]
        space = self.displacement.space
        global_rhs = self.p.all_rows(b.P(bar_middle, transpose=True), b.owners)
        bar_coarse_rhs = space.solve(space.factor, global_rhs, trans=1)[lo:hi]
        bar_defect = b.P(bar_coarse_rhs)
        coarse_fine = b.P(coarse)
        self.grad_second.outer(a.K(coarse_fine), -bar_coarse_rhs)
        self.grad_second.outer(a.K(-bar_defect, transpose=True), coarse)
        self.first_galerkin(-bar_defect, coarse_fine)

        self.grad_second.outer(defect, bar_coarse_rhs)
        bar_rhs += bar_defect
        self.first_galerkin(-bar_defect, pre)
        bar_pre = bar_middle - a.K(bar_defect, transpose=True)
        bar_rhs += self.middle_transpose(pre, bar_pre)
        self.grad_first.outer(r, bar_rhs)
        return a.P(bar_rhs)

    def transpose(self, gradient, tape):
        fine = self.displacement.fine
        bar_r = self.ras(.5 * gradient)
        bar_middle = gradient - fine(bar_r, transpose=True)
        bar_defect = self.gain_transpose(bar_middle, tape)
        return bar_r + bar_defect + self.ras(bar_middle - fine(bar_defect, transpose=True))

    def transfers(self, legacy):
        from scipy.sparse import csr_matrix
        from .backend import local_csr
        from .factorized import import_matrix, multiply

        p, level = self.p, self.first
        lo, hi = level.owners[p.rank:p.rank+2]
        matrix = self.ilu.matrix
        block = csr_matrix((self.ilu.reverse(self.ilu_gradient), matrix.indices, matrix.indptr),
                           shape=matrix.shape)
        # The ILU numeric derivative is sparse in the process-diagonal K1
        # blocks. Pull it back by (K P) D^T + (K^T P) D.
        for transpose in (False, True):
            local = block if transpose else block.T.tocsr()
            distributed = csr_matrix((local.data, local.indices + lo, local.indptr),
                                     shape=(hi-lo, int(level.owners[-1])))
            d_k = product = pulled = None
            try:
                d_k = import_matrix(p, legacy, distributed, level.owners, level.owners)
                product = multiply(p, self.displacement.fine.matrix, level.transfer,
                                   left_transpose=transpose)
                pulled = multiply(p, product, d_k)
                self.grad_first.add(local_csr(p, pulled, self.bridge)[0])
            finally:
                for handle in (pulled, product, d_k):
                    if handle is not None:
                        p.destroy("Mat", handle)
        return self.grad_first.csr(), self.grad_second.csr()


class PhaseVCycle:
    """Transpose the actual fixed BoomerAMG hierarchy with its C/F sweeps."""

    def __init__(self, p, bridge, pc):
        from scipy.linalg import lu_factor
        from scipy.sparse import tril, triu
        from .backend import local_csr

        self.p = p
        self.actions, self.matrices, self.levels = [], [], []
        bridge.gnn_phase_levels.argtypes = [p.P, C.POINTER(p.I)]
        bridge.gnn_phase_level.argtypes = [p.P, p.I, C.POINTER(p.P), C.POINTER(p.P), C.POINTER(p.I)]
        bridge.gnn_phase_markers.argtypes = [p.P, p.I, p.I, p.P]
        for name in ("gnn_phase_levels", "gnn_phase_level", "gnn_phase_markers"):
            getattr(bridge, name).restype = p.I
        number = p.I()
        try:
            self.checked(bridge.gnn_phase_levels(pc, C.byref(number)), "cycle options")
            for level in range(number.value):
                a, transfer, count = p.P(), p.P(), p.I()
                code = bridge.gnn_phase_level(pc, level, C.byref(a), C.byref(transfer), C.byref(count))
                self.matrices.extend(handle for handle in (a, transfer) if handle.value)
                self.checked(code, "level matrices")
                csr, owners = local_csr(p, a, bridge)
                lo, hi = owners[p.rank:p.rank+2]
                if level == number.value-1:
                    if not 0 < owners[-1] <= 128:
                        raise ValueError("Phase direct coarse system exceeds audited size")
                    dense = np.zeros((owners[-1], owners[-1]))
                    dense[lo:hi] = csr.toarray()
                    self.factor = lu_factor(p.reduce(dense))
                    self.coarse_owners = owners
                    break
                if p.maximum(count.value != hi-lo):
                    raise ValueError("Phase C/F markers do not match the imported row ownership")
                markers = np.empty(count.value, np.int32)
                self.checked(bridge.gnn_phase_markers(pc, level, count, markers.ctypes.data),
                             "C/F markers")
                block = csr[:, lo:hi]
                diagonal = block.diagonal()
                smoothers = []
                for marker in (1, -1):
                    active = np.flatnonzero((markers == marker) & (diagonal != 0))
                    sub = block[active, :][:, active]
                    lower, upper = tril(sub, format="csr"), triu(sub, format="csr")
                    smoothers.append((active, diagonal[active], lower, upper,
                                      upper.T.tocsr(), lower.T.tocsr()))
                action, prolongation = Action(p, a), Action(p, transfer)
                self.actions.extend((action, prolongation))
                self.levels.append((action, prolongation, smoothers))
        except BaseException:
            self.close()
            raise

    def checked(self, code, label):
        maximum = int(self.p.maximum(code))
        if maximum:
            raise ValueError("Cannot copy executed phase BoomerAMG %s, PETSc code %d" % (label, maximum))

    @staticmethod
    def correction(residual, smoother, transpose):
        from scipy.sparse.linalg import spsolve_triangular

        active, diagonal, lower, upper, lower_t, upper_t = smoother
        result = np.zeros_like(residual)
        if len(active):
            left, right = (lower_t, upper_t) if transpose else (lower, upper)
            work = spsolve_triangular(left, residual[active], lower=True)
            result[active] = spsolve_triangular(right, diagonal*work, lower=False)
        return result

    def apply(self, rhs, transpose=False, level=0):
        from scipy.linalg import lu_solve

        if level == len(self.levels):
            p, owners = self.p, self.coarse_owners
            lo, hi = owners[p.rank:p.rank+2]
            return lu_solve(self.factor, p.all_rows(rhs, owners), trans=int(transpose))[lo:hi]
        a, prolongation, smoothers = self.levels[level]
        c, f = smoothers
        # Reverse the full correction sequence and transpose each factor.
        # The down/up C/F order is C,F then F,C for this fixed V-cycle.
        x = self.correction(rhs, c, transpose)
        x += self.correction(rhs - a(x, transpose), f, transpose)
        coarse_rhs = prolongation(rhs-a(x, transpose), transpose=True)
        x += prolongation(self.apply(coarse_rhs, transpose, level+1))
        x += self.correction(rhs - a(x, transpose), f, transpose)
        x += self.correction(rhs - a(x, transpose), c, transpose)
        return x

    def close(self):
        for action in reversed(self.actions):
            action.close()
        self.p.destroy("Mat", *self.matrices)


class FieldSplitAdjoint:
    """Full right-preconditioned action and PETSc's reverse split order."""

    def __init__(self, p, matrix, fs, shell, original_callback, adjoint):
        self.p, self.fs, self.shell = p, fs, shell
        self.original_callback = original_callback
        self.cycle = adjoint
        self.displacement = adjoint.displacement
        self.fine = Action(p, matrix)
        self.vectors = p.vector(matrix), p.vector(matrix)
        self.tape = None
        self.errors = []
        self.dot_errors = []
        self.callbacks = (self.callback(self.forward_child), self.callback(self.reverse_child))
        self.fixed = None
        self.phase_cycle = None
        for name, args in (
            ("PCShellSetApplyTranspose", [p.P, p.P]),
            ("PCApplyTranspose", [p.P, p.P, p.P]),
        ):
            bind(p, name, args)

    def callback(self, action):
        def apply(pc, source, target):
            try:
                action(source, target)
                return 0
            except Exception as error:
                self.errors.append(repr(error))
                return 76

        p = self.p
        return C.CFUNCTYPE(p.I, p.P, p.P, p.P)(apply)

    def forward_child(self, source, target):
        p = self.p
        tape = [] if self.tape is not None else None
        p.assign(target, self.displacement.correction(p.array(source), tape))
        if tape is not None:
            self.tape.append(tape)

    def reverse_child(self, source, target):
        p = self.p
        assert len(self.tape) == 1, "Expected one multiplicative displacement action"
        p.assign(target, self.cycle.transpose(p.array(source), self.tape.pop()))

    def fixed_phase(self):
        """Verify the copied phase V-cycle and use its exact transpose."""
        p = self.p
        n, ptr = p.I(), p.P()
        p.call("PCFieldSplitGetSubKSP", self.fs, C.byref(n), C.byref(ptr))
        try:
            assert n.value == 3
            child = p.P(C.cast(ptr, C.POINTER(p.P))[2])
        finally:
            p.call("PetscFreeA", 1, 0, b"fixed_phase", b"krylov.py", C.byref(ptr))
        original, operator, preconditioner = p.P(), p.P(), p.P()
        kind, family = C.c_char_p(), C.c_char_p()
        p.call("KSPGetType", child, C.byref(kind))
        p.call("KSPGetPC", child, C.byref(original))
        p.call("PCGetType", original, C.byref(family))
        if (kind.value, family.value) != (b"preonly", b"hypre"):
            raise ValueError("The fixed phase PC must use the recorded BoomerAMG preonly solver")
        p.call("KSPGetOperators", child, C.byref(operator), C.byref(preconditioner))
        from .backend import ownership

        lo, hi = ownership(p, operator)[p.rank:p.rank+2]
        indices = np.arange(lo, hi) + 1
        self.phase_cycle = PhaseVCycle(p, self.cycle.bridge, original)
        source, target = p.vector(operator), p.vector(operator)
        forward_errors, dot_errors = [], []
        try:
            for f in (.137, .291, .697):
                u, v = np.sin(f*indices), np.cos((f+.413)*indices)
                p.assign(source, u)
                p.call("PCApply", original, source, target)
                actual = p.array(target)
                reconstructed = self.phase_cycle.apply(u)
                transpose = self.phase_cycle.apply(v, transpose=True)
                diff = reconstructed-actual
                num, norm = p.reduce([diff @ diff, actual @ actual])
                forward_errors.append(float(np.sqrt(num/max(norm, 1e-300))))
                dot_errors.append(self.dot_error(u, actual, v, transpose))
        finally:
            p.destroy("Vec", source, target)
        errors = forward_errors + dot_errors
        if not np.isfinite(errors).all() or max(errors) > 1e-8:
            raise ValueError("Copied phase BoomerAMG cycle differs from the actual PC: " + str(errors))

        wrapper = p.P()
        p.call("PCCreate", p.world, C.byref(wrapper))
        p.call("PCSetOperators", wrapper, operator, preconditioner)
        p.call("PCSetType", wrapper, b"shell")
        callback = self.callback(lambda x, y: p.call("PCApply", original, x, y))
        p.call("PCShellSetApply", wrapper, C.cast(callback, p.P))
        reverse = self.callback(lambda x, y: p.assign(
            y, self.phase_cycle.apply(p.array(x), transpose=True)))
        p.call("PCShellSetApplyTranspose", wrapper, C.cast(reverse, p.P))
        p.call("PetscObjectReference", original)
        p.call("KSPSetPC", child, wrapper)
        self.fixed = child, original, wrapper, callback, reverse
        p.call("PCSetUp", wrapper)
        return dict(levels=len(self.phase_cycle.levels)+1, forward_relative=forward_errors,
                    transpose_dot_relative=dot_errors)

    def dot_error(self, source, result, gradient, transpose):
        left, right, u2, y2, v2, t2 = self.p.reduce(
            [gradient @ result, source @ transpose, source @ source,
             result @ result, gradient @ gradient, transpose @ transpose])
        return float(abs(left-right)/max(abs(left), abs(right), 1e-300,
                                        1e-6*np.sqrt(u2*t2+v2*y2)))

    def open(self):
        errors = self.fixed_phase()
        p = self.p
        p.call("PCShellSetApply", self.shell, C.cast(self.callbacks[0], p.P))
        p.call("PCShellSetApplyTranspose", self.shell, C.cast(self.callbacks[1], p.P))
        return errors

    def apply(self, vector, differentiate=True):
        p, (source, target) = self.p, self.vectors
        self.tape = [] if differentiate else None
        try:
            p.assign(source, vector)
            p.call("PCApply", self.fs, source, target)
            result = self.fine(p.array(target))
            if differentiate:
                assert len(self.tape) == 1
                return result, (self.tape, vector, result)
            return result, None
        finally:
            self.tape = None

    def transpose(self, gradient, tape):
        p, (source, target) = self.p, self.vectors
        self.tape, original, forward = tape
        try:
            p.assign(source, self.fine(gradient, transpose=True))
            p.call("PCApplyTranspose", self.fs, source, target)
            assert not self.tape
            transpose = p.array(target)
            self.dot_errors.append(self.dot_error(original, forward, gradient, transpose))
            return transpose
        finally:
            self.tape = None

    def close(self):
        p = self.p
        p.call("PCShellSetApply", self.shell, C.cast(self.original_callback, p.P))
        p.call("PCShellSetApplyTranspose", self.shell, None)
        if self.fixed is not None:
            child, original, wrapper, _, _ = self.fixed
            p.call("KSPSetPC", child, original)
            p.destroy("PC", wrapper, original)
        if self.phase_cycle is not None:
            self.phase_cycle.close()
        p.destroy("Vec", *self.vectors)
        self.fine.close()



def signed_head(model, vector):
    """Project a parameter vector onto the two coefficient-only output channels."""
    result = np.zeros_like(vector)
    offset = sum(w.size+b.size for w, b in model.layers[:4])
    weight, bias = model.layers[4]
    for begin, end in ((offset+weight.shape[1], offset+weight.size),
                       (offset+weight.size+1, offset+weight.size+bias.size)):
        result[begin:end] = vector[begin:end]
    return result


def coefficient_feedback(p, legacy, bridge, matrix, fs, shell, callback, displacement,
                         model, samples, settings):
    from dataclasses import replace
    from .backend import Hierarchy
    from .factorized import FactorModel, factor_cotangent, frozen_transfer

    # Every direction is normalized by short_gmres. Include mid/late actual
    # residuals and rotate the physical component across coefficient batches.
    actual = [pair for pair in samples if pair[0].startswith("iteration-")]
    local = [pair for pair in samples if pair[0].startswith("displacement-")]
    targets = settings.get("quality_targets")
    if targets is None:
        component = settings.get("component", 0)
        steps = settings["steps"]
        if len(steps) != 3 or min(steps) < 1:
            raise ValueError("Expected one positive length per mid/late/physical direction")
        actual = actual[-2:]
        requested = list(zip(actual, steps[2-len(actual):2]))
        requested.append((local[component % len(local)], steps[2]))
    else:
        requested = [(pair, int(targets[pair[0]]["steps"]))
                     for pair in samples if pair[0] in targets]
    inputs = [(label, x, n) for (label, x), n in requested if p.reduce([x @ x])[0] > 0]
    if not inputs:
        return dict(verified=False, failure="no_nonzero_training_residuals")
    operation = cycle = None
    try:
        cycle = HierarchyAdjoint(p, bridge, displacement)
        operation = FieldSplitAdjoint(p, matrix, fs, shell, callback, cycle)
        phase_errors = operation.open()

        def forward():
            return [short_gmres(lambda v: operation.apply(v, False), p.reduce, r, n)
                    for _, r, n in inputs]

        scores = forward()
        ratios = np.array([item["residual_ratio"] for item in scores])
        if targets is not None:
            a, b = displacement.first, displacement.second
            gradient = np.zeros_like(model.weights, dtype=np.float64)
            probes = {}
            for (name, residual, steps), score in zip(inputs, scores):
                target = targets[name]["target"]
                gap = max(0., score["log_loss"] - np.log(target + 1e-12))
                probe_gradient = np.zeros_like(gradient)
                if gap > 0:
                    cycle.grad_first.values.fill(0.)
                    cycle.grad_second.values.fill(0.)
                    cycle.ilu_gradient.fill(0.)
                    short_gmres(operation.apply, p.reduce, residual, steps, operation.transpose)
                    first, second = cycle.transfers(legacy)
                    probe_gradient = p.reduce(factor_cotangent(p, legacy, bridge, a, first, model)
                                            + factor_cotangent(p, legacy, bridge, b, second, model))
                    gradient += targets[name]["weight"] * probe_gradient
                signed = signed_head(model, probe_gradient)
                probes[name] = dict(score, target=target, active=bool(gap > 0), gap=gap,
                                    weight=targets[name]["weight"],
                                    gradient_norm=float(np.linalg.norm(probe_gradient)),
                                    signed_head_gradient_norm=float(np.linalg.norm(signed)))
            dot_error = max(operation.dot_errors, default=0.)
            if not np.isfinite(dot_error) or dot_error > settings.get("transpose_tolerance", 1e-7):
                return dict(verified=False, failure="full_fieldsplit_transpose", phase=phase_errors,
                            full_dot_errors=operation.dot_errors, ras_dot_error=cycle.ras_dot_error)
            if not np.isfinite(gradient).all():
                raise ValueError("Non-finite network gradient")
            return dict(verified=True, phase=phase_errors,
                        full_dot_errors=operation.dot_errors, ras_dot_error=cycle.ras_dot_error,
                        middle_ilu=cycle.ilu_parity, probes=probes,
                        objective=sum(q["weight"]*q["gap"] for q in probes.values()),
                        gradient_norm=float(np.linalg.norm(gradient)), gradient=gradient.tolist())

        weights = np.full(len(inputs), (1-settings["tail_weight"])/len(inputs))
        tail = int(np.argmax(ratios))
        weights[tail] += settings["tail_weight"]
        for (_, residual, n), weight in zip(inputs, weights):
            cycle.grad_first.weight = cycle.grad_second.weight = weight
            short_gmres(operation.apply, p.reduce, residual, n, operation.transpose)
        dot_error = max(operation.dot_errors)
        if not np.isfinite(dot_error) or dot_error > settings.get("transpose_tolerance", 1e-7):
            return dict(verified=False, failure="full_fieldsplit_transpose", phase=phase_errors,
                        full_dot_errors=operation.dot_errors, ras_dot_error=cycle.ras_dot_error)
        first, second = cycle.transfers(legacy)
        a, b = displacement.first, displacement.second
        first_gradient = p.reduce(factor_cotangent(p, legacy, bridge, a, first, model))
        second_gradient = p.reduce(factor_cotangent(p, legacy, bridge, b, second, model))
        gradient = first_gradient + second_gradient
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(norm) or norm <= 1e-14:
            raise ValueError("Non-finite or negligible network gradient")
        objective = float(weights @ [value["log_loss"] for value in scores])
        report = dict(verified=True, phase=phase_errors, middle_ilu=cycle.ilu_parity,
                      full_dot_errors=operation.dot_errors, ras_dot_error=cycle.ras_dot_error,
                      objective=objective, worst_residual_ratio=float(ratios[tail]),
                      probes=[dict(name=name, weight=float(weight), **value) for
                              (name, _, _), weight, value in zip(inputs, weights, scores)],
                      p1_gradient_norm=float(np.linalg.norm(first_gradient)),
                      p2_gradient_norm=float(np.linalg.norm(second_gradient)),
                      gradient_norm=norm, gradient=gradient.tolist())
        if settings.get("finite_difference"):
            direction = gradient/norm
            epsilon = settings["finite_difference"]
            results = []
            for offset in (-epsilon, 0., epsilon):
                shifted = FactorModel(model.width, (model.weights+offset*direction).astype(model.weights.dtype),
                                      model.adapters, model.unit_sum_depth)
                one = two = rebuilt = None
                try:
                    one = frozen_transfer(p, legacy, bridge, a, displacement.fine.matrix, shifted)
                    two = frozen_transfer(p, legacy, bridge, b, one.coarse, shifted)
                    rebuilt = Hierarchy(p, legacy, one, two, int(b.owners[-1]))
                    operation.displacement = replace(displacement, first=one, second=two, space=rebuilt)
                    results.append(float(weights @ [v["log_loss"] for v in forward()]))
                finally:
                    operation.displacement = displacement
                    for resource in (rebuilt, two, one):
                        if resource is not None:
                            resource.close()
            finite = (results[2]-results[0])/(2*epsilon)
            error = abs(finite-norm)/max(abs(finite), norm)
            center_error = abs(results[1]-objective)
            report["finite_difference"] = dict(epsilon=epsilon, analytic=norm, measured=finite,
                relative_error=error, center_error=center_error, scores=results)
            report["verified"] = bool(np.isfinite(error) and error <= settings["finite_tolerance"]
                                      and center_error <= settings["center_tolerance"])
            if not report["verified"]:
                report["failure"] = "fixed_support_directional_derivative"
        return report
    except (ValueError, RuntimeError) as error:
        return dict(verified=False, failure=str(error), callback_errors=operation.errors if operation else [])
    finally:
        try:
            if operation is not None:
                operation.close()
        finally:
            if cycle is not None:
                cycle.close()
