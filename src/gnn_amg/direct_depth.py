"""Replay fixed coarse-depth controls using the same exact terminal policy."""
import time

import numpy as np
from scipy import sparse
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import splu

from .backend import measured_action, ownership
from .factorized import normalized_coordinates
from .interpolation import collect_csr, harmonic_level, imported_transfer
from .krylov import middle_factor
from .online import DisplacementPC, broadcast_json, solve_policy
from .student_cycle import compact


class InvalidDepth(ValueError):
    """An individual coarse cycle failed a numerical check."""


def shard_csr(shard, name):
    offsets = shard[name + "_indptr"]
    return sparse.csr_matrix((shard[name + "_values"], shard[name + "_indices"], offsets),
                             shape=(len(offsets) - 1, int(shard[name + "_columns"].item())))


def third_endpoints(native, xyz, limit):
    """Deterministic component-balanced farthest points on the small K2 graph."""
    selected = np.zeros(len(native), dtype=bool)
    for component in range(3):
        indices = np.flatnonzero(native % 3 == component)
        indices = indices[np.argsort(native[indices])]
        count = min(len(indices), limit // 3 + int(component < limit % 3))
        if not count:
            continue
        local_xyz = xyz[indices]
        distance = np.sum((local_xyz - local_xyz.mean(axis=0))**2, axis=1)
        for _ in range(count):
            pick = int(np.argmax(distance))
            selected[indices[pick]] = True
            distance = np.minimum(distance, np.sum((local_xyz-local_xyz[pick])**2, axis=1))
            distance[selected[indices]] = -1
    return selected


def action_error(p, truth, actual):
    delta = truth - actual
    error, scale = p.reduce([delta @ delta, truth @ truth])
    return float(np.sqrt(error / max(scale, 1e-300)))


class TerminalLU:
    """One rank factors the actual sparse terminal matrix for every depth."""
    def __init__(self, p, bridge, matrix, limit):
        self.p = p
        self.lu = None
        # Check before gathering so a large terminal cannot concentrate K0.
        self.owners = ownership(p, matrix)
        self.size = int(self.owners[-1])
        if not 0 < self.size <= limit:
            raise InvalidDepth("terminal_sparse_size")
        _, indptr, columns, values = collect_csr(p, bridge, matrix)
        meta = None
        if p.rank == 0:
            try:
                terminal = sparse.csr_matrix((values, columns, indptr),
                                             shape=(self.size, self.size))
                self.lu = splu(terminal.tocsc())
                probe = np.sin(.31 * (np.arange(self.size) + 1))
                closure = float(np.linalg.norm(terminal @ self.lu.solve(probe)-probe)
                                / np.linalg.norm(probe))
                if not np.isfinite(closure) or closure > 1e-7:
                    raise InvalidDepth("terminal_sparse_lu_residual")
                meta = dict(dofs=self.size, matrix_nnz=terminal.nnz,
                            lu_nnz=self.lu.L.nnz + self.lu.U.nnz, probe_error=closure)
            except (RuntimeError, ValueError) as error:
                meta = dict(failure=str(error))
        self.stats = broadcast_json(p, meta)
        if "failure" in self.stats:
            raise InvalidDepth(self.stats["failure"])

    def __call__(self, rhs):
        p = self.p
        full = p.gather_rows(rhs, self.owners)
        result = self.lu.solve(full) if p.rank == 0 else np.zeros(self.size)
        lo, hi = self.owners[p.rank:p.rank+2]
        return p.reduce(result)[lo:hi]


def rank_sparse_smoother(p, bridge, level):
    # PETSc's one-block setup can have an empty sub-KSP on sparse coarse ranks.
    # Reuse the same unshifted ILU(0) algorithm, with tiny triangular solves.
    factor = middle_factor(p, bridge, level)
    if factor is None:
        raise InvalidDepth("coarse_empty_rank_ilu_pivot")
    lower, upper = factor.lower.toarray(), factor.upper.toarray()

    def apply(rhs):
        if not len(rhs):
            return rhs.copy()
        work = solve_triangular(lower, rhs, lower=True, unit_diagonal=True, check_finite=False)
        return solve_triangular(upper, work, check_finite=False)

    return apply


class DepthCycle:
    """The same pre/post block ILU at each nonterminal coarse level."""
    def __init__(self, p, legacy, bridge, levels, terminal_limit):
        self.levels = levels
        self.smoothers, self.resources, self.smoother_types = [], [], []
        begin = time.perf_counter()
        self.terminal = TerminalLU(p, bridge, levels[-1].coarse, terminal_limit)
        self.terminal_setup_s = p.maximum(time.perf_counter()-begin)
        try:
            for level in levels[:-1]:
                lo, hi = map(int, level.owners[p.rank:p.rank+2])
                if np.any(np.diff(level.owners) == 0):
                    self.smoothers.append(rank_sparse_smoother(p, bridge, level))
                    self.smoother_types.append("natural-block-ilu0-empty-rank-compatible")
                else:
                    smoother = legacy.LocalPC(p, level.coarse, None, lo, hi, "bjacobi")
                    self.resources.append(smoother)
                    self.smoothers.append(smoother.apply)
                    self.smoother_types.append("petsc-bjacobi-ilu0")
        except BaseException:
            for smoother in self.resources:
                smoother.close()
            raise

    def gain(self, residual, tape=None, profile=None):
        if tape is not None:
            raise ValueError("Depth control is used only for forward solves")

        def coarse_solve(index, rhs):
            if index == len(self.smoothers):
                terminal = self.terminal if profile is None else measured_action(
                    profile, "terminal_lu", self.terminal)
                return terminal(rhs)
            level = self.levels[index]
            next_transfer = self.levels[index+1].P
            smooth, multiply = self.smoothers[index], level.K
            if profile is not None:
                smooth = measured_action(profile, "middle_ilu", smooth)
                multiply = measured_action(profile, "middle_matvec", multiply)
                next_transfer = measured_action(profile, "coarse_transfer", next_transfer)
            pre = smooth(rhs)
            defect = next_transfer(rhs - multiply(pre), transpose=True)
            answer = pre + next_transfer(coarse_solve(index+1, defect))
            return answer + smooth(rhs - multiply(answer))

        transfer = self.levels[0].P
        if profile is not None:
            transfer = measured_action(profile, "p1_transfer", transfer)
        return transfer(coarse_solve(0, transfer(residual, transpose=True)))

    def close(self):
        for smoother in self.resources:
            smoother.close()
        for level in self.levels[2:]:
            level.close()


def parity(p, levels, fine, reference):
    """Check actual Galerkin actions and equality to archived prefix operators."""
    errors, archived = [], []
    parent = fine
    for index, level in enumerate(levels):
        columns = level.owners
        lo, hi = columns[p.rank:p.rank+2]
        for omega in (.137, .379):
            vector = np.sin(omega*(np.arange(columns[-1])+1))
            owned = vector[lo:hi]
            action = level.K(owned)
            errors.append(action_error(p, level.P(parent(level.P(owned)), transpose=True), action))
            if index < len(reference):
                archived.append(action_error(p, reference[index] @ vector, action))
        parent = level.K
    return errors, archived


def depth_control(config, p, replay, legacy, bridge, row, state, shard, receipt, depth):
    """Use saved P only in the experimental controls, rebuilding every K_i."""
    layout, fields, data = state
    settings = config["direct_final"]["depth"]
    prefixes = [shard_csr(shard, "p%d" % i) for i in range(1, min(depth, 2)+1)]
    references = [shard_csr(shard, "k%d" % i) for i in range(1, min(depth, 2)+1)]
    coordinates = normalized_coordinates(layout["xyz"])

    def prepare(fine, ras):
        levels = []
        parent = fine.matrix
        stats, timings, failure, space = {}, {}, None, None
        try:
            for index, transfer in enumerate(prefixes, 1):
                begin = time.perf_counter()
                parents, owners = shard["c%d_owners" % (index-1)], shard["c%d_owners" % index]
                native = shard["c%d_native" % index]
                report = receipt["levels"][index-1]
                metrics = dict(fine_dofs=report["fine_dofs"], coarse_dofs=report["coarse_dofs"],
                               nnz=report["p"]["nnz"], empty_free_rows=0, injection_error=0)
                level = imported_transfer(p, legacy, parent, transfer, parents, owners, metrics)
                level.native, level.components = native, (native % 3).astype(np.int32)
                levels.append(level)
                parent = level.coarse
                timings["level%d_setup_s" % index] = p.maximum(time.perf_counter()-begin)
                stats["level%d" % index] = metrics
            if depth == 3:
                begin = time.perf_counter()
                previous = levels[-1]
                native = p.all_rows(previous.native, previous.owners).astype(np.int64)
                selected = third_endpoints(native, coordinates[native//3], settings["third_dofs"])
                lo, hi = previous.owners[p.rank:p.rank+2]
                level, metrics, failure = harmonic_level(
                    config, p, legacy, bridge, previous, selected[lo:hi],
                    settings["third_entries"], broadcast_json)
                if failure is not None:
                    raise InvalidDepth(failure)
                level.native = previous.native[selected[lo:hi]]
                levels.append(level)
                stats["level3"] = dict(metrics, fine_dofs=int(previous.owners[-1]))
                timings["level3_setup_s"] = p.maximum(time.perf_counter()-begin)
            started = time.perf_counter()
            stats["galerkin_errors"], stats["prefix_archive_errors"] = parity(p, levels, fine, references)
            timings["galerkin_archive_check_s"] = p.maximum(time.perf_counter()-started)
            if max(stats["galerkin_errors"] + stats["prefix_archive_errors"]) > 1e-9:
                raise InvalidDepth("depth_galerkin_mismatch")
            stats["k1_nnz"] = receipt["levels"][0]["k"]["nnz"]
            space = DepthCycle(p, legacy, bridge, levels, settings["sparse_terminal_limit"])
            stats["terminal"] = dict(space.terminal.stats, smoother_types=space.smoother_types)
            timings["terminal_setup_s"] = space.terminal_setup_s
        except InvalidDepth as error:
            failure = str(error)
            for level in levels[2:]:
                level.close()
        return DisplacementPC(p, fine, ras, levels[0] if levels else None,
                              levels[1] if len(levels) > 1 else None, space, None, None,
                              None, None, stats, timings, failure)

    result = solve_policy(config, p, replay, legacy, bridge, None, row, layout,
                          {"constrained": fields["constrained"]}, data,
                          context=dict(displacement_factory=prepare, archive_choices=False,
                                       factor_profile_stride=settings["pc_profile_stride"]))
    return dict(compact(result, row, "teacher-prefix-%d" % depth), depth=depth,
                prefix_archive_errors=result["numerical"].get("prefix_archive_errors"),
                terminal=result["numerical"].get("terminal"),
                levels=[result["numerical"].get("level%d" % i) for i in range(1, depth+1)],
                setup_parts=result["timings"],
                timing_scope="Stored transfer import, actual Galerkin, smoothers, sparse terminal "
                             "and full solve. Offline teacher construction and file loading excluded.")
