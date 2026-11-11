"""Fixed-support local coarse spaces for an offline representation check."""
from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse

from .backend import local_csr
from .direct_depth import InvalidDepth, TerminalLU
from .direct_final_teacher import sparse_cost
from .direct_solver import allocate
from .factorized import groups, normalized_coordinates
from .interpolation import imported_transfer


class InvalidModalBasis(ValueError):
    """A preregistered modal space violates its representation guard."""


@dataclass
class Patch:
    rows: np.ndarray
    weights: np.ndarray
    center: np.ndarray
    region: int
    basis: np.ndarray
    mandatory: int
    priorities: np.ndarray
    worst_error: np.ndarray
    excluded_directions: int


@dataclass
class ModalTransfer:
    csr: object
    rows: np.ndarray
    columns: np.ndarray
    coordinates: np.ndarray
    regions: np.ndarray
    ordinals: np.ndarray
    rigid_coefficients: np.ndarray
    stats: dict

    def import_level(self, p, legacy, parent):
        return imported_transfer(p, legacy, parent, self.csr, self.rows, self.columns, self.stats)


def rigid_fields(xyz, component, fixed, center):
    """Translations and rotations in physical coordinates, respecting constraints."""
    r = xyz - center
    rotations = np.cross(np.eye(3)[None, :, :], r[:, None, :])
    rotated = np.take_along_axis(rotations, component[:, None, None], axis=2).squeeze(2)
    values = np.column_stack((np.eye(3)[component], rotated))
    values[fixed] = 0.
    return values


def columns_with_norm(vectors, cutoff):
    norm = np.linalg.norm(vectors, axis=0)
    good = norm > cutoff
    return vectors[:, good] / norm[good]


def make_patch(rows, weights, center, region, rigid, directions, extra):
    """Preserve mandatory fields and greedily cover worst relative local errors."""
    required = columns_with_norm(rigid[rows], 1e-14)
    if required.shape[1]:
        q, r, _ = linalg.qr(required, mode="economic", pivoting=True, check_finite=False)
        diagonal = np.abs(np.diag(r))
        q = q[:, :int(np.count_nonzero(diagonal > 1e-11*diagonal[0]))]
    else:
        q = np.empty((len(rows), 0))
    mandatory = q.shape[1]
    # The inputs are globally normalized. The tiny cutoff discards numerical
    # zeros before local normalization; the priorities ignore absolute energy.
    target = columns_with_norm(directions[rows], 1e-10)
    residual = target - q @ (q.T @ target)
    errors = [float(np.linalg.norm(residual, axis=0).max(initial=0.))]
    priorities = []
    for _ in range(extra):
        if errors[-1] <= 1e-10 or q.shape[1] >= len(rows):
            break
        index = int(np.argmax(np.linalg.norm(residual, axis=0)))
        direction = residual[:, index]
        direction = direction - q @ (q.T @ direction)
        length = np.linalg.norm(direction)
        if length <= 1e-10:
            break
        direction /= length
        q = np.column_stack((q, direction))
        priorities.append(min(errors[-1], priorities[-1]) if priorities else errors[-1])
        residual = residual - direction[:, None]*(direction @ residual)[None, :]
        errors.append(float(np.linalg.norm(residual, axis=0).max(initial=0.)))
    return Patch(rows, weights, center, region, q, mandatory, np.asarray(priorities),
                 np.asarray(errors), directions.shape[1]-target.shape[1])


def patch_ranks(patches, budget, p=None):
    base = sum(patch.mandatory for patch in patches)
    scores = np.concatenate([patch.priorities for patch in patches])
    if p is None:
        start, stop = 0, len(scores)
    else:
        counts = p.gather_equal([len(scores), base]).astype(np.int64)
        owners = np.r_[0, np.cumsum(counts[:, 0])]
        start, stop = owners[p.rank:p.rank+2]
        scores = p.all_rows(scores, owners)
        base = int(counts[:, 1].sum())
    if base > budget:
        raise InvalidModalBasis("mandatory_rigid_rank_exceeds_budget")
    keep = np.zeros(len(scores), dtype=bool)
    keep[np.argsort(-scores, kind="stable")[:budget-base]] = True
    owned = keep[start:stop]
    pointer = 0
    ranks = []
    for patch in patches:
        count = len(patch.priorities)
        mask = owned[pointer:pointer+count]
        extra = int(mask.sum())
        if not np.all(mask[:extra]):
            raise InvalidModalBasis("modal_rank_budget_discarded_prefix")
        ranks.append(patch.mandatory+extra)
        pointer += count
    return ranks


def assemble(patches, ranks, rigid, shape, start):
    rows, cols, data, coordinates, regions, ordinals, coefficients = [], [], [], [], [], [], []
    for patch, rank in zip(patches, ranks):
        if rank == 0:
            continue
        q = patch.basis[:, :rank]
        rows.append(np.repeat(patch.rows, rank))
        cols.append(np.tile(np.arange(start, start+rank), len(patch.rows)))
        data.append((patch.weights[:, None]*q).ravel())
        coordinates.append(np.repeat(patch.center[None, :], rank, axis=0))
        regions.append(np.full(rank, patch.region, dtype=np.int64))
        ordinals.append(np.arange(rank, dtype=np.int32))
        coefficients.append(q.T @ rigid[patch.rows])
        start += rank
    if not rows:
        raise InvalidModalBasis("empty_rank_modal_basis")
    transfer = sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=shape)
    transfer.eliminate_zeros()
    return (transfer, np.concatenate(coordinates), np.concatenate(regions),
            np.concatenate(ordinals), np.concatenate(coefficients))


def local_patch_support(matrix, start, labels, fixed):
    """Add a node to its strongest other owned region, with overlap at most two."""
    row = np.repeat(np.arange(len(fixed)), np.diff(matrix.indptr))
    col = matrix.indices.astype(np.int64)-start
    mask = (col >= 0) & (col < len(fixed)) & ~fixed[row] & (matrix.data != 0.)
    row, col, values = row[mask], col[mask], np.abs(matrix.data[mask])
    a, b = labels[row//3], labels[col//3]
    valid = (a >= 0) & (b >= 0) & (a != b) & ~fixed[col]
    if not np.any(valid):
        return np.full(len(labels), -1, dtype=np.int32)
    contact = sparse.csr_matrix((values[valid], (row[valid]//3, b[valid])),
                                shape=(len(labels), int(labels.max()+1)))
    halo = np.full(len(labels), -1, dtype=np.int32)
    for node in np.flatnonzero(np.diff(contact.indptr)):
        begin, end = contact.indptr[node:node+2]
        halo[node] = contact.indices[begin+np.argmax(contact.data[begin:end])]
    return halo


def distributed_owners(p, count):
    return np.r_[0, np.cumsum(p.gather_equal([count]).ravel().astype(np.int64))]


def matrix_cost(p, csr, owners):
    labels = ("nnz", "off_rank_entries", "remote_columns_rank_sum")
    metrics = p.reduce(sparse_cost(csr, owners[p.rank:p.rank+2])).astype(int)
    return dict(zip(labels, metrics.tolist()))


def gram_condition(gram):
    spectrum = linalg.eigvalsh(gram, check_finite=False)
    if not len(spectrum) or spectrum[0] <= 0:
        return float("inf")
    return float(spectrum[-1]/spectrum[0])


def local_quality(p, patches, ranks):
    own = max((patch.worst_error[rank-patch.mandatory]
               for patch, rank in zip(patches, ranks)), default=0.)
    return p.maximum(own)


def build_first(p, bridge, fine, layout, fields, directions, settings):
    stiffness, owners = local_csr(p, fine.matrix, bridge)
    lo, hi = owners[p.rank:p.rank+2]
    np.testing.assert_array_equal(owners, layout["ownership"])
    comp = layout["component"][lo:hi].astype(int)
    nodes = layout["native_node"][lo:hi]
    if len(comp) % 3 or not np.all(comp.reshape((-1, 3))==[0, 1, 2]):
        raise InvalidModalBasis("fine_owned_rows_do_not_form_displacement_blocks")
    fixed = fields["constrained"][3*nodes+comp]
    xyz = layout["xyz"]
    normalized = normalized_coordinates(xyz)[nodes[::3]]
    active = (~fixed).reshape((-1, 3)).any(1)
    counts = p.gather_equal([int(active.sum())]).ravel().astype(np.int64)
    required = (counts > 0).astype(np.int64)
    quotas = required + allocate(counts, counts-required, settings["fine_regions"]-required.sum())
    local_count = int(quotas[p.rank])
    if p.maximum(local_count==0):
        raise InvalidModalBasis("no_active_fine_core_on_a_rank")
    labels = groups(3*nodes[::3], normalized, active, int(active.sum())/local_count)
    if p.maximum(labels.max(initial=-1)+1 != local_count):
        raise InvalidModalBasis("fine_core_count_did_not_match_fixed_budget")
    halo = local_patch_support(stiffness, lo, labels, fixed)
    weights = np.repeat(1./(1+(halo >= 0)), 3)
    offset = np.r_[0, np.cumsum(quotas)][p.rank]
    rigid = rigid_fields(xyz[nodes], comp, fixed, xyz.mean(axis=0))
    patches = []
    for group in range(local_count):
        included = np.repeat((labels == group) | (halo == group), 3) & ~fixed
        rows = np.flatnonzero(included)
        center = normalized[labels == group].mean(axis=0)
        patches.append(make_patch(rows, weights[rows], center, offset+group, rigid,
                                  directions, settings["max_extra"]))
    ranks = patch_ranks(patches, settings["max_c1"], p)
    cols = distributed_owners(p, sum(ranks))
    if np.any(np.diff(cols) == 0):
        raise InvalidModalBasis("empty_first_modal_ownership")
    transfer, coord, regions, ordinals, coarse_rigid = assemble(
        patches, ranks, rigid, (int(hi-lo), int(cols[-1])), int(cols[p.rank]))
    view = transfer[:, cols[p.rank]:cols[p.rank+1]]
    condition = p.maximum(gram_condition((view.T @ view).toarray()))
    errors = p.reduce([np.count_nonzero(np.diff(transfer.indptr)[~fixed] == 0),
                       np.count_nonzero(np.diff(transfer.indptr)[fixed]),
                       int(not np.isfinite(transfer.data).all())])
    if np.any(errors) or not np.isfinite(condition) or condition > settings["gram_limit"]:
        raise InvalidModalBasis("modal_first_rank_or_coverage")
    stats = dict(fine_dofs=int(owners[-1]), coarse_dofs=int(cols[-1]),
                 regions=int(quotas.sum()), max_local_worst_error=local_quality(p, patches, ranks),
                 max_gram_condition=condition, empty_free_rows=0, constrained_entries=0,
                 max_support_memberships=int(p.maximum(1+(halo >= 0).any())),
                 mandatory_modes=int(p.reduce([sum(q.mandatory for q in patches)])[0]),
                 excluded_local_directions=int(p.reduce([sum(q.excluded_directions for q in patches)])[0]),
                 **matrix_cost(p, transfer, cols))
    return ModalTransfer(transfer, owners, cols, coord, regions, ordinals, coarse_rigid, stats), rigid


def build_second(p, legacy, bridge, fine, first, metadata, directions, settings):
    """Use actual Galerkin solutions and ILU errors in the modal coordinates."""
    owners = first.owners
    start, stop = map(int, owners[p.rank:p.rank+2])
    terminal = TerminalLU(p, bridge, first.coarse, settings["max_c1"])
    ilu = legacy.LocalPC(p, first.coarse, None, start, stop, "bjacobi")
    try:
        # g=P1^T K_uu e, q=K1^{-1}g, and q-B1g are coarse error directions.
        coarse = []
        for column in directions.T:
            rhs = first.P(fine(column), transpose=True)
            q = terminal(rhs)
            coarse.append(q-ilu.apply(rhs))
    finally:
        ilu.close()
    coarse = np.column_stack(coarse)
    norms = np.sqrt(p.reduce(np.sum(coarse**2, axis=0)))
    keep = norms > 1e-30
    coarse = coarse[:, keep] / norms[keep]
    if not coarse.shape[1]:
        raise InvalidModalBasis("no_actual_coarse_errors")
    n = int(owners[-1])

    def all_matrix(values):
        width = values.shape[1]
        return p.all_rows(values.ravel(), owners*width).reshape((n, width))

    rigid = all_matrix(metadata.rigid_coefficients)
    directions = all_matrix(coarse)
    xyz = all_matrix(metadata.coordinates)
    region = p.all_rows(metadata.regions, owners).astype(np.int64)
    ids, first_row, patch_index = np.unique(region, return_index=True, return_inverse=True)
    count = settings["coarse_regions"]
    if len(ids) < count:
        raise InvalidModalBasis("too_few_first_regions")
    groups_of_patches = groups(3*ids, xyz[first_row], np.ones(len(ids), bool), len(ids)/count)
    if groups_of_patches.max()+1 != count:
        raise InvalidModalBasis("second_region_count_did_not_match_fixed_budget")
    labels = groups_of_patches[patch_index]
    patches = []
    for group in range(count):
        rows = np.flatnonzero(labels == group)
        patches.append(make_patch(rows, np.ones(len(rows)), xyz[rows].mean(axis=0),
                                  group, rigid, directions, settings["max_extra"]))
    ranks = patch_ranks(patches, settings["max_c2"])
    dim = sum(ranks)
    columns = (np.arange(p.size+1, dtype=np.int64)*dim)//p.size
    transfer, coord, region, ordinal, coefficients = assemble(patches, ranks, rigid, (n, dim), 0)
    condition = gram_condition((transfer.T @ transfer).toarray())
    if np.any(np.diff(transfer.indptr)==0) or not np.isfinite(transfer.data).all() \
            or not np.isfinite(condition) or condition > settings["gram_limit"]:
        raise InvalidModalBasis("modal_second_rank_or_coverage")
    local = transfer[start:stop]
    clo, chi = columns[p.rank:p.rank+2]
    stats = dict(fine_dofs=n, coarse_dofs=dim, regions=count,
                 max_local_worst_error=max(patch.worst_error[rank-patch.mandatory]
                                           for patch, rank in zip(patches, ranks)),
                 max_gram_condition=condition, empty_free_rows=0,
                 mandatory_modes=sum(patch.mandatory for patch in patches),
                 excluded_local_directions=sum(patch.excluded_directions for patch in patches),
                 excluded_coarse_directions=int(np.count_nonzero(~keep)),
                 exact_coarse_probe_error=terminal.stats["probe_error"],
                 **matrix_cost(p, local, columns))
    return ModalTransfer(local, owners, columns, coord[clo:chi], region[clo:chi], ordinal[clo:chi],
                         coefficients[clo:chi], stats)


def rigid_error(p, level, target, coefficients):
    errors = []
    for i in range(target.shape[1]):
        truth = target[:, i]
        residual = level.P(coefficients[:, i]) - truth
        energy, length = p.reduce([residual@residual, truth@truth])
        errors.append(float(np.sqrt(energy/max(length, 1e-300))))
    return errors


def oracle_spaces(p, legacy, bridge, layout, fields, displacement, directions, coarse_bank,
                  settings, first_cost_guard=None):
    first = second = None
    fine = displacement.fine
    try:
        p1, rigid = build_first(p, bridge, fine, layout, fields, directions, settings)
        first = p1.import_level(p, legacy, fine.matrix)
        p1.stats["rigid_reproduction_error"] = rigid_error(p, first, rigid, p1.rigid_coefficients)
        if max(p1.stats["rigid_reproduction_error"]) > 1e-10:
            raise InvalidModalBasis("first_rigid_reproduction_failed")
        k1, _ = local_csr(p, first.coarse, bridge)
        k1_cost = matrix_cost(p, k1, first.owners)
        if first_cost_guard is not None and not first_cost_guard(p1, k1_cost):
            return p1, None, k1_cost
        p2 = build_second(p, legacy, bridge, fine, first, p1, coarse_bank, settings)
        second = p2.import_level(p, legacy, first.coarse)
        p2.stats["rigid_reproduction_error"] = rigid_error(
            p, second, p1.rigid_coefficients, p2.rigid_coefficients)
        if max(p2.stats["rigid_reproduction_error"]) > 1e-10:
            raise InvalidModalBasis("second_rigid_reproduction_failed")
        return p1, p2, k1_cost
    except InvalidDepth as error:
        raise InvalidModalBasis(str(error)) from error
    finally:
        if second is not None:
            second.close()
        if first is not None:
            first.close()
