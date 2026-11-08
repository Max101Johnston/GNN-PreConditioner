"""Offline supervised examples from accepted labels and the actual new K1.

This module's feature functions also serve online inference. Only export_state
reads teacher labels; graph_features and pooled_attributes are label-free.
"""
from pathlib import Path
import os
import numpy as np

from .backend import digest, graph_features, pooled_attributes, record


ATTRIBUTE_NAMES = ["phi", "phi_squared", "phasefield_previous_ip", "active_degradation",
                   "target_degradation", "log10_bulk_modulus", "log10_shear_modulus",
                   "biot_coefficient", "porosity", "log10_aperture", "x", "y", "z"]
FEATURE_NAMES = ATTRIBUTE_NAMES + ["log10_diagonal", "log10_abs_row_sum", "negative_fraction",
    "positive_fraction", "log1p_strong_degree", "mpi_border_fraction", "diagonal_fraction",
    "diagonal_sign", "ux", "uy", "uz", "constrained", "stage", "log1p_support_mass"]


def physical_attributes(fields, xyz, native):
    """Fresh physical observations in requested native-DOF order."""
    nodes = np.asarray(native, dtype=np.int64) // 3
    material = fields["features"][nodes]
    phi = fields["phi"][nodes]
    aperture = fields["aperture"][nodes]
    # Coordinate normalization is geometry-local, independent of all labels.
    center = (xyz.max(axis=0) + xyz.min(axis=0))/2
    length = np.maximum(np.ptp(xyz, axis=0), 1e-30)
    coordinates = (xyz[nodes] - center)/length
    values = np.column_stack([phi, phi**2, material[:, 1:4],
        np.log10(np.maximum(material[:, 5:7], 1e-30)), material[:, 11:13],
        np.log10(np.maximum(aperture, 1e-30)), coordinates])
    assert values.shape[1] == len(ATTRIBUTE_NAMES) and np.isfinite(values).all()
    return values


def gather_array(p, values, owners):
    a = np.asarray(values)
    width = int(np.prod(a.shape[1:])) if a.ndim > 1 else 1
    result = p.gather_rows(a.ravel(), owners * width)
    return result.reshape((-1,) + a.shape[1:])


def gather_graph(p, graph, owners):
    lengths = p.gather_rows(np.diff(graph.indptr), owners)
    counts = p.gather_equal([len(graph.data)]).ravel().astype(np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    columns = p.gather_rows(graph.indices, offsets)
    weights = p.gather_rows(graph.data, offsets)
    if p.rank:
        return None
    return dict(indptr=np.r_[0, np.cumsum(lengths)].astype(np.int64),
                indices=columns.astype(np.int32), weights=weights.astype(np.float32))


def export_state(config, p, bridge, row, layout, teacher, matrix, first, second):
    old = Path(config["baseline_directory"])
    owners = layout["ownership"]
    lo, hi = map(int, owners[p.rank:p.rank+2])
    native = teacher["native"][lo:hi]
    with np.load(old / "data" / row["key"] / "fields.npz") as fields:
        attributes = physical_attributes(fields, layout["xyz"], native)
    coarse_attributes, support = pooled_attributes(p, first, attributes)
    results = []
    for stage, k, ids, comp, attrs, mass, constrained, labels, executed in (
        (0, matrix, native, layout["component"][lo:hi], attributes, np.ones(len(native)),
         teacher["fixed"][lo:hi], teacher["depth"][lo:hi] >= 3, first.executed_markers == 1),
        (1, first.coarse, first.native, first.components, coarse_attributes, support,
         np.zeros(len(first.native), bool), np.isin(first.native, teacher["terminal_native"]), second.executed_markers == 1)):
        features, graph, owner = graph_features(p, bridge, k, comp, attrs, constrained, stage, mass, config["interpolation"]["strength_threshold"])
        assert features.shape[1] == len(FEATURE_NAMES)
        arrays = gather_graph(p, graph, owner)
        gathered = dict(features=gather_array(p, features, owner).astype(np.float32),
            native=gather_array(p, ids, owner).astype(np.int64),
            eligible=gather_array(p, ~constrained, owner).astype(bool),
            raw_teacher=gather_array(p, labels, owner).astype(bool),
            executed_teacher=gather_array(p, executed, owner).astype(bool))
        if p.rank == 0:
            arrays.update(gathered)
            arrays["feature_names"] = np.asarray(FEATURE_NAMES)
            arrays["stage"] = np.asarray(stage)
            assert np.unique(arrays["native"]).size == len(arrays["native"])
            assert not np.any(arrays["raw_teacher"] & ~arrays["eligible"])
            # In-degree prior is computed from the current graph only.
            incoming = np.bincount(arrays["indices"], weights=arrays["weights"], minlength=len(arrays["native"]))
            arrays["connection_prior"] = np.clip(np.log1p(incoming) - np.log(2), -2, 2).astype(np.float32)
            path = Path(config["output_directory"]) / "data/supervised" / (os.environ.get("SLURM_JOB_ID", "local") + "-" + row["key"] + "-level%d.npz" % stage)
            with path.open("xb") as stream:
                np.savez(stream, **arrays)
            results.append(dict(path=str(path), sha256=digest(path), stage=stage,
                nodes=len(ids) if p.size == 1 else int(owner[-1]), edges=len(arrays["weights"]),
                raw_coarse=int(arrays["raw_teacher"].sum()), executed_coarse=int(arrays["executed_teacher"].sum()),
                bytes=path.stat().st_size))
    p.barrier()
    if p.rank == 0:
        record(config, "supervised_state_exported", key=row["key"], split=row["split"], files=results)
    return results
