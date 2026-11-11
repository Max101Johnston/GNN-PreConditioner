"""A single four-state gate for an idealized local modal hierarchy."""
import os
from pathlib import Path
import time

import numpy as np

from .backend import digest, ownership, read_json, record
from .factorized import load_model
from .modal_basis import InvalidModalBasis, oracle_spaces
from .online import (DisplacementPC, finish_hierarchy, global_objects,
                     load_observed_state, solve_policy)
from .student import load_student
from .student_cycle import compact, save_once


def error_probe_bank(p, state, displacement, iterates, settings, index):
    """Extract actual displacement errors and apply the RAS error propagator."""
    layout, fields, data = state
    full_owners = ownership(p, data[2])
    full_lo, full_hi = full_owners[p.rank:p.rank+2]
    if "reference_check" not in iterates or "reference" not in iterates:
        raise InvalidModalBasis("strict_reference_check_unavailable")
    local_rows = data[1] - full_lo
    if np.any((local_rows < 0) | (local_rows >= full_hi-full_lo)):
        raise InvalidModalBasis("full_displacement_error_mapping")
    ref_step, ref, ref_residual = iterates["reference"]
    near_step, near, near_residual = iterates["reference_check"]
    if near_step >= ref_step or not 0. <= ref_residual < near_residual:
        raise InvalidModalBasis("distinct_tight_reference_levels_unavailable")
    ref, near = ref[local_rows], near[local_rows]
    delta = ref-near
    uncertainty = float(np.sqrt(p.reduce([delta@delta])[0]))
    checkpoints = {step:(vector[local_rows], residual)
                   for name, (step, vector, residual) in iterates.items()
                   if name not in ("reference", "reference_check")}
    if len(checkpoints) < 3:
        raise InvalidModalBasis("missing_distinct_actual_krylov_iterates")
    lo, hi = layout["ownership"][p.rank:p.rank+2]
    comp = layout["component"][lo:hi]
    native = 3*layout["native_node"][lo:hi] + comp
    fixed = fields["constrained"][native]
    directions, coarse_input, named, errors, source_steps = [], [], [], [], []

    def store(label, vector, coarse=False):
        vector = vector.copy()
        vector[fixed] = 0.
        norm = float(np.sqrt(p.reduce([vector@vector])[0]))
        if not np.isfinite(norm):
            raise InvalidModalBasis("nonfinite_probe_error")
        if norm > 1e-30:
            directions.append(vector/norm)
            coarse_input.append(coarse)
        named.append(dict(name=label, norm=norm, kept=norm > 1e-30, coarse_input=coarse))
        return norm

    def after_ras(vector):
        return vector-displacement.local.apply(displacement.fine(vector))

    for step, (iterate, residual) in sorted(checkpoints.items()):
        error = ref-iterate
        error[fixed] = 0.
        norm = store("actual-error-%d" % step, error)
        if norm <= 1e-30:
            raise InvalidModalBasis("zero_actual_krylov_error")
        errors.append(error)
        store("ras-error-%d" % step, after_ras(error), coarse=True)
        source_steps.append(dict(step=step, true_residual=residual, displacement_error_norm=norm))
    uncertainty_ratio = uncertainty/min(row["displacement_error_norm"] for row in source_steps)
    if uncertainty_ratio > settings["max_reference_uncertainty"]:
        raise InvalidModalBasis("late_error_not_resolved_by_reference")

    rng = np.random.default_rng(np.random.SeedSequence([settings["seed"], index, p.rank]))
    for component in range(3):
        error = np.where((comp == component) & ~fixed, rng.standard_normal(len(comp)), 0.)
        for _ in range(settings["random_ras_sweeps"]):
            error = after_ras(error)
            error[fixed] = 0.
        store("ras-random-component-%d" % component, error, coarse=True)
    return (np.column_stack(directions), np.asarray(coarse_input, bool), ref, np.column_stack(errors),
            dict(reference_step=ref_step, reference_true_residual=ref_residual,
                 check_step=near_step, check_true_residual=near_residual,
                 reference_change_ratio=uncertainty_ratio, sources=source_steps, directions=named))


def cost_gate(p1, k1, reference, limit):
    limits = {}
    for name, values, base in (("p1", p1.stats, reference["p"]), ("k1", k1, reference["k"])):
        limits[name] = {field: dict(actual=values[field], baseline=base[field],
                                    passed=values[field] <= limit*base[field])
                        for field in ("nnz", "off_rank_entries", "remote_columns_rank_sum")}
    return dict(passed=all(x["passed"] for category in limits.values() for x in category.values()),
                limit=limit, metrics=limits)


def archive_basis(p, folder, first, second, ref, errors, directions, coarse_input):
    if p.rank == 0:
        folder.mkdir()
    p.barrier()
    arrays = dict(reference_displacement=ref, actual_displacement_errors=errors,
                  normalized_error_bank=directions, coarse_input=coarse_input)
    for name, transfer in (("p1", first), ("p2", second)):
        if transfer is None:
            continue
        csr = transfer.csr
        for field, values in (("indptr", csr.indptr), ("indices", csr.indices),
                              ("data", csr.data), ("row_owners", transfer.rows),
                              ("column_owners", transfer.columns), ("region", transfer.regions),
                              ("ordinal", transfer.ordinals), ("centroid", transfer.coordinates),
                              ("rigid_coefficients", transfer.rigid_coefficients)):
            arrays[name+"_"+field] = values
    path = folder/("rank-%03d.npz" % p.rank)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    sha = p.gather_equal(np.frombuffer(bytes.fromhex(digest(path)), np.uint8)).astype(np.uint8)
    sizes = p.gather_equal([path.stat().st_size]).ravel().astype(np.int64)
    return dict(directory=str(folder), pattern="rank-%03d.npz",
                parts=[dict(rank=i, bytes=int(size), sha256=h.tobytes().hex())
                       for i, (h, size) in enumerate(zip(sha, sizes))])


def summarize_gate(cases, keys, settings):
    checks = []
    for repeat in (1, 2):
        rows = {(r["method"], r["key"]):r for c in cases for r in c["solves"]
                if r["repeat"] == repeat}
        metrics = {}
        for method in ("factorized", "student", "modal-oracle"):
            by_method = [rows[method, key] for key in keys if (method, key) in rows]
            metrics[method] = dict(cases=len(by_method), passed=sum(r["passed"] for r in by_method),
                                   steps=sum(r["iterations"] or 0 for r in by_method),
                                   total_s=sum(r["total_s"] for r in by_method))
        full = len(rows) == 3*len(keys) and all(row["passed"] for row in rows.values())
        matched = full and all(len({rows[m, key]["true_limit"] for m in metrics})==1 for key in keys)
        oracle, baseline = metrics["modal-oracle"], metrics["factorized"]
        iteration_passed = (matched and oracle["steps"] <= settings["steps_ratio"]*baseline["steps"]
            and all(rows["modal-oracle", key]["iterations"]
                    <= settings["single_step_ratio"]*rows["factorized", key]["iterations"]
                    for key in keys))
        time_passed = (matched and oracle["total_s"]
            <= settings["wall_ratio"]*min(baseline["total_s"], metrics["student"]["total_s"]))
        checks.append(dict(repeat=repeat, totals=metrics, matched_truth=bool(matched),
                           iteration_passed=bool(iteration_passed), wall_passed=bool(time_passed)))
    affordable = all(c["basis"].get("budget", {}).get("passed", False) for c in cases)
    return dict(cost_passed=affordable, repeats=checks,
                passed=affordable and all(c["iteration_passed"] and c["wall_passed"] for c in checks))


def run_modal_gate(config, p, replay, legacy, bridge, inventory):
    settings = config["modal_gate"]
    origin = settings["parent"]
    teacher_receipt = settings["teacher_data"]
    if digest(origin["path"]) != origin["sha256"] or digest(teacher_receipt["path"]) != teacher_receipt["sha256"]:
        raise ValueError("Changed modal experiment provenance")
    baseline = read_json(origin["path"])
    teachers = {r["key"]:r for r in read_json(teacher_receipt["path"])["rows"]}
    keys = baseline["keys"]
    if len(keys) != 4 or any(teachers[key]["role"] != "diagnostic" for key in keys):
        raise ValueError("Modal feasibility uses the four existing diagnostic states")
    model = load_model(baseline["teacher"])
    if not model.matrix_only:
        raise ValueError("Modal reference must use the verified matrix-only teacher")
    student = config["factor_reference"]
    for level, entry in student["students"].items():
        load_student(entry["path"], entry["sha256"], int(level[1:]))
    frames = {row["key"]:row for row in inventory["snapshots"]}
    output = Path(config["output_directory"])
    job = os.environ["GNN_RUN_ID"]
    directory = output/("data/modal-gate-" + job)
    if p.global_rank == 0:
        directory.mkdir()
    global_objects(p, [str(directory)] if p.global_rank == 0 else [])
    local = []
    for i, key in enumerate(keys):
        if i % config["parallel_workers"] != p.worker:
            continue
        row, teacher = frames[key], teachers[key]
        if row["manifest_sha256"] != teacher["snapshot_sha256"]:
            raise ValueError("Changed modal diagnostic matrix")
        state = load_observed_state(config, p, replay, row)
        layout, fields, data = state
        fields = {"constrained": fields["constrained"]}
        starts = time.perf_counter()
        captures, iterates = {}, {}
        compare = []

        def capture(displacement, result):
            try:
                directions, coarse_input, ref, errors, labels = error_probe_bank(
                    p, state, displacement, iterates, settings, i)
                captures["errors"] = labels
                def size_guard(p1, k1):
                    captures["budget"] = cost_gate(p1, k1, teacher["levels"][0],
                                                   settings["fill_ratio"])
                    return captures["budget"]["passed"]

                p1, p2, k1 = oracle_spaces(p, legacy, bridge, layout, fields, displacement,
                                            directions, directions[:, coarse_input], settings,
                                            size_guard)
                captures.update(p1=p1, p2=p2, k1=k1)
                if p2 is None:
                    captures["stop_stage"] = "first_level_fill_and_communication"

                captures["archive"] = archive_basis(p, directory/key, p1, p2, ref, errors,
                                                    directions, coarse_input)
            except InvalidModalBasis as error:
                captures["failure"] = str(error)

        def solve(context):
            return solve_policy(config, p, replay, legacy, bridge, None, row, layout, fields, data,
                                context=dict(context, archive_choices=False))

        def modal(fine, ras):
            first = captures["p1"].import_level(p, legacy, fine.matrix)
            second = captures["p2"].import_level(p, legacy, first.coarse)
            stats = dict(level1=first.stats, level2=second.stats)
            space, failure = finish_hierarchy(config, p, legacy, bridge, fine, first, second, stats)
            return DisplacementPC(p, fine, ras, first, second, space, None, None, None, None,
                                  stats, {}, failure)

        try:
            if p.rank == 0:
                record(config, "modal_gate_begin", job=job, key=key)
            factor_context = dict(factor_model=model, factor_fusion=True)
            ref = solve(dict(factor_context, modal_iterates=iterates,
                             displacement_observer=capture))
            elapsed = p.maximum(time.perf_counter()-starts)
            if not ref["reached_true_target"] or ref["failure"]:
                captures["failure"] = "strict_reference_" + str(ref["failure"])
            basis = {k:v for k,v in captures.items() if k not in ("p1", "p2")}
            basis.update(reference=compact(ref, row, "strict-reference"), offline_s=elapsed)
            if "p1" in captures:
                basis.update(p1=captures["p1"].stats,
                             p2=captures["p2"].stats if captures["p2"] is not None else None)
            if p.rank == 0:
                save_once(output/"records"/("modal-basis-%s-%s.json" % (job, key)), basis)
                record(config, "modal_gate_basis", job=job, key=key,
                       budget=basis.get("budget"), failure=basis.get("failure"),
                       offline_s=elapsed)
            contexts = [("factorized", factor_context),
                        ("student", dict(policy="prior_only", fast_prior=True,
                                         p1_refinement_sweeps=2, students=student["students"],
                                         controls_p1=student["controls_p1"],
                                         controls_p2=student["controls_p2"]))]
            if basis.get("budget", {}).get("passed") and not basis.get("failure"):
                contexts.append(("modal-oracle", dict(displacement_factory=modal)))
            offset = i % len(contexts)
            contexts = contexts[offset:] + contexts[:offset]
            for repeat, order in enumerate((contexts, contexts[::-1]), 1):
                for name, context in order:
                    solved = solve(context)
                    receipt = compact(solved, row, name)
                    receipt.update(repeat=repeat, basis_known=name=="modal-oracle",
                                   terminal_dofs=solved["numerical"].get("level2", {}).get("coarse_dofs"))
                    if p.rank == 0:
                        compare.append(receipt)
                        record(config, "modal_gate_solve", job=job, key=key, method=name,
                               repeat=repeat, passed=receipt["passed"], steps=receipt["iterations"],
                               true_residual=receipt["final_true_residual"], limit=receipt["true_limit"],
                               total_s=receipt["total_s"])
            if p.rank == 0:
                local.append(dict(key=key, basis=basis, solves=compare))
        finally:
            p.destroy("Vec", data[3], data[4])
            p.destroy("Mat", data[2])
    cases = [c for group in global_objects(p, local) for c in group]
    if len(cases) != 4 or {c["key"] for c in cases} != set(keys):
        raise ValueError("Incomplete modal feasibility experiment")
    gate = summarize_gate(cases, keys, settings)
    if p.global_rank == 0:
        report = dict(job=job, keys=keys, parent=origin, teacher=baseline["teacher"],
                      settings=settings, cases=cases, gate=gate,
                      scope="Offline ideal modal bases. Timed path includes sparse transfer import, "
                            "actual Galerkin, all smoothing and terminal setup, and full solve. "
                            "Strict reference solves, basis fitting and archives are timed separately.")
        save_once(output/"records"/("modal-gate-"+job+".json"), report)
        record(config, "modal_gate_complete", job=job, **gate)
    return 0
