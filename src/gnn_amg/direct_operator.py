"""Verify and train direct signed heads using the actual FieldSplit operator."""
from dataclasses import replace
import os
from pathlib import Path

import numpy as np

from .backend import Hierarchy, digest, ownership, read_json, record
from .direct_solver import DirectWeights, prepare_direct
from .factorized import load_model
from .krylov import (FieldSplitAdjoint, HierarchyAdjoint, probe_trajectories, short_gmres,
                     verify_short_segments)
from .online import global_objects, load_observed_state, solve_policy
from .student_cycle import compact, save_once


def load_probes(p, data, receipt, settings):
    part = receipt["shards"][p.rank]
    path = Path(receipt["directory"]) / (receipt["shard_pattern"] % p.rank)
    if digest(path) != part["sha256"]:
        raise ValueError("Actual teacher probe shard identity changed")
    with np.load(path) as z:
        np.testing.assert_array_equal(ownership(p, data[2]), z["full_owners"])
        np.testing.assert_array_equal(data[1], z["fine_full_rows"])
        phases = z["gmres_phase"].tolist()
        gmres = [(phase, z["gmres_inputs"][phases.index(phase)], int(settings["steps"][phase]))
                 for phase in settings["steps"]]
        index = receipt["actual_pcapply"] - 1
        name = str(z["pcapply_names"][index])
        source, target = z["pcapply_inputs"][index], z["pcapply_teacher"][index]
    return gmres, (name, source, target)


def action_error(p, displacement, action, adjoint=None, weight=0.):
    name, source, target = action
    tape = [] if adjoint is not None else None
    answer = displacement.correction(source, tape)
    difference = answer - target
    norm2, err2 = p.reduce([target @ target, difference @ difference])
    if not np.isfinite(err2) or not np.isfinite(norm2) or norm2 <= 0:
        raise ValueError("Invalid same-input PCApply objective")
    if adjoint is not None:
        adjoint.grad_first.weight = adjoint.grad_second.weight = 1.
        adjoint.transpose((2.*weight/norm2)*difference, tape)
    return dict(name=name, normalized_squared_error=float(err2/norm2))


def score(settings, trajectories, action):
    return (sum(settings["phase_weight"][name]*entry["log_loss"]
                for name, entry in trajectories.items())
            + settings["teacher_action_weight"]*action["normalized_squared_error"])


def actual_forward(p, matrix, fs, displacement, probes, settings):
    gmres, action = probes
    trajectories = probe_trajectories(p, matrix, fs, [(name, r) for name, r, _ in gmres],
                                      {name:[length] for name, _, length in gmres})
    if set(trajectories) != {name for name, _, _ in gmres}:
        raise ValueError("Missing actual GMRES direction")
    pc = action_error(p, displacement, action)
    return dict(objective=score(settings, trajectories, pc), trajectories=trajectories,
                pcapply=pc, verified=True)


def direct_adjoint(p, legacy, bridge, matrix, fs, shell, callback, displacement, model,
                   probes, settings):
    """Verify signed-head gradients with frozen supports and actual reassembled K1,K2."""
    gmres, action = probes
    cycle = operation = None
    try:
        parity = verify_short_segments(p, matrix, fs,
                                       [(name, r) for name, r, _ in gmres if name == "middle"], (4, 6))
        cycle = HierarchyAdjoint(p, bridge, displacement)
        operation = FieldSplitAdjoint(p, matrix, fs, shell, callback, cycle)
        phase = operation.open()
        windows = {}
        for name, residual, length in gmres:
            cycle.grad_first.weight = cycle.grad_second.weight = settings["phase_weight"][name]
            windows[name] = short_gmres(operation.apply, p.reduce, residual, length, operation.transpose)
        pc = action_error(p, displacement, action, cycle, settings["teacher_action_weight"])
        base = score(settings, windows, pc)
        first_grad, second_grad = cycle.transfers(legacy)
        first = displacement.first.direct_stencil
        second = displacement.second.direct_stencil
        gradient = p.reduce([first.pullback(first_grad), second.pullback(second_grad)])
        norm = float(np.linalg.norm(gradient))
        dot = max(operation.dot_errors, default=0.)
        if not np.isfinite(norm) or norm <= 1e-14:
            raise ValueError("Direct signed head gradient is zero or non-finite")
        if not np.isfinite(dot) or dot > settings["transpose_tolerance"]:
            raise ValueError("Full FieldSplit transpose parity failed")

        def value(coefficient):
            a = b = h = None
            try:
                a = first.rebuild(p, legacy, displacement.fine.matrix, coefficient[0])
                b = second.rebuild(p, legacy, a.coarse, coefficient[1])
                h = Hierarchy(p, legacy, a, b, int(b.owners[-1]))
                trial = replace(displacement, first=a, second=b, space=h)
                operation.displacement = trial
                trajectories = {name:short_gmres(lambda v: operation.apply(v, False), p.reduce,
                                                   residual, length)
                                for name, residual, length in gmres}
                return score(settings, trajectories, action_error(p, trial, action))
            finally:
                operation.displacement = displacement
                for resource in (h, b, a):
                    if resource is not None:
                        resource.close()

        reference = model.coefficient
        direction = gradient/norm
        center_error = abs(value(reference)-base)
        slopes = []
        for step in settings["finite_steps"]:
            minus = value(reference-step*direction)
            plus = value(reference+step*direction)
            finite = (plus-minus)/(2.*step)
            error = abs(finite-norm)/max(abs(finite), norm, 1e-30)
            slopes.append(dict(step=step, minus=minus, plus=plus, analytic=norm, measured=finite,
                               relative_error=float(error)))
        stable = any(max(a["relative_error"], b["relative_error"]) <= settings["finite_tolerance"]
                     for a, b in zip(slopes, slopes[1:]))
        passed = bool(stable and center_error <= settings["center_tolerance"])
        return dict(verified=passed, failure=None if passed else "signed_head_finite_difference",
                    objective=base, trajectories=windows, pcapply=pc, phase=phase,
                    short_petsc_parity=parity, ras_dot_error=cycle.ras_dot_error,
                    fieldsplit_dot_error=dot, middle_ilu=cycle.ilu_parity,
                    head_gradient_norms=np.linalg.norm(gradient, axis=1).tolist(),
                    gradient=gradient.tolist(), finite_difference=slopes, center_error=center_error)
    except (RuntimeError, ValueError) as error:
        return dict(verified=False, failure=str(error),
                    callback_errors=operation.errors if operation is not None else [])
    finally:
        try:
            if operation is not None:
                operation.close()
        finally:
            if cycle is not None:
                cycle.close()


def run_direct(config, p, replay, legacy, bridge, inventory):
    settings = config["direct_solve"]
    run_id = os.environ["GNN_RUN_ID"]
    output = Path(config["output_directory"])
    model_info = settings["distillation"]
    if digest(model_info["path"]) != model_info["sha256"]:
        raise ValueError("Distillation receipt identity changed")
    fitted = read_json(model_info["path"])
    seed = DirectWeights.load(fitted["checkpoint"])
    teacher_info = config["direct_final"]["teacher"]
    if digest(teacher_info["path"]) != teacher_info["sha256"]:
        raise ValueError("Actual probe archive identity changed")
    source = read_json(teacher_info["path"])
    teacher_rows = {r["key"]:r for r in source["rows"]}
    frames = {row["key"]:row for row in inventory["snapshots"]}
    operator_settings = settings["operator"]
    fit_keys = operator_settings["fit_keys"]
    diagnose = fitted["diagnostic"]
    if (len(fit_keys) != config["parallel_workers"] or not set(fit_keys) <= set(fitted["fit"])
            or set(fit_keys) & set(diagnose)):
        raise ValueError("Operator fitting must use separate training states")
    reference = load_model(config["factor_model"])

    def share(local):
        return [record for part in global_objects(p, local) for record in part]

    def destroy(state):
        data = state[2]
        p.destroy("Vec", data[3], data[4])
        p.destroy("Mat", data[2])

    def solve(key, state, model, observe=None, retain_gradient=False):
        layout, fields, data = state
        fixed = fields["constrained"]
        if model is None:
            context = dict(factor_model=reference, factor_fusion=True, archive_choices=False)
        else:
            def prepare(fine, local):
                return prepare_direct(config, p, legacy, bridge, layout, fixed, fine, local,
                                      model, retain_gradient)
            context = dict(displacement_factory=prepare, archive_choices=False,
                           displacement_probe=observe)
        return solve_policy(config, p, replay, legacy, bridge, None, frames[key], layout,
                            {"constrained": fixed}, data, context=context)

    fit_key = fit_keys[p.worker]
    fit_state = load_observed_state(config, p, replay, frames[fit_key])
    fit_receipt = teacher_rows[fit_key]
    if fit_receipt["snapshot_sha256"] != frames[fit_key]["manifest_sha256"]:
        raise ValueError("Actual operator training snapshot identity changed")
    probes = load_probes(p, fit_state[2], fit_receipt, operator_settings)

    def audited(displacement, matrix, fs, shell, callback):
        return direct_adjoint(p, legacy, bridge, matrix, fs, shell, callback, displacement,
                              seed, probes, operator_settings)

    def objective(displacement, matrix, fs, shell, callback):
        try:
            return actual_forward(p, matrix, fs, displacement, probes, operator_settings)
        except (ValueError, RuntimeError) as error:
            return dict(verified=False, failure=str(error))

    def observation(model, measure, retain_gradient=False):
        result = solve(fit_key, fit_state, model, measure, retain_gradient)
        if "verified" not in result:
            result = dict(verified=False, failure=result.get("failure") or "no_operator_observation")
        if p.rank == 0:
            record(config, "direct_actual_operator", job=run_id, key=fit_key,
                   result={k:v for k,v in result.items() if k != "gradient"})
        return share([dict(key=fit_key, **result)] if p.rank == 0 else [])

    current, history, accepted = seed, [], None
    try:
        initial = observation(seed, audited, True)
        if all(row["verified"] for row in initial):
            gradient = np.mean([row["gradient"] for row in initial], axis=0)
            norm = float(np.linalg.norm(gradient))
            if np.isfinite(norm) and norm > 1e-14:
                before = {row["key"]:row for row in initial}
                for step in operator_settings["step_sizes"]:
                    proposed = seed.adjusted(seed.coefficient-step*gradient/norm)
                    values = observation(proposed, objective)
                    good = (all(row["verified"] for row in values)
                            and sum(row["objective"] for row in values)
                            < sum(row["objective"] for row in initial)-operator_settings["min_improvement"]
                            and all(row["objective"] <= before[row["key"]]["objective"]
                                    + operator_settings["state_guard"] for row in values))
                    history.append(dict(step=step, accepted=bool(good), cases=values))
                    if good:
                        current, accepted = proposed, step
                        break
    finally:
        destroy(fit_state)

    model_receipt = None
    if accepted is not None and p.global_rank == 0:
        path = output / "checkpoints" / ("generator-%s-direct-operator.npz" % run_id)
        arrays = dict(current.arrays, signed_coefficients=current.coefficient)
        with path.open("xb") as stream:
            np.savez(stream, **arrays)
        model_receipt = dict(path=str(path), sha256=digest(path))
    receipts = share([model_receipt] if model_receipt is not None else [])
    tuned = receipts[0] if receipts else None
    if p.global_rank == 0:
        record(config, "direct_actual_operator_fitted", job=run_id,
               verified_states=sum(row["verified"] for row in initial),
               accepted_step=accepted, checkpoint=tuned)
    rows = []
    methods = [("factorized-teacher", None), ("direct-distilled", seed)]
    if tuned:
        methods.append(("direct-operator", current))
    for index, key in enumerate(diagnose):
        if index % config["parallel_workers"] != p.worker:
            continue
        state = load_observed_state(config, p, replay, frames[key])
        try:
            for repeat in range(settings["repeats"]):
                order = methods if (index//config["parallel_workers"]+repeat)%2 == 0 else methods[::-1]
                for label, model in order:
                    actual = solve(key, state, model)
                    row = compact(actual, frames[key], label)
                    row["repeat"] = repeat+1
                    row["levels"] = [actual["numerical"].get("level1"), actual["numerical"].get("level2")]
                    if model is not None:
                        row["timing_scope"] = ("Fresh graph, regional point and edge selection, signed coefficients, "
                                               "actual Galerkin, RAS, audits and full true-residual solve.")
                    if p.rank == 0:
                        rows.append(row)
                        record(config, "direct_full_solve", job=run_id, **row)
        finally:
            destroy(state)

    rows = share(rows)
    expected = {(key,label,r+1) for key in diagnose for label, _ in methods for r in range(settings["repeats"])}
    if len(rows) != len(expected) or {(r["key"],r["method"],r["repeat"]) for r in rows} != expected:
        raise ValueError("Missing paired actual direct solves")
    totals = {}
    for label, _ in methods:
        group = [row for row in rows if row["method"] == label]
        totals[label] = dict(passed=sum(row["passed"] for row in group), frames=len(group),
            iterations=sum(row["iterations"] or 0 for row in group),
            setup_s=sum(row["setup_s"] for row in group),
            solve_s=sum(row["solve_s"] or 0 for row in group),
            total_s=sum(row["total_s"] for row in group))
    if p.global_rank == 0:
        receipt = dict(job=run_id, distillation=model_info, teacher=config["factor_model"],
                       fit_keys=fit_keys, diagnostic=diagnose, probes=teacher_info,
                       operator=dict(initial=initial, trials=history, accepted_step=accepted, checkpoint=tuned,
                           signed_head_delta=np.linalg.norm(current.coefficient-seed.coefficient, axis=1).tolist()),
                       diagnostic_rows=rows, diagnostic_totals=totals)
        save_once(output / "records" / ("direct-solve-" + run_id + ".json"), receipt)
        record(config, "direct_solve_complete", job=run_id, accepted_step=accepted,
               checkpoint=tuned, totals=totals)
    return 0
