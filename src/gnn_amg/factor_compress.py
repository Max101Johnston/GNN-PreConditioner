"""Learn a sparse explicit P1 on the generated candidates."""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy import sparse

from .backend import local_csr, record
from .factorized import (PRODUCT_BUDGET_FEATURES, PRODUCT_SUPPORT_FEATURES,
                         normalized_coordinates, sparse_entries)


def product_context(csr, fine, coarse, xyz):
    degree = np.diff(csr.indptr)
    rows = np.repeat(np.arange(csr.shape[0]),degree)
    magnitude = np.abs(csr.data)
    peak = np.zeros(len(fine))
    np.maximum.at(peak,rows,magnitude)
    relative = magnitude/np.maximum(peak[rows],1e-300)
    group = 3*rows+coarse[csr.indices]%3
    group_peak = np.zeros(3*len(fine))
    np.maximum.at(group_peak,group,magnitude)
    positions = np.arange(csr.nnz)
    anchor = np.full(len(group_peak),csr.nnz)
    largest = magnitude==group_peak[group]
    np.minimum.at(anchor,group[largest],positions[largest])
    mandatory = positions==anchor[group]
    groups = np.bincount(rows[mandatory],minlength=len(fine))
    distance = np.linalg.norm(xyz[fine[rows]//3]-xyz[coarse[csr.indices]//3],axis=1)
    same = fine[rows]%3==coarse[csr.indices]%3
    edges = np.column_stack((np.ones(csr.nnz),np.log(np.maximum(relative,1e-12)),distance,
                             same,np.sign(csr.data)*relative,np.log1p(degree[rows])/3.))
    divisor = np.maximum(1,degree)
    budget = np.column_stack((np.ones(len(fine)),np.log1p(degree)/3.,groups/3.,peak,
              np.bincount(rows,weights=relative,minlength=len(fine))/divisor,
              np.bincount(rows,weights=distance,minlength=len(fine))/divisor,
              np.bincount(rows,weights=~same,minlength=len(fine))/divisor,xyz[fine//3]))
    return rows,degree,group,anchor,mandatory,groups,edges,budget


@dataclass
class ProductDrop:
    raw: object
    kept: object
    slots: np.ndarray

    def apply(self, source):
        rows = np.repeat(np.arange(self.raw.shape[0]),np.diff(self.raw.indptr))
        values = sparse_entries(source,rows,self.raw.indices)
        coefficients = np.bincount(self.slots,weights=values,minlength=self.kept.nnz)
        return sparse.csr_matrix((coefficients,self.kept.indices,self.kept.indptr),shape=self.kept.shape)

    def pullback(self, gradient):
        rows = np.repeat(np.arange(self.kept.shape[0]),np.diff(self.kept.indptr))
        entries = sparse_entries(gradient,rows,self.kept.indices)
        return sparse.csr_matrix((entries[self.slots],self.raw.indices,self.raw.indptr),shape=self.raw.shape)


def hard_product(csr, context, support, budget):
    rows,degree,group,anchor,mandatory,groups,edges,features = context
    score = edges@support
    optional = degree-groups
    quota = np.minimum(degree,np.maximum(4,groups+np.rint(optional*np.clip(features@budget,0.,1.)).astype(int)))
    order = np.lexsort((csr.indices,-score,~mandatory,rows))
    rank = np.empty(csr.nnz,dtype=np.int64)
    rank[order] = np.arange(csr.nnz)-csr.indptr[rows[order]]
    keep = rank<quota[rows]
    # Dropped entries contribute to the anchor of their own component.
    def assemble(mask):
        chosen = np.flatnonzero(mask)
        output_position = np.zeros(csr.nnz, dtype=np.int64)
        output_position[chosen] = np.arange(len(chosen))
        slots = np.where(mask, output_position, output_position[anchor[group]])
        indptr = np.r_[0, np.cumsum(np.bincount(rows[mask], minlength=csr.shape[0]))]
        coefficients = np.bincount(slots, weights=csr.data, minlength=len(chosen))
        result = sparse.csr_matrix((coefficients, csr.indices[chosen], indptr), shape=csr.shape)
        return result, slots, chosen

    result, slots, chosen = assemble(keep)
    cancelled = np.unique(group[chosen[result.data == 0.]])
    if len(cancelled):
        keep |= np.isin(group, cancelled)
        result, slots, _ = assemble(keep)
    return result, ProductDrop(csr, result, slots), len(cancelled)


def compress_product(p, csr, fine, coarse, xyz, weights):
    context = product_context(csr,fine,coarse,xyz)
    support,budget = np.split(weights,[PRODUCT_SUPPORT_FEATURES])
    result,tape,restored = hard_product(csr,context,support,budget)
    raw_groups = np.bincount(context[2],weights=csr.data,minlength=3*len(fine))
    rows = np.repeat(np.arange(len(fine)),np.diff(result.indptr))
    new_groups = np.bincount(3*rows+coarse[result.indices]%3,weights=result.data,minlength=len(raw_groups))
    error = p.maximum(np.max(np.abs(raw_groups-new_groups),initial=0.))
    after_degree = np.diff(result.indptr)
    sizes = p.reduce([csr.nnz, result.nnz, restored, np.count_nonzero(after_degree < context[1]),
                      np.count_nonzero(after_degree), int(not np.isfinite(result.data).all())])
    if sizes[5] or not np.isfinite(error) or error > 1e-10:
        raise ValueError("P1 compression changed a signed component sum")
    return result, tape, dict(before_nnz=int(sizes[0]), after_nnz=int(sizes[1]),
                              restored_component_groups=int(sizes[2]), changed_rows=int(sizes[3]),
                              active_rows=int(sizes[4]), component_sum_error=error)


def select_probes(probes, count):
    """Sample both input families, cycling through archived phases."""
    groups = {}
    for probe in probes:
        phase, name = probe[0].split("/", 1)
        groups.setdefault((name.split("-", 1)[0], phase), []).append(probe)
    for group in groups.values():
        if len(group) > 1:
            group.insert(1, group.pop())
    result = []
    for index in range(max(map(len, groups.values()))):
        result.extend(group[index] for group in groups.values() if index < len(group))
        if len(result) >= count:
            break
    return result[:count]


def sample_product_rows(p, bridge, displacement, layout, probes, rng, count, reference_probes):
    first = displacement.first
    lo, hi = map(int, layout["ownership"][p.rank:p.rank+2])
    fine = (3*layout["native_node"] + layout["component"])[lo:hi]
    coarse = p.all_rows(first.native, first.owners).astype(np.int64)
    csr = local_csr(p, first.transfer, bridge)[0]
    csr.eliminate_zeros()
    csr.sort_indices()
    xyz = normalized_coordinates(layout["xyz"])
    _, degree, group, anchor, mandatory, groups, edge, row_features = product_context(
        csr, fine, coarse, xyz)
    directions = []
    for name, residual, _, weights2, _ in probes:
        tape = []
        answer = displacement.correction(residual, tape)
        direction = p.all_rows(tape[-1], first.owners)
        directions.append(direction / max(np.linalg.norm(direction)/np.sqrt(len(direction)), 1e-300))
        if reference_probes is not None:
            norm = p.reduce([answer @ (weights2*answer)])[0]
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("Compression reference action has no positive probe norm")
            reference_probes.append((name, residual, answer, weights2, norm))
    coarse_action = np.column_stack(directions)
    eligible = np.flatnonzero(degree > 4)
    samples = []
    for row in rng.choice(eligible, min(len(eligible), count), replace=False):
        start, stop = csr.indptr[row:row+2]
        positions = np.arange(start, stop)[~mandatory[start:stop]]
        source = coarse_action[csr.indices[positions]]
        anchors = coarse_action[csr.indices[anchor[group[positions]]]]
        lost = csr.data[positions, None]*(source-anchors)
        scale = max(np.linalg.norm(csr.data[start:stop] @ coarse_action[csr.indices[start:stop]]),
                    .1*np.abs(csr.data[start:stop]).sum())
        samples.append((edge[positions], row_features[row], lost/scale, max(4-groups[row], 0),
                        csr.indices[positions]))
    return samples


def ridge_weights(p, share, normal, rhs, count, penalty):
    local = p.reduce(np.r_[normal.ravel(), rhs, count])
    totals = np.sum(share([local.tolist()] if p.rank == 0 else []), axis=0)
    if totals[-1] <= 0 or not np.isfinite(totals).all():
        raise ValueError("No finite P1 support training rows")
    features = len(rhs)
    matrix = totals[:normal.size].reshape(normal.shape) / totals[-1]
    matrix += penalty*np.eye(features)
    return np.linalg.solve(matrix, totals[normal.size:-1]/totals[-1]), int(totals[-1])


def learn_support(p, share, samples, penalty):
    normal = np.zeros((PRODUCT_SUPPORT_FEATURES, PRODUCT_SUPPORT_FEATURES))
    rhs = np.zeros(PRODUCT_SUPPORT_FEATURES)
    for edges, _, lost, _, _ in samples:
        impact = np.linalg.norm(lost, axis=1)
        target = np.log(np.maximum(impact/max(impact.max(), 1e-300), 1e-6))
        normal += (edges.T @ edges)/len(edges)
        rhs += (edges.T @ target)/len(edges)
    return ridge_weights(p, share, normal, rhs, len(samples), penalty)


def learn_budget(p, share, samples, support, tolerance, penalty):
    normal = np.zeros((PRODUCT_BUDGET_FEATURES, PRODUCT_BUDGET_FEATURES))
    rhs = np.zeros(PRODUCT_BUDGET_FEATURES)
    for edges, features, lost, minimum, columns in samples:
        # Drop in reverse deployment order. Stop at the first prefix whose
        # signed action error exceeds the row's allowance.
        order = np.lexsort((-columns, edges @ support))[:len(edges)-minimum]
        error = np.linalg.norm(np.cumsum(lost[order], axis=0), axis=1)
        excess = np.flatnonzero(error > tolerance)
        dropped = int(excess[0]) if len(excess) else len(order)
        normal += np.outer(features, features)
        rhs += features*(1.-dropped/len(edges))
    return ridge_weights(p, share, normal, rhs, len(samples), penalty)[0]


def refit_coefficients(p, legacy, bridge, config, model, states, probes, solve, share, settings, job, label):
    from .factor_operator import operator_observation

    def observe(candidate, gradient=False, audit=False):
        local = []
        for key, loaded in states.items():
            def action(displacement):
                return operator_observation(p, legacy, bridge, displacement, candidate, probes[key],
                                            gradient, settings["numerical"], audit=audit)
            row = solve(key, loaded, dict(name=label, factor_model=candidate,
                factor_context=dict(factor_fusion=True, factor_tape=gradient, factor_operation=action)),
                "compression-action", details=False)
            if p.rank == 0:
                local.append(dict(key=key, **row))
        return share(local)

    def summary(rows):
        return [{k:v for k,v in r.items() if k!="gradient"} for r in rows]

    def valid(rows):
        return len(rows)==len(settings["pilot_keys"]) and all(row["failure"] is None for row in rows)

    history = []
    for step in range(settings["steps"]):
        source = observe(model, gradient=True, audit=step==0)
        row = dict(step=step+1, source=summary(source), attempts=[], accepted=False)
        history.append(row)
        if not valid(source):
            row["stop"] = "invalid_hierarchy"
            break
        if step==0 and not all(r["directional_derivatives"]["passed"] for r in source):
            row["stop"] = "directional_derivative"
            break
        gradient = np.mean([r["gradient"] for r in source], axis=0)
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(norm) or norm < 1e-12:
            row["stop"] = "no_coefficient_gradient"
            break
        row["gradient_norm"] = norm
        before = {r["key"]: r for r in source}
        for rate in (settings["step_size"], settings["step_size"]*.5):
            weights = model.coefficient_weights - rate*gradient/norm
            trial = replace(model, coefficient_weights=weights.astype(model.weights.dtype))
            checked = observe(trial)
            accepted = bool(valid(checked) and all(
                after["loss"] <= 1.05*before[after["key"]]["loss"]
                and all(a <= 1.05*b for a,b in zip(after["nonzeros"][:2], before[after["key"]]["nonzeros"][:2]))
                for after in checked)
                and sum(r["loss"] for r in checked) < sum(r["loss"] for r in source))
            row["attempts"].append(dict(step_size=rate, rows=checked, accepted=accepted))
            if accepted:
                model, row["accepted"] = trial, True
                break
        if p.global_rank == 0:
            record(config, "factor_compress_coefficient", job=job, method=label, step=step+1,
                   accepted=row["accepted"], gradient_norm=norm)
        if not row["accepted"]:
            break
    return model, history


def run_compression(config, p, legacy, bridge, frames, load, destroy, solve, share, evaluate, job):
    from .factor_operator import checked_json
    from .factor_replay import load_probe_shards
    from .factorized import load_model, save_model
    from .student_cycle import method_totals

    settings = config["factor_compress"]
    source = checked_json(settings["source_report"])
    fit_keys, pilot_keys = settings["fit_keys"], settings["pilot_keys"]
    validation = config["training"]["validation_keys"]
    if (not source["comparisons"]["solver"]["compression"]
            or source["candidates"]["solver"]["model"] != config["factor_model"]
            or source["keys"] != validation
            or not set(pilot_keys) <= set(fit_keys)
            or len(fit_keys) != len(set(fit_keys))
            or any(frames[k]["split"] != "training" for k in fit_keys)):
        raise ValueError("Compression needs the qualified solver and disjoint training inputs")
    baseline = load_model(config["factor_model"])
    if baseline.product_compression is not None or baseline.coefficient_weights is None:
        raise ValueError("The qualified solver needs uncompressed independent coefficient networks")

    def method(name, model):
        return dict(name=name, factor_model=model, factor_context=dict(factor_fusion=True))

    local_samples, receipts, references = [], [], {}
    for index, key in enumerate(fit_keys):
        if index % config["parallel_workers"] != p.worker:
            continue
        loaded = load(key)
        try:
            probes = select_probes(load_probe_shards(p, source["probe_bank"][key]), settings["probes"])
            reference_probes = [] if key in pilot_keys else None
            rng = np.random.default_rng([settings["seed"], index, p.rank])

            def observe(displacement):
                local_samples.extend(sample_product_rows(p, bridge, displacement, loaded[0],
                    probes, rng, settings["rows_per_rank"], reference_probes))
                return dict(failure=None)

            value = solve(key, loaded, dict(method("fit_support", baseline),
                factor_context=dict(factor_fusion=True, factor_operation=observe)), "compression-fit", details=False)
            if value["failure"] is None and reference_probes is not None:
                references[key] = reference_probes
            if p.rank == 0:
                receipts.append(dict(key=key, failure=value["failure"], probes=[probe[0] for probe in probes]))
        finally:
            destroy(loaded)
    fit = share(receipts)
    if len(fit)!=len(fit_keys) or any(row["failure"] is not None for row in fit):
        raise ValueError("Unable to rebuild the qualified reference on all compression frames")
    support, count = learn_support(p, share, local_samples, settings["ridge"])
    if p.global_rank == 0:
        record(config, "factor_compress_support_fit", job=job, keys=fit_keys, rows=count,
               support=support.tolist())

    models, trials = {}, {}
    reference = method("uncompressed", baseline)
    output = Path(config["output_directory"])
    states = {}
    try:
        for key in references:
            states[key] = load(key)
        for index, tolerance in enumerate(settings["tolerances"], 1):
            label = "compressed-%d" % index
            budget = learn_budget(p, share, local_samples, support, tolerance, settings["ridge"])
            trial = replace(baseline, product_compression=np.r_[support, budget])
            model, history = refit_coefficients(p, legacy, bridge, config, trial, states, references,
                                               solve, share, settings, job, label)
            checkpoint = []
            if p.global_rank == 0:
                checkpoint.append(save_model(output/"checkpoints"/("generator-%s-%s.npz" % (job,label)), model))
            receipt, = share(checkpoint)
            trials[label] = dict(tolerance=tolerance, budget=budget.tolist(), model=receipt, history=history,
                                 accepted_updates=sum(r["accepted"] for r in history), qualified=False)
            if history and history[0].get("stop") not in ("invalid_hierarchy", "directional_derivative"):
                models[label] = method(label, model)
    finally:
        for loaded in states.values():
            destroy(loaded)

    pilot_cases = evaluate(pilot_keys, [reference, *models.values()], "compression-pilot")
    pilot_totals = method_totals(pilot_cases)
    base = pilot_totals[reference["name"]]

    def fill_ok(a, b):
        return all(a[k] <= (1.-settings["min_fill_reduction"])*b[k] for k in ("p1_nnz", "k1_nnz"))

    def passed(total, keys):
        return total["frames"] == len(keys) and not total["failed"] and not total["fallback_count"]

    for label in models:
        a = pilot_totals[label]
        trials[label]["qualified"] = (passed(a, pilot_keys) and passed(base, pilot_keys)
            and a["iterations"] <= settings["pilot_iteration_ratio"]*base["iterations"] and fill_ok(a, base))
    shortlist = [k for k in models if trials[k]["qualified"]]
    selected = min(shortlist, key=lambda label: (pilot_totals[label]["total_s"], label)) if shortlist else None
    if p.global_rank == 0:
        record(config, "factor_compress_pilot", job=job, selected=selected, totals=pilot_totals)

    paired, repeats, speed = [], [], False
    if selected:
        paired = evaluate(validation, [reference, models[selected], dict(config["factor_reference"], fallback=False)],
                          "compression-final", repeats=2)
        repeats = [method_totals([r for r in paired if r["repeat"]==i]) for i in range(2)]
        speed = all(all(passed(totals[label], validation) for label in (selected, "uncompressed", "student"))
            and fill_ok(totals[selected], totals["uncompressed"])
            and totals[selected]["iterations"] <= min(totals["student"]["iterations"],
                settings["validation_iteration_ratio"]*totals["uncompressed"]["iterations"])
            and totals[selected]["total_s"] <= (1.-settings["speed_margin"])*totals["student"]["total_s"]
            for totals in repeats)
    return dict(source=settings["source_report"], initial_model=config["factor_model"], fit=fit,
                support=support.tolist(), training_rows=count, pilot_keys=pilot_keys, pilot_cases=pilot_cases,
                pilot_totals=pilot_totals, trials=trials, selected=selected, keys=validation, cases=paired,
                repeat_totals=repeats, speed_gate=speed)
