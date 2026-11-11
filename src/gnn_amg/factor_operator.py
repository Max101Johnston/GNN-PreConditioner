"""Structure supervision and hard two-level displacement operator training."""
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .backend import Hierarchy, digest, local_csr, neighbour_blocks, read_json, record
from .dataset import ATTRIBUTE_NAMES
from .factor_search import SearchProposal, probabilities
from .factorized import (DEPTHS, MAX_SUPPORT, FactorModel, candidate_context, candidate_context_pullback,
                         edge_pullback, factor_cotangent,
                         frozen_transfer, hierarchy_signature, load_model, normalized_coordinates,
                         packed_gradient, region_ids, save_model, support_mask, support_order)
from .krylov import HierarchyAdjoint
from .student_cycle import method_totals, save_once


def checked_json(receipt):
    if digest(receipt["path"]) != receipt["sha256"]:
        raise ValueError("Operator input receipt changed: " + receipt["path"])
    return read_json(receipt["path"])


def recovered_plans(settings, origin):
    report = checked_json(settings["search_report"])
    if report["job"] != settings["search_job"] or report["model"] != origin:
        raise ValueError("Search supervision uses a different origin checkpoint")
    plans, margin_audit, saved = [], [], []
    for rr, receipt in zip(report["rounds"], settings["screens"]):
        screens = {row["key"]: row for row in checked_json(receipt)}
        rebuilt = {(r["key"], r["method"]): r["examples"] for r in rr["restored_labels"] if r["matches"]}
        for row in rr["checked"]:
            if not row["accepted"] or row["method"] == "coefficient":
                continue
            key, name = row["key"], row["method"]
            case = row["candidate"]
            plan = screens[key]["proposals"][int(name.split("-")[-1])]
            decision = case["factor_policy"]
            plans.append(dict(key=key, method=name, round=rr["round"], policy=plan,
                              hierarchy=case["hierarchy_signature"], actions=decision["signature"],
                              expected_changes=decision["changes"]))
            saved.extend(rebuilt.get((key, name), decision["examples"]))
        lower = []
        for example in saved:
            if example["unchanged"]:
                continue
            features, logits = np.asarray(example["features"]), np.asarray(example["logits"])
            winner, desired = int(np.argmax(logits)), example["chosen"]
            gap = float(logits[winner]-logits[desired])
            separation = float(np.linalg.norm(features[winner]-features[desired]))
            lower.append(gap/separation if separation > 0 else (np.inf if gap else 0.))
        finite = [x for x in lower if np.isfinite(x)]
        margin_audit.append(dict(round=rr["round"], examples=len(saved), targets=len(lower),
                                 attainable_at_004=sum(x<=.04 for x in lower),
                                 median_required=float(np.median(finite)) if finite else None))
    return plans, margin_audit


class ContextProposal(SearchProposal):
    """Collect every changed decision on its real candidate graph."""

    capture_context = True

    def __init__(self, p, model, settings, guards):
        super().__init__(p, model, dict(settings, examples=0))
        self.example_limit = guards
        self.decisions = []
        self.pending = []
        self.archive = {}

    def representative_choice(self, features, scores, chosen, original, native, record,
                              inputs=None, regions=None):
        if chosen != original:
            self.added.append(native[chosen])
        if record:
            self.decisions.append(dict(kind="representative", stage=self.stage, domain=native,
                fine=inputs, regions=regions, logits=scores, chosen=int(chosen),
                unchanged=bool(chosen==original), temperature=self.temperature))

    def example(self, layer, features, scores, chosen, original, ids, row=None):
        self.pending.append(dict(kind="support" if layer==4 else "budget", row=int(row),
                                 domain=np.asarray(ids), logits=np.asarray(scores), chosen=int(chosen),
                                 unchanged=bool(chosen==original), temperature=self.temperature))

    def observe_edges(self, p, stage, x, owned_native, selected, active, owners, columns,
                      native, edges, valid, same, keep, scores, quota, regions):
        pending = self.pending if stage == self.stage else []
        active_counts = keep.sum(1)
        available = valid.sum(1)
        used = np.asarray([e["row"] for e in pending], dtype=np.int64)
        background = np.flatnonzero((regions != self.region) & (active_counts >= 2)
                                    & (available > active_counts))
        background = np.setdiff1d(background, used, assume_unique=False)
        if len(background):
            rows = background[np.linspace(0, len(background)-1,
                                          min(len(background), self.example_limit), dtype=int)]
            order = support_order(scores[rows], valid[rows], same[rows])
            for i, row in enumerate(rows):
                slots = np.r_[order[i,active_counts[row]-1], np.flatnonzero(valid[row] & ~keep[row])]
                pending.append(dict(kind="support", row=int(row), domain=native[columns[row, slots]],
                                    logits=scores[row, slots], chosen=0, unchanged=True,
                                    temperature=self.temperature))
                budgets = np.arange(1, min(MAX_SUPPORT, available[row])+1)
                pending.append(dict(kind="budget", row=int(row), domain=budgets,
                                    logits=-(budgets-3.-quota[row])**2, chosen=int(active_counts[row]-1),
                                    unchanged=True, temperature=self.temperature))
        rows = np.unique([e["row"] for e in pending]).astype(np.int64)
        neighbours = neighbour_blocks(p, owners, x[selected], columns[rows])
        for e in pending:
            row = e.pop("row")
            candidate_ids = np.where(valid[row], native[np.maximum(columns[row], 0)], -1)
            if e["kind"]=="support":
                slots = np.flatnonzero(np.isin(candidate_ids, e["domain"]))
                lookup = {candidate_ids[slot]:slot for slot in slots}
                e["options"] = np.asarray([lookup[j] for j in e["domain"]], dtype=np.int64)
            e.update(stage=stage, fine=x[active[row]].copy(), coarse=neighbours[np.searchsorted(rows, row)].copy(),
                     edge=edges[row].copy(), valid=valid[row].copy(), same=same[row].copy(), regions=int(regions[row]),
                     keep=keep[row].copy(), native=int(owned_native[active[row]]), candidates=candidate_ids)
            self.decisions.append(e)
        if stage == self.stage:
            self.pending = []

    def archive_stage(self, stage, native, selected, rows, owners, csr):
        prefix = "factor%d_" % (stage+1)
        self.archive.update({prefix+name: value for name,value in dict(
            native=native, selected=native[selected], row_owners=rows, coarse_owners=owners,
            indptr=csr.indptr, indices=csr.indices, values=csr.data).items()})

    def export(self, path, displacement, bridge):
        matrix, _ = local_csr(self.p, displacement.first.coarse, bridge)
        # Each decision keeps its full legal candidate row and stable native IDs.
        payload = json.dumps([{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in row.items()}
                              for row in self.decisions], allow_nan=False).encode()
        with path.open("xb") as stream:
            np.savez_compressed(stream, k1_indptr=matrix.indptr, k1_indices=matrix.indices,
                k1_values=matrix.data, decisions=np.frombuffer(payload, np.uint8), **self.archive)
        return digest(path)


def collect_structure(config, p, bridge, settings, model, load, destroy, solve, share, job):
    local = []
    if p.global_rank == 0:
        plans, audit = recovered_plans(settings, config["factor_model"])
        if not plans:
            raise ValueError("No verified winners are available for structure training")
        root = Path(config["output_directory"])/"data"/("generator-context-"+job)
        root.mkdir()
        local = [dict(path=str(root), plans=plans, margin_audit=audit)]
    inventory = share(local)[0]
    records, examples = [], []
    for index, plan in enumerate(inventory["plans"]):
        if index % config["parallel_workers"] != p.worker:
            continue
        key = plan["key"]
        policy = ContextProposal(p, model, plan["policy"], settings["guard_rows_per_rank"])
        path = Path(inventory["path"])/("%02d-r%03d.npz" % (index, p.rank))

        def capture(displacement):
            signature = hierarchy_signature(displacement.numerical)
            receipt = policy.receipt()
            matched = (signature == tuple(plan["hierarchy"]) and receipt["signature"] == plan["actions"]
                       and receipt["changes"] == plan["expected_changes"])
            if not matched:
                return dict(failure="changed_winner_signature", numerical=displacement.numerical)
            sha = policy.export(path, displacement, bridge)
            count = sum(not row["unchanged"] for row in policy.decisions)
            index_rows = p.gather_equal(np.r_[np.frombuffer(bytes.fromhex(sha), np.uint8),
                                              len(policy.decisions), count])
            return dict(failure=None, shards=[dict(path="%02d-r%03d.npz" % (index,rank),
                        sha256=values[:32].astype(np.uint8).tobytes().hex(), rows=int(values[32]),
                        changed=int(values[33])) for rank,values in enumerate(index_rows)],
                        hierarchy=signature, actions=receipt["signature"])

        loaded = load(key)
        try:
            got = solve(key, loaded, dict(name="structure_context", factor_model=model,
                    factor_context=dict(factor_operation=capture, factor_policy=policy)),
                    "operator-context", details=False)
        finally:
            destroy(loaded)
        if got["failure"] is None:
            examples.extend(policy.decisions)
        if p.rank == 0:
            records.append(dict(plan, **got))
    receipts = share(records)
    if p.global_rank == 0:
        save_once(Path(config["output_directory"])/"records"/("generator-context-%s.json" % job),
                  dict(inventory, receipts=receipts))
        record(config, "generator_context", job=job, proposals=len(receipts),
               matched=sum(r["failure"] is None for r in receipts), margin_audit=inventory["margin_audit"])
    return examples, receipts, inventory["margin_audit"]


def structure_example(model, example, differentiate):
    kind, stage, temperature = example["kind"], example["stage"], example["temperature"]
    x = example["fine"]
    if kind == "representative":
        hidden, logits = model.node_outputs(x, stage, example["regions"])
        scores = logits[:,0]
        hard_match = None
    else:
        node, coarse_x = x[None], example["coarse"][None]
        fine = np.maximum(0, model.linear(0, node))
        coarse = np.maximum(0, model.linear(0, coarse_x))
        edges, valid, same = (example[key][None] for key in ("edge","valid","same"))
        tape = [] if differentiate else None
        output, quota = model.edge_outputs(fine, coarse, edges, valid, stage,
                                           np.array([example["regions"]]), tape)
        domain = example["domain"]
        if kind == "support":
            scores = output[0, example["options"], 0]
        else:
            scores = -(domain-3.-quota[0])**2
        keep = support_mask(output[:,:,0], quota, valid, same)[0]
        hard_match = bool(np.array_equal(keep, example["keep"]))
    p = probabilities(scores/temperature)
    target = probabilities(example["logits"]/temperature) if example["unchanged"] else np.eye(1,len(p),example["chosen"])[0]
    derivative = (p-target)/temperature
    entropy = -np.dot(target, np.log(np.maximum(p, 1e-300)))
    if example["unchanged"]:
        entropy += np.dot(target, np.log(np.maximum(target, 1e-300)))
    gradient = None
    if differentiate:
        if kind == "representative":
            gradients = [(np.zeros(w.shape),np.zeros(b.shape)) for w,b in model.layers]
            gradients[1][0][0] = derivative @ hidden
            gradients[1][1][0] = derivative.sum()
            adapter = model.adapters[stage, example["regions"],0]/model.width
            d_node = derivative[:,None]*(model.layers[1][0][0]+adapter[:,None])*(hidden>0)
            gradients[0] = d_node.T @ x, d_node.sum(0)
        else:
            d_output = np.zeros(output.shape)
            d_quota = None
            if kind=="support":
                d_output[0,example["options"],0] = derivative
            else:
                d_quota = np.array([np.dot(derivative, 2*(domain-3.-quota[0]))])
            gradients, a, b = edge_pullback(model, fine, coarse, edges, valid, tape, d_output, d_quota)
            a *= fine>0
            b *= coarse>0
            b = b.reshape((-1,model.width))
            gradients[0] = a.T @ node + b.T @ coarse_x.reshape((-1,node.shape[-1])), a.sum(0)+b.sum(0)
        gradient = packed_gradient(gradients)
    chosen = bool(np.argmax(scores)==example["chosen"])
    return float(entropy), chosen, hard_match, gradient


def structure_batch(p, model, examples, share, differentiate):
    counts = np.zeros((2,5))
    gradient = np.zeros((2,model.weights.size))
    for row in examples:
        loss, match, exact, direction = structure_example(model,row,differentiate)
        group = int(row["unchanged"])
        counts[group] += (1., loss, match, exact is not None, bool(exact))
        if differentiate:
            gradient[group] += direction
    block = np.c_[counts, gradient]
    packed = p.reduce(block.ravel()).tolist()
    total = np.sum(share([packed]),axis=0).reshape(block.shape)
    stats = []
    for row in total:
        n, loss, match, hard, exact = row[:5]
        stats.append(dict(examples=int(n), loss=float(loss/max(n,1.)),
                          match=float(match/max(n,1.)), hard_rows=int(hard),
                          hard_exact=float(exact/max(hard,1.))))
    return stats, total[:,5:]/np.maximum(total[:,0,None],1.)


def fit_structure(config, p, model, examples, settings, share, job):
    weights = model.weights.copy()
    first, second = np.zeros_like(weights,np.float64), np.zeros_like(weights,np.float64)
    best = model
    history = []
    score = None
    for step in range(settings["structure_steps"]+1):
        trial = FactorModel(model.width, weights.copy(), model.adapters, model.unit_sum_depth)
        stats, gradient = structure_batch(p,trial,examples,share,step<settings["structure_steps"])
        if step==0:
            if not all(row["examples"] for row in stats):
                raise ValueError("Structure training needs both changed decisions and behaviour guards")
            guard = max(0., stats[1]["match"]-settings["allowed_guard_loss"])
        metric = (stats[0]["match"], stats[0]["hard_exact"], -stats[0]["loss"])
        if score is None or (stats[1]["match"]>=guard and metric>score):
            score, best, best_stats, best_step = metric, trial, stats, step
        history.append(dict(step=step, target=stats[0], guard=stats[1]))
        if p.global_rank == 0 and (step%10==0 or step==settings["structure_steps"]):
            record(config,"generator_structure_step",job=job,step=step,target=stats[0],guard=stats[1])
        if step == settings["structure_steps"]:
            break
        gradient = gradient[0]+settings["guard_weight"]*gradient[1]
        if not np.isfinite(gradient).all():
            raise ValueError("Non-finite full structure-network gradient")
        first = .9*first+.1*gradient
        second = .999*second+.001*gradient**2
        direction = (first/(1-.9**(step+1)))/(np.sqrt(second/(1-.999**(step+1)))+1e-8)
        weights -= (settings["structure_lr"]*direction).astype(weights.dtype)
    return best, dict(initial=history[0], selected_step=best_step, selected=best_stats,
                      guard_floor=guard, history=history)


def make_probes(p, loaded, observed, solve, key, seed, settings, teacher="retained_base", provenance=None,
                all_observed=False):
    layout, fields, _ = loaded
    lo, hi = layout["ownership"][p.rank:p.rank+2]
    native = (3*layout["native_node"]+layout["component"])[lo:hi].astype(np.int64)
    regions = region_ids(normalized_coordinates(layout["xyz"])[native//3])
    components = native % 3
    groups = 3*regions+components
    active = ~fields["constrained"][native]
    actual = [(f"pcapply-{call}", residual) for call,residual in observed
              if all_observed or call in (0,8,24,48) or call==observed[-1][0]]
    rng = np.random.default_rng(np.random.SeedSequence([seed, p.rank]))
    counts = p.reduce(np.bincount(groups[active], minlength=24))
    random = []
    for component in range(3):
        available = np.flatnonzero(counts[component::3]>0)
        if not len(available):
            continue
        region = int(available[(seed+component)%len(available)])
        mask = active & (groups==3*region+component)
        residual = np.zeros(len(native))
        residual[mask] = rng.standard_normal(int(mask.sum()))
        random.append((f"region-{region}-component-{component}",residual))
    candidates = [(name,r) for name,r in actual+random if p.reduce([r@r])[0]>0]
    if not actual or not candidates:
        raise ValueError("Missing actual displacement PCApply inputs on "+key)
    target = solve(key,loaded,dict(name=teacher,displacement_inputs=[r for _,r in candidates]),
                   "operator-teacher",details=False)
    if target is None or target["failure"] is not None or len(target["action_targets"])!=len(candidates):
        raise ValueError("Could not apply the displacement teacher on "+key)
    if provenance is not None:
        provenance.update(method=teacher,numerical=target["numerical"])
    probes = []
    for (name,residual), answer in zip(candidates,target["action_targets"]):
        energy = p.reduce(np.bincount(groups,weights=answer**2,minlength=24))
        if not np.isfinite(energy).all() or energy.sum()<=0:
            raise ValueError("Invalid displacement teacher action on "+key+"/"+name)
        floor = max(settings["scale_floor"]*energy.mean(),1e-300)
        weights2 = 1./np.maximum(energy,floor)
        normalization = float(weights2@energy)
        probes.append((name,residual,answer,weights2[groups],normalization))
    return probes


def archive_probes(p, directory, key, phase, probes):
    path = directory/("%s-%s-r%03d.npz" % (key,phase,p.rank))
    names, inputs, answers, weights2, normalization = zip(*probes)
    with path.open("xb") as stream:
        np.savez_compressed(stream,names=names,inputs=inputs,teacher=answers,
                            weights_squared=weights2,normalization=normalization)
    sha = np.frombuffer(bytes.fromhex(digest(path)),np.uint8)
    checksums = p.gather_equal(sha).astype(np.uint8)
    return dict(key=key,phase=phase,probes=list(names),
                shards=[r.tobytes().hex() for r in checksums])


def action_losses(p, displacement, probes, adjoint=None, probe_weights=None):
    losses, dot = {}, []
    for name,residual,teacher,weights2,normalization in probes:
        tape = [] if adjoint is not None else None
        answer = displacement.correction(residual,tape)
        difference = answer-teacher
        loss = float(p.reduce([difference@(weights2*difference)])[0]/normalization)
        if not np.isfinite(loss):
            raise ValueError("Non-finite displacement action objective")
        losses[name] = loss
        if adjoint is not None:
            scale = (2./(normalization*len(probes)) if probe_weights is None
                     else 2.*probe_weights[name]/normalization)
            output = scale*weights2*difference
            pullback = adjoint.transpose(output,tape)
            left,right,u2,z2,g2,b2 = p.reduce(
                [output@answer,residual@pullback,residual@residual,answer@answer,output@output,pullback@pullback])
            dot.append(float(abs(left-right)/max(abs(left),abs(right),1e-6*np.sqrt(u2*b2+z2*g2),1e-300)))
    return losses, dot


def fixed_action_value(p, legacy, bridge, displacement, model, probes):
    first = second = space = None
    try:
        first = frozen_transfer(p,legacy,bridge,displacement.first,displacement.fine.matrix,
                                model.coefficients_at(0))
        second = frozen_transfer(p,legacy,bridge,displacement.second,first.coarse,
                                 model.coefficients_at(DEPTHS[0]))
        space = Hierarchy(p,legacy,first,second,int(second.owners[-1]))
        values,_ = action_losses(p,replace(displacement,first=first,second=second,space=space),probes)
        return float(np.mean(list(values.values())))
    finally:
        for resource in (space,second,first):
            if resource is not None:
                resource.close()


def directional_slopes(center, minus, plus, step, analytic):
    slopes = dict(minus=(center-minus)/step, plus=(plus-center)/step,
                  central=(plus-minus)/(2*step))
    errors = {name:abs(value-analytic)/max(abs(value),abs(analytic),1e-300)
              for name,value in slopes.items()}
    return dict(step=step, center=center, minus=minus, plus=plus,
                slopes=slopes, relative_errors=errors)


def attention_factor_point(p, factor, incoming, model, direction, steps):
    """Audit the fixed local map with the actual Galerkin cotangent."""
    dtype = np.float64
    precise = replace(model, weights=model.weights.astype(dtype),
                      adapters=model.adapters.astype(dtype),
                      context_parameters=model.context_parameters.astype(dtype))
    factor = replace(factor, csr=factor.csr.astype(dtype, copy=False))
    d_summary, base_network, production_network = [], [], []
    gradient = factor.backward(p,precise,incoming,context_dtype=dtype,summary_cotangents=d_summary)
    fine = factor.network(p,precise,base_network,context_dtype=dtype)[1]
    hidden, summary, _, context_tape = base_network
    parameters = precise.context_parameters
    _, _, d_pool = candidate_context_pullback(
        "attention",parameters,fine,hidden,factor.valid,context_tape,d_summary[0])
    analytic = p.reduce([np.sum(d_pool*direction),gradient[-model.context_size:]@direction.ravel()])
    base = factor.fixed_matrix(p,precise,context_dtype=dtype)
    logit = factor.network(p,model,production_network)[0][:,:,1]
    primary = factor.keep & factor.same
    if factor.stage < model.unit_sum_depth:
        primary = np.where(primary.any(1,keepdims=True),primary,factor.keep)
    at_center = production_network[2] > 0
    peak = lambda array: p.maximum(np.max(np.abs(array),initial=0.))
    measurements = []
    for step in steps:
        local, crossings = [], []
        for sign in (-1,1):
            shifted = parameters+sign*step*direction
            pooled,_ = candidate_context("attention",shifted,fine,hidden,factor.valid,dtype=dtype)
            trial = replace(precise, context_parameters=shifted)
            csr = factor.fixed_matrix(p,trial,context_dtype=dtype)
            local.append(p.reduce([np.sum(d_summary[0]*(pooled-summary)), incoming@(csr.data-base.data)]))

            tape = []
            trial32 = replace(model, context_parameters=shifted.astype(model.weights.dtype))
            logits = factor.network(p,trial32,tape)[0][:,:,1]
            relu = ((tape[2]>0)!=at_center) & factor.keep[:,:,None]
            clip = (((logits<=-12)!=(logit<=-12)) | ((logits>=12)!=(logit>=12))) & primary
            n_relu, n_clip = map(int,p.reduce([relu.sum(),clip.sum()]))
            crossings.append(dict(sign=sign,relu=n_relu,clip=n_clip,
                                  max_logit_change=peak((logits-logit)[factor.keep])))
        measurements.append(dict(step=step, production_crossings=crossings, **{
            name:directional_slopes(0.,float(local[0][i]),float(local[1][i]),step,float(analytic[i]))
            for i,name in enumerate(("pooling_fp64","signed_coefficients_fp64"))}))
    return dict(stage=factor.stage, pooling_ad=float(analytic[0]), coefficients_ad=float(analytic[1]),
                measurements=measurements)


def diagnose_attention_point(p, legacy, bridge, displacement, model, probes, expected, settings):
    """Keep the original decoded tapes and revisit only the failed level."""
    begin = time.perf_counter()
    signature = hierarchy_signature(displacement.numerical)
    if list(signature)!=expected["signature"]:
        raise ValueError("The failed frame's fixed hierarchy did not reproduce")
    original = expected["directional_derivatives"]
    level_index = 1
    leaf = model.coefficients_at(DEPTHS[0])
    tapes = []
    adjoint = HierarchyAdjoint(p,bridge,displacement)
    try:
        losses,dot = action_losses(p,displacement,probes,adjoint)
        if list(losses)!=list(expected["probes"]) or not all(
                np.isclose(loss,expected["probes"][name],rtol=2e-7,atol=1e-10)
                for name,loss in losses.items()):
            raise ValueError("The original probe actions did not reproduce")
        if max(dot,default=0.)>settings["adjoint_tolerance"]:
            raise ValueError("The failed frame's transpose parity did not reproduce")
        transfer = adjoint.transfers(legacy)[level_index]
        gradient = p.reduce(factor_cotangent(p,legacy,bridge,displacement.second,transfer,leaf,
                                             observe=lambda tape, entries: tapes.append((tape,entries))))
    finally:
        adjoint.close()
    vector = gradient[-leaf.context_size:]
    norm = float(np.linalg.norm(vector))
    center = fixed_action_value(p,legacy,bridge,displacement,model,probes)
    if not (np.isclose(norm,original["levels"][level_index]["analytic"],rtol=2e-6,atol=1e-10)
            and np.isclose(center,original["center"],rtol=2e-7,atol=1e-10)):
        raise ValueError("The archived context direction or center did not reproduce")
    direction = vector/norm
    steps = settings["steps"]
    factors = [attention_factor_point(p,tape,entries,leaf,direction.reshape(2,leaf.width),steps)
               for tape,entries in tapes]
    measurements = []
    for step in steps:
        values = []
        for sign in (-1,1):
            shifted = model.coefficient_weights.copy()
            shifted[level_index,-leaf.context_size:] += sign*step*direction
            trial = replace(model,coefficient_weights=shifted)
            values.append(fixed_action_value(p,legacy,bridge,displacement,trial,probes))
        measurements.append(directional_slopes(center,*values,step,norm))
    return dict(signature=signature, probes=list(losses), loss=float(np.mean(list(losses.values()))),
                level=level_index+1, analytic=norm, direction=direction.tolist(), dot_errors=dot,
                local_factors=factors, full_operator_fp32=measurements,
                original_center=original["center"], original_level=original["levels"][level_index],
                seconds=p.maximum(time.perf_counter()-begin))


def audit_operator_gradient(p, legacy, bridge, displacement, model, probes, value, gradients, settings):
    epsilon = settings["finite_difference"]
    multipliers = settings.get("context_fd_multipliers", (1.,)) if model.context_size else (1.,)
    tolerance = settings["finite_tolerance"]
    center = fixed_action_value(p,legacy,bridge,displacement,model,probes)
    checks = []
    stable = True
    for level,gradient in enumerate(gradients):
        if model.context_size:
            gradient = gradient.copy()
            gradient[:-model.context_size] = 0.
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(norm) or norm<=1e-12:
            raise ValueError("The coefficient encoder lacks a usable level gradient")
        direction = gradient/norm
        measurements = []
        for multiplier in multipliers:
            step = epsilon*multiplier
            results = []
            for sign in (-1,1):
                shifted = model.coefficient_weights.copy()
                shifted[level] = (shifted[level]+sign*step*direction).astype(shifted.dtype)
                trial = replace(model, coefficient_weights=shifted)
                results.append(fixed_action_value(p,legacy,bridge,displacement,trial,probes))
            finite = (results[1]-results[0])/(2*step)
            error = abs(finite-norm)/max(abs(finite),norm)
            measurements.append(dict(step=step, finite=finite, relative_error=error,
                                     minus=results[0], plus=results[1]))
        selected = measurements[0]
        window = None
        if len(measurements)>1:
            # Require a neighboring pair of resolved slopes. A single match
            # can be caused by roundoff or by using too large a perturbation.
            for low,high in zip(measurements,measurements[1:]):
                a,b = low["finite"],high["finite"]
                agreement = abs(a-b)/max(abs(a),abs(b),norm)
                if max(low["relative_error"],high["relative_error"])<tolerance and agreement<tolerance/2:
                    window = dict(steps=[low["step"],high["step"]], relative_difference=agreement)
                    selected = low
                    break
            stable = stable and window is not None
        row = dict(level=level+1, analytic=norm, finite=selected["finite"],
                   relative_error=selected["relative_error"])
        if len(measurements)>1:
            row.update(step_measurements=measurements, stable_window=window)
        checks.append(row)
    center_error = abs(center-value)/max(1.,abs(value))
    return dict(center=center,center_relative_error=center_error,levels=checks,
                passed=bool(stable and center_error<settings["center_tolerance"] and
                            all(row["relative_error"]<tolerance for row in checks)),
                scope="Executed factors, Galerkin products and coarse solves on fixed decoded feature tapes.",
                parameter_scope="candidate_context" if model.context_size else "coefficient_network")


def operator_observation(p, legacy, bridge, displacement, model, probes, gradient, settings,
                         audit=False, objective=None):
    begin = time.perf_counter()
    result = dict(failure=None, signature=hierarchy_signature(displacement.numerical),
                  structure=hierarchy_signature(displacement.numerical,("representatives","support")),
                  nonzeros=[displacement.first.stats["nnz"],displacement.numerical["k1_nnz"],
                            displacement.second.stats["nnz"]])
    adjoint = HierarchyAdjoint(p,bridge,displacement) if gradient else None
    try:
        if objective is None:
            losses,dot = action_losses(p,displacement,probes,adjoint)
            value = float(np.mean(list(losses.values())))
        else:
            losses,_ = action_losses(p,displacement,probes)
            value,weights,breakdown = objective(losses)
            result["objective"] = breakdown
            dot = action_losses(p,displacement,probes,adjoint,weights)[1] if gradient else []
        result.update(loss=value, probes=losses)
        if gradient:
            if max(dot,default=0.)>settings["adjoint_tolerance"]:
                raise ValueError("Displacement transpose failed the probe dot-product audit")
            transfers = adjoint.transfers(legacy)
            vectors = np.stack([p.reduce(factor_cotangent(p,legacy,bridge,level,cotangent,
                                                         model.coefficients_at(stage)))
                                for level,cotangent,stage in zip((displacement.first,displacement.second),
                                                                  transfers,(0,DEPTHS[0]))])
            if not np.isfinite(vectors).all():
                raise ValueError("Non-finite full coefficient-network gradient")
            result.update(gradient=vectors.tolist(), dot_errors=dot, ras_dot_error=adjoint.ras_dot_error,
                          ilu=adjoint.ilu_parity)
            if audit:
                result["directional_derivatives"] = audit_operator_gradient(
                    p,legacy,bridge,displacement,model,probes,value,vectors,settings)
        result["training_seconds"] = p.maximum(time.perf_counter()-begin)
        return result
    finally:
        if adjoint is not None:
            adjoint.close()


def layer_norms(model, vector):
    offset, norms = 0,[]
    for w,b in model.layers:
        size = w.size+b.size
        norms.append(float(np.linalg.norm(vector[offset:offset+size])))
        offset += size
    return norms


def split_model(structure, origin):
    return FactorModel(structure.width,structure.weights,structure.adapters,structure.unit_sum_depth,
                       np.stack([origin.coefficients_at(stage).weights.copy() for stage in (0,DEPTHS[0])]))


def displacement_fit(config, p, legacy, bridge, settings, origin, seed, states, batches, fixed_batches,
                     solve, share, refresh, job):
    def measure(model, active, gradient=False, audit=False):
        local = []
        for key,loaded in states.items():
            def observation(displacement):
                return operator_observation(p,legacy,bridge,displacement,model,active[key],
                                            gradient,settings,audit)
            answer = solve(key,loaded,dict(name="operator_action",factor_model=model,
                factor_context=dict(factor_tape=gradient,factor_operation=observation)),
                "operator-action",details=False)
            if p.rank == 0:
                local.append(dict(key=key,**answer))
        return share(local)

    def accepted(base, trial):
        current = {row["key"]:row for row in base}
        if len(trial)!=len(base) or any(row["failure"] is not None for row in trial):
            return False
        if any(row["loss"] > current[row["key"]]["loss"]*(1+settings["max_frame_loss_increase"])
               or any(nnz>limit for nnz,limit in zip(row["nonzeros"],fill_ceiling[row["key"]]))
               for row in trial):
            return False
        return sum(r["loss"] for r in trial) < sum(r["loss"] for r in base)*(1-settings["min_step_improvement"])

    model, history = seed, []
    first_pass = None
    fill_ceiling = {}
    initial_origin = measure(split_model(origin,origin), fixed_batches)
    step_size = settings["coefficient_step"]
    updates = 0
    for step in range(settings["coefficient_steps"]):
        if step == settings["refresh_at"] and updates:
            batches = refresh(model)
        source = measure(model,batches,gradient=True,audit=step==0)
        if any(r["failure"] is not None for r in source):
            raise ValueError("Could not build a hard learner for coefficient training")
        if step==0:
            first_pass = source
            for row in source:
                fill_ceiling[row["key"]] = [max(nnz*settings["fill_guard"],nnz+1) for nnz in row["nonzeros"]]
            if not all(r["directional_derivatives"]["passed"] for r in source):
                return model,dict(status="directional_derivative_gate",initial_origin=initial_origin,
                                  first_pass=source,history=[],updates=0, final=measure(model,fixed_batches))
        gradient = np.mean([r["gradient"] for r in source],axis=0)
        if step < settings["second_only_steps"]:
            gradient[0] = 0.
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(norm) or norm<1e-12:
            break
        attempts = []
        for fraction in (1.,.5,.25):
            weights = (model.coefficient_weights - step_size*fraction*gradient/norm).astype(model.weights.dtype)
            trial = FactorModel(model.width,model.weights,model.adapters,model.unit_sum_depth,weights)
            rows = measure(trial,batches)
            current = {r["key"]:r for r in source}
            for row in rows:
                if row["failure"] is None:
                    row["structure_stages_changed"] = [
                        i//2+1 for i in range(0,len(row["structure"]),2)
                        if row["structure"][i:i+2] != current[row["key"]]["structure"][i:i+2]]
            ok = accepted(source,rows) and (step>=settings["second_only_steps"]
                 or all(not row["structure_stages_changed"] for row in rows))
            attempts.append(dict(step_size=step_size*fraction,accepted=ok,cases=rows))
            if ok:
                updates += 1
                model = trial
                step_size = min(settings["max_coefficient_step"],1.2*step_size*fraction)
                break
        else:
            step_size *= .5
        entry = dict(step=step+1, source=[{k:v for k,v in r.items() if k!="gradient"} for r in source],
                     attempts=attempts, gradient_norm=norm,
                     gradient_layers=[layer_norms(model,part) for part in gradient])
        history.append(entry)
        if p.global_rank == 0:
            record(config,"generator_operator_step",job=job,step=step+1,accepted=attempts[-1]["accepted"],
                   before=[r["loss"] for r in source],
                   after=[r.get("loss") for r in attempts[-1]["cases"]], gradient_norm=norm)
        if step_size<settings["min_coefficient_step"]:
            break
    final = measure(model,fixed_batches)
    initial_seed = first_pass or measure(seed,fixed_batches)
    return model,dict(status="trained" if updates else "no_accepted_update", initial_origin=initial_origin,
                      first_pass=[{k:v for k,v in r.items() if k!="gradient"} for r in initial_seed],
                      final=final, history=history, updates=updates,
                      layer_delta=[layer_norms(model,delta) for delta in model.coefficient_weights-seed.coefficient_weights])


def run_operator(config, p, legacy, bridge, frames, load, destroy, solve, share, evaluate, job):
    settings = config["factor_operator"]
    output = Path(config["output_directory"])
    keys = settings["keys"]
    if len(set(keys))!=len(keys) or any(frames[key]["split"]!="training" for key in keys):
        raise ValueError("Operator fitting requires distinct training frames")
    origin = load_model(config["factor_model"])
    examples, contexts, margin_audit = collect_structure(
        config,p,bridge,settings,origin,load,destroy,solve,share,job)
    fitted, structure = fit_structure(config,p,origin,examples,settings,share,job)
    trained_structure = split_model(fitted,origin)
    structure["layer_delta"] = layer_norms(origin,fitted.weights-origin.weights)
    receipts = []
    if p.global_rank == 0:
        receipt = save_model(output/"checkpoints"/("generator-structure-%s.npz" % job),trained_structure)
        receipts = [receipt]
        save_once(output/"records"/("generator-structure-%s.json" % job),dict(structure,model=receipt))
    structure["model"] = share(receipts)[0]
    workers = config["parallel_workers"]
    states, batches, full_probes = {}, {}, {}
    original_rows, structure_rows, pilot_curves = [], [], []
    probe_records = []
    probe_directory = output/"data"/("generator-probes-"+job)
    if p.global_rank == 0:
        probe_directory.mkdir()
    share([str(probe_directory)] if p.global_rank==0 else [])
    windows = dict.fromkeys(("iteration-24","late"),settings["krylov_prefixes"])
    try:
        for i,key in enumerate(keys):
            if i%workers != p.worker:
                continue
            state = states[key] = load(key)
            observed, full = [], []
            base = solve(key,state,dict(name="origin",factor_model=origin,
                factor_context=dict(capture_inputs=observed),factor_probe_capture=full,
                factor_probe_windows=windows),"generator-capture")
            if not base["passed"] or not full:
                raise ValueError("Cannot capture the baseline displacement and full-system probes on "+key)
            batches[key] = make_probes(p,state,observed,solve,key,settings["seed"]+i,settings)
            receipt = archive_probes(p,probe_directory,key,"initial",batches[key])
            full_probes[key] = full
            if p.rank == 0:
                original_rows.append(base)
                probe_records.append(receipt)
            if structure["selected_step"]:
                trial = solve(key,state,dict(name="structure",factor_model=trained_structure),
                              "generator-structure-check")
                if p.rank == 0:
                    structure_rows.append(trial)
        before, after = share(original_rows), share(structure_rows)
        start_by_key = {row["key"]:row for row in before}
        use_structure = len(after)==len(before) and all(
            row["passed"] and row["iterations"] <= start_by_key[row["key"]]["iterations"]+settings["structure_iteration_guard"]
            and all(row[field]<=settings["fill_guard"]*start_by_key[row["key"]][field]
                    for field in ("p1_nnz","k1_nnz","p2_nnz")) for row in after)
        seed = trained_structure if use_structure else split_model(origin,origin)
        if p.global_rank == 0:
            record(config,"generator_seed",job=job,structure_used=use_structure,
                   before=method_totals(before),after=method_totals(after))

        def refresh(model):
            refreshed, local = {}, []
            for index,(key,state) in enumerate(states.items()):
                inputs = []
                receipt = solve(key,state,dict(name="operator_refresh",factor_model=model,
                    factor_context=dict(capture_inputs=inputs)),"generator-refresh")
                if not receipt["passed"]:
                    refreshed[key] = batches[key]
                else:
                    refreshed[key] = make_probes(p,state,inputs,solve,key,
                                                  settings["seed"]+100+index,settings)
                    artifact = archive_probes(p,probe_directory,key,"refresh",refreshed[key])
                    if p.rank == 0:
                        probe_records.append(artifact)
                if p.rank == 0:
                    local.append(dict(key=key,pass_solve=receipt["passed"],probes=[q[0] for q in refreshed[key]]))
            items = share(local)
            if p.global_rank == 0:
                record(config,"generator_probe_refresh",job=job,frames=items)
            return refreshed

        model, operator = displacement_fit(config,p,legacy,bridge,settings,origin,seed,states,batches,
                                           batches,solve,share,refresh,job)
        for row in operator["initial_origin"]:
            initial_levels = dict(zip(("level1","level2"),start_by_key[row["key"]]["factor_levels"]))
            if row.get("failure") or tuple(row["signature"])!=hierarchy_signature(initial_levels):
                raise ValueError("Independent coefficient initialization changed the baseline hierarchy")
        for key,state in states.items():
            controls = (dict(name="operator",factor_model=model,factor_probe_only=True),
                        dict(name="retained_base"))
            local = {}
            for method in controls:
                row = solve(key,state,dict(method,factor_probe_inputs=full_probes[key],
                    factor_probe_windows=windows),"generator-short-check",details=False)
                local[method["name"]] = row
            if p.rank == 0:
                pilot_curves.append(dict(key=key,origin=start_by_key[key]["factor_trajectories"],**local))
        curves = share(pilot_curves)
        probe_inventory = share(probe_records)
    finally:
        for loaded in states.values():
            destroy(loaded)
    receipts = []
    if p.global_rank == 0:
        receipts = [save_model(output/"checkpoints"/("generator-operator-%s.npz" % job),model)]
        save_once(output/"records"/("generator-operator-%s.json" % job),
                  dict(operator,model=receipts[0],seed="structure" if use_structure else "origin"))
    checkpoint = share(receipts)[0]
    methods = [dict(name="origin",factor_model=origin),dict(name="operator",factor_model=model),
               dict(config["factor_reference"],fallback=False)]
    pilot = evaluate(keys,methods,"generator-pilot")
    totals = method_totals(pilot)
    initial = {r["key"]:r["loss"] for r in operator["initial_origin"]}
    measured = {r["key"]:r["loss"] for r in operator["final"] if r["failure"] is None}
    reduction = (1.-sum(measured.values())/sum(initial.values())
                 if set(initial)==set(measured) and sum(initial.values())>0 else None)
    quality = (operator["updates"]>0 and reduction is not None
               and reduction>=settings["quality_gain"] and not any(row["failed"] for row in totals.values())
               and all(measured[key]<=initial[key]*(1+settings["max_frame_loss_increase"]) for key in initial)
               and totals["operator"]["iterations"]<=min(totals["origin"]["iterations"],
                                      settings["student_iteration_ratio"]*totals["student"]["iterations"]))
    validation, pairs = [], []
    speed = compression = False
    if quality:
        validation = config["training"]["validation_keys"]
        pairs = evaluate(validation,methods,"generator-validation")
        comparisons = method_totals(pairs)
        compression = (not any(row["failed"] for row in comparisons.values())
                       and comparisons["operator"]["iterations"]<=min(comparisons["origin"]["iterations"],
                           settings["student_iteration_ratio"]*comparisons["student"]["iterations"]))
        def faster(rows):
            return (all(not row["failed"] for row in rows.values())
                    and all(rows["operator"]["total_s"] < (1-settings["speed_margin"])*rows[name]["total_s"]
                            for name in ("origin","student")))
        if faster(comparisons):
            reverse = evaluate(validation,methods,"generator-validation",start_repeat=1)
            speed = faster(method_totals(reverse))
            pairs += reverse
    return dict(structure=structure,context_receipts=contexts,margin_audit=margin_audit, model=checkpoint,
                operator=operator,operator_seed="structure" if use_structure else "origin",
                probes=dict(directory=str(probe_directory),frames=probe_inventory),
                structure_pilot=before+after,pilot_keys=keys,pilot=pilot,pilot_totals=totals,
                pilot_curves=curves,action_reduction=reduction,quality_gate=bool(quality),
                validation_keys=validation,validation=pairs,speed_gate=bool(speed),
                compression_gate=bool(compression),
                training_scope="Hard deployed factors. The adjoint holds decoded graph features fixed, "
                               "and every proposed update redecodes both levels from the actual K1.")
