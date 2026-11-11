"""Persistent training probes and paired checkpoint selection for the factor generator."""
from collections import defaultdict, deque
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .backend import digest, record
from .factor_operator import action_losses, archive_probes, checked_json, make_probes, operator_observation, split_model
from .factorized import DEPTHS, PHYSICAL_COLUMNS, STAGES, FactorModel, hierarchy_signature, load_model, save_model
from .student_cycle import method_totals, save_once


def balanced_action(losses, reference, settings):
    """Return the frame objective and its subgradient on the same probes."""
    if set(losses)!=set(reference):
        raise ValueError("Action scoring requires the same reference inputs")
    groups = {"pcapply": [], "region": []}
    for name in losses:
        kind = name.rsplit("/",1)[-1].split("-",1)[0]
        groups[kind].append(name)
    if not all(groups.values()):
        raise ValueError("Both actual and regional probes are required")
    weights, terms = {}, {}
    for kind,names in groups.items():
        weight = settings["pc_weight"] if kind=="pcapply" else 1.-settings["pc_weight"]
        weights.update({n:weight/len(names) for n in names})
        terms[kind] = float(np.mean([losses[n] for n in names]))
    denominator = {name:max(reference[name],settings["tail_floor"]) for name in losses}
    increases = {name:(losses[name]-reference[name])/denominator[name] for name in losses}
    count = max(1,int(np.ceil(settings["tail_fraction"]*len(losses))))
    worst = sorted(losses,key=lambda n:(-increases[n],n))[:count]
    terms["tail"] = sum(max(0.,increases[n]) for n in worst)/count
    terms["tail_names"] = worst
    for name in worst:
        if increases[name]>0:
            weights[name] += settings["tail_weight"]/(count*denominator[name])
    loss = (settings["pc_weight"]*terms["pcapply"]+(1-settings["pc_weight"])*terms["region"]
            +settings["tail_weight"]*terms["tail"])
    return loss,weights,terms


def load_probe_shards(p, receipts, namespace=True):
    probes = []
    for receipt in receipts:
        phase = receipt["phase"]
        path = Path(receipt["directory"])/("%s-%s-r%03d.npz" % (receipt["key"],phase,p.rank))
        if digest(path)!=receipt["shards"][p.rank]:
            raise ValueError("Archived probe shard changed: "+str(path))
        with np.load(path) as data:
            names = data["names"].tolist()
            if names != receipt["probes"]:
                raise ValueError("Archived probe names changed")
            probes.extend(((phase+"/"+n) if namespace else n,r,z,w,float(norm))
                for n,r,z,w,norm in zip(names,data["inputs"],data["teacher"],
                                        data["weights_squared"],data["normalization"]))
    if len({probe[0] for probe in probes})!=len(probes):
        raise ValueError("Probe identity duplicated in a frame pool")
    return probes


def training_order(frames, workers, seed):
    """Round-robin cases and time strata, visiting every training state once."""
    cases = defaultdict(list)
    for key,row in frames.items():
        if row["split"]=="training":
            cases[row["case"]].append(key)
    rng = np.random.default_rng(seed)
    queues = []
    for case,keys in sorted(cases.items()):
        ordered = sorted(keys,key=lambda key:(frames[key]["t"],key))
        bands = [deque(rng.permutation(band).tolist()) for band in np.array_split(ordered,3)]
        sequence = []
        while any(bands):
            for band in bands:
                if band:
                    sequence.append(band.popleft())
        queues.append(deque(sequence))
    order = []
    while any(queues):
        for q in queues:
            if q:
                order.append(q.popleft())
    return [order[i:i+workers] for i in range(0,len(order),workers)]


def operator_rows(p,legacy,bridge,model,states,batches,solve,share,numerical,fusion,
                  gradient=False,references=None,settings=None,inference=None,audit=False):
    local = []
    for key,state in states.items():
        def apply(displacement):
            objective = None if references is None else lambda losses: balanced_action(
                losses,references[key],settings)
            return operator_observation(p,legacy,bridge,displacement,model,batches[key],gradient,
                                        numerical,audit=audit,objective=objective)
        result = solve(key,state,dict(name="operator_replay",factor_model=model,
            factor_context=dict(factor_tape=gradient, factor_operation=apply, factor_fusion=fusion,
                                factor_inference=inference)),
            "operator-replay",details=False)
        if p.rank==0:
            local.append(dict(key=key,**result))
    return share(local)


def restore_step_seven(origin, archived, measure, numerical):
    """Replay the seven archived accepted updates on their unchanged initial inputs."""
    model = split_model(origin,origin)
    checks = []

    def matches(rows, expected):
        previous = {r["key"]:r for r in expected}
        return (len(rows)==len(previous) and all(r["failure"] is None
                and r["key"] in previous and r["signature"]==previous[r["key"]]["signature"]
                and np.isclose(r["loss"],previous[r["key"]]["loss"],rtol=2e-7,atol=1e-10) for r in rows))

    for entry in archived["operator"]["history"][:7]:
        step = entry["step"]
        source = measure(model,gradient=True)
        if not matches(source,entry["source"]):
            return None,dict(status="source_signature_or_loss",step=step,checks=checks)
        expected = entry["attempts"][-1]
        if not expected["accepted"]:
            return None,dict(status="original_step_not_accepted",step=step,checks=checks)
        indexed = {r["key"]:r for r in source}
        gradient = np.mean([indexed[r["key"]]["gradient"] for r in entry["source"]],axis=0)
        if step<=numerical["second_only_steps"]:
            gradient[0] = 0.
        norm = np.linalg.norm(gradient)
        if not np.isclose(norm,entry["gradient_norm"],rtol=2e-7,atol=1e-10):
            return None,dict(status="gradient_norm_mismatch",step=step,checks=checks)
        weights = (model.coefficient_weights-expected["step_size"]*gradient/norm).astype(model.weights.dtype)
        trial = FactorModel(model.width,model.weights,model.adapters,model.unit_sum_depth,weights)
        checked = measure(trial)
        if not matches(checked,expected["cases"]):
            return None,dict(status="updated_signature_or_loss",step=step,checks=checks)
        checks.append(dict(step=step,gradient_norm=float(norm),frames=[dict(key=r["key"],loss=r["loss"],
                           signature=r["signature"]) for r in checked]))
        model = trial
    if len(checks)!=7:
        return None,dict(status="incomplete_archived_history",checks=checks)
    return model,dict(status="reproduced",checks=checks)


def training_update(p, legacy, bridge, model, states, batches, references, solve, share, numerical,
                    settings, fusion, step_size):
    def measure(model, gradient=False):
        return operator_rows(p,legacy,bridge,model,states,batches,solve,share,numerical,fusion,gradient,
                             references,settings)
    source = measure(model,True)
    if not source or any(r["failure"] is not None for r in source):
        return model,dict(accepted=False,failure="source_hierarchy",source=source),step_size
    gradient = np.mean([r["gradient"] for r in source],axis=0)
    norm = np.linalg.norm(gradient)
    if not np.isfinite(norm) or norm<1e-12:
        return model,dict(accepted=False,failure="coefficient_gradient"),step_size
    current = {r["key"]:r for r in source}
    record_rows = [{k:v for k,v in r.items() if k!="gradient"} for r in source]
    scaled = gradient
    if model.context_size:
        scaled = gradient.copy()
        scaled[:, -model.context_size:] *= settings["context_lr_scale"]
    attempts = []
    for factor in (1.,.5,.25):
        step = step_size*factor
        weights = (model.coefficient_weights-step*scaled/norm).astype(model.weights.dtype)
        trial = replace(model, coefficient_weights=weights)
        rows = measure(trial)
        for row in rows:
            if row["failure"] is None:
                row["structure_stages_changed"] = [i//2+1 for i in range(0,len(row["structure"]),2)
                    if row["structure"][i:i+2]!=current[row["key"]]["structure"][i:i+2]]
        admissible = all(r["failure"] is None and all(a<=settings["fill_guard"]*b for a,b in
                         zip(r["nonzeros"],current[r["key"]]["nonzeros"])) for r in rows)
        accepted = bool(admissible and len(rows)==len(source) and
                    sum(r["loss"] for r in rows)<(1-settings["min_step_improvement"])*sum(r["loss"] for r in source))
        attempts.append(dict(step=step,accepted=accepted,rows=rows))
        if accepted:
            return trial,dict(accepted=True,gradient_norm=float(norm),source=record_rows,attempts=attempts), min(
                settings["max_step"],1.15*step)
    return model,dict(accepted=False,gradient_norm=float(norm),source=record_rows,attempts=attempts),max(
        settings["min_step"],.5*step_size)


def equivalent_probe_actions(plain, fused):
    reference = {r["key"]:r for r in plain}
    if len(reference)!=len(plain) or set(reference)!={r["key"] for r in fused} or len(plain)!=len(fused):
        return False
    for row in fused:
        expected = reference[row["key"]]
        if (row["failure"] is not None or expected["failure"] is not None
                or row["signature"]!=expected["signature"] or row["nonzeros"]!=expected["nonzeros"]
                or set(row["probes"])!=set(expected["probes"])
                or not all(np.isclose(value,expected["probes"][key],rtol=1e-12,atol=1e-14)
                           for key,value in row["probes"].items())):
            return False
    return bool(reference)


def replay_prefix(config):
    resume = config.get("factor_replay_resume")
    if resume is None:
        return None
    previous = checked_json(resume["config"])
    fields = ("factor_model","factor_replay","factor_reference","training","inventory_sha256",
              "baseline_directory","mpi_ranks","parallel_workers","image_sha256","bridge_sha256",
              "dependency_hashes","rtol","atol","max_iterations")
    if any(previous[k]!=config[k] for k in fields):
        raise ValueError("Probe replay resume differs from the original schedule or model")
    source = resume["history"]
    if digest(source["path"])!=source["sha256"]:
        raise ValueError("Probe replay history receipt changed")
    with Path(source["path"]).open() as stream:
        history = [json.loads(line) for line in stream]
    event = resume["milestone"]
    batch = event["batch"]
    if batch!=len(history) or [x["batch"] for x in history]!=list(range(1,batch+1)) or any(
            event[role] not in (0,batch) for role in ("best_action","best_solve")):
        raise ValueError("Resume requires the complete prefix through its saved milestone")
    setup = checked_json(resume["setup"])
    if setup["reconstruction"]["status"]=="reproduced":
        load_model(setup["reconstruction"]["model"])
    cases = []
    for receipt in resume["cases"]:
        raw = checked_json(receipt)
        row,result = raw["compact"],raw["result"]
        if (not row["passed"] or not result["reached_true_target"]
                or row["final_true_residual"]!=result["final_true_residual"]
                or not 0<=row["final_true_residual"]<=row["true_limit"]
                or row["total_s"]!=result["total_wall_s"]):
            raise ValueError("Saved milestone solve is invalid")
        cases.append(dict(row,repeat=receipt["repeat"]))
    if method_totals(cases)!=event["totals"]:
        raise ValueError("Saved milestone totals differ from the recorded solves")
    return dict(job=resume["job"],setup=setup,history=history,model=resume["model"],
                milestone=dict(event,cases=cases,checkpoint=resume["model"]))


def run_probe_replay(config,p,legacy,bridge,frames,load,destroy,solve,share,evaluate,job):
    settings = config["factor_replay"]
    original = checked_json(settings["pilot_config"])
    archive = checked_json(settings["pilot_report"])
    if (archive["model"]!=config["factor_model"] or original["factor_model"]!=settings["tail_reference"]
            or archive["operator_seed"]!="origin"):
        raise ValueError("Archived operator and reference checkpoints do not match")
    if any(original[k]!=config[k] for k in ("inventory_sha256","image_sha256","mpi_ranks",
                                              "bridge_sha256","dependency_hashes")):
        raise ValueError("The input, PETSc build, or MPI partition changed")
    numerical = original["factor_operator"]
    prefix = replay_prefix(config)
    restored_model = load_model(prefix["model"]) if prefix else None
    baseline = load_model(config["factor_model"])
    origin = load_model(settings["tail_reference"])
    reference = split_model(origin,origin)
    if baseline.coefficient_weights is None or not np.array_equal(baseline.weights,origin.weights):
        raise ValueError("Probe replay requires the fixed structural checkpoint")
    diagnostics = archive["pilot_keys"]
    validation = config["training"]["validation_keys"]
    if any(frames[k]["split"]!="training" for k in diagnostics) or any(
            frames[k]["split"]!="validation" for k in validation):
        raise ValueError("Training and fixed validation frames must remain separate")
    workers = config["parallel_workers"]
    order = training_order(frames,workers,settings["seed"])
    expected = {k for k,row in frames.items() if row["split"]=="training"}
    assert sum(map(len,order))==len(expected)==55 and {k for batch in order for k in batch}==expected
    hard = settings["hard_replay_keys"]
    assert len(hard)<=workers and set(hard)<=set(diagnostics)
    output = Path(config["output_directory"])
    bank = defaultdict(list)
    for receipt in archive["probes"]["frames"]:
        bank[receipt["key"]].append(dict(receipt,directory=archive["probes"]["directory"]))
    fixed = {k:list(v) for k,v in bank.items()}
    directory = output/"data"/("replay-probes-"+job)
    if p.global_rank==0:
        directory.mkdir()
    share([str(directory)] if p.global_rank==0 else [])

    def local_states(keys):
        return {key:load(key) for i,key in enumerate(keys) if i%workers==p.worker}

    def checkpoint(model,name):
        local = []
        if p.global_rank==0:
            local = [save_model(output/"checkpoints"/("generator-%s-%s.npz"%(job,name)),model)]
        return share(local)[0]

    def method(name,model,fusion):
        return dict(name=name,factor_model=model,factor_context=dict(factor_fusion=fusion))

    def score(rows):
        return float(np.mean([r["loss"] for r in rows])) if rows and all(
            r["failure"] is None for r in rows) else None

    states = local_states(diagnostics)
    try:
        frozen = {k:load_probe_shards(p,fixed[k]) for k in states}
        if prefix:
            reconstruction = prefix["setup"]["reconstruction"]
            plain = prefix["setup"]["fusion"]["plain"]
            fused = prefix["setup"]["fusion"]["fused"]
        else:
            start = {k:load_probe_shards(p,[r for r in fixed[k] if r["phase"]=="initial"],False) for k in states}

            def historical(model,gradient=False):
                return operator_rows(p,legacy,bridge,model,states,start,solve,share,numerical,False,gradient)
            step7,reconstruction = restore_step_seven(origin,archive,historical,numerical)
            if step7 is not None:
                reconstruction["model"] = checkpoint(step7,"step7")
            plain = operator_rows(p,legacy,bridge,baseline,states,frozen,solve,share,numerical,False)
            fused = operator_rows(p,legacy,bridge,baseline,states,frozen,solve,share,numerical,True)
        if any(row["failure"] is not None for row in plain):
            raise ValueError("The frozen operator checkpoint cannot rebuild its training hierarchy")
        fusion = equivalent_probe_actions(plain,fused)
        ref_rows = operator_rows(p,legacy,bridge,reference,states,frozen,solve,share,numerical,fusion)
        if any(r["failure"] is not None for r in ref_rows):
            raise ValueError("Could not score the fixed reference on diagnostic inputs")
        diag_ref = {r["key"]:r["probes"] for r in ref_rows}
        seed_rows = operator_rows(p,legacy,bridge,baseline,states,frozen,solve,share,numerical,fusion,
                                  references=diag_ref,settings=settings)
        if prefix:
            resumed_rows = operator_rows(p,legacy,bridge,restored_model,states,frozen,solve,share,
                                         numerical,fusion,references=diag_ref,settings=settings)
            observed = score(resumed_rows)
            if (observed is None or not np.isclose(observed,prefix["milestone"]["action"],rtol=1e-10,atol=1e-12)
                    or not np.array_equal(restored_model.weights,baseline.weights)
                    or not np.array_equal(restored_model.adapters,baseline.adapters)):
                raise ValueError("Restored milestone failed its fixed training diagnostics")
    finally:
        for state in states.values():
            destroy(state)
    if score(seed_rows) is None:
        raise ValueError("Could not score the fixed operator checkpoint on its diagnostic inputs")
    names = [method("separate",baseline,False)]
    if fusion:
        names.append(method("shared",baseline,True))
    if prefix and fusion==prefix["setup"]["fusion"]["exact"]:
        fusion_cases = prefix["setup"]["fusion"]["cases"]
    else:
        fusion_cases = evaluate(diagnostics,names,"factor-fusion")
    fuse_totals = method_totals(fusion_cases)
    if not all(r["passed"] for r in fusion_cases):
        raise ValueError("Frozen checkpoint failed on the diagnostic solves")
    seed_totals = fuse_totals["shared" if fusion else "separate"]
    fusion_report = dict(exact=bool(fusion),plain=plain,fused=fused,cases=fusion_cases,totals=fuse_totals,
                         probe_rtol=1e-12,probe_atol=1e-14,setup_source=prefix["job"] if prefix else None)
    if p.global_rank==0:
        save_once(output/"records"/("generator-replay-setup-%s.json"%job),
                  dict(reconstruction=reconstruction,fusion=fusion_report,diagnostics=seed_rows))
        record(config,"generator_replay_setup",job=job,step7=reconstruction["status"],
               fused=bool(fusion),diagnostic_loss=score(seed_rows),totals=fuse_totals)

    model = baseline
    best_action = (score(seed_rows),model,0)
    best_solve = ((seed_totals["iterations"],1.),model,0)
    saved = {}
    step_size = settings["step_size"]
    trained, visited, milestones, history = set(),set(),[],[]
    refresh_plan = []
    completed = 0
    if prefix:
        model = restored_model
        history = prefix["history"]
        completed = len(history)
        milestone = prefix["milestone"]
        totals = milestone["totals"]
        saved[completed] = prefix["model"]
        if milestone["best_action"]==completed:
            best_action = (milestone["action"],model,completed)
        if milestone["best_solve"]==completed:
            best_solve = ((totals["candidate"]["iterations"],
                           totals["candidate"]["total_s"]/totals["initial"]["total_s"]),model,completed)
        milestones.append(dict(batch=completed,action=milestone["action"],actions=resumed_rows,
                               cases=milestone["cases"],totals=totals,checkpoint=saved[completed],
                               best_action_batch=best_action[2],best_solve_batch=best_solve[2],
                               reused_from=prefix["job"]))
        for entry in history:
            visited.update(entry["active"])
            for receipt in entry["probe_updates"]:
                bank[receipt["key"]].append(receipt)
                if receipt["phase"]=="added":
                    refresh_plan.append(receipt["key"])
            for update in entry["updates"]:
                attempts = update.get("attempts",[])
                if update["accepted"]:
                    trained.update(entry["active"])
                    step_size = min(settings["max_step"],1.15*attempts[-1]["step"])
                elif attempts:
                    step_size = max(settings["min_step"],.5*step_size)
        if p.global_rank==0:
            record(config,"generator_replay_resumed",job=job,source=prefix["job"],batches=completed,
                   checkpoint=prefix["model"],coverage=len(visited),next_step_size=step_size)
    schedule = [(hard,False,False)]
    for i,batch in enumerate(order,1):
        schedule.append((batch,False,True))
        if i%settings["revisit_every"]==0:
            schedule.append((hard,i==settings["revisit_every"],False))
    for number,(keys,refresh,coverage) in enumerate(schedule,1):
        if number<=completed:
            saved_batch = history[number-1]
            if (saved_batch["keys"]!=keys or saved_batch["refresh"]!=refresh
                    or saved_batch["coverage_batch"]!=coverage):
                raise ValueError("Recovered prefix keys differ from the frozen replay schedule")
            continue
        states = local_states(keys)
        pool, local_receipts, local_captures = {},[],[]
        try:
            for key,state in states.items():
                probes = load_probe_shards(p,bank[key]) if bank[key] else []
                if not probes or refresh:
                    phase = "added" if refresh else "initial"
                    observed = []
                    source = "current"
                    case = solve(key,state,dict(method("probe_capture",model,fusion),
                        factor_context=dict(factor_fusion=fusion,capture_inputs=observed)),
                        "replay-"+phase)
                    if not case["passed"]:
                        source = "initial"
                        case = solve(key,state,dict(method("probe_capture_seed",baseline,fusion),
                            factor_context=dict(factor_fusion=fusion,capture_inputs=observed)),
                            "replay-"+phase+"-seed")
                    if case["passed"]:
                        source_info = dict(learner=source)
                        retained = Path(config["baseline_directory"])/"records"/(
                            "solver-export-"+config["retained_export_prefixes"]["training"]+"-"+key+".json")
                        teacher = "retained_base" if retained.exists() else "factor_teacher"
                        values = make_probes(p,state,observed,solve,key,
                            settings["seed"]+100*number, numerical,teacher,source_info)
                        receipt = archive_probes(p,directory,key,phase,values)
                        if p.rank==0:
                            local_receipts.append(dict(receipt,directory=str(directory),source=source_info))
                            local_captures.append(case)
                        probes.extend((phase+"/"+n,r,z,w,s) for n,r,z,w,s in values)
                    elif not probes:
                        if p.rank==0:
                            local_captures.append(case)
                        continue
                pool[key] = probes
            probe_updates = share(local_receipts)
            for receipt in probe_updates:
                bank[receipt["key"]].append(receipt)
                if receipt["phase"]=="added":
                    refresh_plan.append(receipt["key"])
            captures = share(local_captures)
            active = {key:states[key] for key in pool}
            rows = operator_rows(p,legacy,bridge,reference,active,pool,solve,share,numerical,fusion)
            refs = {r["key"]:r["probes"] for r in rows if r["failure"] is None}
            active = {k:v for k,v in active.items() if k in refs}
            valid = sorted(refs)
            visited.update(valid)
            updates = []
            if valid:
                for _ in range(settings["updates_per_batch"]):
                    model, report, step_size = training_update(
                        p,legacy,bridge,model,active,pool,refs,solve,share,numerical,settings,fusion,step_size)
                    updates.append(report)
                    if report["accepted"]:
                        trained.update(valid)
        finally:
            for state in states.values():
                destroy(state)
        entry = dict(batch=number,keys=keys,coverage_batch=coverage,refresh=refresh,active=valid,
                     captures=captures,probe_updates=probe_updates,updates=updates,coverage=len(visited))
        history.append(entry)
        if p.global_rank==0:
            path=output/"records"/("generator-replay-%s.jsonl"%job)
            with path.open("a") as f:
                f.write(json.dumps(entry,allow_nan=False)+"\n")
            record(config,"generator_replay_batch",job=job,batch=number,keys=keys,active=valid,
                   accepted=sum(r["accepted"] for r in updates),coverage=len(visited))
        if number!=1 and number%settings["check_every"]!=0 and number!=len(schedule):
            continue
        states = local_states(diagnostics)
        try:
            probes = {k:load_probe_shards(p,fixed[k]) for k in states}
            actions = operator_rows(p,legacy,bridge,model,states,probes,solve,share,numerical,fusion,
                                    references=diag_ref,settings=settings)
        finally:
            for state in states.values():
                destroy(state)
        paired = evaluate(diagnostics,[method("initial",baseline,fusion),method("candidate",model,fusion)],
                          "replay-milestone-%02d"%number,start_repeat=(number//settings["check_every"])%2)
        totals = method_totals(paired)
        value = score(actions)
        if not any(r["failed"] for r in totals.values()):
            candidate = totals["candidate"]
            rank = (candidate["iterations"],candidate["total_s"]/totals["initial"]["total_s"])
            if rank < best_solve[0]:
                best_solve = (rank,model,number)
            if value is not None and value < best_action[0]:
                best_action = (value,model,number)
        if number in (best_action[2],best_solve[2]):
            saved[number] = checkpoint(model,"milestone-%02d"%number)
        milestones.append(dict(batch=number,action=value,actions=actions,cases=paired,totals=totals,
                               checkpoint=saved.get(number),
                               best_action_batch=best_action[2],best_solve_batch=best_solve[2]))
        if p.global_rank==0:
            record(config,"generator_replay_milestone",job=job,batch=number,action=value,
                   best_action=best_action[2],best_solve=best_solve[2],totals=totals)

    candidates, receipts = [], {}
    for label, item in (("action",best_action),("solver",best_solve)):
        metric,selected,batch = item
        same = next((m for m in candidates if np.array_equal(m["factor_model"].coefficient_weights,
                                                             selected.coefficient_weights)),None)
        if same is None:
            candidates.append(method(label,selected,fusion))
            receipt = saved[batch] if batch else config["factor_model"]
            name = label
        else:
            name = same["name"]
            receipt = receipts[name]["model"]
        receipts[label] = dict(model=receipt,method=name,batch=batch,metric=metric)
    candidates.append(dict(config["factor_reference"],fallback=False))
    paired = evaluate(validation,candidates,"replay-final",repeats=2)
    repeats = [method_totals([r for r in paired if r["repeat"]==i]) for i in range(2)]
    comparison = {}
    for candidate in candidates[:-1]:
        label = candidate["name"]
        def clears(limit):
            return all(row[label]["frames"]==row["student"]["frames"]==len(validation)
                       and not row[label]["failed"] and not row["student"]["failed"]
                       and row[label]["iterations"]<=row["student"]["iterations"]
                       and row[label]["total_s"]<=limit*row["student"]["total_s"] for row in repeats)
        comparison[label] = dict(speed=clears(.98),compression=clears(1.05))
    return dict(training_keys=sorted(expected),coverage=sorted(visited),updated_on=sorted(trained),
                coverage_complete=visited==expected,training=history,milestones=milestones,
                original_model=config["factor_model"],step7=reconstruction,fusion=fusion_report,
                resume=dict(job=prefix["job"],batches=completed,model=prefix["model"]) if prefix else None,
                candidates=receipts,probe_bank=dict(bank),refresh_keys=refresh_plan,
                keys=validation,cases=paired,repeat_totals=repeats,totals=method_totals(paired),
                comparisons=comparison,speed_gate=any(row["speed"] for row in comparison.values()),
                compression_gate=any(row["compression"] for row in comparison.values()))


def cost_pair(cases, keys, baseline, candidate, margin):
    """Require the same accuracy in each repeat and rank by elapsed time."""
    totals = [method_totals([row for row in cases if row["repeat"] == repeat]) for repeat in (0, 1)]
    chosen = [row for row in cases if row["method"] in (baseline, candidate)]
    complete = len(chosen) == 4*len(keys) and {(row["repeat"], row["key"], row["method"])
        for row in chosen} == {(repeat, key, name) for repeat in (0, 1)
                               for key in keys for name in (baseline, candidate)}
    converged = complete and all(row["passed"] and row["fallback"] is None
        and 0. <= row["final_true_residual"] <= row["true_limit"]
        and np.isfinite(row["total_s"]) and row["total_s"] > 0. for row in chosen)
    if converged:
        limits = {(row["repeat"], row["key"]):row["true_limit"]
                  for row in chosen if row["method"] == baseline}
        converged = all(row["true_limit"] == limits[row["repeat"], row["key"]]
                        for row in chosen if row["method"] == candidate)
    ratios = [t[candidate]["total_s"]/t[baseline]["total_s"] for t in totals] if converged else []
    return dict(totals=totals, ratios=ratios,
                eligible=bool(ratios and all(ratio <= 1.-margin for ratio in ratios)))


def cost_rank(pair):
    return max(pair["ratios"]), float(np.mean(pair["ratios"]))


def run_cost_replay(config, p, legacy, bridge, frames, load, destroy, solve, share, evaluate, job):
    """Measure equivalent construction paths and tune the uncompressed coefficients."""
    settings = config["factor_cost"]
    source = checked_json(settings["source_report"])
    numerical = settings["numerical"]
    update_settings = settings["update"]
    validation = config["training"]["validation_keys"]
    diagnostics = settings["diagnostic_keys"]
    batches = settings["batches"]
    fit = [key for batch in batches for key in batch]
    focus = settings["fresh_keys"]
    anchor = frames[settings["anchor"]]
    workers = config["parallel_workers"]
    if (source["candidates"]["solver"]["model"] != config["factor_model"] or not source["fusion"]["exact"]
            or source["keys"] != validation or anchor["key"] not in validation
            or len(set(fit+diagnostics+validation)) != len(fit+diagnostics+validation)
            or len(set(focus)) != len(focus) or not set(focus) <= set(fit)
            or any(len(batch) > workers for batch in batches)
            or any(frames[k]["split"] != "training" for k in fit+diagnostics)
            or any(frames[k]["split"] != "validation" for k in validation)
            or any(k not in source["probe_bank"] for k in fit+diagnostics)):
        raise ValueError("Construction study requires disjoint training, diagnostics, and fixed validation")
    baseline = load_model(config["factor_model"])
    if baseline.product_compression is not None or baseline.coefficient_weights is None:
        raise ValueError("Construction study requires the uncompressed independent coefficient model")
    output = Path(config["output_directory"])

    def local_states(keys):
        return {key:load(key) for i,key in enumerate(keys) if i%workers == p.worker}

    def close(states):
        for loaded in states.values():
            destroy(loaded)

    def method(name, model, inference=None, **context):
        return dict(name=name, factor_model=model,
                    factor_context=dict(factor_fusion=True, factor_inference=inference, **context))

    def checkpoint(model, suffix):
        receipts = []
        if p.global_rank == 0:
            path = output/"checkpoints"/("generator-%s-%s.npz" % (job, suffix))
            receipts.append(save_model(path, model))
        receipt, = share(receipts)
        return receipt

    def observe_paths(model, keys, names):
        states = local_states(keys)
        try:
            probes = {key:load_probe_shards(p, source["probe_bank"][key]) for key in states}
            original = operator_rows(p, legacy, bridge, model, states, probes, solve, share, numerical, True)
            if len(original) != len(keys) or any(row["failure"] is not None for row in original):
                raise ValueError("The reference construction failed its operator replay")
            results = {}
            for name in names:
                current = operator_rows(p, legacy, bridge, model, states, probes, solve, share, numerical, True,
                                        inference=name)
                results[name] = dict(exact=equivalent_probe_actions(original, current), cases=current)
            return dict(reference=original, paths=results)
        finally:
            close(states)

    plain = method("incumbent", baseline)
    # Include the nearest difficult training state in the numerical check.
    # Performance selection uses only the separate diagnostic frames.
    parity_keys = diagnostics+[focus[0]]
    parity = observe_paths(baseline, parity_keys, ("packed", "support"))
    paths = [name for name,check in parity["paths"].items() if check["exact"]]
    timed_paths = evaluate(diagnostics, [plain, *(method(name, baseline, name) for name in paths)],
                           "cost-construction", repeats=2)
    reference = {(row["key"], row["repeat"]):row for row in timed_paths if row["method"] == "incumbent"}
    comparisons = {}
    for name in paths:
        rows = [row for row in timed_paths if row["method"] == name]
        same_solve = all(row["passed"] and row["iterations"] == ref["iterations"]
            and row["p1_nnz"] == ref["p1_nnz"] and row["k1_nnz"] == ref["k1_nnz"]
            and row["true_limit"] == ref["true_limit"]
            and np.isclose(row["final_true_residual"], ref["final_true_residual"],
                           rtol=0., atol=1e-5*ref["true_limit"])
            for row in rows for ref in [reference[row["key"], row["repeat"]]] if ref["passed"])
        comparisons[name] = dict(cost_pair(timed_paths, diagnostics, "incumbent", name,
                                          settings["selection_margin"]), same_solve=bool(same_solve))
    # Strict trajectory agreement remains diagnostic. An equivalent
    # construction can qualify at the same true tolerance in both repeats.
    available = [name for name in paths if comparisons[name]["eligible"]]
    deployment = min(available, key=lambda name:cost_rank(comparisons[name])) if available else None
    initial_deployment = deployment
    if p.global_rank == 0:
        record(config, "factor_cost_construction", job=job, equivalent=paths, selected=deployment,
               comparisons={k:dict(ratios=v["ratios"], eligible=v["eligible"], same_solve=v["same_solve"])
                            for k,v in comparisons.items()})

    directory = output/"data"/("cost-probes-"+job)
    if p.global_rank == 0:
        directory.mkdir()
    share([str(directory)] if p.global_rank == 0 else [])
    bank = {k:list(source["probe_bank"][k]) for k in fit}
    captures = []
    for index, key in enumerate(focus):
        if index%workers != p.worker:
            continue
        loaded = load(key)
        try:
            observed = []
            result = solve(key, loaded, method("capture", baseline, capture_inputs=observed,
                capture_apply_calls=settings["capture_calls"], factor_profile_stride=settings["profile_stride"]),
                "cost-focus")
            old = bank[key]
            provenance = dict(model=config["factor_model"])
            receipt = None
            if result["passed"] and observed:
                teacher = next(r["source"]["method"] for r in old if r["phase"] == "initial")
                probes = make_probes(p, loaded, observed, solve, key, settings["seed"]+index, numerical,
                                     teacher, provenance, all_observed=True)
                receipt = archive_probes(p, directory, key, "focus", probes)
                receipt = dict(receipt, directory=str(directory), source=provenance)
            if p.rank == 0:
                captures.append(dict(key=key, result=result, receipt=receipt,
                                     captured_calls=[call for call,_ in observed]))
        finally:
            destroy(loaded)
    captures = share(captures)
    if {row["key"] for row in captures} != set(focus):
        raise ValueError("Incomplete focused capture set")
    for row in captures:
        if row["receipt"] is not None:
            bank[row["key"]].append(row["receipt"])
    if p.global_rank == 0:
        record(config, "factor_cost_inputs", job=job, anchor=anchor["key"],
               captures=[dict(key=r["key"], calls=r["captured_calls"], iterations=r["result"]["iterations"],
                              added=r["receipt"] is not None) for r in captures])

    model = baseline
    step_size = update_settings["step_size"]
    history, milestones, best = [], [], None
    for epoch in range(1, settings["rounds"]+1):
        for keys in batches:
            states = local_states(keys)
            try:
                probes = {k:load_probe_shards(p, bank[k]) for k in states}
                scored = operator_rows(p, legacy, bridge, baseline, states, probes, solve, share,
                                       numerical, True)
                if len(scored) != len(keys) or any(row["failure"] is not None for row in scored):
                    raise ValueError("The frozen reference failed a focused training input")
                targets = {row["key"]:row["probes"] for row in scored}
                updates = []
                for _ in range(settings["updates_per_batch"]):
                    model, update, step_size = training_update(
                        p, legacy, bridge, model, states, probes, targets, solve, share,
                        numerical, update_settings, True, step_size)
                    updates.append(update)
            finally:
                close(states)
            history.append(dict(round=epoch, keys=keys, updates=updates))
            if p.global_rank == 0:
                record(config, "factor_cost_batch", job=job, round=epoch, keys=keys,
                       accepted=sum(row["accepted"] for row in updates))
        receipt = checkpoint(model, "cost-%02d" % epoch)
        cases = evaluate(diagnostics, [plain, method("candidate", model)], "cost-milestone-%02d" % epoch, repeats=2)
        pair = cost_pair(cases, diagnostics, "incumbent", "candidate", settings["selection_margin"])
        change = np.linalg.norm(model.coefficient_weights-baseline.coefficient_weights, axis=1).tolist()
        selected = bool(pair["eligible"] and any(change))
        milestones.append(dict(round=epoch, model=receipt, cases=cases, comparison=pair,
                               coefficient_delta=change, qualified=selected))
        if selected and (best is None or cost_rank(pair) < cost_rank(best[0])):
            best = (pair,model,receipt,epoch)
        if p.global_rank == 0:
            record(config, "factor_cost_milestone", job=job, round=epoch, ratios=pair["ratios"],
                   qualified=selected, coefficient_delta=change, best=best[3] if best else None)

    model, receipt, epoch = best[1:] if best else (baseline, config["factor_model"], 0)
    final_parity = None
    if deployment and best:
        final_parity = observe_paths(model, parity_keys, (deployment,))
        if not final_parity["paths"][deployment]["exact"]:
            deployment = None
    winner = dict(model=receipt, round=epoch, inference=deployment)
    cases, benchmark = [], None
    if epoch or deployment:
        student = dict(config["factor_reference"], fallback=False)
        cases = evaluate(validation, [plain, method("candidate", model, deployment), student],
                         "cost-validation", repeats=2)
        benchmark = dict(incumbent=cost_pair(cases, validation, "incumbent", "candidate", 0.),
                         student=cost_pair(cases, validation, "student", "candidate", settings["speed_margin"]))
    return dict(source=settings["source_report"], baseline=config["factor_model"],
                anchor=dict(key=anchor["key"], case=anchor["case"], time=anchor["t"]),
                focus=[dict(key=k, case=frames[k]["case"], time=frames[k]["t"], dt=frames[k]["dt"],
                            time_offset=frames[k]["t"]-anchor["t"]) for k in focus],
                construction=dict(parity=parity, keys=parity_keys, cases=timed_paths,
                                  comparisons=comparisons, initial_inference=initial_deployment,
                                  final_parity=final_parity),
                diagnostic_keys=diagnostics, training_keys=fit, captures=captures, history=history,
                milestones=milestones, winner=winner, validation_keys=validation, cases=cases,
                benchmark=benchmark, speed_gate=bool(benchmark and benchmark["student"]["eligible"]))


def fit_matrix_encoder(teacher, moments, regularization):
    """Fit the omitted contribution to the teacher's first-layer activation."""
    width, inputs = teacher.layers[0][0].shape
    observed = np.setdiff1d(np.arange(inputs+1), PHYSICAL_COLUMNS)
    if moments.shape != (STAGES, inputs+1, inputs+1) or not regularization > 0:
        raise ValueError("Invalid matrix-input fitting statistics")

    def transfer(leaf, gram):
        original, _ = leaf.layers[0]
        lhs = gram[np.ix_(observed, observed)]
        rhs = gram[np.ix_(observed, PHYSICAL_COLUMNS)] @ original[:, PHYSICAL_COLUMNS].T
        lhs += np.diag(regularization * np.maximum(lhs.diagonal(), 1e-10))
        correction = np.linalg.solve(lhs, rhs).T
        weights = leaf.weights.copy()
        first = weights[:original.size].reshape(original.shape)
        first[:, PHYSICAL_COLUMNS] = 0.
        first[:, observed[:-1]] += correction[:, :-1]
        weights[original.size:original.size+width] += correction[:, -1]
        return weights

    pivot = DEPTHS[0]
    coeff = np.stack([transfer(teacher.coefficients_at(stage), gram)
                      for stage, gram in ((0,moments[:pivot].sum(0)), (pivot,moments[pivot:].sum(0)))])
    return replace(teacher, weights=transfer(teacher, moments.sum(0)),
                   coefficient_weights=coeff, matrix_only=True)


def run_matrix_input(config, p, legacy, bridge, frames, load, destroy, solve, share, evaluate, job):
    """Distill a deployable encoder, then use the real cycles and complete solves."""
    settings = config["factor_input"]
    source = checked_json(settings["source_report"])
    archive = checked_json(settings["probe_report"])
    baseline = load_model(config["factor_model"])
    groups = settings["batches"]
    workers = config["parallel_workers"]
    fit = [key for group in groups for key in group]
    diagnostics, validation = settings["diagnostic_keys"], config["training"]["validation_keys"]
    all_keys = fit + diagnostics + validation
    if (source["winner"]["model"] != config["factor_model"] or source["winner"]["inference"] is not None
            or source["source"] != settings["probe_report"] or source["validation_keys"] != validation
            or baseline.matrix_only or baseline.context_size or baseline.coefficient_weights is None
            or baseline.product_compression is not None or len(set(all_keys)) != len(all_keys)
            or any(not group or len(group)>workers for group in groups)
            or any(frames[key]["split"]!="training" for key in fit+diagnostics)
            or any(frames[key]["split"]!="validation" for key in validation)):
        raise ValueError("Matrix-only fitting requires the frozen teacher and disjoint fixed splits")

    bank = {key:list(archive["probe_bank"][key]) for key in fit}
    for capture in source["captures"]:
        if capture["key"] in bank and capture["receipt"] is not None:
            bank[capture["key"]].append(capture["receipt"])

    def method(name, model, **context):
        return dict(name=name, factor_model=model, factor_context=dict(factor_fusion=True, **context))

    def checkpoint(model, name):
        path = Path(config["output_directory"])/"checkpoints"/("generator-%s-%s.npz" % (job, name))
        receipt, = share([save_model(path,model)] if p.global_rank==0 else [])
        return receipt

    dimension = baseline.layers[0][0].shape[1]+1
    moments = np.zeros((STAGES,dimension,dimension))
    fitting = []
    for index,key in enumerate(fit):
        if index%workers != p.worker:
            continue
        state = load(key)
        try:
            probes = load_probe_shards(p,bank[key])

            def observe(displacement):
                rows = []
                for level in (displacement.first, displacement.second):
                    for factor in level.fixed_factors:
                        x = np.column_stack((factor.x,np.ones(len(factor.x))))
                        count = int(factor.rows[-1])
                        moments[factor.stage] += p.reduce(x.T@x)/count
                        rows.append(count)
                losses,_ = action_losses(p,displacement,probes)
                return dict(failure=None,rows=rows,probes=losses,
                            signature=hierarchy_signature(displacement.numerical))

            result = solve(key,state,method("input_teacher",baseline,factor_tape=True,
                                             factor_operation=observe),"input-distill",details=False)
            if result["failure"] is not None or len(result["rows"])!=STAGES:
                raise ValueError("The frozen factor trajectory failed on "+key)
            if p.rank==0:
                fitting.append(dict(key=key,**result))
        finally:
            destroy(state)
    pieces = share([dict(rows=fitting,moments=moments.tolist())] if p.rank==0 else [])
    fitting = [row for piece in pieces for row in piece["rows"]]
    if sorted(row["key"] for row in fitting)!=sorted(fit):
        raise ValueError("Incomplete matrix-input distillation sample")
    initial = fit_matrix_encoder(baseline,np.sum([r["moments"] for r in pieces],axis=0)/len(fit),
                                 settings["regularization"])
    models = {"matrix_distilled":initial}
    receipts = {"matrix_distilled":checkpoint(initial,"matrix-distilled")}
    if p.global_rank==0:
        record(config,"factor_input_distilled",job=job,model=receipts["matrix_distilled"],
               fit_keys=fit, rows=np.sum([row["rows"] for row in fitting],axis=0).tolist())

    teacher_losses = {row["key"]:row["probes"] for row in fitting}
    update = settings["update"]
    model,step,history = initial,update["step_size"],[]
    for epoch in range(1,settings["rounds"]+1):
        for group in groups:
            states = {key:load(key) for i,key in enumerate(group) if i%workers==p.worker}
            try:
                probes = {key:load_probe_shards(p,bank[key]) for key in states}
                model,result,step = training_update(p,legacy,bridge,model,states,probes,teacher_losses,
                    solve,share,settings["numerical"],update,True,step)
            finally:
                for state in states.values():
                    destroy(state)
            history.append(dict(round=epoch,keys=group,update=result))
            if p.global_rank==0:
                record(config,"factor_input_batch",job=job,round=epoch,keys=group,accepted=result["accepted"])
    if any(row["update"]["accepted"] for row in history):
        models["matrix_tuned"] = model
        receipts["matrix_tuned"] = checkpoint(model,"matrix-tuned")

    frozen = method("frozen",baseline)
    student = dict(config["factor_reference"],fallback=False)
    margin = -settings["student_slack"]
    diagnostic_cases = evaluate(diagnostics,[frozen,student,*(method(name,m) for name,m in models.items())],
                                "input-diagnostics",repeats=2)
    pairs = {name:cost_pair(diagnostic_cases,diagnostics,"student",name,margin) for name in models}
    eligible = [name for name,pair in pairs.items() if pair["eligible"]]
    winner = min(eligible,key=lambda name:cost_rank(pairs[name])) if eligible else None
    cases,comparison = [],{}
    if winner is not None:
        cases = evaluate(validation,[frozen,student,method(winner,models[winner])],"input-validation",repeats=2)
        comparison = {name:cost_pair(cases,validation,name,winner,margin) for name in ("student","frozen")}
    ready = bool(comparison and comparison["student"]["eligible"])
    if p.global_rank==0:
        record(config,"factor_input_result",job=job,selected=winner,live_eligible=ready,
               diagnostic_ratios={name:pair["ratios"] for name,pair in pairs.items()},
               validation_ratios={name:pair["ratios"] for name,pair in comparison.items()})
    return dict(baseline=config["factor_model"],source=settings["source_report"],probes=settings["probe_report"],
                inputs=["displacement_matrix","coordinates","constraints"],fitting=fitting,history=history,
                models=receipts,diagnostic_keys=diagnostics,diagnostic_cases=diagnostic_cases,
                diagnostics=pairs,validation_keys=validation,cases=cases,comparison=comparison,
                winner=receipts.get(winner),winner_name=winner,live_eligible=ready)


def run_attention_point(config, p, legacy, replay, bridge, frames, load, destroy, solve, share, job):
    from .factor_operator import diagnose_attention_point
    from .live import verify_archived_fields

    point = config["factor_attention_point"]
    settings = config["factor_attention"]
    audit = checked_json(point["failed_report"])
    source = checked_json(settings["source_report"])
    archive = checked_json(settings["probe_report"])
    key = point["key"]
    expected, = [r for r in audit["audit"]["attention"]["rows"] if r["key"] == key]
    if (audit["job"] != point["job"] or audit["audit_passed"] or audit["history"]
            or expected["directional_derivatives"]["levels"][1]["stable_window"] is not None
            or key not in settings["batches"][0] or frames[key]["split"] != "training"
            or audit["baseline"] != config["factor_model"] or source["winner"]["model"] != audit["baseline"]
            or settings["source_report"] != audit["source"] or settings["probe_report"] != audit["probes"]
            or p.size*audit["workers"] != audit["cpu_ranks"]):
        raise ValueError("The attention diagnostic does not match the failed audit and MPI layout")
    baseline = load_model(config["factor_model"])
    added = np.zeros((len(baseline.coefficient_weights), 2*baseline.width), dtype=baseline.weights.dtype)
    model = replace(baseline, context_kind="attention",
                    coefficient_weights=np.column_stack((baseline.coefficient_weights, added)))
    bank = list(archive["probe_bank"][key])
    bank.extend(c["receipt"] for c in source["captures"] if c["key"] == key and c["receipt"] is not None)
    probes = load_probe_shards(p, bank)
    if [probe[0] for probe in probes] != audit["probe_names"][key]:
        raise ValueError("The original probe set changed")
    state = load(key)
    try:
        layout, fields, _ = state
        field_parity = verify_archived_fields(p, config, legacy, replay, frames[key], layout, fields)
        if p.global_rank == 0:
            record(config, "live_field_snapshot_parity", job=job, **field_parity)

        def operation(displacement):
            return diagnose_attention_point(p,legacy,bridge,displacement,model,probes,expected,
                                             dict(settings["numerical"], steps=point["steps"]))

        result = solve(key,state,dict(name="attention_point",factor_model=model,
            factor_context=dict(factor_tape=True,factor_operation=operation,factor_fusion=True)),
            "attention-point",details=False)
    finally:
        destroy(state)
    result, = share([result] if p.rank == 0 else [])
    return dict(key=key, failed_report=point["failed_report"], baseline=config["factor_model"],
                field_parity=field_parity, diagnostic=result)


def resolve_attention_audit(config, p, probe_names):
    """Reuse the frozen audit with one branch-local derivative resolution."""
    settings = config["factor_attention"]
    numerical = settings["numerical"]
    receipts = settings["audit_resume"]
    original = checked_json(receipts["original"])
    point = checked_json(receipts["point"])
    diagnostic = point["diagnostic"]
    if (original["audit_passed"] or original["history"] or original["models"]
            or original["cpu_ranks"] != p.global_size or original["workers"] != config["parallel_workers"]
            or original["baseline"] != config["factor_model"] or point["baseline"] != original["baseline"]
            or original["source"] != settings["source_report"] or original["probes"] != settings["probe_report"]
            or original["probe_names"] != probe_names or point["failed_report"] != receipts["original"]
            or point["mode"] != "factor_attention_point" or diagnostic["level"] != 2):
        raise ValueError("Attention audit receipts do not match the unchanged training setup")
    tolerance = numerical["finite_tolerance"]
    unresolved = []
    audit = original["audit"]
    for name, result in audit.items():
        if (not result["exact_initialization"]
                or {row["key"] for row in result["rows"]} != set(settings["batches"][0])):
            raise ValueError("Cannot reuse the initialization and scope of this audit")
        for row in result["rows"]:
            derivatives = row["directional_derivatives"]
            if (row["failure"] or derivatives["center_relative_error"] >= numerical["center_tolerance"]
                    or max(row["dot_errors"]) >= numerical["adjoint_tolerance"]):
                raise ValueError("An archived operator audit failed outside the branch diagnostic")
            for level in derivatives["levels"]:
                if level["stable_window"] is None or level["relative_error"] >= tolerance:
                    unresolved.append((name, row, level))
    if len(unresolved) != 1:
        raise ValueError("Expected one unresolved attention audit")
    name, row, level = unresolved[0]
    if (name != "attention" or row["key"] != point["key"] or level["level"] != diagnostic["level"]
            or level != diagnostic["original_level"] or row["signature"] != diagnostic["signature"]
            or list(row["probes"]) != diagnostic["probes"]):
        raise ValueError("The point diagnostic belongs to a different audit")

    analytic = diagnostic["analytic"]
    factors = diagnostic["local_factors"]
    fp64 = sum(f["coefficients_ad"] for f in factors)
    if abs(fp64-analytic) >= tolerance*abs(analytic):
        raise ValueError("The FP64 and production analytic gradients disagree")
    accepted = []
    for index, full in enumerate(diagnostic["full_operator_fp32"]):
        local = [f["measurements"][index] for f in factors]
        crossings = [m["production_crossings"] for m in local]
        if (full["relative_errors"]["plus"] >= tolerance
                or any(pair[1]["relu"] for pair in crossings)
                or any(side["clip"] for pair in crossings for side in pair)
                or not any(pair[0]["relu"] for pair in crossings)):
            continue
        local_errors = [max(m["pooling_fp64"]["relative_errors"]["central"],
                            m["signed_coefficients_fp64"]["relative_errors"]["plus"]) for m in local]
        minus = sum(m["signed_coefficients_fp64"]["slopes"]["minus"] for m in local)
        minus_gap = abs(minus-full["slopes"]["minus"])/abs(analytic)
        if max(local_errors+[minus_gap]) < tolerance:
            accepted.append((index, full, minus_gap))
    for low, high in zip(accepted, accepted[1:]):
        if high[0] != low[0]+1:
            continue
        a,b = low[1]["slopes"]["plus"],high[1]["slopes"]["plus"]
        agreement = abs(a-b)/max(abs(a),abs(b),abs(analytic))
        if agreement < tolerance/2:
            return audit, dict(key=row["key"], level=diagnostic["level"], side="plus",
                original=receipts["original"], point=receipts["point"], tolerance=tolerance,
                steps=[low[1]["step"],high[1]["step"]], relative_difference=agreement,
                relative_errors=[low[1]["relative_errors"]["plus"],high[1]["relative_errors"]["plus"]],
                minus_explanation=[low[2],high[2]])
    raise ValueError("No adjacent same-branch slopes resolved the failed attention audit")


def run_attention_replay(config, p, legacy, bridge, frames, load, destroy, solve, share, evaluate, job):
    """Test one candidate-local attention model against a matched capacity control."""
    settings = config["factor_attention"]
    source = checked_json(settings["source_report"])
    archive = checked_json(settings["probe_report"])
    validation = config["training"]["validation_keys"]
    diagnostics = settings["diagnostic_keys"]
    groups = settings["batches"]
    fit = [k for group in groups for k in group]
    numerical, update_settings = settings["numerical"], settings["update"]
    workers = config["parallel_workers"]
    if (source["winner"]["model"] != config["factor_model"] or source["winner"]["inference"] is not None
            or source["validation_keys"] != validation or archive["keys"] != validation
            or source["source"] != settings["probe_report"]
            or len(fit+diagnostics+validation) != len(set(fit+diagnostics+validation))
            or any(len(batch)>workers for batch in groups)
            or any(frames[k]["split"]!="training" for k in fit+diagnostics)
            or any(frames[k]["split"]!="validation" for k in validation)):
        raise ValueError("Attention must use the frozen checkpoint and the fixed training/validation split")
    baseline = load_model(config["factor_model"])
    if baseline.context_size or baseline.coefficient_weights is None or baseline.product_compression is not None:
        raise ValueError("Attention starts from the uncompressed mean-context generator")
    bank = {k:list(archive["probe_bank"][k]) for k in fit}
    for capture in source["captures"]:
        if capture["receipt"] is not None and capture["key"] in bank:
            bank[capture["key"]].append(capture["receipt"])
    names = ("attention", "capacity")
    added = np.zeros((len(baseline.coefficient_weights),2*baseline.width),dtype=baseline.weights.dtype)
    models = {name:replace(baseline, context_kind=name,
                           coefficient_weights=np.column_stack((baseline.coefficient_weights,added))) for name in names}

    def method(name, model):
        return dict(name=name, factor_model=model, factor_context=dict(factor_fusion=True))

    def measure(model, states, probes, audit=False):
        return operator_rows(p, legacy, bridge, model, states, probes, solve, share, numerical,
                             True, gradient=audit, audit=audit)

    def local_batch(keys):
        return {k:load(k) for i,k in enumerate(keys) if i%workers==p.worker}

    probe_names = {k:[r["phase"]+"/"+n for r in bank[k] for n in r["probes"]] for k in fit}
    resolution = None
    if "audit_resume" in settings:
        audit,resolution = resolve_attention_audit(config,p,probe_names)
        verified = True
    else:
        states = local_batch(groups[0])
        try:
            probes = {k:load_probe_shards(p, bank[k]) for k in states}
            reference = measure(baseline, states, probes)
            audit = {}
            for name in names:
                rows = measure(models[name], states, probes, audit=True)
                exact = len(reference)==len(groups[0]) and equivalent_probe_actions(reference, rows)
                passed = bool(exact and all(r["directional_derivatives"]["passed"] for r in rows))
                audit[name] = dict(exact_initialization=exact, passed=passed,
                                   rows=[{k:v for k,v in r.items() if k!="gradient"} for r in rows])
        finally:
            for state in states.values():
                destroy(state)
        verified = all(a["passed"] for a in audit.values())
    result = dict(source=settings["source_report"], probes=settings["probe_report"],
        baseline=config["factor_model"], fit_keys=fit, diagnostic_keys=diagnostics, validation_keys=validation,
        probe_names=probe_names, new_parameters_per_model=int(added.size),
        audit=audit, audit_resolution=resolution, audit_passed=verified,
        history=[], models={}, diagnostic_cases=[], diagnostic_comparison={}, cases=[],
        validation_comparison={}, speed_gate=False, research_gate=False)
    if p.global_rank==0:
        record(config, "factor_attention_audit", job=job, keys=groups[0],
               passed={name:r["passed"] for name,r in audit.items()}, resolution=resolution)
    if not verified:
        return result

    steps = {name:update_settings["step_size"] for name in names}
    for epoch in range(1,settings["rounds"]+1):
        for index,keys in enumerate(groups):
            states = local_batch(keys)
            try:
                probes = {k:load_probe_shards(p,bank[k]) for k in states}
                reference = measure(baseline,states,probes)
                if len(reference)!=len(keys) or any(r["failure"] is not None for r in reference):
                    raise ValueError("The frozen generator cannot score a common training batch")
                targets = {r["key"]:r["probes"] for r in reference}
                updates = {}
                for name in names if (epoch+index)%2 else names[::-1]:
                    models[name],updates[name],steps[name] = training_update(
                        p,legacy,bridge,models[name],states,probes,targets,solve,share,
                        numerical,update_settings,True,steps[name])
            finally:
                for state in states.values():
                    destroy(state)
            result["history"].append(dict(round=epoch, keys=keys, updates=updates))
            if p.global_rank==0:
                record(config,"factor_attention_batch",job=job,round=epoch,keys=keys,
                       accepted={k:r["accepted"] for k,r in updates.items()})

    output = Path(config["output_directory"])
    for name in names:
        model = models[name]
        local = []
        if p.global_rank==0:
            local.append(save_model(output/"checkpoints"/("generator-%s-%s.npz"%(job,name)),model))
        receipt, = share(local)
        result["models"][name] = dict(model=receipt,
            context_delta=np.linalg.norm(model.coefficient_weights[:,-model.context_size:],axis=1).tolist(),
            base_coefficient_delta=np.linalg.norm(model.coefficient_weights[:,:baseline.weights.size]
                                                  -baseline.coefficient_weights,axis=1).tolist(),
            accepted=sum(r["updates"][name]["accepted"] for r in result["history"]))
    methods = [method("frozen",baseline), *(method(name,models[name]) for name in names)]
    cases = evaluate(diagnostics,methods,"factor-attention-diagnostic",repeats=2)
    diagnostic = {name:cost_pair(cases,diagnostics,"frozen",name,settings["selection_margin"]) for name in names}
    result.update(diagnostic_cases=cases,diagnostic_comparison=diagnostic)
    selected = diagnostic["attention"]["eligible"] and all(d>0. for d in result["models"]["attention"]["context_delta"])
    if p.global_rank==0:
        record(config,"factor_attention_diagnostics",job=job,selected=bool(selected),
               ratios={k:r["ratios"] for k,r in diagnostic.items()})
    if selected:
        student = dict(config["factor_reference"],fallback=False)
        cases = evaluate(validation,[*methods,student],"factor-attention-validation",repeats=2)
        compare = {name:cost_pair(cases,validation,name,"attention",settings["speed_margin"])
                   for name in ("frozen","capacity","student")}
        research = cost_pair(cases,validation,"student","attention",settings["research_margin"])
        result.update(cases=cases,validation_comparison=compare,
                      speed_gate=all(compare[k]["eligible"] for k in ("frozen","student")),
                      research_gate=research["eligible"],research_comparison=research)
    return result
