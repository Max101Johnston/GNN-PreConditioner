"""MPI entry points for the learned interpolation cycle."""
import json
import os
from pathlib import Path

import numpy as np

from .backend import digest, record, read_json
from .online import global_objects, load_observed_state, solve_policy
from .student import DEFAULT_CONTROL, load_student


def save_once(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)


def compact(result, frame, method, elapsed=None, fallback=None):
    n = result["numerical"]
    first, second = n.get("level1", {}), n.get("level2", {})
    brief = dict(key=frame["key"], split=frame["split"], method=method,
        passed=result["reached_true_target"] and not result.get("timed_out"), failure=result["failure"],
        total_s=result.get("total_wall_s", result["setup_wall_s"]) if elapsed is None else elapsed,
        setup_s=result["setup_wall_s"], solve_s=result.get("solve_s"),
        iterations=result.get("iterations"), final_true_residual=result.get("final_true_residual"),
        true_limit=result.get("true_limit"), pc_apply_s=result["displacement_pc_apply_s"],
        pc_apply_count=result["displacement_pc_apply_count"],
        p1_nnz=first.get("nnz"), fine_dofs=first.get("fine_dofs"),
        k1_nnz=n.get("k1_nnz"), p2_nnz=second.get("nnz"),
        c1=first.get("coarse_dofs"), c2=second.get("coarse_dofs"),
        p1_candidate_s=first.get("candidate_s"), p1_inference_s=first.get("native_inference_s"),
        p2_candidate_s=second.get("candidate_s"), p2_inference_s=second.get("native_inference_s"),
        p1_budget_counts=first.get("budget_counts"),
        p1_budget_options=first.get("budget_options"),
        teacher_guard_rows=first.get("teacher_guard_rows"),
        galerkin_errors=n.get("galerkin_errors"), fallback=fallback,
        feedback_pc_apply_calls=result.get("feedback_pc_apply_calls"),
        factor_levels=[first, second] if result["method"] == "gnn_factorized" else None,
        graph_timings=result.get("timings") if result["method"] == "gnn_factorized" else None,
        cycle_probes=result.get("student_cycle_probes"),
        short_krylov=result.get("short_krylov"),
        middle_ilu_adjoint_parity=result.get("middle_ilu_adjoint_parity"),
        coefficient_feedback=result.get("coefficient_feedback"),
        factor_policy=result.get("factor_policy"),
        hierarchy_signature=result.get("hierarchy_signature"),
        factor_trajectories=result.get("factor_trajectories"), factor_probe_s=result.get("factor_probe_s"),
        teacher=result.get("student_teacher"))
    for name in ("setup_breakdown", "displacement_apply_profile"):
        if name in result:
            brief[name] = result[name]
    return brief


def method_totals(cases):
    totals = {}
    for label in sorted({r["method"] for r in cases}):
        rows = [r for r in cases if r["method"] == label]
        pc_calls = sum(r.get("pc_apply_count") or 0 for r in rows)
        pc_s = sum(r.get("pc_apply_s") or 0.0 for r in rows)
        totals[label] = dict(frames=len(rows), failed=sum(not r["passed"] for r in rows),
            fallback_count=sum(r["fallback"] is not None for r in rows),
            total_s=sum(r["total_s"] for r in rows),
            setup_s=sum(r["setup_s"] for r in rows),
            solve_s=sum(r.get("solve_s") or 0.0 for r in rows),
            iterations=sum(r.get("iterations") or 0 for r in rows),
            p1_nnz=sum(r.get("p1_nnz") or 0 for r in rows),
            k1_nnz=sum(r.get("k1_nnz") or 0 for r in rows),
            p2_nnz=sum(r.get("p2_nnz") or 0 for r in rows),
            fine_dofs=sum(r.get("fine_dofs") or 0 for r in rows),
            pc_apply_count=pc_calls, pc_apply_s=pc_s,
            pc_apply_mean_s=pc_s/pc_calls if pc_calls else None,
            p1_candidate_s=sum(r.get("p1_candidate_s") or 0.0 for r in rows),
            p1_inference_s=sum(r.get("p1_inference_s") or 0.0 for r in rows),
            p2_candidate_s=sum(r.get("p2_candidate_s") or 0.0 for r in rows),
            p2_inference_s=sum(r.get("p2_inference_s") or 0.0 for r in rows))
    return totals


def retained_teacher(config, row):
    old = Path(config["baseline_directory"])
    prefix = config.get("retained_export_prefixes", {}).get(row["split"], "retained-solver")
    path = old / "records" / ("solver-export-"+prefix+"-"+row["key"]+".json")
    if not path.exists():
        return None
    receipt = read_json(path)
    entry = next(e for e in receipt["spaces"] if e["label"] == config["baseline_label"])
    return dict(retained_entry=entry)


def factor_reward(case, reference, config):
    from .objective import time_reward

    settings = config["training"]
    reward = time_reward(reference, case["total_s"], settings["common_time_scale_s"],
                         settings["episode_timeout_seconds"], case["passed"])
    if not case["passed"]:
        probes = [row["residual_ratio"] for row in case["cycle_probes"] or ()]
        if probes:
            # This only ranks failed proposals and remains below any
            # converged reward. Probes run through each student's own PC.
            reward += .25 / (1 + np.mean(probes))
        elif case["final_true_residual"] and case["true_limit"]:
            ratio = case["final_true_residual"] / case["true_limit"]
            reward += .1 / (1 + np.log1p(ratio))
    return float(reward)


def tune_factors(config, p, frames, train, load, destroy, solve, share, evaluate, job):
    from .factorized import FactorModel, REGIONS, load_model, save_model

    setting = config["factorized"]
    model = load_model(config["factor_model"])
    initial = model
    current = config["factor_reference"]
    workers = config["parallel_workers"]
    lanes = [train[i::workers] for i in range(workers)]
    hard_keys = setting["hard_keys"]
    assert set(hard_keys) <= set(train) and all(lanes)
    rng = np.random.default_rng(setting["seed"])
    batch_count = setting["updates"]
    sigma = setting["sigma"]
    output = Path(config["output_directory"])
    receipts, reference, probes, permutations = [], {}, {}, [[] for _ in lanes]
    for step in range(batch_count):
        keys = []
        for lane, members in enumerate(lanes):
            if not permutations[lane]:
                permutations[lane] = list(rng.permutation(members))
            hard = [key for key in hard_keys if key in members]
            key = hard[(step//4) % len(hard)] if hard and step % 4 == 0 else permutations[lane].pop()
            keys.append(str(key))
        region = int(rng.integers(REGIONS))
        direction = np.zeros_like(model.adapters)
        direction[:, region] = rng.choice([-1., 1.], size=(direction.shape[0], direction.shape[-1]))
        key = keys[p.worker]
        loaded = load(key)
        try:
            if key not in reference:
                probes[key] = []
                baseline = solve(key, loaded, dict(current, capture_inputs=probes[key]),
                                 "factor-reference", details=True)
                if not baseline["passed"]:
                    baseline = solve(key, loaded, dict(name="amg"), "factor-reference-amg")
                if not baseline["passed"]:
                    raise ValueError("Factor training reference did not reach its true residual")
                reference[key] = baseline["total_s"]
            local_rows = []
            signs = (1, -1) if (p.worker + step) % 2 == 0 else (-1, 1)
            for sign in signs:
                proposal = FactorModel(model.width, model.weights, model.adapters + sign*sigma*direction,
                                       model.unit_sum_depth)
                name = "plus" if sign == 1 else "minus"
                result = solve(key, loaded, dict(name=name, factor_model=proposal, probe_inputs=probes[key]),
                               "factor-%03d-%s" % (step, name), details=True)
                result.update(reward=factor_reward(result, reference[key], config),
                              reference_seconds=reference[key], sign=sign)
                local_rows.append(result)
        finally:
            destroy(loaded)
        rows = share(local_rows)
        assert len(rows) == workers*2
        reward = {sign: float(np.mean([r["reward"] for r in rows if r["sign"] == sign]))
                  for sign in (1, -1)}
        derivative = np.clip((reward[1] - reward[-1]) / (2*sigma), -2., 2.)
        lr = setting["feedback_lr"]/np.sqrt(1+step/10)
        effects = {}
        for case in rows:
            ratios = [item["residual_ratio"] for item in case["cycle_probes"] or ()]
            if ratios:
                effects[(case["key"], case["sign"])] = float(np.mean(1/(1+np.asarray(ratios))))
        paired = [effects[key, 1]-effects[key, -1] for key in keys
                  if (key, 1) in effects and (key, -1) in effects]
        own_cycle = float(np.clip(np.mean(paired)/(2*sigma), -1., 1.)) if paired else 0.
        gradient = derivative*direction
        gradient[:, :, 4] += setting["probe_weight"]*own_cycle*direction[:, :, 4]
        updated = np.clip(model.adapters + lr*gradient, -2., 2.).astype(np.float32)
        change = float(np.linalg.norm(updated-model.adapters))
        observation = dict(step=step+1, keys=keys, region=region, candidates=rows,
                           mean_reward_plus=reward[1], mean_reward_minus=reward[-1],
                           adapter_change=change, own_cycle_derivative=own_cycle,
                           probe_pairs=len(paired), sigma=sigma,
                           perturbation=direction[:, region].tolist(), adapters=updated.tolist())
        receipts.append(observation)
        model = FactorModel(model.width, model.weights, updated, model.unit_sum_depth)
        if p.global_rank == 0:
            checkpoint = save_model(output/"checkpoints"/("factorized-%s-step%03d.npz" % (job, step+1)),
                                    model)
            observation["checkpoint"] = checkpoint
            with (output/"records"/("factor-pairs-"+job+".jsonl")).open("a") as stream:
                stream.write(json.dumps(observation, allow_nan=False)+"\n")
            record(config, "factor_joint_update", job=job, step=step+1, keys=keys, region=region,
                   reward_plus=reward[1], reward_minus=reward[-1], adapter_change=change,
                   own_cycle_derivative=own_cycle, candidate_passes=sum(r["passed"] for r in rows))
    validation = config["training"]["validation_keys"]
    assert len(set(validation)) == len(validation) and all(frames[key]["split"] == "validation" for key in validation)
    methods = [current, dict(name="amg"), dict(name="retained_base"),
               dict(name="joint_initial", factor_model=initial), dict(name="joint", factor_model=model)]
    cases = evaluate(validation, methods, "factor-validation")
    first_pass = method_totals(cases)
    repeats_run = 1
    if (setting["validation_repeats"] > 1 and not first_pass["joint"]["failed"]
            and first_pass["joint"]["total_s"] < first_pass["student"]["total_s"]):
        repeats_run = setting["validation_repeats"]
        cases += evaluate(validation, methods, "factor-validation",
                          repeats=repeats_run-1, start_repeat=1)
    totals = method_totals(cases)
    required = len(validation)*repeats_run
    for method in methods:
        assert totals[method["name"]]["frames"] == required
    repeats = [method_totals([r for r in cases if r["repeat"] == repeat])
               for repeat in range(repeats_run)]
    return dict(keys=validation, split="validation", cases=cases, totals=totals, repeat_totals=repeats,
                training_batches=batch_count, training_frames=len({k for r in receipts for k in r["keys"]}),
                joint_updates=sum(r["adapter_change"] > 0 for r in receipts), updates=receipts,
                model=receipts[-1].get("checkpoint"),
                scope="Fresh two-level hierarchy per matrix, with both Galerkin operators and full solve")





def schedule_identity(settings):
    import hashlib

    contents = {key:value for key,value in settings.items() if key != "resume"}
    return hashlib.sha256(json.dumps(contents, sort_keys=True, allow_nan=False).encode()).hexdigest()


def coefficient_step(model, cases, step_size, balanced=False):
    from .factorized import FactorModel

    feedback = [r["coefficient_feedback"] for r in cases if r["passed"]]
    if len(feedback) != len(cases) or not feedback or not all((r or {}).get("verified") for r in feedback):
        return None, dict(verified=False, failure="coefficient_feedback_gate")
    gradients = np.asarray([row["gradient"] for row in feedback])
    norms = np.linalg.norm(gradients, axis=1)
    if balanced:
        gradients = gradients/np.maximum(norms[:,None], 1e-12)
    gradient = gradients.mean(0)
    norm = float(np.linalg.norm(gradient))
    if not np.isfinite(norm) or norm <= 1e-14:
        return None, dict(verified=False, failure="cancelled_or_nonfinite_gradient")
    updated = (model.weights - step_size*gradient/norm).astype(model.weights.dtype)
    changes, offset = [], 0
    for weight,bias in model.layers:
        end = offset+weight.size+bias.size
        changes.append(float(np.linalg.norm(updated[offset:end]-model.weights[offset:end])))
        offset = end
    if not (changes[0] > 0 and changes[4] > 0):
        return None, dict(verified=False, failure="no_shared_or_signed_parameter_update")
    return FactorModel(model.width, updated, model.adapters, model.unit_sum_depth), dict(
        verified=True, gradient_norm=norm, frame_gradient_norms=norms.tolist(), layer_delta=changes)



def learning_archive(config):
    """Read a frozen prefix of completed rollouts and coefficient sources."""
    source = config.get("factor_learning_archive")
    if source is None:
        return {}, {}
    history = source["policy"]
    if digest(history["path"]) != history["sha256"]:
        raise ValueError("Archived policy observations changed")
    with Path(history["path"]).open() as stream:
        rows = [json.loads(line) for line in stream]
    if [r["step"] for r in rows] != list(range(1, source["through"]+1)):
        raise ValueError("Archived structural updates are not a complete prefix")
    coefficients = {}
    for value in source["coefficients"]:
        cases = []
        for reference in value["cases"]:
            if digest(reference["path"]) != reference["sha256"]:
                raise ValueError("Archived coefficient case changed")
            cases.append(read_json(reference["path"])["compact"])
        coefficients[value["step"]] = value["update"], cases
    return {r["step"]:r for r in rows}, coefficients


def replayed_candidate(current, original):
    """Verify the action, then use its archived full-solve reward and new score."""
    policy = dict(current["factor_policy"])
    gradient = policy.pop("gradient")
    expected = dict(original["factor_policy"])
    old_prob = expected.pop("log_probability")
    new_prob = policy.pop("log_probability")

    def support(level):
        return [(q["signatures"]["representatives"], q["signatures"]["support"])
                for q in level.get("stages", [])]

    actual_levels = [current["numerical"].get(k, {}) for k in ("level1","level2")]
    previous_levels = original["factor_levels"]
    match = (policy == expected and np.isclose(new_prob, old_prob, rtol=1e-6, atol=1e-8)
             and [support(x) for x in actual_levels] == [support(x) for x in previous_levels]
             and (current["failure"] is None or current["failure"] == original["failure"]))
    row = dict(original, factor_policy=dict(original["factor_policy"], gradient=gradient),
               replay_match=bool(match))
    return row

def tune_structure(config, p, keys, load, destroy, solve, share, job, model, evaluate=None):
    """Evaluate independent constrained choices against same-frame candidates."""
    from .factorized import FactorModel, REGIONS, STAGES, save_model

    settings = config["factor_structure"]
    workers = config["parallel_workers"]
    lanes = [list(keys[i::workers]) for i in range(workers)]
    assert all(lanes) and settings["candidates"] >= 2
    rng = np.random.default_rng(settings["seed"])
    permutations = [[] for _ in lanes]
    history, milestones = [], []
    output = Path(config["output_directory"])
    roles = ("representatives","support","budget")
    coefficient_every = settings.get("coefficient_every", 0)
    validation_every = settings.get("validation_every", 0)
    use_adam = settings.get("adam", False)
    moments = {role: (np.zeros_like(model.weights, dtype=np.float64),
                      np.zeros_like(model.weights, dtype=np.float64), 0) for role in roles}
    start = 0
    covered = set()
    effective = dict.fromkeys(roles, 0)
    coefficient_updates = 0
    if "resume" in settings:
        source = settings["resume"]
        if digest(source["path"]) != source["sha256"]:
            raise ValueError("Structural training resume receipt changed")
        saved = read_json(source["path"])
        assert saved["model"] == config["factor_model"] and saved["keys"] == keys
        assert saved["schedule_identity"] == schedule_identity(settings)
        start, milestones = saved["step"], saved["milestones"]
        covered = set(saved["coverage"])
        effective = saved["effective_structure_updates"]
        coefficient_updates = saved["coefficient_updates"]
        moments = {role: (np.asarray(values[0]), np.asarray(values[1]), values[2])
                   for role,values in saved["moments"].items()}
    if coefficient_every or validation_every:
        assert evaluate is not None
    schedule = settings.get("schedule") or [dict(stage=(i//len(roles))%STAGES,
        region=(i//(len(roles)*STAGES))%REGIONS, role=roles[i%len(roles)])
        for i in range(settings["updates"])]
    assert 0 <= start < len(schedule)
    archive, archived_coefficients = learning_archive(config)
    archive_end = max(archive, default=0)
    rejected = None
    for step, action in enumerate(schedule):
        batch = []
        for lane, members in enumerate(lanes):
            if not permutations[lane]:
                permutations[lane] = list(rng.permutation(members))
            batch.append(str(permutations[lane].pop()))
        subsets = rng.integers(0, 2**32, size=workers, dtype=np.uint32)
        seeds = rng.integers(0, 2**32, size=(workers,settings["candidates"]), dtype=np.uint32)
        if step < start:
            continue
        number = step+1
        covered.update(batch)
        order = np.roll(np.arange(settings["candidates"]), (step+p.worker)%settings["candidates"])
        if (step//settings["candidates"]+p.worker)%2:
            order = order[::-1]
        key = batch[p.worker]
        before = model
        archived = archive.get(number)
        coefficient_update = None
        while True:
            model = before
            restored = archived is not None and rejected is None
            cached_coefficient = archived_coefficients.get(number) if rejected is None else None
            coefficient_update = None
            if coefficient_every and number % coefficient_every == 0:
                feedback = dict(config["factor_feedback"], component=(number//coefficient_every)%3)
                audited = bool(validation_every and number % validation_every == 0)
                feedback["finite_difference"] = feedback["finite_difference"] if audited else 0.
                if audited:
                    feedback["steps"] = [4,12,4]
                if cached_coefficient:
                    known, cases = cached_coefficient
                    indexed = {r["key"]:r for r in cases}
                    assert set(indexed) == set(batch)
                    source = [indexed[name] for name in batch]
                else:
                    source = evaluate(batch, [dict(name="coefficient_source", factor_model=model,
                                                   factor_feedback=feedback)], "factor-coefficient-%03d"%number)
                stepped, info = coefficient_step(model, source, feedback["step_size"], balanced=True)
                if cached_coefficient and (not info["verified"] or not known["verified"] or
                        not np.allclose(info["layer_delta"], known["layer_delta"], rtol=1e-5, atol=2e-8)):
                    rejected = number
                    if p.global_rank == 0:
                        record(config, "factor_learning_replay_rejected", job=job, step=number,
                               check="coefficient_update")
                    continue
                coefficient_update = dict(info, finite_difference_audit=audited,
                    replayed=bool(cached_coefficient), cases=[
                    dict(key=r["key"], passed=r["passed"], iterations=r["iterations"],
                         feedback={k:v for k,v in (r["coefficient_feedback"] or {}).items() if k!="gradient"})
                    for r in source])
                if stepped is not None:
                    model = stepped
            local = []
            if restored:
                assert archived["keys"] == batch and all(archived[k]==v for k,v in action.items())
                previous = {(r["key"],r["method"]):r for r in archived["candidates"]}
            loaded = load(key)
            try:
                for slot in order:
                    sample = dict(action, seed=int(seeds[p.worker,slot]), subset_seed=int(subsets[p.worker]),
                                  temperature=settings["temperature"], actions_per_rank=settings["actions_per_rank"])
                    name = "structure-%d" % (slot+1)
                    candidate = dict(name=name, factor_model=model, factor_policy=sample)
                    if restored:
                        candidate["factor_policy_replay"] = True
                    row = solve(key, loaded, candidate, "policy-%03d-%s" % (number,name), details=False)
                    local.append(replayed_candidate(row, previous[key,name]) if restored else row)
            finally:
                destroy(loaded)
            cases = share(local)
            if restored and not all(r["replay_match"] for r in cases):
                rejected = number
                if p.global_rank == 0:
                    record(config, "factor_learning_replay_rejected", job=job, step=number,
                           check="executed_actions", cases=[r["key"]+"/"+r["method"] for r in cases
                                                            if not r["replay_match"]])
                continue
            delta = np.zeros_like(model.weights, dtype=np.float64)
            for frame in batch:
                group = [row for row in cases if row["key"] == frame]
                assert len(group) == settings["candidates"]
                reference = config["training"]["common_time_scale_s"]
                rewards = np.array([factor_reward(row, reference, config) for row in group])
                baseline = (rewards.sum()-rewards)/(len(group)-1)
                for row, reward, base in zip(group, rewards, baseline):
                    row.update(reward=float(reward), leave_one_out=float(base), advantage=float(reward-base))
                    receipt = row["factor_policy"]
                    delta += row["advantage"] * np.asarray(receipt.pop("gradient"))
            delta /= len(cases)
            norm = float(np.linalg.norm(delta))
            if not np.isfinite(norm):
                raise ValueError("Non-finite structural score gradient")
            clip = min(1., settings["gradient_clip"]/max(norm, 1e-30))
            direction = clip*delta
            role = action["role"]
            state = None
            if use_adam and norm > 0:
                mean, variance, count = moments[role]
                count += 1
                mean = .9*mean + .1*direction
                variance = .99*variance + .01*direction**2
                state = mean, variance, count
                direction = (mean/(1-.9**count))/(np.sqrt(variance/(1-.99**count))+1e-7)
            increment = settings["learning_rate"]*direction
            if "max_parameter_step" in settings:
                increment *= min(1., settings["max_parameter_step"]/max(np.linalg.norm(increment), 1e-30))
            updated = (model.weights + increment).astype(model.weights.dtype)
            change = float(np.linalg.norm(updated-model.weights))
            if restored and not np.allclose([norm,change], [archived["gradient_norm"],archived["parameter_change"]],
                                            rtol=1e-5, atol=2e-8):
                rejected = number
                if p.global_rank == 0:
                    record(config, "factor_learning_replay_rejected", job=job, step=number,
                           check="parameter_increment", gradient_norm=norm, parameter_change=change)
                continue
            if state is not None:
                moments[role] = state
            model = FactorModel(model.width, updated, model.adapters, model.unit_sum_depth)
            if coefficient_update:
                coefficient_updates += int(coefficient_update["verified"])
                if p.global_rank == 0:
                    record(config, "factor_coefficient_update", job=job, step=number,
                           **{k:v for k,v in coefficient_update.items() if k!="cases"})
            break
        observation = dict(step=number, keys=batch, **action, coefficient_update=coefficient_update,
            replayed=restored,
            candidate_passes=sum(row["passed"] for row in cases),
            actions=sum(row["factor_policy"]["actions"] for row in cases),
            changes=sum(row["factor_policy"]["changes"] for row in cases),
            gradient_norm=norm, parameter_change=change, candidates=cases)
        history.append(observation)
        if change > 0 and observation["changes"] > 0 and observation["candidate_passes"] > 0:
            effective[action["role"]] += 1
        summary = None
        if validation_every and (number % validation_every == 0 or number == len(schedule)):
            validation = config["training"]["validation_keys"]
            current = dict(name="learning-%03d"%number, factor_model=model)
            cases = evaluate(validation, [current], "factor-learning-validation-%03d"%number)
            assert len(cases) == len(validation)
            summary = method_totals(cases)[current["name"]]
        if p.global_rank == 0 and (summary is not None or (restored and number == archive_end)):
            path = output/"checkpoints"/("factorized-learning-%s-step%03d.npz" % (job,number))
            receipt = save_model(path, model)
            if summary is not None:
                milestones.append(dict(step=number, model=receipt, totals=summary, cases=cases))
            state = dict(step=number, model=receipt, keys=keys, milestones=milestones,
                         schedule_identity=schedule_identity(settings), coverage=sorted(covered),
                         effective_structure_updates=effective, coefficient_updates=coefficient_updates,
                         moments={name:[m.tolist(),v.tolist(),n] for name,(m,v,n) in moments.items()})
            state_path = output/"records"/("factor-learning-%s-step%03d.json" % (job,number))
            save_once(state_path, state)
            event = "factor_learning_recovered" if summary is None else "factor_learning_validation"
            record(config, event, job=job, step=number,
                   model=receipt, resume=dict(path=str(state_path),sha256=digest(state_path)), **(summary or {}))
        if p.global_rank == 0:
            with (output/"records"/("factor-policy-"+job+".jsonl")).open("a") as stream:
                stream.write(json.dumps(observation, allow_nan=False)+"\n")
            record(config, "factor_structure_update", job=job, step=step+1, role=action["role"],
                   stage=action["stage"]+1, region=action["region"], sampled=observation["actions"],
                   changed=observation["changes"], parameter_change=change, replayed=restored,
                   candidate_passes=observation["candidate_passes"])
    checkpoint = None
    if p.global_rank == 0:
        checkpoint = save_model(output/"checkpoints"/("factorized-structure-"+job+".npz"), model)
    checkpoint, = [value for value in global_objects(p, checkpoint) if value]
    measured = all(row["actions"] > 0 and row["changes"] > 0 and row["parameter_change"] > 0
                   and row["candidate_passes"] > 0 for row in history)
    milestones = global_objects(p, milestones)[0]
    return model, dict(model=checkpoint, verified=measured, steps=history, milestones=milestones,
                       resumed_after=start, coverage=sorted(covered), effective_structure_updates=effective,
                       coefficient_updates=coefficient_updates, completed_steps=start+len(history),
                       replay_rejected_at=rejected,
                       replayed_steps=sum(row["replayed"] for row in history),
                       candidate_solves=sum(len(row["candidates"]) for row in history if not row["replayed"]))


def run_cycle(config, p, replay, legacy, bridge, inventory):
    from .gate import RetainedFirst, load_teacher, run_method

    mode = config["student_cycle"]
    workers = config["parallel_workers"]
    job = os.environ["GNN_RUN_ID"]
    records = Path(config["output_directory"])/"records"
    frames = {row["key"]: row for row in inventory["snapshots"]}
    train = [row["key"] for row in frames.values() if row["split"] == "training"]
    methods = config.get("student_methods", [])
    for method in methods:
        for name, model in method.get("students", {}).items():
            load_student(model["path"], model["sha256"], int(name[1:]))
    refresh = config.get("refresh_p1")
    if refresh:
        load_student(refresh["path"], refresh["sha256"], 1)
    feedback = {}

    def load(key):
        return load_observed_state(config, p, replay, frames[key])

    def destroy(loaded):
        data = loaded[2]
        p.destroy("Vec", data[3], data[4])
        p.destroy("Mat", data[2])

    def solve(key, loaded, method, tag, export_level=0, details=True):
        layout, fields, data = loaded
        row, label = frames[key], method["name"]
        models = method.get("students", {})
        direct = label == "direct_p1" or method.get("direct", False)
        if "factor_model" in method:
            if method["factor_model"].matrix_only:
                fields = dict(constrained=fields["constrained"])
            context = dict(factor_model=method["factor_model"], archive_choices=False, tag=tag,
                           probe_inputs=method.get("probe_inputs"), short_krylov=method.get("short_krylov"),
                           factor_feedback=method.get("factor_feedback"))
            context.update(method.get("factor_context", {}))
            for setting in ("factor_probe_inputs", "factor_probe_capture", "factor_probe_windows",
                            "factor_probe_only", "factor_signatures"):
                if setting in method:
                    context[setting] = method[setting]
            if "factor_policy" in method:
                from .factorized import FactorPolicy

                policy = method["factor_policy"]
                actor = FactorPolicy
                if policy.get("search"):
                    from .factor_search import SearchProposal

                    actor = SearchProposal
                context["factor_policy"] = actor(p, method["factor_model"], policy)
                context["factor_policy_replay"] = method.get("factor_policy_replay", False)
            result = solve_policy(config, p, replay, legacy, bridge, None, row, layout, fields, data,
                                  context=context)
            if (context.get("factor_policy_replay") or context.get("factor_probe_only")
                    or context.get("factor_operation") is not None):
                return result
            if result.get("duplicate"):
                return dict(result, key=key, method=label)
        elif label in ("amg", "retained_base", "factor_teacher"):
            teacher = retained_teacher(config,row) if label=="retained_base" else None
            if label=="factor_teacher":
                teacher = dict(fields=fields)
            if label == "retained_base" and teacher is None:
                return None
            result = run_method(config, p, replay, legacy, bridge, row, layout, data, label, teacher,
                                method.get("factor_probe_inputs"), method.get("factor_probe_windows"),
                                method.get("displacement_inputs"))
            if "factor_probe_inputs" in method or "displacement_inputs" in method:
                return result
        elif label == "teacher_roundtrip" or direct:
            teacher = load_teacher(config, p, row, layout)
            teacher["pair"] = RetainedFirst(config, p, legacy, row, teacher, assemble=False)
            teacher["geometry"] = (layout["xyz"], fields["phi"])
            if direct:
                teacher.update(direct_model=models["p1"], direct_control=method.get("teacher_control"))
            if method.get("capture_apply"):
                teacher["capture_apply"] = feedback.setdefault(key, [])
            if method.get("export_direct"):
                lo, hi = teacher["pair"].fine_owners[p.rank:p.rank+2]
                teacher.update(export_direct=True, phase=fields["phi"][teacher["native"][lo:hi]//3],
                               feedback_inputs=feedback.pop(key, []))
            solver_method = "direct_p1" if direct else label
            result = run_method(config, p, replay, legacy, bridge, row, layout, data, solver_method, teacher)
        else:
            context = dict(policy="prior_only", fast_prior=True, tag=tag, p1_refinement_sweeps=2,
                           students=models, export_teacher_level=export_level,
                           archive_choices=False, capture_inputs=method.get("capture_inputs"))
            if export_level == 2:
                context.update(p2_builder="weights", p2_expand_budget=8)
            for level in ("p1", "p2"):
                if "controls_"+level in method:
                    context["controls_"+level] = method["controls_"+level]
            if "terminal_min_radius" in method:
                context["terminal_min_radius"] = method["terminal_min_radius"]
            result = solve_policy(config, p, replay, legacy, bridge, None, row, layout, fields, data,
                                  context=context)
        fallback = None
        elapsed = result.get("total_wall_s", result["setup_wall_s"])
        if not direct and models and not result["reached_true_target"] and method.get("fallback", False):
            safe = solve_policy(config, p, replay, legacy, bridge, None, row, layout, fields, data,
                context=dict(policy="prior_only", fast_prior=True, tag=tag+"-fallback",
                             p1_refinement_sweeps=2, archive_choices=False))
            elapsed += safe.get("total_wall_s", safe["setup_wall_s"])
            fallback = compact(safe, row, "reference")
        brief = compact(result, row, label, elapsed, fallback)
        if p.rank == 0:
            if details:
                path = records/("student-%s-%s-%s.json" % (job, key, tag))
                save_once(path, dict(result=result, compact=brief))
            record(config, "student_solve", mode=mode, key=key, method=label,
                   passed=brief["passed"], iterations=brief["iterations"],
                   seconds=brief["total_s"], worker=p.worker)
        return brief

    def share(local):
        return [r for part in global_objects(p, local) for r in part]

    def evaluate(keys, candidates, phase, repeats=1, start_repeat=0):
        result = []
        for i, key in enumerate(keys):
            if i % workers != p.worker:
                continue
            loaded = load(key)
            for repeat in range(start_repeat, start_repeat+repeats):
                # Rotate all methods and reverse consecutive repeats. This
                # keeps comparisons inside the same matrix and solver group.
                offset = i % len(candidates)
                ordered = candidates[offset:] + candidates[:offset]
                if repeat % 2:
                    ordered.reverse()
                for method in ordered:
                    result_row = solve(key, loaded, method,
                        "%s-r%d-%s" % (phase, repeat, method["name"]))
                    if result_row is not None:
                        result_row["repeat"] = repeat
                        result.append(result_row)
            destroy(loaded)
        return share(result)

    report = dict(job=job, mode=mode, workers=workers, cpu_ranks=p.global_size)
    status = 0
    if mode == "factor_data":
        from .factorized import export_state

        keys = config["student_keys"]
        assert len(set(keys)) == len(keys) and set(keys) <= set(train)
        exported = []
        for index, key in enumerate(keys):
            if index % workers != p.worker:
                continue
            loaded = load(key)
            try:
                example = export_state(config, p, replay, legacy, bridge, frames[key], *loaded, job)
            finally:
                destroy(loaded)
            if p.rank == 0:
                exported.append(example)
                record(config, "factor_supervision_verified", job=job, key=key,
                       raw_product_errors=example["raw_product_errors"], cache=example["cache"])
        exported = share(exported)
        assert len(exported) == len(keys) and all(
            next(item for item in exported if item["key"] == key)["cache"] is not None
            for key in config["gate_keys"])
        report.update(keys=keys, frames=exported, accepted=True)
    elif mode == "factor_krylov":
        from .factorized import load_model

        keys = config["student_keys"]
        assert len(set(keys)) == len(keys) and set(keys) <= set(train)
        candidate = dict(name="hard_model", factor_model=load_model(config["factor_model"]),
                         short_krylov=config["factor_hard"]["krylov_steps"])
        cases = evaluate(keys, [candidate], mode)
        checked = all(x["short_krylov"] for x in cases)
        report.update(keys=keys, completed=checked and len(cases)==len(keys), cases=cases,
                      totals=method_totals(cases))
        status = 0 if report["completed"] else 2
    elif mode == "factor_gradient":
        from .factorized import load_model, save_model

        settings = config["factor_feedback"]
        keys = config["student_keys"]
        assert len(set(keys)) == len(keys) and set(keys) <= set(train)
        model = load_model(config["factor_model"])
        cases = evaluate(keys, [dict(name="gradient_source", factor_model=model,
                                     factor_feedback=settings)], mode)
        verified = len(cases) == len(keys) and all(
            row["passed"] and (row["coefficient_feedback"] or {}).get("verified") for row in cases)
        report.update(keys=keys, cases=cases, totals=method_totals(cases), verified=verified,
                      initial_model=config["factor_model"])
        status = 0 if verified else 2
        if verified:
            candidate, update = coefficient_step(model, cases, settings["step_size"])
            assert candidate is not None, update
            checkpoint = None
            if p.global_rank == 0:
                checkpoint = save_model(Path(config["output_directory"])/"checkpoints"/
                                        ("factorized-gradient-"+job+".npz"), candidate)
            checkpoint, = [value for value in global_objects(p, checkpoint) if value]
            methods = [dict(name="gradient_initial", factor_model=model),
                       dict(name="gradient_updated", factor_model=candidate)]
            paired = evaluate(keys, methods, "factor-gradient-pair", repeats=settings["repeats"])
            if "factor_structure" in config:
                selected_keys = config["factor_structure"].get("training_keys", train)
                assert len(set(selected_keys)) == len(selected_keys) and set(selected_keys) <= set(train)
                structured, structure = tune_structure(config, p, selected_keys, load, destroy,
                                                        solve, share, job, candidate)
                report["structure"] = structure
                status = 0 if structure["verified"] else 2
                methods.append(dict(name="structure_updated", factor_model=structured))
            if settings.get("validation"):
                validation = config["training"]["validation_keys"]
                assert all(frames[key]["split"] == "validation" for key in validation)
                comparators = [config["factor_reference"], dict(name="amg"), dict(name="retained_base")]
                results = evaluate(validation, methods+comparators, "factor-gradient-validation")
                paired += results
                timing = method_totals(results)
                winners = [item for item in timing if item in ("gradient_updated", "structure_updated")
                           and not timing[item]["failed"]
                           and timing[item]["total_s"] < timing["student"]["total_s"]]
                if winners:
                    paired += evaluate(validation, methods+comparators, "factor-gradient-validation",
                                       start_repeat=1)
            report.update(model=checkpoint, layer_delta=update["layer_delta"],
                          gradient_norm=update["gradient_norm"], paired=paired, paired_totals={split:method_totals(
                              [row for row in paired if row["split"] == split])
                              for split in sorted({r["split"] for r in paired})})
    elif mode == "factor_compare":
        from .factorized import load_model

        keys = config["training"]["validation_keys"]
        assert len(keys)==len(set(keys))==8 and all(frames[key]["split"]=="validation" for key in keys)
        methods = [dict(name="operator",factor_model=load_model(config["factor_model"])),
                   dict(config["factor_reference"],fallback=False)]
        cases = evaluate(keys,methods,"factor-check",repeats=2)
        repeats = [method_totals([row for row in cases if row["repeat"]==i]) for i in range(2)]
        def compared(limit):
            return all(len(row)==2 and all(s["frames"]==len(keys) and not s["failed"] for s in row.values())
                       and row["operator"]["iterations"]<=row["student"]["iterations"]
                       and row["operator"]["total_s"]<=limit*row["student"]["total_s"] for row in repeats)
        report.update(model=config["factor_model"],keys=keys,cases=cases,repeat_totals=repeats,
                      totals=method_totals(cases),speed_gate=compared(.98),compression_gate=compared(1.05))
    elif mode == "factor_final":
        from .factor_operator import checked_json
        from .factor_replay import cost_pair
        from .factorized import load_model

        settings = config["factor_final"]
        source = checked_json(settings["source_report"])
        keys = settings["keys"]
        if (source["winner"]["model"] != config["factor_model"] or source["winner"]["inference"] is not None
                or len(keys) != len(set(keys)) or set(keys) != {k for k,r in frames.items() if r["split"]=="test"}):
            raise ValueError("The final evaluation requires the frozen generator and the complete test split")
        model = load_model(config["factor_model"])
        methods = [dict(name="frozen", factor_model=model, factor_context=dict(factor_fusion=True)),
                   dict(config["factor_reference"], fallback=False), dict(name="amg")]
        cases = evaluate(keys, methods, "factor-final", repeats=2)
        cached_keys = [key for key in keys if retained_teacher(config, frames[key]) is not None]
        cached = evaluate(cached_keys, [dict(name="retained_base")], "factor-final-cached") if cached_keys else []
        complete = {(r["repeat"], r["key"], r["method"]) for r in cases}
        expected = {(i,k,m["name"]) for i in range(2) for k in keys for m in methods}
        status = 0 if complete == expected and len(cases) == len(expected) else 2
        comparisons = {name:cost_pair(cases, keys, name, "frozen", settings["speed_margin"])
                       for name in ("student", "amg")}
        report.update(source=settings["source_report"], model=config["factor_model"], keys=keys, cases=cases,
                      totals=method_totals(cases), comparisons=comparisons,
                      cached_teacher=dict(keys=cached_keys, cases=cached, totals=method_totals(cached)))
    elif mode == "factor_input":
        from .factor_replay import run_matrix_input

        report.update(run_matrix_input(config,p,legacy,bridge,frames,load,destroy,solve,share,evaluate,job))
    elif mode == "factor_attention_point":
        from .factor_replay import run_attention_point

        report.update(run_attention_point(config,p,legacy,replay,bridge,frames,load,destroy,solve,share,job))
    elif mode == "factor_attention":
        from .factor_replay import run_attention_replay

        result = run_attention_replay(config, p, legacy, bridge, frames, load, destroy, solve, share, evaluate, job)
        report.update(result)
        status = 0 if result["audit_passed"] else 2
    elif mode == "factor_cost":
        from .factor_replay import run_cost_replay

        report.update(run_cost_replay(config, p, legacy, bridge, frames, load, destroy, solve, share, evaluate, job))
    elif mode == "factor_compress":
        from .factor_compress import run_compression

        report.update(run_compression(config,p,legacy,bridge,frames,load,destroy,solve,share,evaluate,job))
    elif mode == "factor_replay_train":
        from .factor_replay import run_probe_replay

        report.update(run_probe_replay(config,p,legacy,bridge,frames,load,destroy,solve,share,evaluate,job))
        status = 0 if report["coverage_complete"] else 2
    elif mode == "factor_operator":
        from .factor_operator import run_operator

        report.update(run_operator(config, p, legacy, bridge, frames, load, destroy, solve, share, evaluate, job))
    elif mode == "factor_search":
        from .factor_search import search_distill

        report.update(search_distill(config, p, train, load, destroy, solve, share, evaluate, job))
    elif mode == "factor_learn":
        from .factorized import load_model

        settings = config["factor_structure"]
        keys = settings["training_keys"]
        validation = config["training"]["validation_keys"]
        assert len(set(keys)) == len(keys) and set(keys) == set(train)
        assert len(set(validation)) == len(validation) and all(
            frames[key]["split"] == "validation" for key in validation)
        initial = load_model(config["factor_model"])
        _, learned = tune_structure(config, p, keys, load, destroy, solve, share,
                                    job, initial, evaluate)
        assert learned["completed_steps"] == settings["updates"]
        assert set(learned["coverage"]) == set(keys)
        status = 0 if learned["coefficient_updates"] and all(
            learned["effective_structure_updates"].values()) else 2
        ranked = sorted(learned["milestones"], key=lambda row:(
            row["totals"]["failed"], row["totals"]["total_s"]))[:2]
        origin = load_model(config["factor_learning_origin"]) if "factor_learning_origin" in config else initial
        references = [config["factor_reference"], dict(name="amg"), dict(name="retained_base"),
                      dict(name="learning_start", factor_model=origin)]
        finalists = [dict(name="learned-%03d"%r["step"], factor_model=load_model(r["model"]))
                     for r in ranked]
        repeats = settings["validation_repeats"]
        cases = evaluate(validation, references+finalists, "factor-learning-pair", repeats=repeats)
        totals = method_totals(cases)
        repetitions = [method_totals([r for r in cases if r["repeat"] == repeat])
                       for repeat in range(repeats)]
        winner = min(finalists, key=lambda method:(totals[method["name"]]["failed"],
                                                  totals[method["name"]]["total_s"]))["name"]
        def beat(reference):
            return all(reference in row and row[winner]["frames"] == len(validation)
                       and row[reference]["frames"] == len(validation) and not row[winner]["failed"]
                       and not row[reference]["failed"] and row[winner]["total_s"] < row[reference]["total_s"]
                       for row in repetitions)
        comparison = {method:beat(method) for method in ("student","amg","retained_base")}
        winner_record = next(row for row in ranked if winner=="learned-%03d"%row["step"])
        report.update(keys=validation, training_keys=keys, learning=learned, cases=cases, totals=totals,
                      repeat_totals=repetitions, winner=winner, model=winner_record["model"],
                      repeatable_speedups=comparison, speed_gate=comparison["student"] and comparison["amg"])
    elif mode == "factor_tune":
        report.update(tune_factors(config, p, frames, train, load, destroy, solve, share, evaluate, job))
    elif mode == "direct_data":
        keys = config["student_keys"]
        assert len(set(keys)) == len(keys) and set(keys) <= set(train)
        feedback_cases = []
        kind = config.get("direct_teacher_kind")
        if kind in ("E", "F"):
            source = config["direct_feedback_model"] if kind == "F" else config["direct_prior_model"]
            prior = dict(name="direct_feedback", direct=True, capture_apply=True,
                         students={"p1":source})
            feedback_cases = evaluate(keys, [prior], "direct_feedback")
            assert all(row["passed"] for row in feedback_cases)
        controls = [dict(name="teacher_roundtrip", export_direct=True)]
        if kind != "F":
            controls.insert(0,dict(name="retained_base"))
        results = evaluate(keys, controls, mode)
        cases = {(r["key"], r["method"]): r for r in results}
        roundtrip = [cases[(key, "teacher_roundtrip")] for key in keys]
        accepted = all(r["passed"] and r["teacher"] for r in roundtrip)
        if kind != "F":
            accepted &= all(cases[(r["key"],"retained_base")]["passed"] and
                r["iterations"]==cases[(r["key"],"retained_base")]["iterations"] for r in roundtrip)
        report.update(keys=keys, cases=results, totals=method_totals(results), accepted=accepted,
                      feedback_cases=feedback_cases,
                      samples={"1": [r["teacher"] for r in roundtrip if r["teacher"]]})
        status = 0 if accepted else 2
    elif mode in ("data", "refresh"):
        keys = config.get("student_keys") or train
        assert len(set(keys)) == len(keys) and set(keys) <= set(train)
        levels = [2] if mode == "refresh" else config.get("student_teacher_levels", [1, 2])
        result = []
        for index, key in enumerate(keys):
            if index % workers != p.worker:
                continue
            loaded = load(key)
            for level in levels:
                method = dict(name="teacher_L%d" % level,
                    students={"p1": refresh} if refresh else {})
                tag = "teacher-L%d" % level
                row_result = solve(key, loaded, method, tag, export_level=level)
                if level == 1 and not row_result["passed"]:
                    # Keep C1 and its transfer construction fixed. Retry the
                    # terminal graph radius only after a recorded failure.
                    for radius in (6, 7):
                        retry = dict(method, terminal_min_radius=radius)
                        previous = row_result["failure"]
                        row_result = solve(key, loaded, retry, tag+"-r%d" % radius, export_level=level)
                        row_result.update(retry_radius=radius, previous_failure=previous)
                        if row_result["passed"]:
                            break
                result.append(row_result)
            destroy(loaded)
        result = share(result)
        records_by_level = {str(level): [r["teacher"] for r in result
            if r["method"] == "teacher_L%d" % level and r["passed"] and r["teacher"]] for level in levels}
        p2_minimum = max(1, (9*len(keys))//10) if mode == "refresh" else min(8, len(keys))
        minimum = {str(level): len(keys) if level == 1 else p2_minimum for level in levels}
        accepted = all(len(records_by_level[str(level)]) >= minimum[str(level)] for level in levels)
        report.update(keys=keys, samples=records_by_level, cases=result, totals=method_totals(result),
                      minimum_samples=minimum, accepted=accepted)
        status = 0 if accepted or mode == "refresh" else 2
    elif mode in ("evaluate", "confirm", "diagnose"):
        keys = config["student_keys"]
        split = config["student_evaluation_split"]
        assert len(set(keys)) == len(keys) and all(frames[k]["split"] == split for k in keys)
        cases = evaluate(keys, methods, mode, repeats=config.get("student_repeats", 1))
        totals = method_totals(cases)
        report.update(keys=keys, split=split, methods=methods, cases=cases, totals=totals)
        if mode == "diagnose":
            assert len(cases) == len(keys)*len(methods)
        else:
            reference = config.get("student_reference", "reference")
            assert totals[reference]["frames"] == len(keys)*config.get("student_repeats", 1)
            status = 2 if totals[reference]["failed"] else 0
    elif mode == "tune":
        assert workers == 3 and len(methods) == 1
        method = methods[0]
        active = [name for name in ("p1", "p2") if name in method["students"]]
        assert active
        settings = config["student_tuning"]
        milestones = set(settings["milestones"])
        updates = settings["batches"]
        assert 0 in milestones and updates in milestones
        controls = np.concatenate([method.get("controls_"+name, DEFAULT_CONTROL) for name in active])
        base = dict(name="reference")
        validation = config["training"]["validation_keys"]
        assert all(frames[k]["split"] == "validation" for k in validation)
        best, best_score = None, (float("inf"), float("inf"))
        observations, checkpoints = [], []
        sigma = float(settings["sigma"])
        scale = float(settings["common_time_scale_s"])
        rng = np.random.default_rng(settings["seed"])
        lower = np.tile([-2, -2, .25, .0], len(active))
        upper = np.tile([2, 2, 1.75, 1.75], len(active))
        lanes = [train[i::workers] for i in range(workers)]

        def controlled(phi, name):
            return dict(method, name=name, **{"controls_"+level: phi[4*i:4*(i+1)].tolist()
                        for i, level in enumerate(active)})

        for step in range(updates+1):
            if step in milestones:
                candidate = controlled(controls, "student")
                cases = evaluate(validation, [base, candidate], "tune-%03d" % step)
                totals = method_totals(cases)
                score = (totals["student"]["failed"], totals["student"]["total_s"])
                checkpoint = dict(step=step, method=candidate, totals=totals, cases=cases)
                checkpoints.append(checkpoint)
                if score < best_score:
                    best, best_score = checkpoint, score
                if p.global_rank == 0:
                    save_once(records/("student-tune-%s-step%03d.json" % (job, step)), checkpoint)
            if step == updates:
                break
            direction = rng.choice([-1.0, 1.0], size=len(controls))
            keys = [str(rng.choice(lane)) for lane in lanes]
            loaded = load(keys[p.worker])
            plus = np.clip(controls+sigma*direction, lower, upper)
            minus = np.clip(controls-sigma*direction, lower, upper)
            pair = [("+", plus), ("-", minus)]
            if (step+p.worker) % 2:
                pair.reverse()
            rows = [solve(keys[p.worker], loaded, dict(controlled(phi, label), fallback=False),
                          "pair-%04d-%s" % (step,label), details=False) for label,phi in pair]
            destroy(loaded)
            rows = share(rows)
            cost = {side:sum(r["total_s"] if r["passed"] else 120.0 for r in rows if r["method"]==side)
                    for side in ("+","-")}
            derivative = np.clip((cost["+"]-cost["-"])/(2*sigma*scale*workers), -2, 2)
            lr = settings["learning_rate"]/np.sqrt(1+step/20)
            previous = controls.tolist()
            controls = np.clip(controls-lr*derivative*direction, lower, upper)
            observation = dict(step=step, keys=keys, cases=rows, plus=plus.tolist(), minus=minus.tolist(),
                               controls_before=previous, controls_after=controls.tolist(), cost=cost)
            observations.append(observation)
            if p.global_rank == 0:
                with (records/("student-pairs-%s.jsonl" % job)).open("a") as f:
                    f.write(json.dumps(observation, allow_nan=False)+"\n")
        report.update(batches=updates, candidate_solves=updates*workers*2, tuning=observations,
                      checkpoints=checkpoints, best=best)
    else:
        raise ValueError("Unknown student cycle phase: "+mode)
    if p.global_rank == 0:
        save_once(records/("student-cycle-%s.json" % job), report)
        record(config, "student_cycle_complete", job=job, mode=mode, exit_code=status)
    return status
