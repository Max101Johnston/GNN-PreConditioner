"""MPI entry points for the learned interpolation cycle."""
import json
import os
from pathlib import Path

import numpy as np

from .backend import record, read_json
from .online import global_objects, load_observed_state, solve_policy
from .student import DEFAULT_CONTROL, load_student


def save_once(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)


def compact(result, frame, method, elapsed=None, fallback=None):
    n = result["numerical"]
    first, second = n.get("level1", {}), n.get("level2", {})
    return dict(key=frame["key"], split=frame["split"], method=method,
        passed=result["reached_true_target"], failure=result["failure"],
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
        teacher=result.get("student_teacher"))


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
        if label in ("amg", "retained_base"):
            teacher = retained_teacher(config, row) if label == "retained_base" else None
            if label == "retained_base" and teacher is None:
                return None
            result = run_method(config, p, replay, legacy, bridge, row, layout, data, label, teacher)
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
                           archive_choices=False)
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

    def evaluate(keys, candidates, phase, repeats=1):
        result = []
        for i, key in enumerate(keys):
            if i % workers != p.worker:
                continue
            loaded = load(key)
            for repeat in range(repeats):
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
    if mode == "direct_data":
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
