"""Offline improvement of the deterministic, two-level factor generator."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .backend import digest, read_json, record
from .factorized import (DEPTHS, MAX_SUPPORT, REGIONS, STAGES, FactorModel, FactorPolicy,
                         hierarchy_signature, load_model, save_model, support_order)
from .krylov import signed_head
from .student_cycle import method_totals, save_once


def probabilities(logits):
    logits = np.asarray(logits, dtype=np.float64)
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


class SearchProposal(FactorPolicy):
    """A regional representative/support move or a zero-sum budget move."""

    def __init__(self, p, model, settings):
        self.kind = settings["role"]
        role = "representatives" if self.kind == "exchange" else self.kind
        super().__init__(p, model, dict(settings, role=role))
        self.modify_edges = True
        self.added = []
        self.examples = []
        self.example_limit = (settings.get("examples", 0)
                              if p.rank % max(1, p.size//8) == 0 else 0)

    def example(self, layer, features, scores, chosen, original, ids):
        same = chosen == original
        if sum(item["layer"] == layer and item["unchanged"] == same
               for item in self.examples) >= self.example_limit:
            return
        self.examples.append(dict(layer=layer, features=np.asarray(features).tolist(),
                                  logits=np.asarray(scores).tolist(), chosen=int(chosen),
                                  domain=np.asarray(ids).tolist(), unchanged=bool(same),
                                  stage=self.stage+1, region=self.region,
                                  temperature=self.temperature))

    def representative_choice(self, features, scores, chosen, original, native, record):
        if chosen != original:
            self.added.append(native[chosen])
        if record:
            self.example(1, features, scores, chosen, original, native)

    def edges(self, outputs, quota, valid, same, keep, fine, summary, refined, native, regions,
              candidate_ids=None, xyz=None):
        count = keep.sum(1)
        capacity = np.minimum(MAX_SUPPORT, valid.sum(1))
        scores = outputs[:, :, 0]
        order = support_order(scores, valid, same)
        limit = self.settings["actions_per_rank"]
        eligible = regions == self.region
        if self.kind == "budget":
            context = np.column_stack((fine, summary, np.ones(len(fine))))
            touched = []
            for component in range(3):
                rows = np.flatnonzero(eligible & (capacity > 1) & (native % 3 == component))
                rows = rows[np.lexsort((native[rows], xyz[rows, 2], xyz[rows, 1], xyz[rows, 0]))]
                # Sample within adjacent pairs. Each move retains the anchor
                # and keeps the pair's total budget fixed.
                shift = self.settings["seed"] % 2
                rows = rows[shift:]
                pairs = rows[:len(rows)-len(rows)%2].reshape(-1, 2)
                pairs = self.subset.permutation(pairs)[:max(1, limit//3)]
                for left, right in pairs:
                    options = [(i, j) for i, j in ((left, right), (right, left))
                               if count[i] > 1 and count[j] < capacity[j]]
                    if not options:
                        continue
                    i, j = options[self.rng.integers(len(options))]
                    for row, change in ((i, -1), (j, 1)):
                        self.budget_example(row, count[row]+change, count, capacity, context, quota)
                        keep[row] = False
                        keep[row, order[row, :count[row]+change]] = True
                        self.record(native[row], count[row]+change, count[row], 1., True)
                        touched.append(row)
            background = np.flatnonzero(eligible & (capacity > 1))
            for row in np.setdiff1d(background, touched)[:self.example_limit]:
                self.budget_example(row, count[row], count, capacity, context, quota)
            return keep

        eligible &= (count >= 2) & (valid.sum(1) > count)
        novel = None
        if self.kind == "exchange":
            # Coarse stages repeat the same representative choices on all
            # ranks. Fine stages contribute disjoint local choices.
            added = np.full(limit, -1, np.int64)
            added[:len(self.added)] = self.added
            ids = self.p.gather_equal(added).ravel()
            novel = np.isin(candidate_ids, ids[ids >= 0]) & valid & ~keep
        rows = np.flatnonzero(eligible)
        pool = rows if novel is None else rows[novel[rows].any(1)]
        # Take adjacent rows of this region in a single displacement
        # component so a support move explores a localized group.
        if len(pool):
            pool = pool[np.lexsort((native[pool], xyz[pool, 2], xyz[pool, 1],
                                    xyz[pool, 0], native[pool] % 3))]
            start = self.subset.integers(max(1, len(pool)-limit+1))
            selected = pool[start:start+limit]
        else:
            selected = pool
        for row in selected:
            previous = order[row, count[row]-1]
            options = np.r_[previous, np.flatnonzero(valid[row] & ~keep[row])]
            candidate_scores = scores[row, options]
            allowed = (np.arange(len(options)) if novel is None
                       else np.r_[0, np.flatnonzero(novel[row, options])])
            choice, prob = self.choose(candidate_scores[allowed])
            index = allowed[choice]
            chosen = options[index]
            keep[row, previous], keep[row, chosen] = False, True
            self.record(native[row], candidate_ids[row, chosen], candidate_ids[row, previous],
                        prob[choice], True)
            self.example(4, refined[row, options], candidate_scores, index, 0, candidate_ids[row, options])
        for row in np.setdiff1d(rows, selected)[:self.example_limit]:
            options = np.r_[order[row, count[row]-1], np.flatnonzero(valid[row] & ~keep[row])]
            self.example(4, refined[row, options], scores[row, options], 0, 0, candidate_ids[row, options])
        return keep

    def budget_example(self, row, chosen, count, capacity, features, quota):
        options = np.arange(max(1, count[row]-1), min(capacity[row], count[row]+1)+1)
        self.example(5, 2*(options[:, None]-3.)*features[row],
                     -(options-3.-quota[row])**2, chosen-options[0], count[row]-options[0], options)

    def receipt(self):
        result = super().receipt()
        result.pop("gradient")
        result["role"] = self.kind
        if self.settings.get("examples"):
            payload = np.frombuffer(json.dumps(self.examples, allow_nan=False).encode(), np.uint8)
            padded = np.zeros(int(self.p.maximum(len(payload))), np.uint8)
            padded[:len(payload)] = payload
            packets = self.p.gather_equal(padded)
            if self.p.rank == 0:
                result["examples"] = [row for part in packets
                                      for row in json.loads(part.astype(np.uint8).tobytes().rstrip(b"\0"))]
        return result


def historical_audit(first_path, resumed_path):
    """Use executed signatures and within-frame numerics from the 200 batches."""
    def batches(path):
        with Path(path).open() as stream:
            return {item["step"]: item for item in map(json.loads, stream)}

    first, resumed = batches(first_path), batches(resumed_path)
    merged = [first[i] if i <= 120 else resumed[i] for i in range(1, 201)]
    duplicates = equal_steps = repeated = 0
    identical_spreads, equal_step_spreads, priorities = [], [], defaultdict(list)
    useful_by_frame = defaultdict(list)
    focus = {}
    for batch in merged:
        setting = (batch["stage"], batch["region"], batch["role"])
        assert len(batch["keys"]) == 2 and len(batch["candidates"]) == 8
        for key in batch["keys"]:
            group = [r for r in batch["candidates"] if r["key"] == key]
            by_signature = defaultdict(list)
            for row in group:
                if row["passed"]:
                    numerical = dict(zip(("level1", "level2"), row["factor_levels"]))
                    by_signature[hierarchy_signature(numerical)].append(row)
            duplicates += sum(len(rows)-1 for rows in by_signature.values())
            repeated += bool(any(len(rows)>1 for rows in by_signature.values()))
            identical_spreads.extend(np.ptp([r["total_s"] for r in rows])
                                     for rows in by_signature.values() if len(rows)>1)
            iterations = [r["iterations"] for r in group if r["passed"]]
            if len(iterations)==4 and min(iterations)==max(iterations):
                equal_steps += 1
                equal_step_spreads.append(np.ptp([r["total_s"] for r in group]))
            unique = [rows[0] for rows in by_signature.values()]
            if len(unique) >= 2:
                best = min(unique, key=lambda r:r["iterations"])
                similar = [r["iterations"] for r in unique
                           if all(abs(r[col]/best[col]-1.) <= .05 for col in ("p1_nnz", "k1_nnz"))]
                gain = float(np.median(similar)-best["iterations"])
                priorities[setting].append(gain)
                if gain >= 2:
                    useful_by_frame[key].append(dict(stage=setting[0], region=setting[1], role=setting[2],
                                                     iterations_gain=gain))
            if batch["step"] == 165 and key == "test-1103-407":
                source = next(r for r in batch["coefficient_update"]["cases"] if r["key"] == key)
                focus = dict(source_iterations=source["iterations"], candidate_iterations=iterations,
                             candidate_total_s=[r["total_s"] for r in group],
                             candidate_signatures=len(by_signature),
                             coefficient_update_verified=batch["coefficient_update"]["verified"],
                             logged_layer_delta=batch["coefficient_update"].get("layer_delta"),
                             all_structure_changes_zero=not any(r["factor_policy"]["changes"] for r in group))
    scores = [dict(stage=s, region=r, role=k, comparable_groups=len(gains),
                   clear_groups=sum(g>=2 for g in gains), mean_gain=float(np.mean(gains)))
              for (s,r,k), gains in priorities.items()]
    rank = lambda q:(q.get("clear_groups", 0), q.get("mean_gain", q.get("iterations_gain", 0.)))
    return dict(batches=len(merged), candidates=sum(len(b["candidates"]) for b in merged),
                frame_groups=2*len(merged), duplicate_candidates=duplicates, repeated_groups=repeated,
                equal_iteration_groups=equal_steps,
                equal_iteration_spread_median_s=float(np.median(equal_step_spreads)),
                identical_hierarchy_spread_median_s=float(np.median(identical_spreads)),
                identical_hierarchy_spread_max_s=float(max(identical_spreads)),
                priorities=sorted(scores, key=rank, reverse=True),
                frame_priorities={key: sorted(rows, key=rank, reverse=True)[:4]
                                  for key, rows in useful_by_frame.items()}, step_165=focus)


def calibrated_targets(calibration, settings):
    windows = settings["prefixes"]
    methods = ("origin", "late_checkpoint", "retained_base")
    required = settings["min_calibration_frames"]
    threshold = settings["probe_separation"]
    targets, diagnostics = {}, {}
    for name in ("iteration-24", "late", "displacement-0", "displacement-1", "displacement-2"):
        rows = [tuple(frame["controls"][method]["trajectories"][name] for method in methods)
                for frame in calibration
                if all(name in frame["controls"][method]["trajectories"] for method in methods)]
        stats = []
        for n in windows:
            ratios = np.asarray([[row["prefix_ratios"][str(n)] for row in group] for group in rows])
            if len(ratios):
                stats.append(dict(steps=n, frames=len(ratios), medians=np.median(ratios, axis=0).tolist(),
                                  teacher_separations=int(np.count_nonzero(ratios[:,0]-ratios[:,2]>=threshold))))
        diagnostics[name] = stats
        if name.startswith("displacement"):
            if stats:
                goal = max(settings["displacement_target"], stats[0]["medians"][2])
                targets[name] = dict(steps=windows[0], target=goal, weight=.2)
        else:
            selected = next((r for r in stats if r["teacher_separations"] >= required), None)
            if selected is not None:
                origin, _, teacher = selected["medians"]
                # Begin between the teacher and the learned operating point.
                targets[name] = dict(steps=selected["steps"], target=max(.02, .5*(origin+teacher)), weight=1.)
    return targets, diagnostics


def proposal_settings(audit, settings, key, seed):
    rng = np.random.default_rng(seed)
    suggestions = audit["frame_priorities"].get(key, []) + audit["priorities"]
    chosen = []
    for kind, historical in (("exchange", "representatives"), ("support", "support"), ("budget", "budget"),
                             ("exchange", "representatives")):
        choices = [(row["stage"], row["region"]) for row in suggestions
                   if row["role"] == historical and (kind,row["stage"],row["region"]) not in chosen]
        stage, region = choices[min(len(choices)-1, int(rng.integers(3)))] if choices else (
            int(rng.integers(STAGES)), int(rng.integers(REGIONS)))
        chosen.append((kind, stage, region))
    return [dict(search=True, role=role, stage=stage, region=region,
                 seed=int(rng.integers(2**32)), subset_seed=int(rng.integers(2**32)),
                 temperature=settings["temperature"], actions_per_rank=settings["actions_per_rank"])
            for role, stage, region in chosen]


def physical_improvement(base, trial):
    if not trial.get("passed") or not base.get("passed"):
        return False
    count = base["iterations"]-trial["iterations"]
    delta_pc = base["pc_apply_s"]-trial["pc_apply_s"]
    delta_setup = base["setup_s"]-trial["setup_s"]
    if count >= 2:
        similar = all(trial[k] <= 1.05*base[k] for k in ("p1_nnz", "k1_nnz"))
        if similar:
            return True
    return delta_pc >= .10 or delta_setup >= .10


def noise_floor(base, settings):
    return max(settings["absolute_margin_s"], settings["relative_margin"]*base["total_s"])


def confirmed_improvement(first_base, first, second_base, second, settings):
    if any(not row.get("passed") for row in (first_base, first, second_base, second)):
        return False
    if (first_base["hierarchy_signature"] != second_base["hierarchy_signature"]
            or first["hierarchy_signature"] != second["hierarchy_signature"]):
        return False
    gaps = [base["total_s"]-trial["total_s"] for base,trial in ((first_base,first),(second_base,second))]
    return (min(gaps) > 0 and np.mean(gaps) >= noise_floor(first_base, settings)
            and all(physical_improvement(base, trial) for base,trial in ((first_base,first),(second_base,second))))


def coefficient_candidate(model, case, settings):
    feedback = case.get("coefficient_feedback")
    if not case["passed"] or not feedback or not feedback["verified"]:
        return None
    direction = signed_head(model, np.asarray(feedback["gradient"]))
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 1e-14:
        return None
    weights = (model.weights - settings["coefficient_step"]*direction/norm).astype(model.weights.dtype)
    return FactorModel(model.width, weights, model.adapters, model.unit_sum_depth)


def structural_model(model, examples, deltas, settings):
    """Fit confirmed same-domain choices under a policy KL penalty."""
    base = model.weights.astype(np.float64)
    trial = FactorModel(model.width, base.copy(), model.adapters, model.unit_sum_depth)
    counts = Counter((row["layer"], row["unchanged"]) for row in examples)
    cache = []
    for row in examples:
        layer = row["layer"]
        features = np.asarray(row["features"], dtype=np.float64)/row["temperature"]
        if layer != 5:
            features = np.column_stack((features, np.ones(len(features))/row["temperature"]))
        old = np.asarray(row["logits"])/row["temperature"]
        current = np.r_[model.layers[layer][0][0], model.layers[layer][1][0]]
        reference = probabilities(old)
        cache.append((layer, features, old-features@current, reference, row["chosen"],
                      row["unchanged"], counts[layer, row["unchanged"]]))
    for _ in range(settings["distill_steps"]):
        gradients = [np.zeros(w.shape[1]+1) for w, _ in model.layers]
        for layer, features, offset, reference, chosen, unchanged, count in cache:
            w, b = trial.layers[layer]
            predicted = probabilities(offset+features@np.r_[w[0], b[0]])
            derivative = settings["preservation_weight"]*(predicted-reference)
            if not unchanged:
                derivative += predicted
                derivative[chosen] -= 1.
            gradients[layer] += features.T @ derivative / count
        for layer in (1, 4, 5):
            w, b = trial.layers[layer]
            update = settings["distill_lr"]*gradients[layer]
            w[0] -= update[:-1]
            b[0] -= update[-1]
        delta = trial.weights - base
        trial.weights[:] = base + delta*min(1., settings["distill_radius"]/max(np.linalg.norm(delta), 1e-30))
    if deltas:
        # Reuse the signed-head target of every confirmed coefficient move.
        trial.weights[:] += settings["coefficient_distill"]*np.mean(deltas, axis=0)
    return FactorModel(model.width, trial.weights.astype(model.weights.dtype), model.adapters, model.unit_sum_depth)


def search_distill(config, p, train, load, destroy, solve, share, evaluate, job):
    settings = config["factor_search"]
    history = settings["historical_audit"]
    if digest(history["path"]) != history["sha256"]:
        raise ValueError("Historical audit receipt changed")
    audit = read_json(history["path"])
    if audit["candidates"] != 1600:
        raise ValueError("The proposal prior must cover all historical candidates")
    rounds = settings["round_keys"]
    validation = config["training"]["validation_keys"]
    assert len(rounds)==3 and all(len(keys)==len(set(keys))==24 for keys in rounds)
    assert set(key for group in rounds for key in group) == set(train)
    assert len(validation)==len(set(validation))==8 and not set(validation)&set(train)
    maximum_calls = (len(config["gate_keys"])+6*sum(map(len, rounds))
                     +2*len(rounds)*(settings["confirm_pairs"]+len(validation))+6*len(validation))
    assert maximum_calls <= 600
    output = Path(config["output_directory"])
    origin = load_model(config["factor_model"])
    worse = load_model(settings["late_model"])
    incumbent, incumbent_receipt = origin, config["factor_model"]
    control = dict(config["factor_reference"], fallback=False)
    baseline = dict(name="origin", factor_model=origin)
    calibration = []
    windows = dict.fromkeys(("iteration-24", "late", "displacement-0", "displacement-1",
                             "displacement-2"), settings["prefixes"])
    workers = config["parallel_workers"]
    for index, key in enumerate(config["gate_keys"]):
        if index % workers != p.worker:
            continue
        state = load(key)
        inputs = []
        try:
            anchor = solve(key, state, dict(baseline, factor_probe_capture=inputs,
                                            factor_probe_windows=windows), "search-calibration")
            if not anchor["passed"] or not inputs:
                raise ValueError("Could not capture full-solve probes on "+key)
            controls = dict(origin=dict(trajectories=anchor["factor_trajectories"],
                                        probe_s=anchor["factor_probe_s"]))
            for method in (dict(name="retained_base"), dict(name="late_checkpoint", factor_model=worse)):
                controls[method["name"]] = solve(key, state, dict(method, factor_probe_only=True,
                    factor_probe_inputs=inputs, factor_probe_windows=windows), "search-probes")
                if not controls[method["name"]] or controls[method["name"]]["failure"]:
                    raise ValueError("Cannot calibrate against "+method["name"]+" on "+key)
            calibration.append(dict(key=key, anchor=anchor, controls=controls))
        finally:
            destroy(state)
    calibration = share(calibration)
    targets, diagnostics = calibrated_targets(calibration, settings)
    if p.global_rank == 0:
        save_once(output/"records"/("factor-search-calibration-%s.json" % job),
                  dict(frames=calibration, targets=targets, windows=diagnostics))
        record(config, "factor_search_calibration", job=job, targets=targets, windows=diagnostics)
    observations, replay_bank, coefficient_bank = [], [], []
    confirmation_count = 0
    counts = Counter(calibration=len(calibration))
    for round_index, keys in enumerate(rounds, 1):
        phase = "search-%d" % round_index
        local = []
        for index, key in enumerate(keys):
            if index % workers != p.worker:
                continue
            state, inputs, seen = load(key), [], set()
            proposals = proposal_settings(audit, settings, key, settings["seed"]+100*round_index+index)
            component = "displacement-%d" % ((round_index+index)%3)
            quality = {name: value for name,value in targets.items()
                       if not name.startswith("displacement") or name == component}
            feedback = dict(config["factor_feedback"], quality_targets=quality,
                            finite_difference=0.) if quality else None
            try:
                anchor = solve(key, state, dict(name="incumbent", factor_model=incumbent, factor_signatures=seen,
                               factor_probe_capture=inputs, factor_feedback=feedback), phase, details=False)
                cases = [anchor]
                order = np.roll(np.arange(len(proposals)), (round_index+index)%len(proposals))
                for slot in order:
                    cases.append(solve(key, state, dict(name="proposal-%d" % slot, factor_model=incumbent,
                        factor_policy=proposals[slot], factor_signatures=seen), phase, details=False))
                candidate = coefficient_candidate(incumbent, anchor, settings)
                if candidate is not None:
                    windows = {name:[row["steps"]] for name,row in quality.items()}
                    case = solve(key, state, dict(name="coefficient", factor_model=candidate,
                                 factor_probe_inputs=inputs, factor_probe_windows=windows,
                                 factor_signatures=seen), phase, details=False)
                    if case.get("passed"):
                        before = hierarchy_signature(dict(zip(("level1","level2"), anchor["factor_levels"])),
                                                     ("representatives","support"))
                        after = hierarchy_signature(dict(zip(("level1","level2"), case["factor_levels"])),
                                                    ("representatives","support"))
                        case["fixed_structure"] = before == after
                        old_probes = anchor["coefficient_feedback"]["probes"]
                        measured = case["factor_trajectories"] or {}
                        case["same_probe_ratios"] = {name: [row["residual_ratio"], measured[name]["residual_ratio"]]
                                                    for name,row in old_probes.items() if name in measured}
                    cases.append(case)
            finally:
                destroy(state)
            local.append(dict(key=key, proposals=proposals, cases=cases))
        screening = share(local)
        counts["screen_attempts"] += sum(len(frame["cases"]) for frame in screening)
        counts["screened"] += sum(row.get("iterations") is not None for frame in screening for row in frame["cases"])
        counts["duplicates"] += sum(bool(row.get("duplicate")) for frame in screening for row in frame["cases"])
        counts["coefficient_proposals"] += sum(row["method"] == "coefficient"
                                              for frame in screening for row in frame["cases"])
        shortlist = []
        for frame in screening:
            base = frame["cases"][0]
            promising = [row for row in frame["cases"][1:] if physical_improvement(base, row)
                         and row.get("fixed_structure", True)
                         and base["total_s"]-row["total_s"] >= .4*noise_floor(base, settings)]
            if promising:
                first = min(promising, key=lambda r:r["total_s"])
                shortlist.append((base["total_s"]-first["total_s"], frame["key"], first["method"], frame))
        shortlist.sort(key=lambda r:(-r[0], r[1]))
        if p.global_rank == 0:
            save_once(output/"records"/("factor-search-screen-%s-r%d.json" % (job, round_index)), screening)
            record(config, "factor_search_screened", job=job, round=round_index, shortlist=len(shortlist),
                   counts=dict(counts))
        local = []
        for _, key, name, frame in shortlist[:settings["confirm_pairs"]]:
            if keys.index(key)%workers != p.worker:
                continue
            state = load(key)
            anchor = frame["cases"][0]
            if name == "coefficient":
                candidate = coefficient_candidate(incumbent, anchor, settings)
                method = dict(name=name, factor_model=candidate)
            else:
                policy = frame["proposals"][int(name.split("-")[-1])]
                method = dict(name=name, factor_model=incumbent, factor_policy=dict(policy, examples=3))
            try:
                # Reverse the screen order. Compare rebuilt hierarchies.
                second = solve(key, state, dict(method, factor_signatures=set()), phase+"-confirm", details=False)
                base = solve(key, state, dict(name="incumbent", factor_model=incumbent, factor_signatures=set()),
                             phase+"-confirm", details=False)
            finally:
                destroy(state)
            first = next(row for row in frame["cases"] if row["method"] == name)
            accepted = confirmed_improvement(anchor, first, base, second, settings)
            local.append(dict(key=key, method=name, accepted=bool(accepted), base=base, candidate=second))
        checked = share(local)
        counts["confirmed_calls"] += sum(r[field].get("iterations") is not None
                                         for r in checked for field in ("base", "candidate"))
        confirmation_count += len(checked)
        round_winners = []
        for row in checked:
            if not row["accepted"]:
                continue
            if row["method"]=="coefficient":
                anchor = next(item["cases"][0] for item in screening if item["key"]==row["key"])
                candidate = coefficient_candidate(incumbent, anchor, settings)
                coefficient_bank.append(candidate.weights.astype(np.float64)-incumbent.weights)
            else:
                replay_bank.extend(row["candidate"]["factor_policy"]["examples"])
            round_winners.append((row["key"], row["method"]))
        trial = structural_model(incumbent, replay_bank, coefficient_bank, settings)
        change = float(np.linalg.norm(trial.weights-incumbent.weights))
        summary = dict(round=round_index, keys=keys, shortlist=len(shortlist), checked=checked,
                       winners=round_winners, examples=len(replay_bank), coefficient_examples=len(coefficient_bank),
                       parameter_delta=change, promoted=False)
        methods = [dict(name="incumbent", factor_model=incumbent)]
        if change:
            methods.append(dict(name="distilled", factor_model=trial))
        cases = evaluate(validation, methods, phase+"-validation", start_repeat=round_index%2)
        counts["round_validation"] += sum(row.get("iterations") is not None for row in cases)
        totals = method_totals(cases)
        summary.update(validation=cases, totals=totals)
        if change:
            base, new = totals["incumbent"], totals["distilled"]
            accepted = (new["frames"]==len(validation) and not new["failed"] and not base["failed"]
                        and new["total_s"] + noise_floor(base, settings) < base["total_s"])
            receipt = None
            if p.global_rank == 0:
                receipt = save_model(output/"checkpoints"/("factor-search-%s-r%d.npz" % (job, round_index)), trial)
            receipt = share([receipt])[0]
            summary.update(trial=receipt, promoted=bool(accepted))
            if accepted:
                incumbent, incumbent_receipt = trial, receipt
        summary.update(incumbent=incumbent_receipt, counts=dict(counts))
        observations.append(summary)
        if p.global_rank == 0:
            save_once(output/"records"/("factor-search-round-%s-r%d.json" % (job, round_index)), summary)
            record(config, "factor_search_round", job=job, round=round_index, winners=round_winners,
                   examples=len(replay_bank), promoted=summary["promoted"], totals=summary.get("totals"))
    finalists = [baseline, control]
    if incumbent_receipt != config["factor_model"]:
        finalists.insert(1, dict(name="best_distilled", factor_model=incumbent))
    final = evaluate(validation, finalists, "search-final", repeats=2)
    counts["final_validation"] += sum(row.get("iterations") is not None for row in final)
    repetitions = [method_totals([row for row in final if row["repeat"]==index]) for index in range(2)]
    def faster(name, reference):
        return all(row[name]["frames"] == row[reference]["frames"] == len(validation)
                   and not row[name]["failed"] and not row[reference]["failed"]
                   and row[name]["total_s"]+noise_floor(row[reference], settings)<row[reference]["total_s"]
                   for row in repetitions)

    improved = len(finalists) == 3 and faster("best_distilled", "origin")
    winner = "best_distilled" if improved else "origin"
    receipt = incumbent_receipt if improved else config["factor_model"]
    solved = sum(counts[name] for name in ("calibration", "screened", "confirmed_calls",
                                            "round_validation", "final_validation"))
    return dict(training_keys=train, keys=validation, calibration=calibration, targets=targets,
                windows=diagnostics, rounds=observations, repeat_totals=repetitions, cases=final,
                totals=method_totals(final), model=receipt, winner=winner,
                model_improved=bool(improved), speed_gate=bool(improved and faster(winner, "student")),
                full_solves=solved, maximum_full_calls=maximum_calls, solve_counts=dict(counts),
                confirmations=confirmation_count,
                timing_scope="Replay cold construction and true residual. Short probes are recorded separately.")
