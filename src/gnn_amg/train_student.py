"""Train compact interpolation students and verify their native exports."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .backend import digest, read_json, record
from .student import (CONTEXT_BUDGETS, CONTEXT_SCORE_SCALE, DEFAULT_CONTROL, EDGE_FEATURES,
                      LOCAL_EDGE_FEATURES, ROW_FEATURES, SCHEMA, STUDENT_KINDS,
                      context_edge_output, context_layers, context_node_output,
                      context_numpy_decode, context_parameters, direct_graph, predict, predict_local)


class InterpolationStudent(nn.Module):
    def __init__(self, kind, width):
        super().__init__()
        self.kind = kind
        self.row = nn.Sequential(nn.Linear(ROW_FEATURES, width), nn.ReLU(), nn.Linear(width, 4))
        self.edge = (nn.Sequential(nn.Linear(EDGE_FEATURES, width), nn.ReLU(), nn.Linear(width, 2))
                     if kind in ("B", "D") else None)
        with torch.no_grad():
            self.row[-1].weight.mul_(0.02)
            self.row[-1].bias.copy_(torch.tensor([0., 0., .3, 0.]))
            if self.edge is not None:
                self.edge[-1].weight.mul_(0.02)
                self.edge[-1].bias.zero_()

    def forward(self, x, hints, norm, same, valid, temperature=0.15):
        r = self.row(x)
        omega = 1+(r[:, 0:1]).clamp(-.5, .5)
        raw = hints[:, :, 0]+omega*hints[:, :, 1]
        shift = 0
        if self.edge is not None:
            shape = valid.shape
            z = torch.cat((hints/norm[:, None, None], same[:, :, None],
                x[:, None, [1, 4, 6]].expand(-1, shape[1], -1)), dim=2)
            edge = self.edge(z)
            bounded = edge.clamp(-1, 1)
            raw = edge[:, :, 0] if self.kind == "D" else raw+hints[:, :, 2]+norm[:, None]*bounded[:, :, 0]
            shift = bounded[:, :, 1]
        scores = (raw.abs() if self.kind == "D" else raw.abs()/norm[:, None])+shift
        available = valid & (raw != 0)
        masked = torch.where(available, scores, -1e8)
        order = torch.argsort(masked, dim=1, descending=True, stable=True)
        rank = torch.empty_like(order).scatter_(1, order,
            torch.arange(order.shape[1], device=x.device).expand_as(order))
        budgets = r.new_tensor([6., 8., 12.])
        probabilities = torch.softmax(r[:, 1:], dim=1)
        chosen = budgets[r[:, 1:].argmax(dim=1)]
        hard = (rank < chosen[:, None]) & available
        if self.training:
            sorted_score = torch.gather(masked, 1, order)
            thresholds = sorted_score[:, [5, 7, 11]]
            soft = torch.sigmoid((scores[:, :, None]-thresholds[:, None, :])/temperature)
            gate = (soft*probabilities[:, None, :]).sum(2)*valid
            # Hard values are used in every training forward. The smooth
            # derivative gives edge priorities and budgets a learning signal.
            mask = hard.to(raw.dtype)+(gate-gate.detach())
        else:
            mask = hard
        return raw*mask, probabilities@budgets, chosen

    def packed(self):
        return np.ascontiguousarray(np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in self.parameters()]), dtype=np.float32)


class LocalInterpolationStudent(InterpolationStudent):
    def __init__(self, width, options):
        nn.Module.__init__(self)
        self.kind = "E"
        self.edge = nn.Sequential(nn.Linear(LOCAL_EDGE_FEATURES, width), nn.ReLU(), nn.Linear(width, 2))
        self.options = options

    def forward(self, edges):
        return self.edge(edges).unbind(dim=-1)

    def select(self, weights, scores, valid):
        available = valid & (weights != 0)
        order = torch.argsort(scores.masked_fill(~available, -torch.inf), dim=1,
                              descending=True, stable=True)
        rank = torch.empty_like(order).scatter_(1, order,
            torch.arange(order.shape[1], device=order.device).expand_as(order))
        return available & (rank < self.options["max_entries"]) & (
            (rank < self.options["min_entries"]) | (scores >= self.options["score_cutoff"]))


class ContextInterpolationStudent(InterpolationStudent):
    def __init__(self, width):
        nn.Module.__init__(self)
        self.kind = "F"
        self.layers = nn.ModuleList(nn.Linear(i,o) for i,o in context_layers(width))
        with torch.no_grad():
            self.layers[6].bias.copy_(torch.tensor([0.,0.,.1,.2]))

    def encode(self, nodes, neighbor_nodes, strengths):
        center = self.layers[0](nodes).relu()
        neighbors = self.layers[0](neighbor_nodes).relu()
        message = torch.cat((center,(neighbors*strengths.abs().unsqueeze(2)).sum(1),
                             (neighbors*strengths.unsqueeze(2)).sum(1)),dim=1)
        return self.layers[1](message).relu()

    def forward(self, frame, rows):
        columns = frame["cols"][rows]
        valid = columns >= 0
        selected, inverse = torch.unique(columns.clamp_min(0),return_inverse=True)
        fine = self.encode(frame["nodes"][rows],frame["neighbor_nodes"][rows],frame["strengths"][rows])
        coarse = self.encode(frame["coarse_nodes"][selected],frame["coarse_neighbor_nodes"][selected],
                             frame["coarse_strengths"][selected])
        pooled = self.layers[2](frame["pooled"][selected]).relu()
        edges = frame["edges"][rows]
        width, edge_width = fine.shape[1], edges.shape[2]
        w, bias = self.layers[3].weight, self.layers[3].bias
        anchor = (coarse @ w[:,edge_width+width:edge_width+2*width].T+
                  pooled @ w[:,edge_width+2*width:].T)
        local = fine @ w[:,edge_width:edge_width+width].T
        hidden = (edges @ w[:,:edge_width].T+local[:,None,:]+anchor[inverse]+bias).relu()
        masked = hidden*valid.unsqueeze(2)
        summary = torch.cat((masked.sum(1)/valid.sum(1,keepdim=True).clamp_min(1),masked.amax(1)),dim=1)
        w, bias = self.layers[4].weight, self.layers[4].bias
        refined = (hidden @ w[:,:width].T+(summary @ w[:,width:].T)[:,None,:]+bias).relu()
        signed, scores = self.layers[5](refined).unbind(dim=2)
        return signed,scores,self.layers[6](torch.cat((fine,summary),dim=1))


def context_support(scores, budget, valid, temperature=None):
    budgets = torch.as_tensor(CONTEXT_BUDGETS,device=scores.device)
    order = torch.argsort(torch.round(CONTEXT_SCORE_SCALE*scores).masked_fill(~valid,-torch.inf),
                          dim=1,descending=True,stable=True)
    rank = torch.empty_like(order).scatter_(1,order,
        torch.arange(order.shape[1],device=scores.device).expand_as(order))
    counts = budgets[torch.round(CONTEXT_SCORE_SCALE*budget).argmax(1)]
    hard = (rank < counts[:,None]) & valid
    if temperature is None:
        return hard
    cutoffs = scores.masked_fill(~valid,-torch.inf).gather(1,order)[:,budgets-1].detach()
    probability = torch.sigmoid((scores[:,:,None]-cutoffs[:,None,:])/temperature)
    soft = (probability*budget.softmax(dim=1)[:,None,:]).sum(2)*valid
    return hard.to(scores.dtype)+(soft-soft.detach())


def verified_frames(config):
    records = config["student_datasets"]
    frames = {}
    for path in records:
        report = read_json(path)
        assert report["accepted"] and report["mode"] in ("data", "refresh", "direct_data")
        cases = {(r["key"], r["method"]):r for r in report["cases"]}
        for level, receipts in report["samples"].items():
            for r in receipts:
                assert r["split"] == "training" and r["real_directions"] > 0
                for entry in ("samples", "transfers"):
                    assert digest(r[entry]["path"]) == r[entry]["sha256"], r[entry]["path"]
                if report["mode"] == "direct_data":
                    method = "teacher_roundtrip"
                    kind = r["graph"].get("kind", "D")
                    assert r["graph"] == direct_graph(config, kind)
                else:
                    method = "teacher_L"+level
                case = cases[(r["key"], method)]
                assert case["passed"] and case["final_true_residual"] <= case["true_limit"]
                frames.setdefault(int(level), []).append(dict(r, iterations=case["iterations"]))
    if config.get("student_full_training"):
        assert len({r["key"] for r in frames.get(1, [])}) == 55
    return frames


def load_training_rows(receipts, kind, device):
    all_frames = []
    median = max(float(np.median([r["iterations"] for r in receipts])), 1)
    probe = None
    for receipt in receipts:
        with np.load(receipt["samples"]["path"], allow_pickle=False) as d:
            names = ("x", "cols", "target", "norm_teacher")
            names += ("edges", "row_sum") if kind == "E" else ("hints", "norm")
            a = {n: d[kind+"_"+n] for n in names}
            n, cap = a["cols"].shape
            assert cap >= 12
            comp = (d["coarse_native"]%3).astype(np.int32)
            assert np.all((a["cols"] >= -1) & (a["cols"] < len(comp)))
            valid = a["cols"] >= 0
            safe = np.maximum(a["cols"], 0)
            q = d["q"][safe].astype(np.float32)
            q *= valid[:, :, None]
            action = d["action"]
            if kind == "E":
                assert a["edges"].shape == (n, cap, LOCAL_EDGE_FEATURES)
                component = (2*a["x"][:, 5]).astype(np.int32)
                frame_weight = n/(3*np.bincount(component, minlength=3)[component])
                inputs = dict(edges=a["edges"], row_sum=a["row_sum"])
            else:
                assert a["hints"].shape == (n, cap, 3)
                same = (comp[safe] == (2*a["x"][:, 5:6])).astype(np.float32)
                frame_weight = np.full(n, np.clip(np.sqrt(receipt["iterations"]/median), .7, 1.6))
                inputs = dict(hints=a["hints"], norm=a["norm"], same=same)
            all_frames.append(dict(x=a["x"], valid=valid, target=a["target"], norm_teacher=a["norm_teacher"],
                q=q, action=action, frame_weight=frame_weight.astype(np.float32), **inputs))
            if probe is None and n:
                keep = min(n, 192)
                probe_names = ("x", "cols") + (("edges",) if kind == "E" else ("hints", "norm"))
                probe = ({name:np.ascontiguousarray(a[name][:keep]) for name in probe_names}, comp)
    assert probe is not None
    data = {name: torch.as_tensor(np.concatenate([f[name] for f in all_frames]), device=device)
            for name in all_frames[0]}
    return data, probe


def loss_terms(model, data, indexes, budget, temperature):
    d = {name: value[indexes] for name, value in data.items()}
    weights, expected, _ = model(d["x"], d["hints"], d["norm"], d["same"], d["valid"], temperature)
    coefficient_scale = d["norm_teacher"].clamp_min(.05)
    coeff = ((weights-d["target"])/coefficient_scale[:, None]).square().sum(1)
    action = (weights[:, :, None]*d["q"]).sum(1)
    action_scale = torch.maximum(d["action"].square().mean(1).sqrt(), .4*coefficient_scale)
    effect = ((action-d["action"])/action_scale[:, None]).square().mean(1)
    wd = d["frame_weight"]
    coeff_loss, action_loss = (coeff*wd).mean(), (effect*wd).mean()
    budget_loss = .025*torch.relu(expected.mean()-budget).square()+.001*expected.mean()
    coeff_weight = .85 if model.kind == "D" else .55
    total = coeff_weight*coeff_loss+(1-coeff_weight)*action_loss+budget_loss
    return total, torch.stack([coeff_loss, action_loss, expected.mean(),
                               (weights != 0).sum(1).float().mean()])


def local_loss_terms(model, data, indexes, hard):
    d = {name: value[indexes] for name, value in data.items()}
    raw, scores = model(d["edges"])
    valid, target = d["valid"], d["target"]
    selected = model.select(raw, scores, valid) if hard else valid
    dense = raw*valid
    weights = raw*selected
    norm = d["norm_teacher"].clamp_min(.05)
    coeff = ((weights-target)/norm[:, None]).square().sum(1)
    dense_error = ((dense-target)/norm[:, None]).square().sum(1)
    # Independent ranking supervision favors large signed teacher entries and
    # keeps the ranking trainable when the deployment mask is already hard.
    positive = (target != 0) & valid
    negative = ~positive & valid
    ranking = nn.functional.smooth_l1_loss(scores, torch.log1p(8*target.abs()), reduction="none")
    support = ((ranking*positive).sum(1)/positive.sum(1).clamp_min(1) +
               (ranking*negative).sum(1)/negative.sum(1).clamp_min(1))/2
    action = (weights[:, :, None]*d["q"]).sum(1)
    action_scale = torch.maximum(d["action"].square().mean(1).sqrt(), .4*norm)
    effect = ((action-d["action"])/action_scale[:, None]).square().mean(1)
    sum_error = ((weights.sum(1)-d["row_sum"])/norm.clamp_min(.25)).square()
    balance = d["frame_weight"]
    total = ((.43*coeff+.10*dense_error+.25*support+.14*effect+.08*sum_error)*balance).mean()
    return total, torch.stack([(coeff*balance).mean(), (effect*balance).mean(),
                               (support*balance).mean(), selected.sum(1).float().mean()])


def context_training_frames(receipts, device):
    frames = []
    row_keys = ("cols","edges","nodes","neighbor_nodes","strengths","target","norm_teacher","row_sum")
    coarse_keys = ("coarse_nodes","coarse_neighbor_nodes","coarse_strengths","pooled","q","action")
    for receipt in receipts:
        with np.load(receipt["samples"]["path"],allow_pickle=False) as f:
            frame = {k:torch.as_tensor(f["F_"+k],device=device) for k in row_keys}
            frame.update({k:torch.as_tensor(f[k],device=device) for k in coarse_keys})
            regions = f["F_region"]
        frame["cols"] = frame["cols"].long()
        frame["groups"] = [np.flatnonzero(regions==g) for g in np.unique(regions)]
        component = frame["nodes"][:,9:12].argmax(1)
        frame["balance"] = len(component)/(3*torch.bincount(component,minlength=3)[component])
        frames.append(frame)
    return frames


def context_batches(frames, rows_per_batch, rng, device):
    groups = [(frame, rows) for frame in frames for rows in frame["groups"]]
    for index in rng.permutation(len(groups)):
        frame, rows = groups[index]
        shuffled = rng.permutation(rows)
        for start in range(0,len(rows),rows_per_batch):
            yield frame,torch.as_tensor(shuffled[start:start+rows_per_batch],device=device)


def context_loss_terms(model, frame, rows, hard, temperature):
    signed,scores,budget = model(frame,rows)
    columns = frame["cols"][rows]
    valid = columns >= 0
    target = frame["target"][rows]
    positive = target != 0
    mask = context_support(scores,budget,valid,temperature) if hard else positive
    weights = signed*mask
    norm = frame["norm_teacher"][rows].clamp_min(.05)
    coeff = ((weights-target)/norm[:,None]).square().sum(1)
    dense = (((signed*valid-target)/norm[:,None]).square()).sum(1)
    distribution = target.abs().pow(.75)
    distribution = distribution/distribution.sum(1,keepdim=True).clamp_min(1e-20)
    order = -(distribution*scores.masked_fill(~valid,-1e8).log_softmax(dim=1)).sum(1)
    binary = nn.functional.binary_cross_entropy_with_logits(scores,positive.to(scores.dtype),reduction="none")
    negative = ~positive & valid
    support = (binary*positive).sum(1)/positive.sum(1).clamp_min(1)
    support += (binary*negative).sum(1)/negative.sum(1).clamp_min(1)
    energy = torch.sort(target.square(),dim=1,descending=True).values.cumsum(1)
    max_count = torch.as_tensor(CONTEXT_BUDGETS,device=target.device)-1
    budget_target = (energy[:,max_count] < .998*energy[:,-1:]).sum(1).clamp_max(3)
    budget_loss = nn.functional.cross_entropy(budget,budget_target,reduction="none")
    action = (weights[:,:,None]*frame["q"][columns.clamp_min(0)]).sum(1)
    reference = frame["action"][rows]
    effect = ((action-reference)/torch.maximum(reference.square().mean(1).sqrt(),.4*norm)[:,None]).square().mean(1)
    moment = ((weights.sum(1)-frame["row_sum"][rows])/norm.clamp_min(.25)).square()
    balance = frame["balance"][rows]
    total = ((.44*coeff+.08*dense+.13*order+.08*support+.04*budget_loss+.16*effect+.07*moment)*balance).mean()
    return total,torch.stack(((coeff*balance).mean(),(effect*balance).mean(),(order*balance).mean(),
                              (weights != 0).sum(1).float().mean()))


def verify_context(model, frame, width):
    rows = torch.linspace(0,len(frame["cols"])-1,192,device=frame["cols"].device).long()
    model.eval()
    with torch.no_grad():
        raw,scores,budget = model(frame,rows)
        mask = context_support(scores,budget,frame["cols"][rows]>=0)
        expected,counts = (raw*mask).cpu().numpy(),mask.sum(1).cpu().numpy()
    layers = context_parameters(width,model.packed())
    def encoded(node, neighbor, strength):
        a = np.maximum(node.cpu().numpy() @ layers[0][0].T+layers[0][1],0)
        b = np.maximum(neighbor.cpu().numpy() @ layers[0][0].T+layers[0][1],0)
        return context_node_output(layers,a,b,strength.cpu().numpy())
    fine = encoded(*(frame[key][rows] for key in ("nodes","neighbor_nodes","strengths")))
    coarse = encoded(*(frame[key] for key in ("coarse_nodes","coarse_neighbor_nodes","coarse_strengths")))
    pool = np.maximum(frame["pooled"].cpu().numpy() @ layers[2][0].T+layers[2][1],0)
    cols = frame["cols"][rows].cpu().numpy()
    raw,scores,budget = context_edge_output(layers,fine,coarse,pool,frame["edges"][rows].cpu().numpy(),cols)
    values,counts_numpy = context_numpy_decode(raw,scores,budget,cols>=0)
    np.testing.assert_array_equal(counts,counts_numpy)
    np.testing.assert_array_equal(values!=0,expected!=0)
    np.testing.assert_allclose(values,expected,rtol=3e-5,atol=2e-6)
    return dict(rows=len(rows),exact_support=True,max_coefficient_difference=float(np.max(np.abs(values-expected))))


def verify_native(config, model, probe, width):
    arrays, comp = probe
    markers = np.full(len(arrays["x"]), -1, np.int32)
    # Compare on CPU with TF32 disabled in training; both use float32
    # and a stable candidate-order tie break.
    device = next(model.parameters()).device
    model.eval()
    if model.kind == "E":
        with torch.no_grad():
            raw, score = model(torch.as_tensor(arrays["edges"], device=device))
            mask = model.select(raw, score, torch.as_tensor(arrays["cols"] >= 0, device=device))
            expected, budget = raw*mask, mask.sum(1)
        values, actual = predict_local(config, arrays, markers, width, model.packed())
    else:
        safe = np.maximum(arrays["cols"], 0)
        same = (comp[safe] == (2*arrays["x"][:, 5:6])).astype(np.float32)
        args = [arrays["x"], arrays["hints"], arrays["norm"], same, arrays["cols"] >= 0]
        with torch.no_grad():
            expected, _, budget = model(*[torch.as_tensor(a, device=device) for a in args])
        values, actual = predict(config, arrays, markers, comp, width, STUDENT_KINDS[model.kind],
                                  model.packed(), DEFAULT_CONTROL)
    expected, budget = expected.cpu().numpy(), budget.cpu().numpy()
    assert np.array_equal(budget, actual), "Export changed the hard budget"
    assert np.array_equal(values != 0, expected != 0), "Export changed the selected support"
    np.testing.assert_allclose(values, expected, rtol=3e-5, atol=2e-6)
    return dict(rows=len(actual), exact_support=True,
                max_coefficient_difference=float(np.max(np.abs(values-expected))))


def train(config):
    job = os.environ.get("SLURM_JOB_ID", "local")
    output = Path(config["output_directory"])
    checkpoint_dir = output/"checkpoints"
    device = torch.device(config["student_train_device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This allocation did not provide the configured GPU")
    torch.set_num_threads(config.get("student_train_threads", 4))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    frames = verified_frames(config)
    reports = []
    for run in config["student_runs"]:
        level, kind, seed, width = (run[k] for k in ("level","kind","seed","width"))
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        local = kind == "E"
        context = kind == "F"
        if context:
            model = ContextInterpolationStudent(width).to(device)
        else:
            model = (LocalInterpolationStudent(width, config["direct_local"]) if local
                     else InterpolationStudent(kind, width)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=run.get("learning_rate", .002))
        if "resume" in run:
            receipt = run["resume"]
            assert digest(receipt["path"]) == receipt["sha256"]
            saved = torch.load(receipt["path"], map_location=device)
            assert saved["level"] == level and saved["kind"] == kind and saved["width"] == width
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
        if context:
            data = context_training_frames(frames[level],device)
            rows = sum(len(frame["nodes"]) for frame in data)
            cap = data[0]["cols"].shape[1]
        else:
            data, probe = load_training_rows(frames[level], kind, device)
            rows = len(data["x"])
            cap = probe[0]["cols"].shape[1]
        epochs = run["epochs"]
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=.0003)
                     if local or context else None)
        history, exports = [], []
        assert set(run["checkpoints"]) <= set(range(1,epochs+1)) and epochs in run["checkpoints"]
        for epoch in range(1, epochs+1):
            model.train()
            aggregate = torch.zeros(5, device=device)
            batch_size = run.get("batch_rows", 2048)
            target = 8. if level == 1 else 7.
            budget = 10-(10-target)*min(1., epoch/max(1,epochs//2))
            hard = epoch > epochs//(4 if context else 2)
            if context:
                rng = np.random.default_rng(seed+epoch)
                batches = context_batches(data,batch_size,rng,device)
            else:
                order = torch.randperm(rows,device=device)
                batches = ((data,indices) for indices in order.split(batch_size))
            for batch,indices in batches:
                optimizer.zero_grad(set_to_none=True)
                temperature = .22-.1*epoch/epochs
                if context:
                    loss,metrics = context_loss_terms(model,batch,indices,hard,temperature)
                elif local:
                    loss,metrics = local_loss_terms(model,batch,indices,hard)
                else:
                    loss,metrics = loss_terms(model,batch,indices,budget,temperature)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
                optimizer.step()
                aggregate += torch.cat((loss.detach().reshape(1), metrics.detach()))*len(indices)
            if scheduler is not None:
                scheduler.step()
            values = (aggregate/rows).detach().cpu().tolist()
            stats = dict(epoch=epoch, loss=values[0], coefficient_loss=values[1],
                action_loss=values[2], soft_budget=values[3], hard_entries=values[4])
            if local or context:
                stats.update(support_loss=stats.pop("soft_budget"), hard_selection=hard)
            history.append(stats)
            record(config, "student_epoch", job=job, level=level, kind=kind, seed=seed, **stats)
            if epoch in run["checkpoints"]:
                parity = (verify_context(model,data[0],width) if context
                          else verify_native(config,model,probe,width))
                stem = "student-%s-L%d-%s-seed%d-e%02d" % (job,level,kind,seed,epoch)
                native_path = checkpoint_dir/(stem+".npz")
                train_path = checkpoint_dir/(stem+".pt")
                with native_path.open("xb") as f:
                    np.savez(f, schema=np.int32(SCHEMA), width=np.int32(width), level=np.int32(level),
                        kind=np.int32(STUDENT_KINDS[kind]), candidates=np.int32(cap),
                        params=model.packed())
                with train_path.open("xb") as f:
                    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        level=level, kind=kind, width=width, seed=seed, epoch=epoch,
                        datasets=config["student_datasets"], history=history,
                        rng=torch.get_rng_state()), f)
                exports.append(dict(name=stem, path=str(native_path), sha256=digest(native_path),
                    level=level, kind=kind, seed=seed, epoch=epoch, native_parity=parity,
                    training=dict(path=str(train_path), sha256=digest(train_path))))
                if local or context:
                    exports[-1]["graph"] = direct_graph(config,kind)
                    exports[-1]["parameters"] = int(model.packed().size)
        reports.append(dict(level=level, kind=kind, width=width, seed=seed, samples=rows,
                            frames=len(frames[level]), history=history, checkpoints=exports))
        del data, model, optimizer
    result = dict(job=job, device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
        elapsed_seconds=time.perf_counter()-start, runs=reports, datasets=config["student_datasets"])
    with (output/"records"/("student-trained-"+job+".json")).open("x") as f:
        json.dump(result, f, allow_nan=False, indent=2)
    record(config, "student_training_complete", job=job, runs=len(reports), device=result["device_name"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    train(read_json(parser.parse_args().config))
