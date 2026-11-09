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
from .student import DEFAULT_CONTROL, EDGE_FEATURES, ROW_FEATURES, SCHEMA, predict


class InterpolationStudent(nn.Module):
    def __init__(self, kind, width):
        super().__init__()
        self.kind = kind
        self.row = nn.Sequential(nn.Linear(ROW_FEATURES, width), nn.ReLU(), nn.Linear(width, 4))
        self.edge = (nn.Sequential(nn.Linear(EDGE_FEATURES, width), nn.ReLU(), nn.Linear(width, 2))
                     if kind == "B" else None)
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
            e = self.edge(z).clamp(-1, 1)
            raw = raw+hints[:, :, 2]+norm[:, None]*e[:, :, 0]
            shift = e[:, :, 1]
        scores = raw.abs()/norm[:, None]+shift
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


def verified_frames(config):
    records = config["student_datasets"]
    frames = {}
    for path in records:
        report = read_json(path)
        assert report["accepted"] and report["mode"] in ("data", "refresh")
        cases = {(r["key"], r["method"]):r for r in report["cases"]}
        for level, receipts in report["samples"].items():
            for r in receipts:
                assert r["split"] == "training" and r["real_directions"] > 0
                for entry in ("samples", "transfers"):
                    assert digest(r[entry]["path"]) == r[entry]["sha256"], r[entry]["path"]
                case = cases[(r["key"], "teacher_L"+level)]
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
            names = ("x", "hints", "norm", "cols", "target", "norm_teacher")
            a = {n: d[kind+"_"+n] for n in names}
            n, cap = a["cols"].shape
            assert cap >= 12 and a["hints"].shape == (n, cap, 3)
            comp = (d["coarse_native"]%3).astype(np.int32)
            assert np.all((a["cols"] >= -1) & (a["cols"] < len(comp)))
            valid = a["cols"] >= 0
            safe = np.maximum(a["cols"], 0)
            q = d["q"][safe].astype(np.float32)
            q *= valid[:, :, None]
            same = (comp[safe] == (2*a["x"][:, 5:6])).astype(np.float32)
            action = d["action"]
            all_frames.append(dict(x=a["x"], hints=a["hints"], norm=a["norm"], same=same,
                valid=valid, target=a["target"], norm_teacher=a["norm_teacher"], q=q,
                action=action, frame_weight=np.full(n, np.clip(np.sqrt(receipt["iterations"]/median),
                    .7, 1.6), dtype=np.float32)))
            if probe is None and n:
                keep = min(n, 192)
                probe = ({name:np.ascontiguousarray(a[name][:keep]) for name in ("x","hints","norm","cols")}, comp)
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
    total = .55*coeff_loss+.45*action_loss+budget_loss
    return total, torch.stack([coeff_loss, action_loss, expected.mean(),
                               (weights != 0).sum(1).float().mean()])


def verify_native(config, model, probe, width):
    arrays, comp = probe
    markers = np.full(len(arrays["x"]), -1, np.int32)
    safe = np.maximum(arrays["cols"], 0)
    same = (comp[safe] == (2*arrays["x"][:, 5:6])).astype(np.float32)
    # Compare on CPU with TF32 disabled in training; both use float32
    # and a stable candidate-order tie break.
    device = next(model.parameters()).device
    args = [arrays["x"], arrays["hints"], arrays["norm"], same, arrays["cols"] >= 0]
    model.eval()
    with torch.no_grad():
        expected, _, budget = model(*[torch.as_tensor(a, device=device) for a in args])
    values, actual = predict(config, arrays, markers, comp, width, int(model.kind=="B"),
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
        model = InterpolationStudent(kind, width).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=run.get("learning_rate", .002))
        if "resume" in run:
            receipt = run["resume"]
            assert digest(receipt["path"]) == receipt["sha256"]
            saved = torch.load(receipt["path"], map_location=device)
            assert saved["level"] == level and saved["kind"] == kind and saved["width"] == width
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
        data, probe = load_training_rows(frames[level], kind, device)
        rows = len(data["x"])
        epochs = run["epochs"]
        history, exports = [], []
        assert set(run["checkpoints"]) <= set(range(1,epochs+1)) and epochs in run["checkpoints"]
        for epoch in range(1, epochs+1):
            model.train()
            aggregate = torch.zeros(5, device=device)
            order = torch.randperm(rows, device=device)
            batch_size = run.get("batch_rows", 2048)
            target = 8. if level == 1 else 7.
            budget = 10-(10-target)*min(1., epoch/max(1,epochs//2))
            for indices in order.split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = loss_terms(model, data, indices, budget, .22-.1*epoch/epochs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
                optimizer.step()
                aggregate += torch.cat((loss.detach().reshape(1), metrics.detach()))*len(indices)
            values = (aggregate/rows).detach().cpu().tolist()
            stats = dict(epoch=epoch, loss=values[0], coefficient_loss=values[1],
                action_loss=values[2], soft_budget=values[3], hard_entries=values[4])
            history.append(stats)
            record(config, "student_epoch", job=job, level=level, kind=kind, seed=seed, **stats)
            if epoch in run["checkpoints"]:
                parity = verify_native(config, model, probe, width)
                stem = "student-%s-L%d-%s-seed%d-e%02d" % (job,level,kind,seed,epoch)
                native_path = checkpoint_dir/(stem+".npz")
                train_path = checkpoint_dir/(stem+".pt")
                with native_path.open("xb") as f:
                    np.savez(f, schema=np.int32(SCHEMA), width=np.int32(width), level=np.int32(level),
                        kind=np.int32(kind=="B"), candidates=np.int32(config["student_candidates"]),
                        params=model.packed())
                with train_path.open("xb") as f:
                    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        level=level, kind=kind, width=width, seed=seed, epoch=epoch,
                        datasets=config["student_datasets"], history=history,
                        rng=torch.get_rng_state()), f)
                exports.append(dict(name=stem, path=str(native_path), sha256=digest(native_path),
                    level=level, kind=kind, seed=seed, epoch=epoch, native_parity=parity,
                    training=dict(path=str(train_path), sha256=digest(train_path))))
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
