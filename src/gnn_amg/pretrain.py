"""GPU supervised warm start; numerical prediction validation is still required.

Raw accepted C-point labels are targets. Coverage repairs are recorded by the
numerical environment and never substituted for those raw targets. Checkpoint
likelihood is diagnostic, not an end-to-end speed or convergence claim.
"""
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .backend import digest, read_json, record
from .dataset import FEATURE_NAMES
from .policy import TwoStagePolicy, supervised_loss


def examples(config):
    root = Path(config["output_directory"])
    job = str(config["dataset_job"])
    gate = read_json(root / "records" / ("gate-" + job + ".json"))
    if not gate["passed"]:
        raise ValueError("Supervised dataset did not pass the numerical gate")
    inventory = read_json(Path(config["baseline_directory"]) / "data/inventory.json")
    assert digest(Path(config["baseline_directory"]) / "data/inventory.json") == config["inventory_sha256"]
    rows = [r for r in inventory["snapshots"] if r["split"] == "training"]
    assert gate["methods"] == len(rows)
    result = []
    for row in rows:
        receipt = read_json(root / "records" / ("gate-%s-%s-teacher_multipass_repaired.json" % (job, row["key"])))
        assert receipt["reached_true_target"] and receipt["failure"] is None
        files = receipt["numerical"]["supervised_files"]
        assert [f["stage"] for f in files] == [0, 1]
        for f in files:
            assert digest(f["path"]) == f["sha256"]
            result.append(dict(f, key=row["key"], manifest=row["manifest_sha256"]))
    return result


def fit_scaling(rows):
    """Separate-stage scalers, fitted exclusively on training examples."""
    width = len(FEATURE_NAMES)
    counts = np.zeros(2, np.int64)
    means = np.zeros((2, width))
    m2 = np.zeros_like(means)
    positives, eligible = np.zeros(2, np.int64), np.zeros(2, np.int64)
    for row in rows:
        stage = row["stage"]
        with np.load(row["path"]) as f:
            assert list(f["feature_names"]) == FEATURE_NAMES
            x = f["features"][f["eligible"]].astype(np.float64)
            n = len(x)
            block_mean = x.mean(axis=0)
            block_m2 = np.sum((x-block_mean)**2, axis=0)
            delta = block_mean - means[stage]
            total = counts[stage] + n
            m2[stage] += block_m2 + delta**2 * (counts[stage]*n/total)
            means[stage] += delta*n/total
            counts[stage] = total
            positives[stage] += np.count_nonzero(f["raw_teacher"] & f["eligible"])
            eligible[stage] += np.count_nonzero(f["eligible"])
    scale = np.sqrt(m2/counts[:, None])
    scale[scale < 1e-6] = 1
    return means, scale, positives/eligible


def tensor_example(path, mean, scale, device):
    with np.load(path) as f:
        stage = int(f["stage"])
        x = np.clip((f["features"]-mean[stage])/scale[stage], -12, 12).astype(np.float32)
        n = len(x)
        graph = torch.sparse_csr_tensor(torch.from_numpy(f["indptr"].astype(np.int64)),
            torch.from_numpy(f["indices"].astype(np.int64)), torch.from_numpy(f["weights"]),
            size=(n, n), device=device)
        return (torch.as_tensor(x, device=device), graph,
                torch.as_tensor(f["connection_prior"], device=device),
                torch.as_tensor(f["raw_teacher"], device=device),
                torch.as_tensor(f["eligible"], device=device), stage)


def train(config):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU pretraining requires the allocated CUDA device")
    settings = config["pretraining"]
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(settings.get("cpu_threads", 8))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = examples(config)
    mean, scale, coarse_fraction = fit_scaling(rows)
    device = torch.device("cuda")
    model = TwoStagePolicy(len(FEATURE_NAMES), config["training"]["width"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    output = Path(config["output_directory"])
    job = os.environ.get("SLURM_JOB_ID", "local")
    started = time.perf_counter()
    history, updates = [], 0
    record(config, "pretrain_begin", job=job, device=torch.cuda.get_device_name(),
        examples=len(rows), raw_coarse_fraction=coarse_fraction.tolist(),
        seed=seed, dataset_job=str(config["dataset_job"]), torch=torch.__version__,
        nondeterminism="CUDA sparse reductions may vary at floating-point roundoff; seeds and software are recorded.")
    for epoch in range(1, settings["epochs"]+1):
        order = list(rows)
        random.shuffle(order)
        losses, recalls = [[], []], [[], []]
        epoch_start = time.perf_counter()
        model.train()
        for row in order:
            features, graph, prior, labels, eligible, stage = tensor_example(row["path"], mean, scale, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, graph, prior, stage)
            loss = supervised_loss(logits, labels, eligible)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite supervised loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                scores = logits.masked_fill(~eligible, -torch.inf)
                count = int(labels.sum().item())
                selected = torch.topk(scores, count, sorted=False).indices
                recall = labels[selected].float().mean().item()
            losses[stage].append(loss.item())
            recalls[stage].append(recall)
            updates += 1
            if updates % 20 == 0:
                record(config, "pretrain_update", job=job, epoch=epoch, updates=updates,
                    key=row["key"], stage=stage, loss=loss.item(), raw_label_recall=recall,
                    gradient_norm=float(gradient), peak_gpu_bytes=torch.cuda.max_memory_allocated())
            del features, graph, prior, labels, eligible, logits, loss, scores, selected
        torch.cuda.synchronize()
        detail = dict(epoch=epoch, updates=updates, seconds=time.perf_counter()-epoch_start,
            loss_by_stage=[float(np.mean(x)) for x in losses],
            raw_label_recall_by_stage=[float(np.mean(x)) for x in recalls])
        history.append(detail)
        path = output / "checkpoints" / ("pretrain-%s-epoch%02d.pt" % (job, epoch))
        if path.exists():
            raise FileExistsError(path)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            feature_mean=mean, feature_scale=scale, coarse_fraction=coarse_fraction,
            feature_names=FEATURE_NAMES, settings=settings, epoch=epoch, updates=updates,
            dataset_job=str(config["dataset_job"]), dataset_sources=rows,
            rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all(), seed=seed, cuda=torch.version.cuda,
            torch_version=torch.__version__, config=config), path)
        detail.update(checkpoint=str(path), sha256=digest(path))
        record(config, "pretrain_epoch_complete", job=job, **detail)
        (output / "records" / ("pretrain-"+job+".json")).write_text(json.dumps(dict(
            state="running", epochs=history, scope="supervised warm start; independent predicted-C solve still required"), indent=2)+"\n")
    result = dict(state="supervised_complete_not_numerically_validated", job=job,
        epochs=history, seconds=time.perf_counter()-started, training_states=len(rows)//2,
        validation_states_used=0, test_states_used=0,
        next="Freeze checkpoint; independently predict C1/C2 without reading teacher hierarchies, verify full solve, then test one real on-policy update before24h RL.")
    (output / "records" / ("pretrain-"+job+".json")).write_text(json.dumps(result, indent=2)+"\n")
    record(config, "pretrain_complete", job=job, epochs=len(history), updates=updates,
        seconds=result["seconds"], checkpoint=history[-1]["checkpoint"], validated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    train(read_json(args.config))


if __name__ == "__main__":
    main()
