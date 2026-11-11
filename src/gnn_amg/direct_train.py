"""Distill direct coarse points, sparse addresses and signed final weights."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

from .backend import digest, read_json, record
from .student_cycle import save_once


class DirectNet(nn.Module):
    """One stiffness encoder serves the point, routing and interpolation heads."""
    def __init__(self, features, width):
        super().__init__()
        self.encoder = nn.Linear(features, width)
        self.region_query = nn.Linear(width, width)
        self.region_key = nn.Linear(width, width)
        self.region_value = nn.Linear(width, width)
        self.fuse = nn.Linear(2*width, width)
        self.coarse = nn.Linear(width, 1)
        self.region_budget = nn.Linear(width, 1)
        self.route_query = nn.Linear(width, width)
        self.route_key = nn.Linear(width, width)
        self.query = nn.Linear(width, width//2)
        self.key = nn.Linear(width, width//2)
        self.row_budget = nn.Linear(width, 1)
        self.edge = nn.Linear(2*width+6, width)
        self.edge_heads = nn.Linear(width, 2)

    def context(self, features, available):
        encoded = torch.tanh(self.encoder(features))
        score = self.region_query(encoded) @ self.region_key(encoded).T / np.sqrt(encoded.shape[-1])
        attention = F.softmax(score.masked_fill(~available[None, :], -1e9), dim=1)
        return encoded + attention @ self.region_value(encoded)

    def nodes(self, features, ids, context):
        hidden = torch.tanh(self.encoder(features))
        return hidden + torch.tanh(self.fuse(torch.cat([hidden, context[ids]], dim=-1)))

    def routing(self, fine, context, available):
        query = self.route_query(fine)
        key = self.route_key(context)
        scores = query @ key.T / np.sqrt(fine.shape[-1])
        return scores.masked_fill(~available[None, :], -1e9)

    def edge_outputs(self, fine, coarse, descriptors, regions, route):
        shape = (*coarse.shape[:-1], fine.shape[-1])
        query = F.normalize(self.query(fine), dim=-1)
        key = F.normalize(self.key(coarse), dim=-1)
        addressing = 4.0 * (query[:, None, :] * key).sum(-1)
        pair = torch.cat([fine[:, None, :].expand(shape), coarse, descriptors], dim=-1)
        score, coefficient = self.edge_heads(torch.tanh(self.edge(pair))).unbind(-1)
        return addressing, score + addressing + route.gather(1, regions), coefficient


def support_gate(logits, valid, quota):
    """Exact hard ranks in the forward pass, with sigmoid proxy derivatives."""
    order = torch.argsort(logits.masked_fill(~valid, -1e9), dim=1, descending=True, stable=True)
    rank = torch.argsort(order, dim=1)
    hard = valid & (rank < quota[:, None])
    soft = torch.sigmoid(logits) * valid
    return hard, hard.to(logits.dtype) + (soft - soft.detach())


def final_weights(coefficients, gate, hard, same_component, unit_sum):
    """One signed output per edge. Fine-level rows reproduce constants."""
    weights = coefficients * gate
    if unit_sum:
        preferred = hard & same_component
        preferred = torch.where(preferred.any(dim=1, keepdim=True), preferred, hard)
        distribution = preferred / preferred.sum(dim=1, keepdim=True).clamp_min(1)
        weights = weights + distribution * (1.0 - weights.sum(dim=1, keepdim=True))
    return weights


class Frame:
    def __init__(self, folder, depth, shards):
        self.depth = depth
        self.key = folder.name
        with np.load(folder / ("l%d-context.npz" % depth)) as c:
            for key in c.files:
                setattr(self, key, c[key])
        names = ("features", "region", "native", "selected", "xyz", "difficulty", "rank")
        samples = {key: [] for key in names}
        counts, columns, values = [], [], []
        for rank in range(shards):
            with np.load(folder / ("l%d-r%03d.npz" % (depth, rank))) as shard:
                for key in names:
                    samples[key].append(shard[key])
                counts.append(np.diff(shard["indptr"]))
                columns.append(shard["columns"])
                values.append(shard["values"])
        for key, data in samples.items():
            setattr(self, key, np.concatenate(data))
        self.offsets = np.r_[0, np.cumsum(np.concatenate(counts))]
        self.columns = np.concatenate(columns)
        self.values = np.concatenate(values)
        if len(self.features) == 0 or len(self.values) == 0:
            raise ValueError("Empty direct sample bank for " + self.key)

    def batch(self, rows):
        start, stop = self.offsets[rows], self.offsets[rows+1]
        offsets = np.arange(max(1, int(np.max(stop-start))))
        mask = offsets[None, :] < (stop-start)[:, None]
        index = np.where(mask, start[:, None]+offsets[None, :], 0)
        cols = self.columns[index]
        target = np.where(mask, self.values[index], 0)
        delta = self.endpoint_xyz[cols] - self.xyz[rows, None, :]
        same = self.endpoint_native[cols] % 3 == self.native[rows, None] % 3
        remote = self.endpoint_rank[cols] != self.rank[rows, None]
        edges = np.concatenate([delta, np.linalg.norm(delta, axis=-1)[..., None],
                                same[..., None], remote[..., None]], axis=-1).astype(np.float32)
        return dict(fine=self.features[rows], region=self.region[rows], selected=self.selected[rows],
                    difficulty=self.difficulty[rows], coarse=self.endpoint_features[cols],
                    coarse_region=self.endpoint_region[cols], descriptors=edges,
                    valid=mask, target=target, same=same, remote=remote)


def objective(model, frame, rows, warmup, cap, cost):
    x = {key: torch.from_numpy(array) for key, array in frame.batch(rows).items()}
    available = torch.from_numpy(frame.region_counts > 0)
    context = model.context(torch.from_numpy(frame.regions), available)
    fine = model.nodes(x["fine"], x["region"].long(), context)
    coarse = model.nodes(x["coarse"], x["coarse_region"].long(), context)
    route = model.routing(fine, context, available)
    addressing, score, raw = model.edge_outputs(fine, coarse, x["descriptors"],
                                               x["coarse_region"].long(), route)
    point = model.coarse(fine).squeeze(-1)
    label = x["selected"]
    log_point = F.binary_cross_entropy_with_logits(point, label.float(), reduction="none")
    point_loss = (log_point[label].sum()/label.sum().clamp_min(1) +
                  log_point[~label].sum()/(~label).sum().clamp_min(1))/2
    distribution = torch.from_numpy(frame.region_coarse_count).float()
    distribution = distribution / distribution.sum().clamp_min(1)
    allocation = model.region_budget(context).squeeze(-1).masked_fill(~available, -1e9)
    budget_loss = -(distribution * F.log_softmax(allocation, dim=0)).sum()
    present, valid = x["target"] != 0, x["valid"]
    count = present.sum(dim=1)
    weights = (1 + .3*x["difficulty"]) * (count > 0)
    normalize = weights.sum().clamp_min(1)

    def active_mean(value):
        return (weights * value).sum()/normalize

    def balanced_bce(logits):
        raw_loss = F.binary_cross_entropy_with_logits(logits, present.float(), reduction="none")
        absent = valid & ~present
        return active_mean((raw_loss * present).sum(1)/count.clamp_min(1)
                           + (raw_loss * absent).sum(1)/absent.sum(1).clamp_min(1))/2

    rank_loss = balanced_bce(score) + .5*balanced_bce(addressing)
    route_loss = active_mean(-(F.log_softmax(route, dim=1).gather(1, x["coarse_region"].long())
                              * present).sum(1)/count.clamp_min(1))
    log_budget = torch.sigmoid(model.row_budget(fine).squeeze(-1)) * np.log1p(cap)
    quota_loss = active_mean(F.smooth_l1_loss(log_budget, torch.log1p(count.float()),
                                              reduction="none"))
    quotas = torch.round(torch.expm1(log_budget.detach())).long().clamp(1, cap)
    hard, gate = support_gate(score, valid, quotas)
    if warmup:
        gate = present.float()
    pred = final_weights(raw, gate, present if warmup else hard, x["same"], frame.depth == 1)
    target = x["target"]
    magnitude = target.square().sum(1).clamp_min(.04)
    coefficient_loss = active_mean((pred-target).square().sum(1)/magnitude)
    cost_loss = active_mean((gate*(1+.3*x["remote"])).sum(1)/cap) if not warmup else 0.
    loss = point_loss + .15*budget_loss + rank_loss + .35*route_loss + .25*quota_loss \
        + coefficient_loss + cost*cost_loss
    physical = final_weights(raw.detach(), hard.float(), hard, x["same"], frame.depth == 1)
    return loss, dict(rows=len(rows), loss=float(loss.detach()), coefficient=float(coefficient_loss.detach()),
        true_points=int(label.sum()), predicted_points=int((point > 0).sum()),
        correct_points=int((label & (point > 0)).sum()), kept=int(hard.sum()), positives=int(count.sum()),
        retained=int((present & hard).sum()), remote=int((hard & x["remote"]).sum()),
        coefficient_error=float((physical-target).square().sum()), coefficient_energy=float(target.square().sum()))


@torch.no_grad()
def retrieval_diagnostic(model, frame, count):
    """Compare online-searchable geometry and learned retrieval at teacher C."""
    available = torch.from_numpy(frame.region_counts > 0)
    context = model.context(torch.from_numpy(frame.regions), available)
    fine = model.nodes(torch.from_numpy(frame.features), torch.from_numpy(frame.region).long(), context)
    coarse = model.nodes(torch.from_numpy(frame.endpoint_features),
                         torch.from_numpy(frame.endpoint_region).long(), context)
    queries = F.normalize(model.query(fine), dim=-1).numpy()
    keys = F.normalize(model.key(coarse), dim=-1).numpy()
    route = model.routing(fine, context, available).numpy()
    geo = cKDTree(frame.endpoint_xyz)
    _, geometric = geo.query(frame.xyz, k=min(count, len(keys)))
    geometric = geometric.reshape(len(queries), -1)
    # Query only the two most promising populated coarse regions. The tree
    # search can retrieve links arbitrarily far from the geometric shortlist.
    pools = [np.flatnonzero(frame.endpoint_region == i) for i in range(len(context))]
    allowed = np.array([len(pool) > 0 for pool in pools])
    route[:, ~allowed] = -np.inf
    target_regions = np.argsort(-route, axis=1, kind="stable")[:, :min(2, allowed.sum())]
    proposals = [[] for _ in queries]
    for i, pool in enumerate(pools):
        rows = np.flatnonzero(np.any(target_regions == i, axis=1))
        if not len(rows):
            continue
        tree = cKDTree(keys[pool])
        _, hits = tree.query(queries[rows], k=min(count//4, len(pool)))
        for row, chosen in zip(rows, pool[hits].reshape(len(rows), -1)):
            proposals[row].extend(chosen.tolist())
    n, by_geo, combined, novel, proposed, n_missing, row_hits = 0, 0, 0, 0, 0, 0, 0
    energy, geo_energy, combined_energy = 0., 0., 0.
    for i, pair in enumerate(zip(frame.offsets[:-1], frame.offsets[1:])):
        a, b = pair
        positive = frame.values[a:b] != 0
        if not positive.any():
            continue
        cols = frame.columns[a:b][positive]
        weights = frame.values[a:b][positive].astype(float)**2
        was = np.isin(cols, geometric[i])
        mixed = np.union1d(geometric[i, :count//2], proposals[i]).astype(np.int64)
        found = np.isin(cols, mixed)
        n += len(cols)
        by_geo += int(was.sum())
        combined += int(found.sum())
        novel += int((found & ~was).sum())
        proposed += len(np.setdiff1d(mixed, geometric[i]))
        energy += float(weights.sum())
        geo_energy += float(weights[was].sum())
        combined_energy += float(weights[found].sum())
        n_missing += int(not found.all())
        row_hits += 1
    return dict(key=frame.key, depth=frame.depth, positives=n, tested_rows=row_hits,
                geometry_edge_recall=by_geo/max(n, 1),
                combined_edge_recall=combined/max(n, 1), novel_true_edges=novel,
                geometry_energy=geo_energy/max(energy, 1e-300),
                combined_energy=combined_energy/max(energy, 1e-300), candidate_budget=count,
                rows_with_missing_edges=n_missing, additional_proposals=proposed)


def metric_summary(rows):
    count = sum(row["rows"] for row in rows)
    total = {name: sum(row[name] for row in rows) for name in
             ("positives", "retained", "coefficient_error", "coefficient_energy",
              "correct_points", "true_points", "predicted_points", "kept", "remote")}
    return dict(rows=count, loss=sum(row["loss"]*row["rows"] for row in rows)/count,
                coefficient_error=float(np.sqrt(total["coefficient_error"]
                                                / max(total["coefficient_energy"], 1e-300))),
                candidate_conditioned_recall=total["retained"]/max(total["positives"], 1),
                point_precision=total["correct_points"]/max(total["predicted_points"], 1),
                point_recall=total["correct_points"]/max(total["true_points"], 1),
                selected_edges=total["kept"], remote_edges=total["remote"])


def train(config):
    job = os.environ["GNN_RUN_ID"]
    settings = config["direct_final"]
    training = settings["distillation"]
    output = Path(config["output_directory"])
    bank = read_json(output / "records" / ("direct-bank-" + job + ".json"))
    if len(bank["fit"]) != 51 or len(bank["diagnostic"]) != 4 or set(bank["fit"]) & set(bank["diagnostic"]):
        raise ValueError("Distillation requires the frozen independent split")
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(min(training["threads"], len(os.sched_getaffinity(0))))
    rng = np.random.default_rng(settings["seed"])
    directory = Path(bank["directory"])
    fit = [Frame(directory/key, depth, bank["shards"]) for key in bank["fit"] for depth in (1, 2)]
    diag = [Frame(directory/key, depth, bank["shards"]) for key in bank["diagnostic"] for depth in (1, 2)]
    input_size = fit[0].features.shape[1]
    model = DirectNet(input_size, training["width"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"])
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    gradient_max = {name: 0.0 for name in ("encoder", "coarse", "route_query", "query",
                                           "edge_heads", "row_budget", "region_budget")}

    def execute(frames, train_model, warmup):
        rows = []
        for i in (rng.permutation(len(frames)) if train_model else range(len(frames))):
            frame = frames[i]
            order = (rng.permutation(len(frame.features)) if train_model
                     else np.arange(len(frame.features)))
            for begin in range(0, len(order), training["batch_size"]):
                indices = order[begin:begin+training["batch_size"]]
                loss, brief = objective(model, frame, indices, warmup,
                                        training["caps"][frame.depth-1], training["cost"])
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite distillation loss on " + frame.key)
                if train_model:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    for name, maximum in gradient_max.items():
                        value = float(torch.linalg.vector_norm(getattr(model, name).weight.grad))
                        gradient_max[name] = max(maximum, value)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                rows.append(brief)
        return metric_summary(rows)

    with torch.no_grad():
        before = execute(diag, False, False)
        retrieval_before = [retrieval_diagnostic(model, frame, training["diagnostic_candidates"])
                            for frame in diag]
    history, best, score, best_epoch = [], None, float("inf"), 0
    for epoch in range(1, training["epochs"]+1):
        warm = epoch <= training["warmup_epochs"]
        model.train()
        fitted = execute(fit, True, warm)
        model.eval()
        with torch.no_grad():
            validation = execute(diag, False, False)
        history.append(dict(epoch=epoch, teacher_support_warmup=warm,
                            fit=fitted, diagnostic=validation))
        record(config, "direct_distillation_epoch", job=job, **history[-1])
        if not warm and validation["loss"] < score:
            score, best_epoch = validation["loss"], epoch
            best = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if best is None:
        raise ValueError("No hard-support checkpoint was evaluated")
    model.load_state_dict(best)
    model.eval()
    retrieval = [retrieval_diagnostic(model, frame, training["diagnostic_candidates"]) for frame in diag]
    deltas = {name: float(torch.linalg.vector_norm(best[name]-initial[name])) for name in best}
    if not all(np.isfinite(list(deltas.values()))) or min(gradient_max.values()) <= 0:
        raise ValueError("A direct parameter did not update")
    model_path = output / "checkpoints" / ("generator-%s-direct-final.npz" % job)
    with model_path.open("xb") as stream:
        np.savez(stream, input_features=input_size, width=training["width"], schema=1,
                 **{name: tensor.numpy() for name, tensor in best.items()})
    receipt = dict(job=job, source=bank["source"], bank=str(directory), fit=bank["fit"],
                   diagnostic=bank["diagnostic"], initial_diagnostic=before, epochs=history,
                   selected_epoch=best_epoch, parameter_delta=deltas, gradient_max=gradient_max,
                   retrieval_before=retrieval_before, retrieval=retrieval,
                   checkpoint=dict(path=str(model_path), sha256=digest(model_path)),
                   stage="supervised signed final-transfer distillation")
    save_once(output / "records" / ("direct-distill-" + job + ".json"), receipt)
    record(config, "direct_distillation_complete", job=job, selected_epoch=best_epoch,
           checkpoint=receipt["checkpoint"], diagnostic=history[best_epoch-1]["diagnostic"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    train(read_json(args.config))


if __name__ == "__main__":
    main()
