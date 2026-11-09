"""Persistent GPU policy service and label-free deterministic coarse decoder.

The wire format is JSON metadata followed by typed numeric buffers, never
pickled objects. A job-private token authenticates the sole MPI-root client.
"""
import argparse
import ctypes as C
from functools import lru_cache
import itertools
import json
import os
from pathlib import Path
import secrets
import socket
import struct
import time

import numpy as np
from scipy import sparse

from .backend import digest, read_json, record


def send_packet(stream, metadata, arrays=None):
    buffers = {k: np.ascontiguousarray(v) for k, v in (arrays or {}).items()}
    items = []
    for name, array in buffers.items():
        if array.dtype.hasobject:
            raise TypeError("Object arrays are forbidden")
        items.append(dict(name=name, dtype=array.dtype.str, shape=list(array.shape), bytes=array.nbytes))
    header = json.dumps(dict(metadata=metadata, arrays=items), allow_nan=False).encode()
    stream.sendall(struct.pack("!Q", len(header)))
    stream.sendall(header)
    for array in buffers.values():
        if array.nbytes:
            stream.sendall(memoryview(array).cast("B"))


def receive_exact(stream, size):
    buffer = bytearray(size)
    view = memoryview(buffer)
    done = 0
    while done < size:
        count = stream.recv_into(view[done:])
        if not count:
            raise EOFError("Policy connection closed")
        done += count
    return buffer


def receive_packet(stream):
    size = struct.unpack("!Q", receive_exact(stream, 8))[0]
    if size > 1 << 20:
        raise ValueError("Oversized message header")
    header = json.loads(receive_exact(stream, size))
    arrays = {}
    total = 0
    for item in header["arrays"]:
        dtype = np.dtype(item["dtype"])
        shape = tuple(item["shape"])
        if dtype.hasobject or any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError("Invalid numeric array")
        size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        total += size
        if size != item["bytes"] or size < 0 or total > 1 << 30:
            raise ValueError("Invalid message size")
        arrays[item["name"]] = np.frombuffer(receive_exact(stream, size), dtype=dtype).reshape(shape)
    return header["metadata"], arrays


class PolicyClient:
    def __init__(self, endpoint, colocated=False):
        cfg = read_json(endpoint)
        self.binding = None
        if colocated:
            host = socket.gethostname()
            cpus = sorted(os.sched_getaffinity(0))
            if cfg["node"] != host or cfg["cpus"] != cpus or cfg["host"] != "127.0.0.1":
                raise RuntimeError("Policy service must share the actual solver root's host and affinity")
            self.binding = dict(node=host, cpus=cpus, service_pid=cfg["pid"])
        self.stream = socket.create_connection((cfg["host"], cfg["port"]), timeout=300)
        self.request(dict(operation="hello", token=cfg["token"]))

    def request(self, metadata, arrays=None):
        send_packet(self.stream, metadata, arrays)
        reply, values = receive_packet(self.stream)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply, values

    def close(self):
        result, _ = self.request(dict(operation="stop"))
        self.stream.close()
        return result


def spatial_decode(scores, features, native, eligible, indptr, indices, fraction):
    """Select the highest GNN score per component in mesh-scale spatial bins.

    The bins provide distributed coverage at the training-derived size scale;
    scores choose the point within each bin. The numerical backend alone checks
    actual strong paths and can add bounded repairs. This decoder reads neither
    teacher IDs nor a traditional AMG hierarchy. Native IDs resolve exact ties.
    """
    scores = np.asarray(scores)
    native = np.asarray(native, dtype=np.int64)
    valid = np.flatnonzero(eligible)
    if len(valid) < 6 or not np.isfinite(scores).all() or not 0 < fraction < 1:
        raise ValueError("Invalid selection input")
    xyz = np.asarray(features[:, 10:13], dtype=np.float64)
    low, high = xyz[valid].min(axis=0), xyz[valid].max(axis=0)
    span = np.maximum(high-low, 1e-12)
    # Estimate spacing from this actual graph. Normalized physical axes alone
    # would misrepresent a geometrically or topologically anisotropic mesh.
    positions = np.linspace(0, max(len(indices)-1, 0), min(len(indices), 65536), dtype=np.int64)
    if len(positions):
        row = np.searchsorted(indptr, positions, side="right")-1
        delta = np.abs(xyz[indices[positions]]-xyz[row])
        step = np.array([np.median(a[a > span[k]*1e-6]) if np.any(a > span[k]*1e-6) else span[k]
                         for k, a in enumerate(delta.T)])
    else:
        step = span.copy()
    shape = np.maximum(span/step, 1)
    shape /= np.prod(shape)**(1/3)
    target = max(1, int(round(len(valid)*fraction/3)))
    nominal = shape * target**(1/3)
    choices = [sorted(set(max(1, int(round(x))+offset) for offset in (-2, -1, 0, 1, 2))) for x in nominal]
    candidates = []
    for counts in itertools.product(*choices):
        counts = np.asarray(counts, np.int64)
        scale = counts/shape
        cost = abs(np.log(np.prod(counts)/target)) + 0.35*np.std(np.log(scale))
        candidates.append((cost, tuple(counts)))
    bins = np.asarray(min(candidates)[1], dtype=np.int64)
    cell = np.minimum(((xyz-low)/span*bins).astype(np.int64), bins-1)
    cell = np.maximum(cell, 0)
    group = ((cell[:, 0]*bins[1]+cell[:, 1])*bins[2]+cell[:, 2])*3 + native % 3
    order = valid[np.lexsort((native[valid], -scores[valid], group[valid]))]
    first = np.r_[True, group[order[1:]] != group[order[:-1]]]
    selected = np.zeros(len(native), dtype=bool)
    selected[order[first]] = True
    return selected, dict(spatial_bins=bins.tolist(), target_per_component=target,
        proposed=int(selected.sum()), eligible=len(valid), graph_spacing=step.tolist(),
        size_fraction=float(selected.sum()/len(valid)), score_rule="highest score per component/bin; native-ID tie break")



def connection_prior(indices, weights, size):
    incoming = np.bincount(indices, weights=weights, minlength=size)
    return np.clip(np.log1p(incoming)-np.log(2), -2, 2).astype(np.float32)


@lru_cache(maxsize=4)
def load_selector(path):
    library = C.CDLL(str(path))
    select = library.gnn_select
    select.argtypes = [C.c_int, C.c_int] + [C.c_void_p]*4 + [C.c_int] + [C.c_void_p]*2
    select.restype = C.c_int
    return select


def graph_decode(scores, native, eligible, indptr, indices, radius, selector=None):
    """One score ordering, followed by directed strong-path exclusion.

    A selected DOF suppresses only rows that can reach it along at most
    ``radius`` actual outgoing strong dependencies. No geometric distances,
    teacher IDs, AMG coarsener, interpolation or extra network calls are used.
    Zero-outgoing-degree F rows follow the backend's special-F convention.
    Native IDs resolve score ties independently of MPI ownership/order.
    The radius is frozen before RL; the number of selected DOFs is an output.
    """
    scores = np.asarray(scores)
    native = np.asarray(native, dtype=np.int64)
    eligible = np.asarray(eligible, dtype=bool)
    n = len(native)
    if scores.shape != (n,) or eligible.shape != (n,) or not np.isfinite(scores).all():
        raise ValueError("Invalid graph decoder arrays")
    if not isinstance(radius, (int, np.integer)) or not 1 <= radius <= 9:
        raise ValueError("Radius must fit the fixed nine-hop multipass limit")
    degree = np.diff(indptr)
    candidates = np.flatnonzero(eligible & (degree > 0))
    # Transpose: for C=j, visit rows i with i -> j, not the reverse direction.
    graph = sparse.csr_matrix((np.ones(len(indices), np.uint8), indices, indptr), shape=(n, n))
    incoming = graph.transpose().tocsr()
    order = candidates[np.lexsort((native[candidates], -scores[candidates]))]
    distance = np.full(n, radius+1, np.int8)
    selected = np.zeros(n, bool)
    if selector is not None:
        order32 = np.ascontiguousarray(order, dtype=np.int32)
        active = np.ascontiguousarray(eligible)
        offsets = np.ascontiguousarray(incoming.indptr, dtype=np.int32)
        rows = np.ascontiguousarray(incoming.indices, dtype=np.int32)
        code = selector(n, len(order), order32.ctypes.data, active.ctypes.data,
                        offsets.ctypes.data, rows.ctypes.data, radius,
                        distance.ctypes.data, selected.ctypes.data)
        if code:
            raise MemoryError("Native selection queue allocation failed")
    else:
        for root in order:
            if distance[root] <= radius:
                continue
            selected[root] = True
            distance[root] = 0
            frontier = np.asarray([root], dtype=np.int64)
            for depth in range(1, radius+1):
                starts = incoming.indptr[frontier]
                lengths = incoming.indptr[frontier+1] - starts
                offsets = np.r_[0, np.cumsum(lengths)]
                if offsets[-1] == 0:
                    break
                positions = np.repeat(starts-offsets[:-1], lengths) + np.arange(offsets[-1])
                parents = incoming.indices[positions]
                frontier = np.unique(parents[eligible[parents] & (distance[parents] > depth)])
                if len(frontier) == 0:
                    break
                distance[frontier] = depth
    if len(candidates) and (not selected.any() or np.any(distance[candidates] > radius)):
        raise RuntimeError("Directed decoder coverage failed")
    return selected, dict(rule="GNN score ordering with directed strong-path exclusion",
        radius=int(radius), proposed=int(selected.sum()), eligible=int(eligible.sum()),
        special_fine=int(np.count_nonzero(eligible & (degree == 0))),
        maximum_path=int(distance[candidates].max()) if len(candidates) else 0,
        size_fraction=float(selected.sum()/max(1, eligible.sum())),
        teacher_read=False, amg_coarsener_called=False)


def local_exploration(frozen, eligible, features, indptr, indices, count, rng):
    """One bounded graph neighbourhood; alternate damage contrast and uniform anchors."""
    n = len(frozen)
    eligible = np.asarray(eligible, bool) & (np.diff(indptr) > 0)
    anchors = np.flatnonzero(frozen & eligible)
    if not len(anchors):
        anchors = np.flatnonzero(eligible)
    if not len(anchors):
        raise ValueError("No eligible exploration anchor")
    graph = sparse.csr_matrix((np.ones(len(indices)), indices, indptr), shape=(n, n))
    focused = bool(rng.integers(2))
    if focused:
        phi = features[:, 0]
        contrast = np.abs(phi - (graph @ phi)/np.maximum(1, np.diff(indptr)))
        threshold = np.quantile(contrast[anchors], .75)
        anchors = anchors[contrast[anchors] >= threshold]
    anchor = int(rng.choice(anchors))
    adjacency = (graph + graph.T).tocsr()
    region = {anchor}
    frontier = np.asarray([anchor], np.int64)
    for _ in range(3):
        neighbours = np.unique(np.concatenate([adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i+1]] for i in frontier]))
        frontier = np.asarray([i for i in neighbours if eligible[i] and i not in region], np.int64)
        region.update(map(int, frontier))
        if len(region) >= count or not len(frontier):
            break
    others = np.asarray(sorted(region - {anchor}), np.int64)
    if len(others) > count-1:
        others = rng.choice(others, count-1, replace=False)
    mask = np.zeros(n, bool)
    mask[np.r_[anchor, others]] = True
    return mask, dict(anchor=anchor, focused=focused, active=int(mask.sum()))


class PolicyLearner:
    """Streaming exact on-policy gradients; optimizer steps only after the whole batch."""
    def __init__(self, config, model, device, saved):
        import torch
        from .objective import RewardBaseline
        self.config, self.model, self.device, self.saved = config, model, device, saved
        self.settings = config["training"]
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.settings["learning_rate"])
        self.baseline = RewardBaseline()
        self.rng = np.random.default_rng(self.settings["seed"])
        torch.manual_seed(self.settings["seed"])
        self.update = 0
        self.keys, self.pending, self.observations, self.action_arrays = [], {}, [], {}
        self.best = None
        self.common_scale = None
        self.reference_times = {}
        self.validation_round = 0
        if saved.get("kind") == "reinforce":
            self.optimizer.load_state_dict(saved["optimizer"])
            self.baseline.values = saved["reward_baseline"]
            self.rng.bit_generator.state = saved["rng_generator"]
            torch.set_rng_state(saved["rng_torch"])
            if device.type == "cuda" and saved.get("rng_cuda"):
                torch.cuda.set_rng_state_all(saved["rng_cuda"])
            self.update = saved["updates"]
            self.best = saved.get("validation_best")
            self.common_scale = saved.get("common_scale")
            self.reference_times = saved.get("reference_times", {})
            self.validation_round = saved.get("validation_round", 0)
        self.initial_update = self.update
        self.worker = int(os.environ.get("GNN_POLICY_WORKER", "0"))
        self.worker_suffix = "-worker%d" % self.worker if config.get("parallel_workers", 1) > 1 else ""

    def identity(self):
        import hashlib
        h = hashlib.sha256()
        for name, value in self.model.state_dict().items():
            h.update(name.encode())
            h.update(value.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def begin(self, request):
        if self.keys:
            raise RuntimeError("An on-policy batch is already active")
        keys = request["keys"]
        if not 2 <= len(keys) <= 4 or len(set(keys)) != len(keys):
            raise ValueError("A batch requires two to four distinct training states")
        scale = float(request["common_scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid shared reward scale")
        if self.common_scale is not None and scale != self.common_scale:
            raise ValueError("The reward scale changed during training")
        self.common_scale = scale
        if self.config.get("parallel_workers", 1) > 1:
            import torch
            seed = self.settings["seed"] + 1009*self.update + self.worker
            self.rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
        self.reference_times = request["reference_times"]
        second, first = self.settings["phase_updates"]
        self.phase = "second" if self.update < second else ("first" if self.update < second+first else "joint")
        self.active_stages = (1,) if self.phase == "second" else ((0,) if self.phase == "first" else (0, 1))
        # Shared features are fixed during the two head-only warm-up phases.
        # Updating the shared trunk while claiming to freeze stage0 would be wrong.
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(self.phase == "joint" or name.startswith("heads.%d." % self.active_stages[0]))
        self.before = {name: p.detach().cpu().clone() for name, p in self.model.named_parameters()}
        self.keys = list(keys)
        self.pending, self.observations, self.action_arrays = {}, [], {}
        self.optimizer.zero_grad(set_to_none=True)
        return dict(update=self.update+1, policy_version=self.update, parameter_sha256=self.identity(),
            phase=self.phase, keys=keys, common_scale=scale)

    def active(self, request):
        return bool(request.get("training") and request.get("key") in self.keys and request["stage"] in self.active_stages)

    def sample(self, request, logits, frozen, arrays):
        import torch
        from .policy import sample_local
        key, stage = request["key"], request["stage"]
        mask, detail = local_exploration(frozen, arrays["eligible"], arrays["features"],
            arrays["indptr"], arrays["indices"], self.settings["exploration_dofs"][stage], self.rng)
        reference = torch.as_tensor(frozen, device=self.device)
        eligible = torch.as_tensor(arrays["eligible"], device=self.device)
        # The frozen reference and its centering are fixed within this batch.
        # A configurable local prior favours retaining its choices; network
        # score differences still alter the actual Bernoulli probabilities.
        center = logits[eligible].median()
        bias = float(np.log(self.settings["reference_probability"]/(1-self.settings["reference_probability"])))
        local_logits = (logits-center)/self.settings["sampling_temperature"] + bias*(2*reference.float()-1)
        sample = sample_local(local_logits, reference, torch.as_tensor(mask, device=self.device), eligible)
        self.pending.setdefault(key, []).append(sample)
        raw = sample.raw.cpu().numpy().copy()
        probabilities = torch.sigmoid(local_logits.detach()).cpu().numpy()
        tag = key+"-stage%d" % stage
        active = np.flatnonzero(mask)
        for name, value in dict(native=arrays["native"][active], probability=probabilities[active],
                                draw=raw[active], frozen=frozen[active]).items():
            self.action_arrays[tag+"-"+name] = np.asarray(value).copy()
        detail.update(stage=stage, log_probability=float(sample.log_probability.detach()),
            entropy=float(sample.entropy.detach()), changes=int(np.count_nonzero(raw != frozen)),
            deletions=int(np.count_nonzero(frozen & ~raw)), additions=int(np.count_nonzero(~frozen & raw)),
            raw_count=int(raw.sum()), reference_count=int(frozen.sum()),
            likelihood="Original Bernoulli draws on active DOFs only; no repaired labels")
        return raw, detail

    def feedback(self, request):
        from .objective import reinforce_loss, time_reward
        key = request["key"]
        if key not in self.keys or any(r["key"] == key for r in self.observations):
            raise ValueError("Unexpected or repeated feedback")
        reward = time_reward(self.reference_times[key], request["seconds"], self.common_scale,
            self.settings["episode_timeout_seconds"], request["converged"])
        samples = self.pending.pop(key, [])
        past = self.baseline.get(key)
        loss_value = None
        if samples:
            loss = reinforce_loss([dict(key=key, reward=reward, samples=samples, importance=request.get("importance", 1.0))], self.baseline)/len(self.keys)
            loss_value = float(loss.detach())
            loss.backward()
        observation = dict(key=key, reward=reward, past_baseline=past, samples=len(samples),
            seconds=request["seconds"], converged=request["converged"], failure=request.get("failure"),
            loss=loss_value, sampling_probability=request.get("sampling_probability", 1/request["training_pool_size"]),
            importance=request.get("importance", 1.0))
        self.observations.append(observation)
        return observation

    def partial_gradient(self):
        if self.pending or len(self.observations) != 1:
            raise RuntimeError("Each parallel worker must complete exactly one state")
        flat, names = [], []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                names.append((name, list(parameter.shape)))
                value = parameter.grad
                flat.extend((value.detach().cpu().numpy().ravel() if value is not None else np.zeros(parameter.numel())).tolist())
        return dict(names=names, gradient=flat, observations=self.observations,
            parameter_sha256=self.identity(), update=self.update, worker=self.worker)

    def merge_gradient(self, request):
        import torch
        if request["parameter_sha256"] != self.identity() or request["update"] != self.update:
            raise ValueError("Cannot merge gradients from stale parameters")
        values = np.asarray(request["gradient"], np.float32)
        offset = 0
        names = []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                names.append([name, list(parameter.shape)])
                count = parameter.numel()
                parameter.grad = torch.as_tensor(values[offset:offset+count].reshape(tuple(parameter.shape)), device=self.device).clone()
                offset += count
        if offset != len(values) or names != request["names"]:
            raise ValueError("Incompatible parallel gradient layout")
        observations = request["observations"]
        if sorted(r["key"] for r in observations) != sorted(self.keys):
            raise ValueError("Parallel batch has missing or duplicate states")
        self.observations = observations
        return dict(merged=True, worker=self.worker)

    def commit(self):
        import torch
        if len(self.observations) != len(self.keys) or self.pending:
            raise RuntimeError("Cannot update an incomplete on-policy batch")
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        gradient = torch.nn.utils.clip_grad_norm_(parameters, self.settings["gradient_clip"], error_if_nonfinite=True)
        self.optimizer.step()
        changes = {name: float(torch.linalg.vector_norm(p.detach().cpu()-self.before[name])) for name, p in self.model.named_parameters()}
        assert all(np.isfinite(v) for v in changes.values())
        self.baseline.update([(r["key"], r["reward"]) for r in self.observations])
        self.update += 1
        run = os.environ.get("GNN_RUN_ID", os.environ.get("SLURM_JOB_ID", "local"))
        root = Path(self.config["output_directory"])
        path = root / "data" / ("rl-%s-update%06d%s-actions.npz" % (run, self.update, self.worker_suffix))
        with path.open("xb") as f:
            np.savez_compressed(f, **self.action_arrays)
        result = dict(update=self.update, worker=self.worker, phase=self.phase, parameter_sha256=self.identity(), gradient_norm=float(gradient),
            parameter_delta_norm=float(np.sqrt(sum(v*v for v in changes.values()))),
            parameter_changes=changes, observations=self.observations, common_scale=self.common_scale,
            actions=dict(path=str(path), sha256=digest(path)), frozen_parameters_until_batch_complete=True)
        self.keys, self.action_arrays = [], {}
        if self.update == 1 or self.update % self.settings["checkpoint_interval"] == 0:
            result["checkpoint"] = self.checkpoint("update")
        record(self.config, "rl_update_complete", **result)
        (root / "records" / ("rl-"+run+self.worker_suffix+".json")).write_text(json.dumps(dict(state="running", latest=result,
            best=self.best, updates_this_allocation=self.update-self.initial_update), indent=2)+"\n")
        return result

    def checkpoint(self, tag):
        import torch
        run = os.environ.get("GNN_RUN_ID", os.environ.get("SLURM_JOB_ID", "local"))
        path = Path(self.config["output_directory"]) / "checkpoints" / ("rl-%s-update%06d%s-%s.pt" % (run, self.update, self.worker_suffix, tag))
        if path.exists():
            raise FileExistsError(path)
        if tag.startswith("validation") and self.best is not None:
            self.best["checkpoint_path"] = str(path)
        state = {name: self.saved[name] for name in ("feature_names", "feature_mean", "feature_scale", "coarse_fraction", "dataset_job")}
        state.update(kind="reinforce", model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
            updates=self.update, reward_baseline=self.baseline.values, rng_generator=self.rng.bit_generator.state,
            rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            common_scale=self.common_scale, reference_times=self.reference_times, config=self.config,
            validation_best=self.best, validation_round=self.validation_round,
            phase_updates=self.settings["phase_updates"])
        with path.open("xb") as f:
            torch.save(state, f)
        return dict(path=str(path), sha256=digest(path), update=self.update)

    def validation(self, request):
        if self.keys:
            raise RuntimeError("Validation cannot interrupt a frozen training batch")
        cases = request["cases"]
        self.validation_round += 1
        if self.best is not None and {r["key"] for r in cases} != {r["key"] for r in self.best["cases"]}:
            self.best = None  # training-smoke and held validation are different pools
        failures = sum(not r["accurate"] for r in cases)
        cost = sum(r["seconds"] if r["accurate"] else self.settings["episode_timeout_seconds"] for r in cases)
        result = dict(update=self.update, failures=failures, cumulative_seconds=cost, cases=cases,
            cumulative_scope="Fixed saved "+("training-smoke" if all(r["split"] == "training" for r in cases) else "validation")+" pool, not a full Newton trajectory")
        score = (failures, cost)
        if self.best is None or score < (self.best["failures"], self.best["cumulative_seconds"]):
            self.best = result.copy()
            result["checkpoint"] = self.checkpoint("validation%04d" % self.validation_round)
            self.best["checkpoint"] = result["checkpoint"]
        record(self.config, "rl_validation_complete", **result)
        return result


def serve(config, endpoint):
    import torch
    from .policy import TwoStagePolicy
    from .dataset import FEATURE_NAMES
    device = torch.device(config.get("inference_device", "cuda"))
    use_cuda = device.type == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA policy device is unavailable")
    colocated = os.environ.get("GNN_COLOCATED_POLICY") == "1"
    if not colocated and not use_cuda and config.get("launch_mode") in ("single_node", "parallel_cpu"):
        # MPI rank0 waits on the socket during inference, so share its first
        # allocated core rather than requesting a second resource group.
        available = sorted(os.sched_getaffinity(0))
        worker = int(os.environ.get("GNN_POLICY_WORKER", "0"))
        os.sched_setaffinity(0, {available[worker % len(available)]})
    torch.set_num_threads(config["gpu_cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    assert digest(config["policy_checkpoint"]) == config["policy_checkpoint_sha256"]
    saved = torch.load(config["policy_checkpoint"], map_location="cpu")
    assert saved["feature_names"] == FEATURE_NAMES
    model = TwoStagePolicy(len(FEATURE_NAMES), config["training"]["width"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    mean = np.asarray(saved["feature_mean"], dtype=np.float32)
    scale = np.asarray(saved["feature_scale"], dtype=np.float32)
    fractions = saved["coarse_fraction"]
    selector = load_selector(config["selector"]) if "selector" in config else None
    audit_models = {}
    for name, checkpoint in config.get("comparison_checkpoints", {}).items():
        assert digest(checkpoint["path"]) == checkpoint["sha256"]
        state = torch.load(checkpoint["path"], map_location="cpu")
        assert state["feature_names"] == FEATURE_NAMES
        candidate = TwoStagePolicy(len(FEATURE_NAMES), config["training"]["width"]).to(device)
        candidate.load_state_dict(state["model"])
        candidate.eval()
        audit_models[name] = (candidate, np.asarray(state["feature_mean"], np.float32),
                             np.asarray(state["feature_scale"], np.float32), checkpoint["sha256"])
    learner = PolicyLearner(config, model, device, saved) if config["stage"] == "rl" else None
    listener = socket.socket()
    listener.bind(("127.0.0.1" if colocated else "0.0.0.0", 0))
    listener.listen(1)
    listener.settimeout(600)
    token = secrets.token_hex(32)
    node = socket.gethostname()
    cpus = sorted(os.sched_getaffinity(0))
    private = dict(host="127.0.0.1" if colocated else node, port=listener.getsockname()[1],
                   token=token, node=node, cpus=cpus, pid=os.getpid())
    pending = Path(str(endpoint)+".pending")
    with os.fdopen(os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
        json.dump(private, f)
        f.flush()
        os.fsync(f.fileno())
    if Path(endpoint).exists():
        raise FileExistsError(endpoint)
    os.rename(pending, endpoint)
    record(config, "policy_server_ready", endpoint=str(endpoint),
           device=torch.cuda.get_device_name() if use_cuda else "cpu",
           checkpoint_sha256=config["policy_checkpoint_sha256"], fractional_sizes=fractions.tolist(),
           node=node, cpus=cpus, colocated=colocated)
    stream, _ = listener.accept()
    stream.settimeout(600)
    hello, _ = receive_packet(stream)
    if hello.get("operation") != "hello" or not secrets.compare_digest(hello.get("token", ""), token):
        raise PermissionError("Invalid policy client")
    send_packet(stream, dict(authenticated=True))
    calls = 0
    try:
        while True:
            request, arrays = receive_packet(stream)
            if request["operation"] == "stop":
                final = learner.checkpoint("final") if learner is not None else None
                send_packet(stream, dict(stopped=True, calls=calls, checkpoint=final))
                break
            if request["operation"] == "ping":
                send_packet(stream, dict(alive=True))
                continue
            if learner is not None and request["operation"] in ("begin_batch", "feedback", "commit", "validation", "partial_gradient", "merge_gradient"):
                operation = request["operation"]
                reply = (learner.begin(request) if operation == "begin_batch" else
                         learner.feedback(request) if operation == "feedback" else
                         learner.partial_gradient() if operation == "partial_gradient" else
                         learner.merge_gradient(request) if operation == "merge_gradient" else
                         learner.commit() if operation == "commit" else learner.validation(request))
                send_packet(stream, reply)
                continue
            if request["operation"] != "select":
                raise ValueError("Unsupported policy operation")
            begin = time.perf_counter()
            policy = request.get("policy", "default")
            network, shift, spread, checkpoint_hash = model, mean, scale, config["policy_checkpoint_sha256"]
            if policy not in ("default", "prior_only"):
                network, shift, spread, checkpoint_hash = audit_models[policy]
            stage = request["stage"]
            if stage not in (0, 1):
                raise ValueError("Invalid stage")
            features = arrays["features"]
            n = len(features)
            prior = connection_prior(arrays["indices"], arrays["weights"], n)
            prior_seconds = time.perf_counter() - begin
            network_seconds = normalization_seconds = 0.0
            if request.get("neutral", False):
                scores = np.zeros(n, np.float32)
            elif policy == "prior_only":
                scores = prior
                checkpoint_hash = None
            else:
                normalization_start = time.perf_counter()
                x = np.empty_like(features, dtype=np.float32)
                np.subtract(features, shift[stage], out=x)
                np.divide(x, spread[stage], out=x)
                np.clip(x, -12, 12, out=x)
                normalization_seconds = time.perf_counter() - normalization_start
                if use_cuda:
                    torch.cuda.synchronize()
                network_start = time.perf_counter()
                graph = torch.sparse_csr_tensor(torch.as_tensor(arrays["indptr"], dtype=torch.int64),
                    torch.as_tensor(arrays["indices"], dtype=torch.int64), torch.as_tensor(arrays["weights"]),
                    size=(n, n), device=device)
                with torch.set_grad_enabled(learner is not None and learner.active(request)):
                    logits = network(torch.as_tensor(x, device=device), graph,
                        torch.as_tensor(prior, device=device), stage)
                scores = logits.detach().cpu().numpy()
                if use_cuda:
                    torch.cuda.synchronize()
                network_seconds = time.perf_counter()-network_start
                del graph
            decode_start = time.perf_counter()
            if config.get("decoder", {}).get("kind") == "directed_graph":
                args = (scores, arrays["native"], arrays["eligible"], arrays["indptr"],
                        arrays["indices"], config["decoder"]["radii"][stage])
                selected, decoded = graph_decode(*args, selector=selector)
                if request.get("check_decoder") and selector is not None:
                    compiled_seconds = time.perf_counter() - decode_start
                    reference_start = time.perf_counter()
                    reference, _ = graph_decode(*args)
                    if not np.array_equal(reference, selected):
                        raise RuntimeError("Compiled decoder changed the selected points")
                    decoded["parity"] = dict(equal=True, compiled_s=compiled_seconds,
                                             reference_s=time.perf_counter()-reference_start)
            else:
                selected, decoded = spatial_decode(scores, features, arrays["native"], arrays["eligible"],
                    arrays["indptr"], arrays["indices"], float(fractions[stage]))
            decode_seconds = time.perf_counter() - decode_start
            sampling_start = time.perf_counter()
            sampling = None
            if learner is not None and learner.active(request):
                selected, sampling = learner.sample(request, logits, selected, arrays)
            calls += 1
            reply = dict(stage=stage, sampling=sampling, decoder=decoded,
                parameter_sha256=learner.identity() if learner is not None else None,
                policy=policy, checkpoint_sha256=checkpoint_hash, prior_s=prior_seconds,
                normalization_s=normalization_seconds, decode_s=decode_seconds,
                sampling_s=time.perf_counter()-sampling_start,
                network_with_copies_s=network_seconds, device=str(device),
                service_compute_s=time.perf_counter()-begin, neutral=request.get("neutral", False))
            send_packet(stream, reply, dict(selected=selected))
    except BaseException as error:
        import traceback
        traceback.print_exc()
        try:
            send_packet(stream, dict(error=str(error)))
        finally:
            raise
    finally:
        stream.close()
        listener.close()
    record(config, "policy_server_stopped", calls=calls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("endpoint", type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    if config.get("parallel_workers", 1) > 1:
        config["progress_name"] = "progress-%s-policy%s.jsonl" % (os.environ.get("SLURM_JOB_ID", "local"), os.environ["GNN_POLICY_WORKER"])
    serve(config, args.endpoint)
