"""Persistent GPU policy service and label-free deterministic coarse decoder.

The wire format is JSON metadata followed by typed numeric buffers, never
pickled objects. A job-private token authenticates the sole MPI-root client.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import secrets
import socket
import struct
import time

import numpy as np

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
    def __init__(self, endpoint):
        cfg = read_json(endpoint)
        self.stream = socket.create_connection((cfg["host"], cfg["port"]), timeout=300)
        self.request(dict(operation="hello", token=cfg["token"]))

    def request(self, metadata, arrays=None):
        send_packet(self.stream, metadata, arrays)
        reply, values = receive_packet(self.stream)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply, values

    def close(self):
        self.request(dict(operation="stop"))
        self.stream.close()


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


def serve(config, endpoint):
    import torch
    from .policy import TwoStagePolicy
    from .dataset import FEATURE_NAMES
    if not torch.cuda.is_available():
        raise RuntimeError("Policy service requires allocated CUDA GPU")
    torch.set_num_threads(config["gpu_cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    assert digest(config["policy_checkpoint"]) == config["policy_checkpoint_sha256"]
    saved = torch.load(config["policy_checkpoint"], map_location="cpu")
    assert saved["feature_names"] == FEATURE_NAMES
    device = torch.device("cuda")
    model = TwoStagePolicy(len(FEATURE_NAMES), config["training"]["width"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    mean, scale = saved["feature_mean"], saved["feature_scale"]
    fractions = saved["coarse_fraction"]
    listener = socket.socket()
    listener.bind(("0.0.0.0", 0))
    listener.listen(1)
    listener.settimeout(600)
    token = secrets.token_hex(32)
    private = dict(host=socket.gethostname(), port=listener.getsockname()[1], token=token)
    pending = Path(str(endpoint)+".pending")
    with os.fdopen(os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
        json.dump(private, f)
        f.flush()
        os.fsync(f.fileno())
    if Path(endpoint).exists():
        raise FileExistsError(endpoint)
    os.rename(pending, endpoint)
    record(config, "gpu_policy_server_ready", endpoint=str(endpoint), device=torch.cuda.get_device_name(),
           checkpoint_sha256=config["policy_checkpoint_sha256"], fractional_sizes=fractions.tolist())
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
                send_packet(stream, dict(stopped=True, calls=calls))
                break
            if request["operation"] != "select":
                raise ValueError("Unsupported policy operation")
            begin = time.perf_counter()
            stage = request["stage"]
            if stage not in (0, 1):
                raise ValueError("Invalid stage")
            features = arrays["features"]
            n = len(features)
            incoming = np.bincount(arrays["indices"], weights=arrays["weights"], minlength=n)
            prior = np.clip(np.log1p(incoming)-np.log(2), -2, 2).astype(np.float32)
            network_seconds = 0.0
            if request.get("neutral", False):
                scores = np.zeros(n, np.float32)
            else:
                x = np.clip((features-mean[stage])/scale[stage], -12, 12).astype(np.float32)
                torch.cuda.synchronize()
                network_start = time.perf_counter()
                graph = torch.sparse_csr_tensor(torch.as_tensor(arrays["indptr"], dtype=torch.int64),
                    torch.as_tensor(arrays["indices"], dtype=torch.int64), torch.as_tensor(arrays["weights"]),
                    size=(n, n), device=device)
                with torch.no_grad():
                    scores = model(torch.as_tensor(x, device=device), graph,
                        torch.as_tensor(prior, device=device), stage).cpu().numpy()
                torch.cuda.synchronize()
                network_seconds = time.perf_counter()-network_start
                del graph
            selected, decoded = spatial_decode(scores, features, arrays["native"], arrays["eligible"],
                arrays["indptr"], arrays["indices"], float(fractions[stage]))
            calls += 1
            reply = dict(stage=stage, decoder=decoded, gpu_network_with_copies_s=network_seconds,
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
    record(config, "gpu_policy_server_stopped", calls=calls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("endpoint", type=Path)
    args = parser.parse_args()
    serve(read_json(args.config), args.endpoint)
