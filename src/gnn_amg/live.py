"""Use the existing OGS SIF to compare frozen displacement PCs on live matrices."""
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import time
import traceback
from xml.etree import ElementTree as ET

import numpy as np

from gnn_amg.backend import bind, digest, load_bridge, load_dependencies, local_csr, ownership, read_json
from gnn_amg.online import prepare_displacement
from gnn_amg.student import load_student


def prepare(config_path, directory):
    """Freeze the original loading inputs and rebuild the RAS overlap at source rank count."""
    config = read_json(config_path)
    settings = config["live"]
    source = Path(settings["source_case"])
    receipt_path = source / "records/loading-inputs.json"
    assert digest(receipt_path) == settings["loading_receipt_sha256"]
    receipt = read_json(receipt_path)
    assert receipt["partitions"] == receipt["mpi_tasks"] == settings["ranks"]
    assert receipt["image_sha256"] == config["image_sha256"]
    assert receipt["source"]["all_192_piece_hashes_verified"]
    for name, identity in receipt["files"].items():
        path = source / name
        assert path.stat().st_size == identity["bytes"] and digest(path) == identity["sha256"], path
    consolidated = source / receipt["source"]["consolidated_vtu"]
    assert digest(consolidated) == receipt["source"]["sha256"]

    project = source / receipt["project"]["project"]
    assert digest(project) == receipt["project"]["sha256"]
    raw = project.read_text()
    assert "<python_script>" not in raw
    changes = [
        ("<OpenGeoSysProject>", "<OpenGeoSysProject>\n    <python_script>live.py</python_script>", 1),
        ("<t_end>300</t_end>", "<t_end>%s</t_end>" % settings["end_s"], 2),
        ("-th2m_ksp_max_it 100", "-th2m_ksp_max_it %d" % config["max_iterations"], 1),
        ("-th2m_fieldsplit_1_pc_type hypre", "-th2m_fieldsplit_1_pc_type none", 1),
        ("<record_matrix_snapshots>true</record_matrix_snapshots>",
         "<record_matrix_snapshots>false</record_matrix_snapshots>", 1),
        ("<record_ksp_history>true</record_ksp_history>",
         "<record_ksp_history>false</record_ksp_history>", 1),
    ]
    for before, after, count in changes:
        assert raw.count(before) == count, before
        raw = raw.replace(before, after)
    prepared_project = directory / project.name
    with prepared_project.open("x") as f:
        f.write(raw)

    baseline = Path(config["baseline_directory"])
    inv_path = baseline / "data/inventory.json"
    assert digest(inv_path) == config["inventory_sha256"]
    rows = [r for r in read_json(inv_path)["snapshots"] if r["case"] == source.name]
    anchor = next(r for r in rows if r["split"] == "training")
    assert anchor["source_ranks"] == settings["ranks"]
    assert digest(Path(anchor["directory"]) / "manifest.json") == anchor["manifest_sha256"]
    _, legacy = load_dependencies(config)
    (directory / "cache").mkdir()
    (directory / "records").mkdir()
    legacy.case_layout(directory, dict(physical_ranks=settings["ranks"],
                         replay_ranks=settings["ranks"], field_workers=8), anchor)

    fixed = None
    for row in rows:
        field_path = baseline / "data" / row["key"] / "fields.npz"
        field_receipt = read_json(baseline / "records" / ("fields-" + row["key"] + ".json"))
        assert field_receipt["manifest"] == row["manifest_sha256"]
        if row is anchor:
            assert digest(field_path) == field_receipt["sha256"]
        with np.load(field_path) as fields:
            mask = fields["constrained"]
        if fixed is None:
            fixed = mask
        else:
            assert np.array_equal(fixed, mask), row["key"]
    with (directory / "constrained.npy").open("xb") as f:
        np.save(f, fixed, allow_pickle=False)
    for level, model in settings["student"]["students"].items():
        load_student(model["path"], model["sha256"], int(level[1:]))
    with (directory / "input-identity.json").open("x") as f:
        json.dump(dict(project=str(prepared_project), project_sha256=digest(prepared_project),
            input_files=len(receipt["files"]), source_loading_job=receipt["job_id"],
            consolidated_sha256=receipt["source"]["sha256"],
            consolidated_time_s=receipt["source"]["final_frame"]["time"],
            mask_frames=len(rows), constrained=int(fixed.sum()),
            layout=read_json(directory / "records" / ("layout-%s.json" % source.name))), f, indent=2)
    print("LIVE_OGS_INPUTS_VERIFIED", prepared_project, flush=True)


class LivePC:
    """Own the live pre/post solve callbacks and all associated PETSc handles."""
    def __init__(self, config, directory, method):
        self.config, self.method = config, method
        settings = config["live"]
        self.cache = directory / "cache" / Path(settings["source_case"]).name
        self.replay, self.legacy = load_dependencies(config)
        self.p = p = self.replay.Petsc()
        assert p.size == settings["ranks"]
        self.replay.pcapply_bind(p)
        self.bridge = load_bridge(p, config["bridge"])
        with np.load(self.cache / "layout.npz") as f:
            self.layout = {name: f[name] for name in ("full_rows", "native_node", "component", "ownership")}
        self.fields = dict(constrained=np.load(directory / "constrained.npy", allow_pickle=False))
        self.lo, self.hi = map(int, self.layout["ownership"][p.rank:p.rank+2])
        self.rows = self.layout["full_rows"][self.lo:self.hi]
        native = (3*self.layout["native_node"] + self.layout["component"])[self.lo:self.hi]
        self.fixed = self.fields["constrained"][native]
        self.context = dict(policy="prior_only", fast_prior=True, p1_refinement_sweeps=2)
        if method == "student":
            frozen = settings["student"]
            for level, model in frozen["students"].items():
                load_student(model["path"], model["sha256"], int(level[1:]))
            self.context.update({key:frozen[key] for key in ("students", "controls_p1", "controls_p2")})
        elif method != "reference":
            raise ValueError("Unknown live method: " + method)
        self.active = None
        self.solve_count = 0
        self.times = []
        self.solve_log = directory / ("solves-%s.jsonl" % method)
        self.register()

    def register(self):
        p = self.p
        I, P, D = p.I, p.P, p.D
        ptr = C.POINTER
        for name, signature in {
            "KSPCreate_GMRES": [P], "KSPRegisterAll": [],
            "KSPRegister": [C.c_char_p, P],
            "KSPGetOptionsPrefix": [P, ptr(C.c_char_p)],
            "KSPGetTolerances": [P, ptr(D), ptr(D), ptr(D), ptr(I)],
        }.items():
            bind(p, name, signature)
        hook = C.CFUNCTYPE(I, P, P, P, P)
        self.pre_hook = hook(self.checked(self.before_solve))
        self.post_hook = hook(self.checked(self.after_solve))
        self.convergence = C.CFUNCTYPE(I, P, I, D, ptr(I), P)(self.checked(self.converge))
        self.constructor = C.CFUNCTYPE(I, P)(self.checked(self.create_gmres))
        p.call("KSPRegisterAll")
        p.call("KSPRegister", b"gmres", C.cast(self.constructor, P))
        if p.rank == 0:
            print("LIVE_OGS_GMRES_HOOK", self.method, p.size, flush=True)

    def checked(self, fn):
        def invoke(*args):
            try:
                fn(*args)
                return 0
            except BaseException:
                traceback.print_exc()
                self.p.mpi.MPI_Abort(self.p.world, 87)
                return 87
        return invoke

    def create_gmres(self, ksp):
        p = self.p
        p.call("KSPCreate_GMRES", ksp)
        prefix = C.c_char_p()
        p.call("KSPGetOptionsPrefix", ksp, C.byref(prefix))
        if prefix.value == b"th2m_":
            p.call("KSPSetPreSolve", ksp, C.cast(self.pre_hook, p.P), None)
            p.call("KSPSetPostSolve", ksp, C.cast(self.post_hook, p.P), None)

    def before_solve(self, ksp, rhs, solution, _):
        assert self.active is None
        p = self.p
        start = time.perf_counter()
        rtol, atol, cap = p.D(), p.D(), p.I()
        p.call("KSPGetTolerances", ksp, C.byref(rtol), C.byref(atol), None, C.byref(cap))
        assert (rtol.value, atol.value, cap.value) == (self.config["rtol"],
            self.config["atol"], self.config["max_iterations"])
        fs, full, pre = p.P(), p.P(), p.P()
        p.call("KSPGetPC", ksp, C.byref(fs))
        p.call("KSPGetOperators", ksp, C.byref(full), C.byref(pre))
        p.call("PCSetUp", fs)
        child, original, operator, k = self.replay.pcapply_child(p, fs, self.rows, expected_pc=b"none")
        assert np.array_equal(ownership(p, k), self.layout["ownership"])
        csr, _ = local_csr(p, k, self.bridge)
        csr.eliminate_zeros()
        diagonal_only = (np.diff(csr.indptr) == 1) & (csr.diagonal(k=self.lo) == 1.)
        if p.maximum(float(np.any(self.fixed != diagonal_only))):
            raise ValueError("Live unit diagonal constraint rows differ from the verified source mask")
        displacement = prepare_displacement(self.config, p, self.legacy, self.bridge, None,
            k, self.layout, self.fields, self.cache, context=self.context)
        if displacement.failure is not None:
            raise ValueError("Live hierarchy failed: " + displacement.failure)
        p.call("PetscObjectReference", original)
        self.times.clear()
        shell, callback = self.replay.pcapply_shell(p, child, operator, k, displacement.apply, self.times)
        build, work = p.vector(full), p.vector(full)
        true_limit = max(atol.value, rtol.value*p.norm(rhs))
        self.active = dict(full=full, rhs=rhs, build=build, work=work, limit=true_limit,
            displacement=displacement, child=child, original=original, shell=shell, callback=callback,
            start=start, cap=cap.value)
        p.call("KSPSetConvergenceTest", ksp, C.cast(self.convergence, p.P), None, None)
        p.call("KSPSetUp", ksp)
        p.call("KSPSetUpOnBlocks", ksp)
        self.active["setup_s"] = p.maximum(time.perf_counter()-start)

    def residual(self, solution):
        p, state = self.p, self.active
        p.call("MatMult", state["full"], solution, state["work"])
        p.call("VecAYPX", state["work"], -1., state["rhs"])
        return p.norm(state["work"])

    def converge(self, ksp, iteration, _internal, reason, _):
        p, state = self.p, self.active
        solution = p.P()
        p.call("KSPBuildSolution", ksp, state["build"], C.byref(solution))
        value = self.residual(solution)
        reason[0] = 2 if value <= state["limit"] else (
            -9 if not math.isfinite(value) else (-3 if iteration >= state["cap"] else 0))

    def after_solve(self, ksp, _rhs, solution, _):
        p, state = self.p, self.active
        elapsed = p.maximum(time.perf_counter()-state["start"])
        residual = self.residual(solution)
        iterations, reason = p.I(), p.I()
        p.call("KSPGetIterationNumber", ksp, C.byref(iterations))
        p.call("KSPGetConvergedReason", ksp, C.byref(reason))
        self.solve_count += 1
        n = state["displacement"].numerical
        pc_s = p.maximum(sum(self.times))
        if p.rank == 0:
            payload = json.dumps(dict(method=self.method, solve=self.solve_count,
                iterations=iterations.value, reason=reason.value, residual=residual,
                true_limit=state["limit"], elapsed_s=elapsed, setup_s=state["setup_s"],
                pc_apply_s=pc_s, pc_apply_count=len(self.times), p1_nnz=n["level1"]["nnz"],
                k1_nnz=n.get("k1_nnz"), p2_nnz=n["level2"]["nnz"],
                galerkin_max_error=max(n["galerkin_errors"])), allow_nan=False)
            with self.solve_log.open("a") as log:
                log.write(payload + "\n")
            print("LIVE_OGS_SOLVE", payload, flush=True)
        if not residual <= state["limit"] or reason.value <= 0:
            raise ValueError("Live TH2M GMRES failed the true residual check")
        p.call("KSPSetPC", state["child"], state["original"])
        p.destroy("PC", state["shell"], state["original"])
        state["displacement"].close()
        p.destroy("Vec", state["build"], state["work"])
        self.active = None


def verify(config_path, directory):
    config = read_json(config_path)
    settings = config["live"]
    job = os.environ["SLURM_JOB_ID"]
    methods = {}
    for method in ("reference", "student"):
        solves = [json.loads(line) for line in (directory / ("solves-%s.jsonl" % method)).read_text().splitlines()]
        assert solves and all(row["reason"] > 0 and row["residual"] <= row["true_limit"] and
                              row["iterations"] <= config["max_iterations"] and
                              row["galerkin_max_error"] <= 1e-9 for row in solves)
        result = Path(config["output_directory"]) / "data" / ("live-%s-%s" % (job, method))
        diagnostic, = result.glob("TH2MHYDRATE_*.jsonl")
        with diagnostic.open() as f:
            records = [json.loads(line) for line in f]
        native = [r for r in records if r["kind"] == "linear_solve"]
        assert len(native) == len(solves)
        assert all(r["iterations"] == s["iterations"] and r["reason"] == s["reason"] and
                   math.isclose(r["final_true_residual"], s["residual"], rel_tol=1e-7, abs_tol=1e-14)
                   for r,s in zip(native, solves))
        decisions = [r for r in records if r["kind"] == "timestep_decision"]
        accepted = [r["t"] for r in decisions if r["accepted"]]
        end = float(settings["end_s"])
        assert accepted and math.isclose(accepted[-1], end, abs_tol=1e-9)
        pvd, = result.glob("*.pvd")
        final = max(ET.parse(pvd).iter("DataSet"), key=lambda e: float(e.attrib["timestep"]))
        assert math.isclose(float(final.attrib["timestep"]), end, abs_tol=1e-9)
        pvtu = pvd.parent / final.attrib["file"]
        pieces = [pvtu.parent / e.attrib["Source"] for e in ET.parse(pvtu).iter("Piece")]
        assert len(set(pieces)) == settings["ranks"] and all(p.stat().st_size > 0 for p in pieces)
        methods[method] = dict(accepted=accepted, rejected=sum(not r["accepted"] for r in decisions),
            solves=len(solves), iterations=sum(row["iterations"] for row in solves),
            linear_seconds=sum(r["elapsed_seconds"] for r in native),
            setup_seconds=sum(s["setup_s"] for s in solves),
            elapsed_seconds=float((directory / ("elapsed-%s.txt" % method)).read_text()),
            max_residual_over_limit=max(s["residual"]/s["true_limit"] for s in solves),
            p1_nnz=sum(s["p1_nnz"] for s in solves), k1_nnz=sum(s["k1_nnz"] for s in solves),
            p2_nnz=sum(s["p2_nnz"] for s in solves), output=str(result), final_pvtu=str(pvtu))
    path = Path(config["output_directory"]) / "records" / ("student-live-%s.json" % job)
    with path.open("x") as f:
        json.dump(dict(job=job, end_s=settings["end_s"], ranks=settings["ranks"],
            same_input=read_json(directory / "input-identity.json"), methods=methods), f, indent=2)
    print("LIVE_OGS_PAIR_VERIFIED", job, path, flush=True)


def install():
    config = read_json(os.environ["GNN_LIVE_CONFIG"])
    directory = Path(os.environ["GNN_LIVE_DIRECTORY"])
    return LivePC(config, directory, os.environ["GNN_LIVE_METHOD"])


if "GNN_LIVE_CONFIG" in os.environ:
    _LIVE_PC = install()
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("config", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    action = prepare if args.action == "prepare" else verify
    action(args.config, args.directory)
