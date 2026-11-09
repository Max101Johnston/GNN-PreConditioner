#!/usr/bin/env python3
"""Run an immutable sequence of Slurm jobs for learned interpolation.

The state file records each actual job ID. Re-running the same plan reuses
submitted jobs and verifies their terminal receipts before continuing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone


def canonical(value):
    return json.dumps(value, indent=2, allow_nan=False, sort_keys=True)+"\n"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_state(path, value):
    path = Path(path)
    temp = path.with_name(path.name+".next")
    temp.write_text(canonical(value))
    temp.replace(path)


REMOTE_SUBMISSION = r"""
import fcntl,json,pathlib,subprocess,sys
receipt,config,identity,expected,argv,since=sys.argv[1:]
p=pathlib.Path(receipt)
args=json.loads(argv)
with pathlib.Path(config).open('rb') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 import hashlib
 assert hashlib.sha256(lock.read()).hexdigest()==expected
 if p.exists():
  data=json.loads(p.read_text())
  assert data['identity']==identity and data['argv']==args
  print(data['job'])
 else:
  # The comment binds recovery to the exact uploaded config even if SSH
  # failed after Slurm accepted the command and before the receipt was saved.
  def call(a):
   return subprocess.check_output(a,universal_newlines=True).strip()
  name=next(a.split('=',1)[1] for a in args if a.startswith('--job-name='))
  old=call(['sacct','-X','-n','-P','-S',since,'--name',name,
            '--format','JobIDRaw,JobName%80,Comment%150,WorkDir%240'])
  queue=call(['squeue','-h','--name',name,'-o','%i|%j|%k|%Z'])
  matches=set()
  for line in (old+'\n'+queue).splitlines():
   fields=line.split('|')
   if len(fields)==4 and fields[1].strip()==name and fields[2].strip()==identity:
    matches.add(fields[0].strip())
  assert len(matches)<=1,matches
  job=next(iter(matches)) if matches else call(args).split(';')[0]
  assert job.isdigit(),job
  p.write_text(json.dumps(dict(job=job,identity=identity,argv=args))+'\n')
  print(job)
"""


class Pipeline:
    def __init__(self, plan):
        self.plan = plan
        self.base = read(plan["base_config"])
        self.case = Path(self.base["case"])
        self.root = self.base["remote_case"]
        self.path = Path(plan["state"])
        self.study = plan["name"]
        assert self.study.isalnum() and len(self.study) <= 18
        self.ssh = ["ssh","-F","/dev/null","-i",self.base["key"],
                    "-o","IdentitiesOnly=yes","-o","IdentityAgent=none",
                    "-o","BatchMode=yes","-o","StrictHostKeyChecking=yes",
                    "-o","ConnectTimeout=10","-p",str(self.base.get("port",22)),self.base["host"]]
        identity = digest(canonical(plan))
        if self.path.exists():
            self.state = read(self.path)
            assert self.state["plan_sha256"] == identity
        else:
            self.state = dict(name=self.study, plan_sha256=identity,
                since=datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d"),
                phases={}, status="prepared")
            self.save()

    def remote(self, args, **kw):
        return subprocess.run(self.ssh+[shlex.join(args)], text=True, check=True,
                              timeout=90, **kw).stdout

    def remote_json(self, path):
        return json.loads(self.remote(["cat",path], capture_output=True))

    def save(self):
        write_state(self.path, self.state)

    def update_run(self, phase, job):
        path = Path(self.plan["run_record"])
        current = read(path)
        current["student_development"] = dict(pipeline=str(self.path), name=self.study,
            phase=phase, active_job_id=job, status=self.state["status"])
        if "final" in self.state:
            current["student_development"]["final"] = self.state["final"]
        current.update(status="student_"+self.state["status"], active_job_id=job)
        write_state(path,current)
        self.remote(["python3","-c","import pathlib,sys;pathlib.Path(sys.argv[1]).write_text(sys.stdin.read())",
            self.root+"/records/run.json"], input=canonical(current), capture_output=True)

    def put_config(self, path, config):
        content = canonical(config)
        local = self.case/"records"/path
        if local.exists():
            assert local.read_text()==content, str(local)
        else:
            local.write_text(content)
        target = self.root+"/records/"+path
        code = ("import pathlib,sys;p=pathlib.Path(sys.argv[1]);s=sys.stdin.read()\n"
                "if p.exists(): assert p.read_text()==s,str(p)\n"
                "else:\n with p.open('x') as f:f.write(s)")
        self.remote(["python3","-c",code,target], input=content, capture_output=True)
        return local,target,digest(content)

    def wait(self, config, job):
        subprocess.run([sys.executable,"-B",self.plan["monitor"],"wait-job",
                        "--config",str(config),"--job",job], check=True)

    def available_gpu(self):
        text = self.remote(["scontrol","show","nodes","-o"], capture_output=True)
        for line in text.splitlines():
            fields = dict(token.split("=",1) for token in line.split() if "=" in token)
            if "gpu" not in fields.get("Partitions","").split(","):
                continue
            if any(x in fields.get("State","") for x in ("DOWN","DRAIN","FAIL","MAINT")):
                continue
            total = dict(t.split("=",1) for t in fields.get("CfgTRES","").split(",") if "=" in t)
            used = dict(t.split("=",1) for t in fields.get("AllocTRES","").split(",") if "=" in t)
            if (int(total.get("gres/gpu","0"))>int(used.get("gres/gpu","0"))
                and int(fields.get("CPUTot","0"))-int(fields.get("CPUAlloc","0"))>=4):
                return fields["NodeName"]
        return None

    def stage(self, phase, updates, training=False):
        entry = self.state["phases"].setdefault(phase,{})
        if training and "device" not in entry:
            node = self.available_gpu() if self.plan.get("prefer_gpu",True) else None
            entry.update(device="cuda" if node else "cpu", gpu_node=node)
        device = entry.get("device")
        config = dict(self.base, **updates)
        if training:
            config.update(stage="student_train", student_train_device=device,
                          student_train_threads=4 if device=="cuda" else 16)
        name = "gnn-"+self.study+"-"+phase
        local,remote,sha = self.put_config("config-"+name+".json",config)
        identity = name+":"+sha
        args = ["sbatch","--parsable","--job-name="+name,"--comment="+identity,
                "--open-mode=append","--chdir="+self.root,
                "--output="+self.root+"/logs/%j.web.log","--error="+self.root+"/logs/%j.web.log"]
        if training and device=="cuda":
            args += ["--partition=gpu","--gres=gpu:1","--nodes=1","--ntasks=1",
                     "--cpus-per-task=4","--mem=48G","--time="+self.plan["gpu_walltime"]]
        elif training:
            args += ["--partition=compute","--nodes=1","--ntasks=1","--cpus-per-task=16",
                     "--mem=64G","--time="+self.plan["cpu_train_walltime"]]
        else:
            walltime = "02:30:00" if phase=="tune" else self.plan["solver_walltime"]
            args += ["--partition=compute","--nodes=3","--ntasks=192","--ntasks-per-node=64",
                     "--cpus-per-task=1","--mem=128G","--time="+walltime]
        args += [self.base["source_directory"]+"/scripts/run-gate.sh",remote]
        entry.update(config=str(local), config_sha256=sha, argv=args)
        self.save()
        job = self.remote(["python3","-c",REMOTE_SUBMISSION,
            self.root+"/records/"+name+"-submission.json",remote,identity,sha,
            json.dumps(args),self.state["since"]],capture_output=True).strip()
        assert job.isdigit()
        path = self.root+"/records/"+("student-trained-" if training else "student-cycle-")+job+".json"
        entry.update(job=job, artifact=path, status="submitted")
        self.state["status"] = "running"
        self.save()
        self.update_run(phase,job)
        print("STUDENT_PHASE",phase,"SLURM",job,device or "3x64",flush=True)
        self.wait(local,job)
        report = self.remote_json(path)
        assert report["job"] == job
        if training:
            exports = [c for r in report["runs"] for c in r["checkpoints"]]
            assert exports and all(c["native_parity"]["exact_support"] for c in exports)
            self.verify_models(exports)
            entry.update(exports=exports,device_name=report["device_name"],elapsed_seconds=report["elapsed_seconds"])
        else:
            if report.get("mode") in ("data","refresh"):
                entry["samples"] = {level:len(rows) for level,rows in report["samples"].items()}
                entry["accepted"] = report["accepted"]
            entry["totals"] = report.get("totals",report.get("best",{}).get("totals",{}))
        entry["status"] = "verified"
        self.save()
        return report,path

    def verify_models(self, exports):
        identities = [(x["path"],x["sha256"]) for x in exports]
        identities += [(x["training"]["path"],x["training"]["sha256"]) for x in exports]
        code = ("import hashlib,json,pathlib,sys\n"
                "for p,h in json.loads(sys.argv[1]):\n"
                " assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,p")
        self.remote(["python3","-c",code,json.dumps(identities)],capture_output=True)

    @staticmethod
    def candidates(trained, level, prefix):
        ans = []
        for run in trained["runs"]:
            if run["level"] != level:
                continue
            for c in run["checkpoints"]:
                name = "%s%s-%d-e%02d" % (prefix,c["kind"],c["seed"],c["epoch"])
                ans.append(dict(name=name, students={"p%d"%level:c}, fallback=True))
        return ans

    @staticmethod
    def best(report, choices):
        return min(choices,key=lambda c:(report["totals"][c["name"]]["failed"],
            report["totals"][c["name"]]["fallback_count"],report["totals"][c["name"]]["total_s"]))

    def evaluate(self, phase, keys, split, methods, repeats=1):
        return self.stage(phase,dict(student_cycle="evaluate",student_keys=keys,
            student_evaluation_split=split,student_methods=methods,student_repeats=repeats))[0]

    def run(self):
        reference = dict(name="reference")
        smoke = self.plan["mode"] == "smoke"
        if smoke:
            job = str(self.plan["smoke_data_job"])
            self.wait(self.plan["smoke_data_config"],job)
            data = self.root+"/records/student-cycle-"+job+".json"
            report = self.remote_json(data)
            assert report["accepted"] and all(len(v)==3 for v in report["samples"].values())
        else:
            report,data = self.stage("data",dict(student_cycle="data"))
            assert report["accepted"] and len(report["samples"]["1"])==55
        epochs = 2 if smoke else self.plan["epochs"]
        save = [2] if smoke else [epochs//3,2*epochs//3,epochs]
        seeds = [20261109] if smoke else [20261109,20261117]
        runs = [dict(level=1,kind=kind,seed=seed,width=8,epochs=epochs,checkpoints=save)
                for kind in ("A","B") for seed in seeds]
        runs.append(dict(level=2,kind="B",seed=20261109,width=8,epochs=epochs,
                         checkpoints=[epochs] if smoke else [epochs//2,epochs]))
        trained,_ = self.stage("train",dict(student_datasets=[data],student_runs=runs,
            student_full_training=not smoke), training=True)
        p1 = self.candidates(trained,1,"P1")
        p2 = self.candidates(trained,2,"P2")
        keys = self.base["gate_keys"] if smoke else self.base["training"]["validation_keys"]
        dev = self.evaluate("dev",keys,"training" if smoke else "validation",[reference]+p1+p2)
        best1 = self.best(dev,p1)
        best2 = self.best(dev,p2)
        if smoke:
            self.state["final"] = dict(scope="development probes",metrics=dev["totals"],
                                       models=best1["students"]|best2["students"])
            self.state["status"] = "smoke_verified"
        else:
            self.finish_full(reference,dev,best1,best2)
        self.save()
        self.update_run("complete",None)
        print("STUDENT_PIPELINE_VERIFIED",self.study,flush=True)

    def finish_full(self, reference, dev, p1, p2):
        valid = self.base["training"]["validation_keys"]
        # P1 and P2-alone remain separate controls even if a joint model fails.
        candidates = [p1,p2]
        if not dev["totals"][p1["name"]]["failed"]:
            refreshed,receipt = self.stage("refresh",dict(student_cycle="refresh",refresh_p1=p1["students"]["p1"]))
            if refreshed["accepted"]:
                epochs = self.plan["refresh_epochs"]
                run = dict(level=2,kind="B",seed=20261117,width=8,epochs=epochs,
                    checkpoints=[epochs//2,epochs],resume=p2["students"]["p2"]["training"])
                trained,_ = self.stage("refreshtrain",dict(student_datasets=[receipt],student_runs=[run],
                    student_full_training=False),training=True)
                for c in self.candidates(trained,2,"joint"):
                    candidates.append(dict(c,students={**p1["students"],**c["students"]}))
        joint = self.evaluate("joint",valid,"validation",[reference]+candidates)
        winner = self.best(joint,candidates)
        batches = self.plan["tune_batches"]
        # Includes worst-case retries of all first teachers and fallbacks
        # outside the exploration pairs. These are maxima, not quotas.
        worst_solve_count = 220+232+55+72+6*batches+120+30+190
        assert 100 <= batches <= 150 and worst_solve_count <= self.plan["full_solve_budget"]
        self.state["planned_maximum_solves"] = worst_solve_count
        self.save()
        tuned,_ = self.stage("tune",dict(student_cycle="tune",student_methods=[winner],
            mpi_timeout_seconds=8700, student_tuning=dict(batches=batches,milestones=[0,25,50,75,batches],
                sigma=.12,learning_rate=.10,common_time_scale_s=5.0,seed=61109)))
        final_method = dict(tuned["best"]["method"],name="student")
        inventory = self.remote_json(self.base["baseline_directory"]+"/data/inventory.json")
        confirm = [row["key"] for row in inventory["snapshots"] if row["split"]=="validation"
                   and row["key"] not in valid]
        test = [row["key"] for row in inventory["snapshots"] if row["split"]=="test"]
        assert len(confirm)==6 and len(test)==19
        methods = [reference,final_method,dict(name="amg"),dict(name="retained_base")]
        self.state["frozen"] = final_method
        self.save()
        confirmation = self.evaluate("confirm",confirm,"validation",methods)
        tested = self.evaluate("test",test,"test",methods,repeats=2)
        v,t = confirmation["totals"],tested["totals"]
        repeat_costs = [{label:sum(r["total_s"] for r in tested["cases"]
                          if r["method"]==label and r["repeat"]==repeat)
                         for label in ("student","reference")} for repeat in range(2)]
        passed = (not v["student"]["failed"] and not t["student"]["failed"] and
                  v["student"]["total_s"]<v["reference"]["total_s"] and
                  all(c["student"]<c["reference"] for c in repeat_costs))
        self.state["final"] = dict(model=final_method,dev=joint["totals"],confirm=v,test=t,
             test_repeats=repeat_costs, cached_test_solves=t.get("retained_base",{}).get("frames",0),
             tuning_batches=tuned["batches"],tuning_best_step=tuned["best"]["step"],
             saved_matrix_speed_gate=passed,live_ogs_validation="pending_candidate_gate" if passed
             else "not_promoted_after_saved_matrix_benchmark")
        self.state["status"] = "saved_matrix_cycle_verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    pipeline = Pipeline(read(args.plan))
    try:
        pipeline.run()
    except BaseException:
        pipeline.state["status"] = "failed"
        pipeline.save()
        raise


if __name__ == "__main__":
    main()
