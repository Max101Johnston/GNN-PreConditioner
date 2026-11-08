#!/bin/bash
set -u -o pipefail
config=$(realpath "${1:?analysis configuration}")
cfg() { python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$config" "$1"; }
task=$(cfg output_directory)
source=$(cfg source_directory)
full="$task/logs/$SLURM_JOB_ID.full.log"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source/src"
(
set -e
printf '[run] Start %s job %s stage %s node %s\n' "$(date --iso-8601=seconds)" "$SLURM_JOB_ID" "$(cfg stage)" "$(hostname)"
python3 - "$config" <<'PY'
import hashlib,json,pathlib,sys
cfg=json.loads(pathlib.Path(sys.argv[1]).read_text())
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
for directory,hashes in [(cfg['source_directory'],cfg['source_hashes']),
                         (cfg['dependency_directory'],cfg['dependency_hashes'])]:
    for name,expected in hashes.items():
        assert sha(pathlib.Path(directory)/name)==expected,name
assert sha(cfg['bridge'])==cfg['bridge_sha256']
assert sha(cfg['image'])==cfg['image_sha256']
print('FROZEN_GNN_INPUTS_VERIFIED',sha(sys.argv[1]),cfg['image'],flush=True)
PY
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR=/dev/shm OMPI_MCA_orte_tmpdir_base=/dev/shm
export SINGULARITYENV_TMPDIR=/dev/shm
export SINGULARITYENV_OMP_NUM_THREADS=1 SINGULARITYENV_MKL_NUM_THREADS=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH:$(cfg container_pythonpath)"
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
unset PETSC_OPTIONS SINGULARITYENV_PETSC_OPTIONS
"$(cfg torch_python)" -B -m unittest discover -s "$source/tests"
if [[ "$(cfg stage)" == "pretrain" ]]; then
    export LD_LIBRARY_PATH="$(cfg cuda_toolkit)/lib64:${LD_LIBRARY_PATH:-}"
    "$(cfg torch_python)" -B -m gnn_amg.pretrain "$config"
elif [[ "$(cfg stage)" == "prediction" || "$(cfg stage)" == "rl" || "$(cfg stage)" == "parallel_smoke" ]]; then
    export GNN_RUN_ID="${SLURM_JOB_ID_HET_GROUP_0:-$SLURM_JOB_ID}"
    export SINGULARITYENV_GNN_RUN_ID="$GNN_RUN_ID"
    endpoint="$task/records/rpc-$GNN_RUN_ID.json"
    workers=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("parallel_workers",1))' "$config")
    policy_pids=()
    stop_gpu() {
        for pid in "${policy_pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
        for pid in "${policy_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
        policy_pids=()
    }
    trap stop_gpu EXIT
    if [[ "$(cfg stage)" != "parallel_smoke" ]]; then
        if [[ "$(cfg launch_mode)" == "parallel_cpu" ]]; then
            for ((worker=0; worker<workers; worker++)); do
                GNN_POLICY_WORKER="$worker" "$(cfg torch_python)" -B -m gnn_amg.inference "$config" "$endpoint-worker$worker" &
                policy_pids+=("$!")
            done
        elif [[ "$(cfg launch_mode)" == "single_node" ]]; then
            "$(cfg torch_python)" -B -m gnn_amg.inference "$config" "$endpoint" &
            policy_pids+=("$!")
        else
            srun --het-group=1 --exact --nodes=1 --ntasks=1 --cpus-per-task="$(cfg gpu_cpu_threads)" \
                env LD_LIBRARY_PATH="$(cfg cuda_toolkit)/lib64:${LD_LIBRARY_PATH:-}" \
                "$(cfg torch_python)" -B -m gnn_amg.inference "$config" "$endpoint" &
            policy_pids+=("$!")
        fi
        python3 - "$endpoint" "$workers" <<'READY'
from pathlib import Path
import sys,time
base=sys.argv[1];workers=int(sys.argv[2]);deadline=time.monotonic()+300
paths=[Path(base+('-worker%d'%i if workers>1 else '')) for i in range(workers)]
while not all(path.exists() for path in paths):
    if time.monotonic()>deadline:raise TimeoutError('Policy endpoints not ready')
    time.sleep(.5)
READY
    fi
    net=$(cfg mpi_tcp_interface)
    if [[ "$(cfg launch_mode)" == "parallel_cpu" ]]; then
        mpi=(mpirun --mca plm slurm --mca ras slurm --map-by slot --tag-output --bind-to core)
    else
        cpu_host=$(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_0:-$SLURM_JOB_NODELIST}")
        [[ "$cpu_host" != *$'\n'* ]]
        mpi=(mpirun --mca plm isolated --mca ras ^slurm --tag-output --bind-to core --host "$cpu_host:$(cfg mpi_ranks)")
    fi
    export SINGULARITYENV_OMPI_MCA_btl_tcp_if_include="$net"
    export SINGULARITYENV_OMPI_MCA_oob_tcp_if_include="$net"
    mpi_status=0
    timeout --signal=TERM --kill-after=30 "$(cfg mpi_timeout_seconds)" \
        "${mpi[@]}" --mca btl self,vader,tcp --mca btl_tcp_if_include "$net" --mca oob_tcp_if_include "$net" \
        -np "$((workers*$(cfg mpi_ranks)))" singularity exec -B "$(cfg remote_root)" "$(cfg image)" \
        /opt/ogs/bin/python3 -B -m gnn_amg.online "$config" "$endpoint" < /dev/null || mpi_status=$?
    if ((mpi_status!=0)); then
        stop_gpu
        printf '[workers] MPI exit: %s; Policy services stopped after MPI failure\n' "$mpi_status"
        exit "$mpi_status"
    fi
    service_status=0
    statuses=()
    for pid in "${policy_pids[@]}"; do
        code=0
        wait "$pid" || code=$?
        statuses+=("$code")
        if ((code!=0)); then service_status=$code; fi
    done
    policy_pids=()
    trap - EXIT
    printf '[workers] MPI exit: %s; Policy service exits: %s\n' "$mpi_status" "${statuses[*]}"
    if ((service_status!=0)); then exit "$service_status"; fi
else
net=$(cfg mpi_tcp_interface)
export SINGULARITYENV_OMPI_MCA_btl_tcp_if_include="$net"
export SINGULARITYENV_OMPI_MCA_oob_tcp_if_include="$net"
timeout --signal=TERM --kill-after=30 "$(cfg mpi_timeout_seconds)" mpirun --tag-output --bind-to core \
    --mca btl self,tcp --mca btl_tcp_if_include "$net" --mca oob_tcp_if_include "$net" \
    -np "$(cfg mpi_ranks)" singularity exec -B "$(cfg remote_root)" "$(cfg image)" \
    /opt/ogs/bin/python3 -B -m gnn_amg.gate "$config" < /dev/null
fi
) 2>&1 | tee -a "$full" | python3 -B -u -m gnn_amg.logs "$task/logs/ogs_tail.log"
codes=("${PIPESTATUS[@]}")
batch=${codes[0]}
if ((codes[1]!=0)); then batch=${codes[1]}; fi
if ((codes[2]!=0)); then batch=${codes[2]}; fi
printf '[log] MPI exit: %s; logger exit: %s; web filter exit: %s\n[run] End: %s; batch exit: %s\n' \
    "${codes[0]}" "${codes[1]}" "${codes[2]}" "$(date --iso-8601=seconds)" "$batch" | tee -a "$full"
footer=("${PIPESTATUS[@]}")
if ((footer[1]!=0)); then batch=${footer[1]}; fi
exit "$batch"
