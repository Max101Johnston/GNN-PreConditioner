#!/bin/bash
set -u -o pipefail
if [[ "${1:-}" == "--policy-rank" ]]; then
    set -e
    config=$2 endpoint=$3 ranks=$4 torch_python=$5 remote_root=$6 image=$7
    rank=${OMPI_COMM_WORLD_RANK:?Open MPI rank required}
    command=(singularity exec -B "$remote_root" "$image"
        /opt/ogs/bin/python3 -B -m gnn_amg.online "$config" "$endpoint")
    if ((rank % ranks != 0)); then exec "${command[@]}"; fi
    worker=$((rank / ranks))
    ready="$endpoint-worker$worker"
    GNN_POLICY_WORKER="$worker" GNN_COLOCATED_POLICY=1 \
        "$torch_python" -B -m gnn_amg.inference "$config" "$ready" &
    pid=$!
    trap 'kill -TERM "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
    for ((attempt=0; attempt<240; ++attempt)); do
        [[ ! -f "$ready" ]] || break
        kill -0 "$pid"
        sleep .25
    done
    [[ -f "$ready" ]]
    "${command[@]}"
    wait "$pid"
    trap - EXIT
    exit 0
fi
config=$(realpath "${1:?analysis configuration}")
cfg() { python3 -c 'import json,sys,functools,operator;print(functools.reduce(operator.getitem,sys.argv[2:],json.load(open(sys.argv[1]))))' "$config" "$@"; }
task=$(cfg output_directory)
source=$(cfg source_directory)
full="$task/logs/$SLURM_JOB_ID.full.log"
if [[ "$(cfg stage)" == "student_live" ]]; then
    full="$(cfg remote_case)/logs/$SLURM_JOB_ID.full.log"
fi
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
if 'selector' in cfg:
    assert sha(cfg['selector'])==cfg['selector_sha256']
if 'student_library' in cfg:
    assert sha(cfg['student_library'])==cfg['student_library_sha256']
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
if [[ "$(cfg stage)" == "student_live" ]]; then
    [[ "$PWD" == "$(cfg remote_case)" && "$SLURM_NTASKS" == "$(cfg live ranks)" ]]
    image=$(cfg image)
    root=$(cfg remote_root)
    net=$(cfg mpi_tcp_interface)
    work="$task/cache/live-$SLURM_JOB_ID"
    mkdir "$work"
    singularity exec -B "$root" "$image" /opt/ogs/bin/python3 -B -m gnn_amg.live prepare "$config" "$work"
    binary_sha=$(singularity exec -B "$root" "$image" sha256sum /opt/ogs/bin/ogs | cut -d ' ' -f 1)
    [[ "$binary_sha" == "$(cfg live ogs_sha256)" ]]
    printf '[run] OGS SHA256: %s; source case: %s\n' "$binary_sha" "$PWD"
    export SINGULARITYENV_OMPI_MCA_btl_tcp_if_include="$net" SINGULARITYENV_OMPI_MCA_oob_tcp_if_include="$net"
    read -r -a methods <<< "$(python3 -c 'import json,sys;print(" ".join(json.load(open(sys.argv[1]))["live"].get("methods",["reference","student"])))' "$config")"
    for method in "${methods[@]}"; do
        project="$work/3D-SCS-1st.prj"
        if [[ "$method" == amg ]]; then project="$work/amg.prj"; fi
        printf '[run] OGS method: %s; project: %s\n' "$method" "$project"
        output="$task/data/live-$SLURM_JOB_ID-$method"
        mkdir "$output"
        export SINGULARITYENV_GNN_LIVE_CONFIG="$config" SINGULARITYENV_GNN_LIVE_DIRECTORY="$work"
        export SINGULARITYENV_GNN_LIVE_METHOD="$method"
        start=$(date +%s.%N)
        timeout --signal=TERM --kill-after=30 "$(cfg mpi_timeout_seconds)" \
            mpirun --mca plm slurm --mca ras slurm --map-by slot --tag-output --bind-to core \
            --mca btl self,vader,tcp --mca btl_tcp_if_include "$net" -np "$SLURM_NTASKS" \
            singularity exec -B "$root" "$image" stdbuf -oL -eL /opt/ogs/bin/ogs \
            -m "$PWD/input" -s "$source/src/gnn_amg" -o "$output" "$project" < /dev/null
        python3 -c 'import sys; print(float(sys.argv[2])-float(sys.argv[1]))' "$start" "$(date +%s.%N)" \
            > "$work/elapsed-$method.txt"
    done
    unset SINGULARITYENV_GNN_LIVE_CONFIG SINGULARITYENV_GNN_LIVE_DIRECTORY SINGULARITYENV_GNN_LIVE_METHOD
    singularity exec -B "$root" "$image" /opt/ogs/bin/python3 -B -m gnn_amg.live verify "$config" "$work"
elif [[ "$(cfg stage)" == "student_train" ]]; then
    export LD_LIBRARY_PATH="$(cfg cuda_toolkit)/lib64:${LD_LIBRARY_PATH:-}"
    "$(cfg torch_python)" -B -m gnn_amg.train_student "$config"
elif [[ "$(cfg stage)" == "pretrain" ]]; then
    export LD_LIBRARY_PATH="$(cfg cuda_toolkit)/lib64:${LD_LIBRARY_PATH:-}"
    "$(cfg torch_python)" -B -m gnn_amg.pretrain "$config"
elif [[ "$(cfg stage)" == "prediction" || "$(cfg stage)" == "rl" || "$(cfg stage)" == "parallel_smoke" || "$(cfg stage)" == "review" ]]; then
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
    mode=$(cfg launch_mode)
    if [[ "$(cfg stage)" != "parallel_smoke" && "$mode" != "colocated_cpu" && "$mode" != "native_cpu" ]]; then
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
    if [[ "$mode" == "parallel_cpu" || "$mode" == "colocated_cpu" || "$mode" == "native_cpu" ]]; then
        mpi=(mpirun --mca plm slurm --mca ras slurm --map-by slot --tag-output --bind-to core)
    else
        cpu_host=$(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_0:-$SLURM_JOB_NODELIST}")
        [[ "$cpu_host" != *$'\n'* ]]
        mpi=(mpirun --mca plm isolated --mca ras ^slurm --tag-output --bind-to core --host "$cpu_host:$(cfg mpi_ranks)")
    fi
    export SINGULARITYENV_OMPI_MCA_btl_tcp_if_include="$net"
    export SINGULARITYENV_OMPI_MCA_oob_tcp_if_include="$net"
    if [[ "$mode" == "colocated_cpu" ]]; then
        application=(/bin/bash "$source/scripts/run-gate.sh" --policy-rank "$config" "$endpoint"
            "$(cfg mpi_ranks)" "$(cfg torch_python)" "$(cfg remote_root)" "$(cfg image)")
    else
        application=(singularity exec -B "$(cfg remote_root)" "$(cfg image)"
            /opt/ogs/bin/python3 -B -m gnn_amg.online "$config" "$endpoint")
    fi
    mpi_status=0
    timeout --signal=TERM --kill-after=30 "$(cfg mpi_timeout_seconds)" \
        "${mpi[@]}" --mca btl self,vader,tcp --mca btl_tcp_if_include "$net" --mca oob_tcp_if_include "$net" \
        -np "$((workers*$(cfg mpi_ranks)))" "${application[@]}" < /dev/null || mpi_status=$?
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
