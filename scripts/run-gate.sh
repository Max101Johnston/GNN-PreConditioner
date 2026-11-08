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
printf '[run] Start %s job %s stage numerical-gate node %s\n' "$(date --iso-8601=seconds)" "$SLURM_JOB_ID" "$(hostname)"
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
net=$(cfg mpi_tcp_interface)
export SINGULARITYENV_OMPI_MCA_btl_tcp_if_include="$net"
export SINGULARITYENV_OMPI_MCA_oob_tcp_if_include="$net"
timeout --signal=TERM --kill-after=30 3300 mpirun --tag-output --bind-to core \
    --mca btl self,tcp --mca btl_tcp_if_include "$net" --mca oob_tcp_if_include "$net" \
    -np "$(cfg mpi_ranks)" singularity exec -B "$(cfg remote_root)" "$(cfg image)" \
    /opt/ogs/bin/python3 -B -m gnn_amg.gate "$config" < /dev/null
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
