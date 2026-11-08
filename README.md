# GNN AMG coarse-DOF selection

This project learns two successive C/F selections. It builds the second graph
from the actual first Galerkin matrix. Interpolation, coarse operators,
smoothing and the complete FieldSplit-GMRES solve remain numerical operations.

The source tree contains reusable ML code, its numerical bridge, tests and
documentation. Datasets, generated features, compiled libraries, checkpoints,
logs, reports and run-specific configuration belong in a separate analysis
directory. Set `PYTHONDONTWRITEBYTECODE=1`; direct compiler, package-install and
test-cache outputs outside this repository. Nothing is published automatically.

## Scientific protocol

- Use the accepted three-dimensional L0/L3/L5 endpoints as supervised labels.
  Their actual teacher is three-component HMIS with ext+i interpolation.
- First verify those endpoints with direct HYPRE multipass interpolation. An
  accepted product of old interpolations does not prove direct multipass works.
- Preserve right GMRES, multiplicative FieldSplit, the other physical blocks,
  original initial vectors and true residual tolerances `1e-8` / `1e-12`.
  The agreed common iteration limit is 1000.
- Use one shared two-message-passing-layer policy, width 32, with two heads.
  Perform local stochastic exploration and retain the original sampled actions.
- Each update uses 2–4 different states with one candidate each. Reward is
  `(reference_seconds - candidate_seconds) / common_time_scale`; failures are
  worse than every valid bounded-time result. The EMA baseline uses past data.
- Periodic validation selects by cumulative elapsed time and convergence.
  Existing training/validation/test identities remain separate. Uniform sampling
  of saved snapshots estimates that snapshot pool, not all Newton events unless
  independently justified event-frequency weights are provided.
- The first full training allocation has a 24-hour budget. Resume checkpoints
  and validation evidence determine continuation. CPU allocation never exceeds
  192; GPU training and measured inference are preferred when beneficial.

## Implemented entry points

`gnn_amg.policy` contains the shared network, raw local sampling and supervised
loss. `gnn_amg.objective` contains the seconds reward, event-frequency correction
and on-policy REINFORCE loss. These are components, not a completed training run.

`python -B -m gnn_amg.gate /path/to/analysis/config.json` runs the numerical
prerequisite on explicit training states, comparing original AMG, the retained
base, and direct multipass from the same coarse endpoints. It uses a frozen,
hashed copy of the existing replay/RAS helpers. The driver does not start RL.

`native/multipass.c` is compiled as a small shared bridge against the existing
PETSc 3.16.3/HYPRE 2.23.0/MPI headers. It does not rebuild OGS, HYPRE or PETSc.
It calls no traditional coarsening routine. Component IDs survive C selection;
MPI borders retain their algebraic connections. A nine-hop strong-path audit
matches this HYPRE version's multipass limit. Empty coarse ownership is handled
explicitly when importing the resulting interpolation into PETSc.

Example tests, with all caches outside the source tree:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B -m unittest discover -s tests
```

The numerical gate additionally checks C injection, empty interpolation rows,
Galerkin actions, full-system true residuals and MPI exit status. Gate diagnostic
timings include checks and are not reported as a deployment speedup. The complete
persistent GPU/MPI rollout controller follows only after this prerequisite passes.

## Reuse and release status

Environment paths, SSH identities and datasets are supplied by external run
configuration. No credentials or production data belong here. The initial
numerical adapter reuses the existing project's audited replay and RAS helpers
via a configured dependency snapshot; a public release must document or package
those dependencies with their provenance. A distribution licence has not yet
been selected.

The bridge preserves HYPRE special-F rows with no outgoing strong connections. They remain fine-smoother responsibilities. The gate independently checks these rows are also empty in the effective base; it does not waive coverage for other F rows.

The fixed decoder can repair coverage by promoting uncovered rows, using native
DOF IDs for deterministic distributed ties. It preserves all proposed C points,
records original and executed markers separately, and caps additions by a
configured fraction (plus the terminal size limit). Nine- and 99-hop diagnostics
distinguish the native interpolation limit from longer/disconnected paths.
A repair-budget failure remains a failed candidate; repaired points never
replace raw policy actions in a REINFORCE log probability.

`gnn_amg.dataset` exports supervised material from the accepted raw labels and
the actually built direct-multipass K1. Fine features contain current phase
field, degradation, moduli, porosity/Biot terms, aperture and normalized geometry.
Coarse physical features are fixed |P1|-weighted support statistics, including
phi and phi squared. Both graphs use the exact HYPRE strength topology with
normalized absolute stiffness weights; matrix sign/scale and MPI-boundary
information are observable features. No teacher information enters feature
construction. Original and repaired teacher labels are stored separately.

Set `export_supervised` in the external gate configuration to export the training
split only and verify every resulting full solve. Validation and final-test
frames are not exported by this preparation step. Generated NPZ files are
offline training material; their construction cost must still be measured in
the online inference path.

`gnn_amg.pretrain` verifies the successful dataset receipts, fits feature scaling
using training examples only, and trains the shared network on one CUDA device.
It stores an immutable checkpoint after every epoch, including optimizer and
random-generator states. Raw-label recall is diagnostic; none of these
checkpoints is accepted until independent predicted-C full solves pass.

`gnn_amg.online` evaluates independent predicted coarse points. It reads the
current full matrix, physical fields and DOF layout, and builds both graphs
fresh. It never reads teacher hierarchy maps or supervised graph caches.
`gnn_amg.inference` keeps one GPU model resident and serves the MPI root through
a job-private authenticated numeric socket protocol. The measured candidate
wall time includes graph extraction, MPI gathering, network transfers, both
policy decisions, repair, numerical setup and the complete solve.

The initial deterministic decoder retains the highest score per component
in spatial bins. Bin aspect ratios use spacing observed on the current strong
graph; the coarse size scale comes from training statistics. A neutral-score
control checks the contribution of network ranking. This is a provisional
decoder: acceptance requires actual full solves, and RL must retain original
sampled masks separately from the executed repaired masks.

CPU64 MPI and a configurable CPU/one-GPU service use two components of one Slurm allocation.
MPI stays on the single allocated CPU node; the GPU service is a separate job
step. See [Slurm heterogeneous jobs](https://slurm.schedmd.com/heterogeneous_jobs.html).

For a single-node allocation, set `launch_mode=single_node`,
`inference_device=cpu` and one inference thread. The existing checkpoint is
reused; the model service shares the first allocated core while MPI rank0
waits for its reply. All64 numerical ranks and the original RAS partitioning
remain unchanged. This mode avoids a second scheduler allocation. Its measured
cost is explicitly CPU inference and must not be called a GPU speedup.

The optional `decoder.kind=directed_graph` uses the single GNN score ordering
and excludes only rows that can reach a chosen representative through the
actual directed strong graph within a configured radius (at most nine hops).
The two radii are frozen before RL. Counts emerge from scores and connectivity;
no AMG hierarchy/coarsener is used. Numerical interpolation and the separate
bounded repair remain unchanged. This replaces the failed spatial decoder for
independent acceptance; the original implementation stays for reproducibility.
The `neutral_graph` control measures what this deterministic constraint alone
achieves. Successful coverage by itself is neither convergence nor evidence
of a learned speed improvement.
