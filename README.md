# GNN AMG coarse-DOF selection

This project studies learned AMG coarsening and interpolation. The factorized
student generates both coarse spaces and their signed interpolation. It builds
the second graph from the actual first Galerkin matrix. PETSc assembles the
coarse operators and executes the complete FieldSplit-GMRES solve.

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

## Persistent multi-frame reinforcement learning

With `stage=rl`, `gnn_amg.online` keeps the 64-rank numerical environment
resident and `gnn_amg.inference.PolicyLearner` keeps the policy and optimizer
resident. Each uniformly sampled batch contains 2–4 different training states.
A bounded neighbourhood is sampled from the current graph, alternating uniform
and physical-field-contrast anchors. Outside it, the current frozen policy's
deterministic decoder is used. Within it, a Bernoulli policy uses network
scores and a fixed local prior favouring the frozen reference. Active draws and
their actual probabilities are saved before numerical repair.

Per-state gradients use true whole-solve seconds and a past-only EMA baseline.
Gradients accumulate while policy parameters stay fixed, and Adam steps once
all batch feedback has arrived. Warm-up updates the second head alone, then the
first head alone; the shared trunk stays frozen in both phases. Joint training
then unfreezes the trunk and both heads. Gradient norms, parameter changes,
parameter hashes and original-vs-executed actions are recorded.

Standard AMG supplies a measurable fixed reference for every saved training
state; the accepted retained-base comparator is also measured for the fixed
evaluation pool where its export exists. One common seconds scale is frozen
across states and carried into continuation. Candidate numerical failures receive
a penalty below every valid bounded-time reward. A collective convergence test
checks the unchanged full-system true residual and the configured episode time
limit. Setup, feature construction, inference and complete solve are included
in the candidate time; parameter backpropagation and artifact writing are
training overhead, excluded from candidate timing but included in the job budget.

A smoke run must make a nonzero real update from three independent full solves
and must revalidate the changed deterministic policy. It uses training states
only. The subsequent 24-hour allocation rotates across all 55 training states
and periodically evaluates a fixed held validation subset. Checkpoints contain
model, optimizer, EMA, random states and update/phase identity. Batch selection
uses the saved global update index, so continuation does not silently restart
the schedule. Validation selection uses failure count and cumulative seconds,
never the average of frame-wise speedup percentages. Saved-frame conclusions
remain separate from actual restarted nonlinear time integration.

## Parallel rollout allocation

`parallel_workers=3`, `mpi_ranks=64` and `launch_mode=parallel_cpu` use one
192-rank Slurm allocation. Three disjoint MPI communicators retain the accepted
64-rank matrix ownership and RAS overlap, and evaluate three different states
concurrently. Three resident policy replicas start from identical parameters.
Each computes its state contribution divided by the full batch size; gradient
sums are exchanged only after all state feedback is complete. Every replica
then applies the same clipped gradient and Adam update. Matching parameter
hashes are required before and after each update. This is one shared policy,
not three independently trained policies.

States have fixed worker assignments, so reference and candidate measurements
use the same placement. One state is sampled from each worker's training slice.
The marginal sampling probability is 1/(3*N_worker); weight 3*N_worker/55 keeps
the objective equal to the uniform saved-state mean. Validation is unweighted
across the same fixed eight states. Native interpolation, coarse solves,
precision and maximum iterations do not change. Node-local MPI uses shared
memory; TCP remains available between nodes.

Each worker saves distinct action and checkpoint files. The canonical worker0
checkpoint contains synchronized model and optimizer state. Per-worker sampling
is seeded by the global update index and worker ID, making continuation explicit.
Parallel training throughput is recorded separately from single-solve latency;
a resource allocation is not by itself proof of a solver speed improvement.

## Learned interpolation cycle

The deterministic connection prior supplies C1 and C2. Student A consumes
one actual `K P0` product and predicts row damping and one of the 6, 8 or
12 entry budgets. Student B adds a bounded signed path product with four
couplings and an eight-entry row seed. Its shared edge model predicts signed
coefficients and priorities on at most 24 candidates. PETSc communicates
matrix rows across the actual ownership boundaries before local inference.

The native library uses the portable NPZ export from
`gnn_amg.train_student`. Each solver rank loads the coefficients once and
generates its own sparse rows. Protected rows come from the original transfer.
The driver computes Galerkin operators, the small terminal solve, and the
original FieldSplit GMRES residual check. Candidate generation and inference
are included in the setup clock.

Use `scripts/student-pipeline.py` with an external plan JSON pointing at a
frozen run config, its monitor and a task record. The workflow verifies
training-only teacher receipts and four sampled coarse directions from
actual solves. It fits A/B on two seeds and the secondary student in a single
training allocation, checks PyTorch/native parity, compares checkpoints in
full solves, refreshes P2 on the selected P1, and uses paired elapsed seconds
for the low-dimensional controller. The remaining validation and test keys
enter after selection. The pipeline state records actual Slurm job IDs and
immutable submission receipts for recovery.

The deployment entry point in this repository is the saved-matrix
FieldSplit replay. A positive speed gate is the prerequisite for the
separate continuous OGS restart comparison.

### Direct first-transfer pilot

`scripts/student-pipeline.py` also accepts a plan with `mode=direct`. Its base
configuration names training-only `gate_keys` with complete retained L0/L3/L5
receipts, `direct_hops`, `direct_candidates`, and a frozen `direct_student_control`.
The data stage imports the retained transfers under their original column
numbering and scaling, rebuilds both Galerkin products, and compares complete
solves with the retained reference. It exports signed unit-injection P1 labels
with per-frame coverage and the original transfer receipt. Data paths and the
pilot state are external to the source tree.

The D student predicts signed weights and support priorities from bounded
path features of the current stiffness matrix and supplied coarse DOFs.
PETSc communicates paths across MPI ownership boundaries. The prediction path
uses no multipass seed or teacher P1 coefficients. The pilot uses the teacher's
C1, C2 and rebased P2 to isolate P1, then compares complete replay time and true
residuals with both the round-trip and the frozen v1 student. Its training-frame
results are a numerical and timing diagnostic.

The `direct_refine` plan runs three coefficient controls for each training
frame, using the frozen D model and its candidates. The controls retain all
teacher entries in the candidates, the six largest teacher entries there, or
the teacher entries selected by D. An empty intersection gets the single
largest original teacher coefficient, and each report counts those rows.
Each transfer rebuilds both Galerkin operators with the same coarse IDs and P2.

Student E generates its own weights using component-local geometric neighbors
and signed one- and two-hop stiffness contacts. The local path uses one PETSc
sparse product. Teacher-derived support labels supervise the ranking alongside
signed coefficients, row sums and correction samples, including early,
middle and late inputs from the frozen D solve evaluated through the teacher.
Training switches from
dense output to the same hard support rule used in native inference. The
`direct_local` configuration controls the candidate widths, support bounds
and score cutoff. The final paired replay compares the selected E checkpoint
with D, the frozen online student, the rebuilt teacher and the cached teacher.

The `direct_context` plan starts from a completed two-arm E teacher-support
comparison, supplied as `e_oracle`. It can extend the algebraic candidate
frontier over bounded strong stiffness paths, using the same candidate width.
One controlled teacher replay selects the topology by convergence. F applies
a shared signed and unsigned graph message layer, pools fine attributes at
coarse endpoints and summarizes the other candidates on each row. Solver
ranks fetch only referenced fine halo rows and exchange coarse summaries.
The deployment uses a learned row budget from 6, 8, 10 or 12 entries.

F samples keep the actual fine and coarse one-hop neighborhoods. Training
rotates batches through MPI regions, first fitting signed coefficients on
teacher supports, then training on the executed hard support. Row ranking
supervision and a soft backward gate let the action loss update edge and
budget scores. The exported model is evaluated with the same hard decoder
in distributed replay. The plan reports numerical quality and total-time
gates separately before extending training to model-selected coarse points.

### Continuous OGS comparison

The opt-in `student_live` stage uses the existing OGS/PETSc SIF and the
production consolidation-to-loading input receipt. It verifies the original
mesh and partition hashes, reuses the original DOF numbering at the physical
MPI rank count, and checks the live displacement split and constraint rows.
Both variants start from the same consolidated input. The OGS Python hook
builds the same displacement hierarchy as replay inside the live TH2M GMRES
setup and measures the full solve with the same true-residual tolerance.
The pilot fails on a missing hook, a numerical guard, or an incomplete final
timestep. Runtime files and OGS results stay outside this repository.

The live `methods` list can include `amg` to measure the original OGS
displacement BoomerAMG on the same loading trajectory. This control retains
the production HYPRE options and uses the same full-system GMRES tolerances,
true-residual checks and output schedule as the learned hierarchy. PETSc
reports the instantiated displacement PC before the first solve.

For timing studies, set `live.output_profile` to `lean` for every method in
the same comparison. It requests the five primary variables and checks that
the output index contains only the start, the optional injection boundary,
and the final time. The extra fracture diagnostics at integration points
are disabled. The full linear and timestep diagnostic records are retained.
Fixed output times also constrain OGS time stepping, so comparisons use the
same profile for every method.

## Factorized joint interpolation

The factorized student composes three learned local transfers into P1 and two
more into P2. A shared graph encoder reads signed stiffness neighbours and
pooled observations. The model ranks representatives in spatial groups, sets
group and row budgets, and generates signed coefficients on its selected
support. Distributed quotient graphs carry the intermediate structure. The
solver uses only P1 and P2, with K1 and K2 assembled from those products.

Run the existing pipeline with an external plan in mode factorized.

    python -B scripts/student-pipeline.py /path/to/analysis/records/plan-factorized.json

The plan lists training-only teacher frames, the frozen base configuration,
Slurm limits and the existing monitor. The factor_data stage verifies original
HMIS shards and native maps, rebuilds the products, and checks the effective
retained cache where present. The comparison restores the cached column
coordinates and accounts for signed component-sum dropping. Sampled shards
contain the same graph observations used in deployment, signed local targets,
and product probes with teacher factors downstream. Extra teacher edges are
available during the initial coefficient warmup. The final supervision phase
uses the deployed candidates and hard decoding.

Joint training starts with the resulting checkpoint. Each batch rotates
training frames and perturbs a shared spatial region across all five stages.
The paired complete solves update representative, support and budget scores
and a signed coefficient adapter. Captured PCApply inputs also train the
coefficient adapter through the student's actual RAS and Galerkin correction.
The same model and decoder generate both candidate and validation hierarchies.

Validation compares the initial and updated models with the existing student,
on-demand AMG and the cached retained hierarchy. It measures fresh graph
generation, selection, transfers, products, Galerkin construction and the full
true-residual solve. The replay input and resident model weights are outside
the timed interval. The pipeline repeats competitive results in reversed
order. An experimental run can complete even if the model misses the speed
target. The report keeps failure counts, per-frame true residuals, factor
sizes, construction timing and model identities.

### Hard-support training

The factor_hard pipeline reuses the accepted teacher shards and exports the
remaining training matrices while the first hard model trains. The pooled fit
uses every training state. All primary losses use the deployment mask from the
first step. The brief teacher auxiliary uses the same available candidates and
row budgets. The shared encoder, representative scores, support scores,
signed coefficients and budget have independent learning rates. A fixed
sample of rows from each training state measures deterministic hard quality
and selects the checkpoint. The fit receipt records teacher coefficient
retention and the error on composed probe actions after projection onto the
deployed budget.

The Krylov pilot uses full-system residuals from early and late iterations and
component-local directions. It compares four, eight and twelve right GMRES
steps with PETSc using the same FieldSplit and RAS. It also checks the sparse
ILU(0) forward and transpose actions on the actual first coarse blocks. Small
directional finite differences exercise the Arnoldi reverse pass and the
sparse ILU numeric reverse pass.


The factor_gradient replay uses the actual mid- and late-solve residuals and a
rotating displacement component. It differentiates the right-preconditioned
short GMRES segment through the multiplicative FieldSplit, RAS, both Galerkin
matrices, middle ILU numerical factors, and all five sparse transfers. The
small terminal solve uses its transpose factor. The frozen minibatch keeps
its current hard supports and graph observations. New replay reconstructs
both coarse matrices and the second-level graph. The pinned numerical bridge
copies the actual fixed phase V-cycle and its C/F markers. The replay checks
the reconstructed phase action and the transpose of each full Arnoldi
operator action. The RAS transpose masks the input on the owned rows before
the existing local ILU transpose solve, then sums all overlapping outputs.
A frozen-support finite difference rebuilds all numerical factors
to verify the coefficient gradient. Paired cold replays use the regular
decoder and separately record the initial and updated checkpoints.


The factor_structure option starts sampled structure updates after verified
coefficient feedback. Each rollout releases a single ranking slot per sampled
local group or row, or one row budget. These conditional choices preserve the
deployment eligibility, anchors, and hard budget limits. Their categorical
likelihoods update the corresponding network output head using four full
same-frame solves and a leave-one-out elapsed-time baseline. The checkpoint is
replayed with deterministic hard decoding on the fixed validation states.


The factor_learn cycle extends the same policy over all training matrices,
alternating structural updates with verified coefficient gradients. It saves
network weights and optimizer state at each periodic deterministic validation.
An interrupted run can resume from a hashed milestone receipt using its
unchanged schedule. The final report replays the two shortlisted checkpoints
and the established solver baselines in reversed paired order. It records
cached and on-demand speed comparisons separately.

If a run terminates after its last milestone, recovery reconstructs the
recorded structural actions without executing their complete solves again.
It verifies the action likelihoods and signatures before reusing saved
elapsed-time rewards. The interrupted step is executed from the last verified
state whenever a reconstruction check fails.

### Search and distillation

The `factor_search` review starts from the retained learned checkpoint. It
calibrates short right-GMRES prefixes on identical full-system residuals for
the start, the late checkpoint, and the retained teacher. The coefficient
objective stops accumulating a gradient on a probe once it meets its target.
An update uses only the signed output channels and records changes in every
stage's deterministic support and representatives before admission.

Each round screens the deterministic incumbent, four regional proposals, and
a separate coefficient proposal when the signed head has a finite nonzero
gradient. Executed hierarchy fingerprints eliminate repeated full solves.
Only a repeatable improvement in true-residual time enters the distillation
bank. KL on the recorded legal candidate sets protects the existing behavior.
The incumbent is promoted on the same eight validation matrices using paired
cold construction and solves. The final comparison reverses the order in its
second repeat and includes the original checkpoint and the signed student.

Select this mode with `stage: review` and `student_cycle: factor_search` in an
external frozen configuration. The analysis records the probe calibration,
each round's screen and confirmation outcomes, and the final paired solves.
