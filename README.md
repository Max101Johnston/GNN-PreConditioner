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
