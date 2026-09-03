# Fathi §4.3 CURRENT Production Inversion Engine

This repository contains the cleaned CURRENT production source for the Fathi §4.3 SEM3D GPU inversion benchmark.

It is intentionally separated from the historical development repository and from heavy numerical runtime data.

## 1. Current production status

Production run:

```text
fathi_s43_repro_p20_t052
```

Validated transitions:

```text
iter001 -> iter002   CLOSED
iter002 -> iter003   CLOSED
iter003 -> iter004   CLOSED
```

Current accepted model:

```text
iter004
```

Current objective sequence:

```text
J0 = 3.78991998304714295e-08
J1 = 3.78817580658295249e-08
J2 = 2.15277530441126652e-08
J3 = 1.79378469968073135e-08
J4 = 1.57501642292417593e-08
```

These are early/mid inversion iterations and should not be described as final convergence.

Next production transition:

```text
K=4
iter004 -> iter005
```

K=4 will also serve as the real migration acceptance test for this cleaned repository.

## 2. CURRENT production workflow

```text
accepted model m_k
        |
external parent forward
        |
objective / residual
        |
exact discrete reverse
        |
material covector
        |
control-space bridge
        |
Mtilde solve
        |
registered physical gradient g_k
        |
L-BFGS history
        |
physical-space L-BFGS
        |
Fathi Eq.25 lambda bias
        |
candidate = parent + alpha * direction
        |
external candidate forward
        |
Armijo decision
        |
accepted-child promotion
        |
accepted model m_{k+1}
        |
final closure audit
```

The mathematical route above is frozen. Historical gradient, search-direction, or line-search routes must not be reintroduced unless a genuine mathematical or physical error is demonstrated.

## 3. Main production entry point

For one CURRENT inversion transition:

```bash
K=4
bash scripts/fathi_benchmark/run_current_iteration.sh "$K"
```

Python entry point:

```bash
python -m scripts.fathi_benchmark.run_current_iteration \
  --parent-iteration 4
```

The exact production command should always be checked against the current runtime configuration and runbook before launching an expensive numerical stage.

## 4. Closure audit

After a completed transition:

```bash
K=4
bash scripts/fathi_benchmark/audit_current_iteration.sh "$K"
```

The audit does not rerun the expensive numerical stages. It checks the existing production artifacts, transition identity, accepted objective, accepted step, hashes, and child state, then writes:

```text
results/<run>/<transition>/closure_audit/final_closure_audit.json
```

The audit framework has already been replayed successfully for K=1, K=2, and K=3 with:

```text
NUMERICAL_RERUNS = 0
```

A transition is frozen only after both:

```text
PASS_CURRENT_ITERATION_XXX_TO_YYY_CLOSED
```

and:

```text
PASS_CURRENT_ITERATION_XXX_TO_YYY_FINAL_CLOSURE_AUDIT
```

are obtained.

## 5. Frozen mathematical contract

- Exact discrete reverse is the algebraic transpose of the certified external discrete forward route.
- The corrected gradient is expressed in the physical parameter space.
- Mtilde defines the physical/control-space metric.
- L-BFGS vectors use physical Pa units.
- L-BFGS history memory target: 15.
- Fathi Eq.25 uses the Euclidean L2 norm.
- Lambda-bias weight:

```text
W(k) = max(1 - k/50, 0)
```

- Armijo parameters:

```text
alpha0 = 1
c1     = 1e-4
rho    = 0.5
```

- Candidate update:

```text
m_candidate = m_parent + alpha * p_parent
```

- No max-absolute normalization is used.

## 6. Source / runtime separation

This Git repository contains source code, configuration, tests, and certification documentation.

Heavy numerical artifacts remain outside Git.

Recommended architecture:

```text
NEW CLEAN SOURCE REPO
~/fathi-sem3d-current-production
│
├── configs/
├── scripts/
├── tests/
├── README.md
└── production documentation
          │
          │ FATHI_RUNTIME_ROOT
          v
CERTIFIED HEAVY RUNTIME
~/fathi-sem3d-gpu-inversion-benchmark
│
├── data/
├── results/
├── states/
├── checkpoints/
└── certified numerical/operator artifacts
```

The existing heavy runtime currently contains the accepted iter004 state required for the K=4 migration acceptance test.

## 7. Git policy

Do not commit generated numerical data:

```text
results/
data/reproduction/
data/diagnostics/
*.h5
*.hdf5
*.npy
*.npz
checkpoints/
checkpoint/
replay_cache/
replay_caches/
numerical logs
```

Avoid destructive Git operations on the production branch.

```text
no git add .
no git reset --hard
no force push
```

Prefer explicit file staging.

## 8. Clean-source migration provenance

The clean project was generated from certified development source commit:

```text
035e7a751aaf92225f936524550d2fb1d8ec88c0
```

Production source manifest:

```text
FILE_COUNT = 53
CONTENT_SIGNATURE_SHA256 =
f7f0010fea1875ea3a5def0105b57def0bd085a376c1416249d89a49e6b4a971
```

Clean archive:

```text
ARCHIVE_SHA256 =
ccbb40e22f9363dc806c0a672756bd6df54ddc99bb8332528c79d12f372331ef
```

Clean-source acceptance results:

```text
53 production source files
DIFF = 0
NUMERICAL_RESULTS_INCLUDED = false

Python syntax = PASS
Shell syntax = PASS

CURRENT entrypoint smoke tests = PASS

51 tests passed
45 subtests passed
```

First commit of this new repository:

```text
8d7fb66a169c653bc96505fb08b9b12c7ad2748f
Import certified CURRENT production source baseline
```

## 9. Historical and superseded routes

Historical development and certification history remain in the previous repository.

The CURRENT production project must not fall back to superseded routes such as:

```text
bridge_stage5o_certified_gradient.py
run_current_t052_*
finalize_current_t052_*
424B_compute_rhs_component_from_traces.py
compute_search_direction.py
prepare_gpu_adjoint_full.py
run_gpu_adjoint_task.py
solve_gpu_mtilde_gradient.py
```

Historical files may only be reused when explicitly retained as immutable certified operator assets or compatibility guards.

## 10. CURRENT tests

```text
tests/test_current_iteration_routing.py
tests/test_exact_reverse_gradient_generic.py
tests/test_current_pipeline_contract_repairs.py
tests/test_current_pipeline_integration_static.py
tests/test_bridge_certified_external_gradient_current.py
```

The clean-source migration baseline passed:

```text
51 passed
45 subtests passed
```

These are source-level regression checks. They do not replace the real SEM3D production acceptance test.

## 11. Next acceptance milestone

The next milestone is the real migration acceptance run:

```text
NEW CLEAN SOURCE
        +
EXISTING CERTIFIED HEAVY RUNTIME
        +
accepted iter004
        |
        v
K=4
iter004 -> iter005
```

Success requires:

```text
PASS_CURRENT_ITERATION_004_TO_005_CLOSED
```

followed by:

```text
PASS_CURRENT_ITERATION_004_TO_005_FINAL_CLOSURE_AUDIT
```

Only after both passes should the new repository be declared:

```text
NEW PROJECT = PRODUCTION CERTIFIED
CURRENT MODEL = iter005
```

## 12. Project principle

This repository preserves the already-certified numerical and physical workflow; it is not a redesign of the inversion algorithm.

Engineering problems should be solved by repairing the existing CURRENT production route.

Changes to the frozen mathematical or physical definition should only be made when a major logic, mathematics, or physics error is explicitly identified and documented.
