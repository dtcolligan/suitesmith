# suitesmith

An RL environment, written on the [verifiers](https://github.com/willccbb/verifiers)
framework, that **trains a small model to write test suites**. Each task is a
generated function: a specification, a correct reference implementation, and
five subtly broken mutants. The model writes pytest tests; reward = the tests
**pass the reference and kill the mutants** (`0` if the reference fails any
test, else `killed / 5`). The trained capability is verifier-writing itself.

Renamed from `covgap` 23 Aug 2026: the smith forges the suite; killing
mutants is how the forge is graded.

Status: environment built and self-tested; nothing trained yet. This README
will never claim otherwise.

## Layout

- `SPEC.md` — the design document (single source of truth; v1's planted-gap
  design is retired, recorded in §9)
- `suitesmith/families.py` — template families: parameterised spec +
  reference generators with per-family mutation menus (SPEC Q3/Q4)
- `suitesmith/build.py` — instance construction: portfolio selection,
  distinguishability enforcement, witness inputs (SPEC Q4 hard rules)
- `suitesmith/harness.py` — sandboxed pytest grader and the Q6 reward
- `suitesmith/dataset.py` — the two fixed prompt templates (white/black box)
  and train/eval row building with the Q5 two-level holdout
- `suitesmith/env.py` — verifiers wiring: `load_environment()` →
  `vf.SingleTurnEnv`; zero-weight rubric funcs carry the §7 instrumentation
- `tests/` — the environment's own test suite
- `scripts/show_instance.py` — print one instance (prompt + mutants) to eyeball

## Run

```bash
.venv/bin/python -m pytest tests/ -q          # self-test
.venv/bin/python scripts/show_instance.py filter_agg 3 black
vf-eval suitesmith -n 50 -r 8 ...             # Q8 calibration (model/endpoint flags per verifiers docs)
```

Set `SUITESMITH_LOG=/path/to/log.jsonl` to append one JSON line per scored
suite (gate, kills by class, suite size) — the S3 net.

Solo project, Dominic Colligan, August 2026.
