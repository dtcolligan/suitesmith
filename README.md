# suitesmith

An RL environment, written on the [verifiers](https://github.com/willccbb/verifiers)
framework, that **trains a small model to write test suites**. Each task is a
generated function: a specification, a correct reference implementation, and
five subtly broken mutants. The model writes pytest tests; reward = the tests
**pass the reference and kill the mutants** (`0` if the reference fails any
test, else `killed / 5`). The trained capability is verifier-writing itself.

Renamed from `covgap` 23 Aug 2026: the smith forges the suite; killing
mutants is how the forge is graded.

Status: **REBUILD IN PROGRESS (from 25 Aug 2026).** The operator-written
v1 lives on the `operator-build` branch; `main` holds Dom's own rebuild.
The 39 tests in `tests/` are the held-out acceptance battery: the rebuild
is done when they pass against Dom's code. Nothing trained yet. This
README will never claim otherwise.

## Layout (prime-envs convention: taskset / verify)

- `SPEC.md` — the design document (single source of truth; v1's planted-gap
  design is retired, recorded in §9)
- `suitesmith/taskset.py` — task construction: instance assembly, portfolio
  selection, witness enforcement, prompt templates, train/eval tiers
  (SPEC Q1/Q4/Q5-ext)
- `suitesmith/families.py` — template families (taskset internals): spec +
  reference generators, mutation menus, twins (SPEC Q3/Q4)
- `suitesmith/verify.py` — verification: runner (sandboxed execution) +
  grader (gates and the Q6 reward policy), kept separate for
  interpretability
- `suitesmith/__init__.py` — verifiers wiring: `load_environment()` →
  `vf.SingleTurnEnv`; zero-weight rubric funcs carry the §7 instrumentation
- `tests/` — the acceptance battery (operator-written, held out)
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
