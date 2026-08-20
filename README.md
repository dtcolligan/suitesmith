# covgap (name under review — the project re-based 20 Aug)

An RL environment that **trains a small model to write test suites**.
Each task is a generated function: a specification, a correct reference
implementation, and a set of subtly broken mutants. The model writes
tests; reward = the tests **pass the reference and kill the mutants**.
The trained capability is verifier-writing itself.

Status: spec v2 in progress. Nothing trained yet; this README will
never claim otherwise.

- `SPEC.md` — the design document (single source of truth; v1's
  planted-gap design is retired, recorded in §9)
- `src/` — generator, mutation engine, grader
- `tests/` — tests for the graders and generator

Solo project, Dominic Colligan, August 2026.
