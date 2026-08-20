# covgap (working name — rename freely)

An RL environment for code tasks where **reward comes from a weak test
suite and ground truth from a held-out strong oracle**. The object of
study is the gap between them under training pressure: what a policy
learns when the grader can't see everything.

Status: spec in progress. Nothing here is trained yet; this README will
never claim otherwise.

- `SPEC.md` — the design document (single source of truth)
- `src/` — environment, generator, graders
- `tests/` — tests for the graders and generator

Solo project, Dominic Colligan, August 2026.
