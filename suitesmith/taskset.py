"""Task construction (prime-envs: the taskset side).

Folded 26 Aug: families.py merged in — one module owns everything
upstream of verification, matching the prime-envs package shape
(__init__ / taskset / verify). Rebuild targets the verifiers.v1 API
(Dom's call, 26 Aug eve — venv rebuilt on py3.12, verifiers from
GitHub main): SuitesmithData(vf.TaskData) / SuitesmithTask(vf.Task,
@vf.reward delegating to verify.py) / SuitesmithTaskset(vf.Taskset)
wrap a framework-free core of plain functions. The battery below
tests that core and is dialect-independent.

TODO(Dom): everything. The acceptance battery in tests/ expects this
module to export BOTH layers:

Instance assembly (SPEC Q1/Q4/Q5-ext — was taskset.py):
  - PORTFOLIOS      subtlety mixes ("easy"/"hard")
  - build_instance(family, seed, visibility, mix, split="train")
  - build_dataset(...)  -> (train_rows, eval_rows) with seen/vocab/template tiers
  - render_prompt(...)
  Witness enforcement, prompt templates, train/eval tiers.

Template families (SPEC Q3/Q4 — was families.py):
  - FAMILIES, TRAINED, EVAL_ONLY registries
  - load_fn(source, fn_name), run_source(...)
  Four families (top_k, filter_agg, dedup, window eval-only). Each
  needs: seeded params (train/eval pool split), render, fully-
  determining spec, mutation menu (10-class taxonomy, subtlety),
  battery, rename-only twin.

Design authority: README.md §Design specification (formerly SPEC.md).
"""
