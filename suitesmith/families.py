"""Template families (taskset internals).

TODO(Dom): four families (top_k, filter_agg, dedup, window eval-only).
The acceptance battery expects this module to export:
  - FAMILIES, TRAINED, EVAL_ONLY registries
  - load_fn(source, fn_name), run_source(...)
Each family needs: seeded params (train/eval pool split), render,
fully-determining spec, mutation menu (10-class taxonomy, subtlety),
battery, rename-only twin. Design authority: SPEC.md Q3/Q4/Q5-ext.
"""
