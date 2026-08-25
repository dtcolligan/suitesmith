"""Task construction (prime-envs: the taskset side).

TODO(Dom): everything below. The acceptance battery in tests/ expects
this module to export:
  - PORTFOLIOS      subtlety mixes ("easy"/"hard")
  - build_instance(family, seed, visibility, mix, split="train")
  - build_dataset(...)  -> (train_rows, eval_rows) with seen/vocab/template tiers

Design authority: SPEC.md (Q1–Q5-ext). Family templates live in
families.py and are imported here.
"""
