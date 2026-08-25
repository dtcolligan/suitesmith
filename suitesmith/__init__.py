"""suitesmith — RL environment: train a model to write test suites.

Layout follows the prime-envs convention:
  taskset.py  — task/dataset construction (families.py holds its internals)
  verify.py   — verification: runner (execution) + grader (reward policy)

TODO(Dom): expose load_environment() -> vf.SingleTurnEnv here, wiring
taskset rows to the verify rubric. Until then this package is a shell.
"""


def load_environment(**kwargs):
    raise NotImplementedError("rebuild in progress: wire SingleTurnEnv here")
