"""suitesmith — RL environment: train a model to write test suites.

Layout follows the prime-envs convention exactly (cf. environments/code/
humaneval): __init__.py / taskset.py / verify.py.
  taskset.py  — task/dataset construction (family templates included)
  verify.py   — verification: runner (execution) + grader (reward policy)

TODO(Dom): expose load_environment() -> vf.SingleTurnEnv here, wiring
taskset rows to the verify rubric. Until then this package is a shell.
"""


def load_environment(**kwargs):
    raise NotImplementedError("rebuild in progress: wire SingleTurnEnv here")
