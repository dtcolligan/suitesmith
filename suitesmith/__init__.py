"""suitesmith — RL environment: train a model to write test suites.

Layout follows the prime-envs convention exactly (cf. environments/code/
humaneval): __init__.py / taskset.py / verify.py, on the verifiers.v1 API
(verifiers >= 0.2.0, installed from GitHub main; Python 3.12 venv).

TODO(Dom): once taskset.py defines them, re-export here per the humaneval
pattern:

    from suitesmith.taskset import SuitesmithTaskset
    __all__ = ["SuitesmithTaskset"]
"""
