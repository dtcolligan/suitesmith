"""suitesmith: train a small model to write test suites.

Reward: suite must pass the reference implementation, scored by how many
mutants it kills (SPEC Q6). verifiers environment entry point below; the
import is lazy so the generator, harness and tests work without verifiers
installed.
"""


def load_environment(**kwargs):
    from suitesmith.env import load_environment as _load

    return _load(**kwargs)
