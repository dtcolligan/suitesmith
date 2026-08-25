"""Verification (prime-envs: the verify side).

Internal split, Dom's design decision 25 Aug:
  runner — sandboxed execution only: run a suite against a target,
           return raw outcomes (which target, return code, timing)
  grader — reward policy only: gates (ref/twin/malformed/no_tests/
           timeout) + reward = kills/N (SPEC Q6, incl. the named
           calibration fallback) + hack-flag instrumentation

TODO(Dom): everything. The acceptance battery expects:
  - extract_suite(text)
  - score_suite(suite_src, info, timeout=...) -> result with
    .gate/.reward/.killed/.flags/...
"""
