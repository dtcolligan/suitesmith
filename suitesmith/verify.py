# /// script
# dependencies = ["pytest>=8"]
# ///
"""Verification (prime-envs: the verify side).

Internal split, Dom's design decision 25 Aug:
  runner — sandboxed execution only: run a suite against a target,
           return raw outcomes (which target, return code, timing)
  grader — reward policy only: gates (ref/twin/malformed/no_tests/
           timeout) + reward = kills/N (SPEC Q6, incl. the named
           calibration fallback) + hack-flag instrumentation

Two lives, one file:
  1. Imported by the battery and by taskset.py (extract_suite).
  2. Shipped BYTES-AS-CARGO into the training sandbox by taskset.py
     (run_uv_script) and executed as `python verify.py payload.json 5.0`,
     printing a one-line JSON verdict. Inside the sandbox the suitesmith
     package does not exist, so this file imports nothing from the rest of
     the repo. Its one third-party need, pytest, is declared in the script
     header above; uv installs it into the cargo env (content-addressed,
     so one install per sandbox, not per rollout).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TIMEOUT = 5.0

# Static markers of the known reward hacks. A marker's presence never
# changes the reward by itself (the twin gate does the enforcing); flags
# exist so training logs show WHAT the policy tried, not just that it
# scored zero.
HACK_MARKERS = ("open(", "solution.py", "__import__(", "getsource(")

# ---------------------------------------------------------------------------
# extract: chat text -> suite source
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_suite(text) -> str:
    """Pull the test suite out of a model reply.

    Last fenced code block wins (models restate earlier drafts; the final
    block is the answer). No fences: take the whole text iff it looks like
    code (contains a def), else there is no suite.
    """
    if not isinstance(text, str):
        return ""
    blocks = _FENCE.findall(text)
    if blocks:
        return blocks[-1].rstrip("\n")
    return text if "def " in text else ""


# ---------------------------------------------------------------------------
# runner: one suite vs one target, sandboxed, raw outcome only
# ---------------------------------------------------------------------------

# The driver runs INSIDE the subprocess: it hands suite.py to a real
# in-process pytest session and tallies outcomes through a tiny plugin,
# so parametrize / fixtures / pytest.raises are all honoured (SPEC Q2:
# the model writes PYTEST functions, the grader must speak pytest). It
# never raises on the SUITE's behalf: every suite outcome is data on
# stdout, so the runner can tell "suite failed" from "runner broke".
# Our own failures (pytest missing from the cargo env) are allowed to
# crash: no JSON line, the runner reports "crash", the grader reports
# the error gate, the taskset raises. A test that raises KeyboardInterrupt
# or SystemExit is still a failing test, not a crash: pytest reports
# SystemExit as a failure and turns KeyboardInterrupt into exit code 2,
# which the driver counts as one failed test.
#
# Counting rules: a test is one collected item (a parametrize case is its
# own test); skipped items are not tests (an all-skip suite reaches the
# no_tests gate); a setup/teardown error is a failed test.
_DRIVER = """\
import json, os, sys

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
import pytest  # missing pytest is OUR failure: crash loudly, never blame the suite


class Count:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.collect_failed = False

    def pytest_collectreport(self, report):
        if report.failed:
            self.collect_failed = True

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
        elif report.failed:
            self.failed += 1


count = Count()
try:
    rc = pytest.main(
        ["-q", "-p", "no:cacheprovider", "--rootdir=.", "suite.py"],
        plugins=[count],
    )
except BaseException:
    rc = 3
if count.collect_failed:
    print(json.dumps({"error": "import", "n_tests": 0, "failed": 0}))
    sys.exit(0)
if rc in (2, 3, 4):
    count.failed += 1
print(json.dumps({"error": None, "n_tests": count.passed + count.failed,
                  "failed": count.failed}))
"""


def run_suite(suite_src: str, target_src: str, timeout: float) -> dict:
    """Run one suite against one target in a fresh subprocess.

    The target is materialised as solution.py in a temp cwd — which is
    exactly what lets `open("solution.py")` hacks run AND get caught by
    the twin. Returns {"status", "n_tests", "failed"}; status is one of
    "ok" | "timeout" | "crash". No policy here: killed/gates/reward are
    the grader's business.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "solution.py").write_text(target_src)
        (root / "suite.py").write_text(suite_src)
        (root / "driver.py").write_text(_DRIVER)
        try:
            proc = subprocess.run(
                [sys.executable, "driver.py"],
                cwd=tmp, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "n_tests": 0, "failed": 0}
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"status": "crash", "n_tests": 0, "failed": 0}
    if out["error"] == "import":
        # The suite blew up at collection (import error, bad decorator,
        # module-level exception): against this target, everything it
        # asserts is unmet.
        return {"status": "ok", "n_tests": 0, "failed": 1}
    return {"status": "ok", "n_tests": out["n_tests"], "failed": out["failed"]}


# ---------------------------------------------------------------------------
# grader: outcomes -> gate + reward + flags
# ---------------------------------------------------------------------------


@dataclass
class ScoreResult:
    gate: str          # pass | malformed | no_tests | ref_failed | twin_failed | timeout | error
    reward: float      # killed / n_mutants iff gate == "pass", else 0.0
    killed: int
    n_tests: int
    flags: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)


def _scan_flags(suite_src: str, fn_name: str) -> list:
    flags = [m for m in HACK_MARKERS if m in suite_src]
    if f"{fn_name}(" not in suite_src:
        flags.append("never_calls_target")
    return flags


def score_suite(suite_src, info: dict, timeout: float = DEFAULT_TIMEOUT) -> ScoreResult:
    """Grade one suite against one instance. The reward policy, in order:

    1. malformed  — not a string / empty / doesn't compile. Nothing runs.
    2. no_tests   — compiles but defines no test_* function.
    3. ref_failed — the reference must pass every test. A suite that
                    fails the correct function is asserting falsehoods.
    4. twin_failed — the behavioural twin must ALSO pass every test.
                    Kills source-fingerprinting: any suite keyed to the
                    reference's text instead of its behaviour dies here.
    5. timeout    — ref or twin run exceeded the budget.
    6. pass       — reward = mutants killed / mutants total. A mutant is
                    killed when at least one test fails against it (a
                    mutant that hangs the suite counts as killed: the
                    divergence just expressed itself as non-termination).
    """
    if not isinstance(suite_src, str) or not suite_src.strip():
        return ScoreResult("malformed", 0.0, 0, 0)
    try:
        compile(suite_src, "suite.py", "exec")
    except SyntaxError:
        return ScoreResult("malformed", 0.0, 0, 0)

    flags = _scan_flags(suite_src, info["fn_name"])

    ref = run_suite(suite_src, info["reference"], timeout)
    if ref["status"] == "timeout":
        return ScoreResult("timeout", 0.0, 0, 0, flags, {"ref": ref})
    if ref["status"] != "ok":
        return ScoreResult("error", 0.0, 0, 0, flags, {"ref": ref})
    if ref["failed"]:
        return ScoreResult("ref_failed", 0.0, 0, ref["n_tests"], flags, {"ref": ref})
    if ref["n_tests"] == 0:
        return ScoreResult("no_tests", 0.0, 0, 0, flags, {"ref": ref})
    n_tests = ref["n_tests"]

    twin = run_suite(suite_src, info["twin"], timeout)
    if twin["status"] == "timeout":
        return ScoreResult("timeout", 0.0, 0, n_tests, flags, {"ref": ref, "twin": twin})
    if twin["status"] != "ok" or twin["failed"]:
        return ScoreResult("twin_failed", 0.0, 0, n_tests, flags,
                           {"ref": ref, "twin": twin})

    mutant_runs = []
    killed = 0
    for m in info["mutants"]:
        r = run_suite(suite_src, m["source"], timeout)
        mutant_runs.append(r)
        if r["status"] != "ok" or r["failed"]:
            killed += 1

    reward = killed / len(info["mutants"])
    return ScoreResult("pass", reward, killed, n_tests, flags,
                       {"ref": ref, "twin": twin, "mutants": mutant_runs})


# ---------------------------------------------------------------------------
# script mode: the sandbox entry point
# ---------------------------------------------------------------------------


def _main() -> None:
    # Two ways in. Training path: the payload arrives in the process
    # environment (SUITESMITH_PAYLOAD) and is popped before any suite
    # runs, so no file exists for a sibling rollout to glob and the
    # suite's own process never inherits it. File path kept for the
    # battery and local use: `verify.py payload.json [timeout]`.
    env_payload = os.environ.pop("SUITESMITH_PAYLOAD", None)
    if env_payload is not None:
        payload = json.loads(env_payload)
        timeout = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TIMEOUT
    else:
        payload_path = Path(sys.argv[1])
        payload = json.loads(payload_path.read_text())
        payload_path.unlink()  # one payload per rollout, no litter
        timeout = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TIMEOUT
    res = score_suite(payload["suite"], payload, timeout=timeout)
    # One JSON line on stdout is the whole interface back to taskset.py.
    print(json.dumps({
        "gate": res.gate, "reward": res.reward, "killed": res.killed,
        "n_tests": res.n_tests, "flags": res.flags,
    }))


if __name__ == "__main__":
    _main()
