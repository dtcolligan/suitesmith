"""Grader: sandboxed pytest execution and the SPEC Q6 reward.

reward = 0 if the reference fails ANY test, else mutants_killed / N.
Strict all-pass gate is Dom's call (Q6); the soften-at-calibration fallback is
NOT implemented — a change to make at calibration only, not before.

Sandbox is subprocess + timeout + a sitecustomize that disables sockets.
Log-first philosophy (Q7): degenerate suites are measured, not blocked —
an assertion-free suite passes the gate and kills nothing, so it earns 0 by
construction.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)

# A plain-function stub breaks ssl's `class SSLSocket(socket)` at import, and
# that takes pytest down with it — the block must stay subclassable.
SITECUSTOMIZE = (
    "import socket\n"
    "class _BlockedSocket(socket.socket):\n"
    "    def __init__(self, *a, **k):\n"
    "        raise RuntimeError('network disabled in suitesmith sandbox')\n"
    "socket.socket = _BlockedSocket\n"
)

PYTEST_ARGS = ["-q", "--tb=no", "-x", "-p", "no:cacheprovider"]


def extract_suite(text: str) -> str:
    """Parser extract_fn: last fenced code block, else raw text if test-like."""
    if not isinstance(text, str):
        return ""
    blocks = FENCE_RE.findall(text)
    if blocks:
        return blocks[-1].strip()
    if "def test" in text:
        return text.strip()
    return ""


@dataclass
class ScoreResult:
    gate: str  # pass | ref_failed | no_tests | malformed | timeout
    reward: float
    killed: int
    n_mutants: int
    n_tests: int
    n_asserts: int
    per_mutant: list = field(default_factory=list)


def _run_pytest(workdir: Path, timeout: float) -> int:
    cmd = [sys.executable, "-m", "pytest", "test_suite.py", *PYTEST_ARGS]
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(workdir),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(workdir),
    }
    try:
        proc = subprocess.run(
            cmd, cwd=workdir, env=env, capture_output=True, timeout=timeout
        )
        return proc.returncode
    except subprocess.TimeoutExpired:
        return -1


def score_suite(suite_src: str | None, info: dict, timeout: float = 15.0) -> ScoreResult:
    mutants = list(info["mutants"])
    n = len(mutants)
    n_tests = len(re.findall(r"^\s*def test", suite_src or "", re.M))
    n_asserts = (suite_src or "").count("assert")
    result = ScoreResult("malformed", 0.0, 0, n, n_tests, n_asserts)

    if suite_src and n_tests > 0:
        with tempfile.TemporaryDirectory(prefix="suitesmith-") as tmp:
            workdir = Path(tmp)
            (workdir / "sitecustomize.py").write_text(SITECUSTOMIZE)
            (workdir / "test_suite.py").write_text(suite_src)
            (workdir / "solution.py").write_text(info["reference"])
            rc = _run_pytest(workdir, timeout)
            if rc == 0:
                result.gate = "pass"
                for m in mutants:
                    (workdir / "solution.py").write_text(m["source"])
                    mrc = _run_pytest(workdir, timeout)
                    killed = mrc != 0
                    result.per_mutant.append(
                        {
                            "mclass": m["mclass"],
                            "name": m["name"],
                            "subtlety": m["subtlety"],
                            "killed": killed,
                            "timeout": mrc == -1,
                        }
                    )
                result.killed = sum(1 for p in result.per_mutant if p["killed"])
                result.reward = result.killed / n if n else 0.0
            elif rc == 1:
                result.gate = "ref_failed"
            elif rc == 5:
                result.gate = "no_tests"
            elif rc == -1:
                result.gate = "timeout"

    log_path = os.environ.get("SUITESMITH_LOG")
    if log_path:
        entry = {
            "t": time.time(),
            "family": info.get("family"),
            "seed": info.get("seed"),
            "visibility": info.get("visibility"),
            **asdict(result),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    return result
