"""Calibration report: why a model is not saturating suitesmith.

Re-grades every rollout in a run directory with per-test detail and
prints, per family: mean reward, gate counts, which mutants survive a
passing suite (by mutation name), and for ref_failed suites the exact
assertion that the correct implementation failed. Then the tasks where
failures concentrate, with spec + failing test side by side, because a
task failing 8/8 is a spec problem and a task failing 1/8 is the model.

    .venv/bin/python scripts/calib_report.py outputs/<run-dir> [--top 4] [--workers 4] [--json out.json]

First used 1 Sep 2026 on the gpt-4o-mini 50x8 run: found the
"ordered by the higher <field>" misread and the k <= 0 hole.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suitesmith.verify import extract_suite, score_suite  # noqa: E402

# (mclass, subtlety) -> mutation name, per family. Mirrors each family's
# mutations() menu in taskset.py; MutantSpec carries class + subtlety but
# not the menu name, and the pair is unique within a family except the
# two window rel_swaps.
NAMES = {
    "top_k": {(7, 1): "flip_primary", (10, 1): "wrong_field", (2, 2): "off_by_one",
              (6, 3): "drop_tie", (7, 3): "flip_tie"},
    "filter_agg": {(4, 1): "negate", (3, 1): "field_swap", (1, 2): "rel_swap",
                   (5, 2): "drop_conjunct", (8, 2): "agg_swap", (9, 3): "bad_empty"},
    "dedup": {(4, 1): "negate", (10, 1): "wrong_key", (5, 1): "drop_membership",
              (7, 2): "flip_precedence", (6, 3): "drop_precedence"},
    "window": {(4, 1): "negate", (5, 2): "drop_hi", (6, 2): "sort_output",
               (1, 3): "rel_swap_lo/hi"},
}


def ref_failure_lines(suite: str, reference: str) -> list[str]:
    """Run the suite against the reference under real pytest, return the
    one-line-per-failure summary (file:line: AssertionError: ...)."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "solution.py").write_text(reference)
        Path(tmp, "suite.py").write_text(suite)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-rA", "--tb=line",
             "-p", "no:cacheprovider", "suite.py"],
            cwd=tmp, capture_output=True, text=True, timeout=120,
        )
    lines = [l for l in proc.stdout.splitlines() if re.match(r"^.*suite\.py:\d+: ", l)]
    return [re.sub(r"^.*suite\.py:", "suite.py:", l) for l in lines] or \
           [l for l in proc.stdout.splitlines() if l.startswith(("FAILED", "ERROR"))]


def regrade(job):
    name, data, reply = job
    suite = extract_suite(reply)
    res = score_suite(suite, data)
    out = {"name": name, "family": data["family"], "visibility": data["visibility"],
           "gate": res.gate, "reward": res.reward, "n_tests": res.n_tests}
    if res.gate == "pass":
        names = NAMES.get(data["family"], {})
        out["survivors"] = [
            names.get((m["mclass"], m["subtlety"]), f"c{m['mclass']}s{m['subtlety']}")
            for m, r in zip(data["mutants"], res.detail["mutants"])
            if r["status"] == "ok" and not r["failed"]
        ]
    elif res.gate == "ref_failed":
        out["lines"] = ref_failure_lines(suite, data["reference"])
        out["suite"] = suite
    return out


def load_jobs(traces: Path):
    tasks, jobs = {}, []
    for line in traces.open():
        t = json.loads(line)
        data = t["task"]["data"]
        tasks[data["name"]] = data
        for tr in t["traces"]:
            sampled = [n["message"] for n in tr["nodes"] if n.get("sampled")]
            jobs.append((data["name"], data, sampled[-1]["content"] if sampled else ""))
    return tasks, jobs


def failing_test(suite: str, line: str) -> str:
    m = re.search(r"suite\.py:(\d+):", line)
    if not m:
        return ""
    ln = int(m.group(1))
    src = suite.splitlines()
    starts = [i for i in range(min(ln, len(src))) if src[i].startswith("def ")]
    return "\n".join(src[starts[-1]:ln]) if starts else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory (contains traces.jsonl) or a traces.jsonl path")
    ap.add_argument("--top", type=int, default=4, help="worst tasks to print in full")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json", help="also dump per-rollout results here")
    a = ap.parse_args()

    traces = Path(a.run)
    if traces.is_dir():
        traces = traces / "traces.jsonl"
    tasks, jobs = load_jobs(traces)
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(regrade, jobs))
    if a.json:
        json.dump(results, open(a.json, "w"))

    n = len(results)
    lost = 1.0 - sum(r["reward"] for r in results) / n
    gate_loss = sum(1 for r in results if r["gate"] != "pass") / n
    print(f"{n} rollouts · mean {1 - lost:.3f} · shortfall {lost:.3f} = "
          f"gates {gate_loss:.3f} + surviving mutants {lost - gate_loss:.3f}")

    by_fam = collections.defaultdict(list)
    for r in results:
        by_fam[r["family"]].append(r)
    for fam, rs in sorted(by_fam.items()):
        gates = collections.Counter(r["gate"] for r in rs)
        surv = collections.Counter(s for r in rs if r["gate"] == "pass" for s in r["survivors"])
        mean = sum(r["reward"] for r in rs) / len(rs)
        print(f"\n{fam:11s} n={len(rs):3d} mean={mean:.3f} {dict(gates)}")
        print(f"            survivors among {gates['pass']} passing suites: "
              f"{ {k: v for k, v in surv.most_common()} or 'none'}")
        for vis in ("white", "black"):
            vs = [r["reward"] for r in rs if r["visibility"] == vis]
            if vs:
                print(f"            {vis}: n={len(vs)} mean={sum(vs) / len(vs):.3f}")

    per_task = collections.defaultdict(lambda: [0, 0])
    for r in results:
        per_task[r["name"]][1] += 1
        per_task[r["name"]][0] += r["gate"] == "ref_failed"
    ranked = sorted(((f, t, name) for name, (f, t) in per_task.items() if f), reverse=True)
    print("\nref_failed concentration (failed/rollouts); 8/8 smells like the spec, 1/8 like the model:")
    for f, t, name in ranked:
        print(f"  {name:30s} {f}/{t}")

    for f, t, name in ranked[: a.top]:
        d = tasks[name]
        ex = next(r for r in results if r["name"] == name and r["gate"] == "ref_failed")
        print(f"\n{'#' * 70}\n{name}  ({f}/{t} ref_failed)\n--- SPEC\n{d['spec'].rstrip()}\n--- REFERENCE\n{d['reference'].rstrip()}")
        for line in ex.get("lines", [])[:2]:
            print(f"--- FAILS ON REFERENCE: {line[-220:]}")
            body = failing_test(ex["suite"], line)
            if body:
                print(body)


if __name__ == "__main__":
    main()
