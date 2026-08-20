# SPEC — v2 (design re-based 20 Aug, Dom's frame)

> v1's weak-reward/planted-gap design is RETIRED (see §9). This project
> trains a capability. Fill the **Q** items by hand (train session, Fri
> 21 Aug); *italics* are fixed constraints.

## 0. Objective (unchanged)

S2 committed: **designed and built an RL environment with a verifiable
grader, and trained models against it — done before 14 Sep.** S1 (one
training step printing a reward) by 31 Aug — GPU day Mon 24, existing-env
fallback held. S3 = *unplanned* exploitation caught by instrumentation —
designed-for via logging, never promised, never staged.

## 1. The capability

**Train a small model to write test suites.** Given a specification (and
possibly an implementation), the model writes tests; good tests are ones
that **pass the correct implementation and kill mutants** (subtly broken
variants). The trained capability is the sub-specialty itself: writing
verifiers that see more.

## 2. Task instance

One instance = a generated function: spec (signature + docstring +
examples) + **reference implementation** (correct by construction) +
**N mutants** (broken variants from a mutation engine). The model
outputs a test suite (pytest-style or assert-based — Q).

**Q1. ✅ DECIDED (20 Aug, Dom's design): visibility is a per-instance
DIFFICULTY KNOB, not a bet.** The dataset mixes white-box instances
(spec + reference shown — isolates adversarial input selection; easier)
and black-box instances (spec only — adds spec-to-oracle derivation;
harder). Mix ratio set empirically at calibration; **static in week
one** (curriculum ramp shelved by name). Safety property: GRPO
advantages are within-instance, so a too-hard black-box group is
wasted compute, never poison — the white-box fraction carries signal
regardless. Riders: two fixed prompt templates, never blended; **all
metrics split by visibility mode** (reward, kill rate, and
reference-fail rate per mode — black-box failures often die at the
reference gate via wrong expected values, a different failure than bad
input selection); the white/black differential learning curve is a
free run-report finding.

**Q2. ✅ DECIDED (20 Aug, Dom): the model writes PYTEST functions.**
The harness provides a fixed import surface (e.g. `from solution import
<fn>`); it swaps `solution.py` between the reference and each mutant
and reruns the emitted suite. Per-test independence and per-test kill
attribution come free with pytest. Malformed output (fails to parse /
collect) → reward 0, logged. Suites with no assertions pass everything
and kill nothing → reward ≈ 0 by construction (self-punishing).

## 3. Generator

**Q3.** Function templates: reuse v1's data-manipulation template
library (filter/aggregate/transform with sampled parameters) — decide
the initial set (~how many templates, what parameter ranges).

**Q4. The mutation engine — the design centre.** Mutation taxonomy:
operator swaps (`<` vs `<=`), boundary shifts (off-by-one), negated
conditions, dropped clauses (e.g., a tie-break rule silently removed),
wrong default, order swap. How many mutants per instance; how is
subtlety tiered (difficulty knob: crude mutants easy to kill, subtle
mutants need pointed tests)?

**Q5.** Seeding + disjoint train/eval splits (contamination-free by
construction — state the mechanism).

## 4. Grader and reward

*Deterministic, execution-based, sandboxed (subprocess, timeout, no
network), no LLM judge.*

- Run the suite against the **reference: must pass** (a suite that
  fails the correct code is worth 0 — correctness of the tests comes
  first).
- Run against each mutant: a mutant is **killed** if ≥1 test fails.
- **Q6. ✅ DECIDED (20 Aug, Dom): `reward = 0 if the reference fails
  ANY test, else mutants_killed / N`.** Strict all-pass gate is Dom's
  call; noted sharpness risk for small models (one wrong expected
  value zeroes the instance). **Named fallback if Sunday's calibration
  shows reward starving: drop reference-failing tests, score kills on
  the surviving subset with a penalty factor** — a change to make at
  calibration only, not before.

**Q7. Degenerate-suite guards** (the natural gaming surface — this is
where unplanned S3 material lives): empty suites, tests with no
assertions, enormous suites, flaky tests (nondeterminism), asserting on
internals (white-box only). Which guards are hard rules vs logged-only?
*Deliberately do NOT over-guard: log first, patch when exploited — that
is the honest break-and-fix loop.*

## 5. Calibration

*Pre-training gate: per-instance rewards must be MIXED (the 7 Aug
variance rule: all-zero or all-max groups give GRPO no signal).*
**Q8.** Calibration protocol: vf-eval the base model on ~50 instances;
tune template complexity + mutant subtlety until the reward
distribution has spread.

## 6. Training

*0.5–3B open-weight + LoRA; GRPO on the verifiers stack; ~$50 gate
budget.* **Q9.** Model pick, rollouts per step, group size, first-run
step count.

## 7. Instrumentation

Per step: mean reward; reference-pass rate; kill rate **by mutation
class** (which flaw types the model learns to catch — the interesting
curve); suite-size and assertion-count distributions (degeneracy
watch); checkpoint transcripts read by hand (the S3 net).

## 8. Milestones (unchanged)

Fri spec → Sat–Sun build + calibration → **Mon 24 GPU/S1** → Tue 25
protected → 26–31 runs + report v1 → ≤7 Sep repo public, **S2
declared** → 14 Sep SPAR starts.

## 9. Retired: v1's planted-gap design (for the record)

v1 rewarded via a deliberately weak suite and measured drift against a
hidden oracle. Retired 20 Aug: Dom's call — he wants to train a
capability, not stage a failure ("I would rather make mistakes on the
way, giving me lessons, rather than force them"). What it got right is
kept: templates, spec-quality tiers (available as a difficulty knob),
differential-vs-reference machinery (now inside mutation generation),
the S2/S3 boundary. The drift experiment survives only as a possible
later arm — a good grader can always be weakened deliberately;
week one never does. Also shelved by name: **structured extraction**
(option 5) as a candidate second environment.

## 10. Out of scope (unchanged)

Hub upload · verifiers-port · paper · multi-turn · big models · GAB ·
insolvency. **Rename pending: "covgap" named the old design — Dom picks
the new name or keeps it (killing mutants IS coverage, so it half
survives).**
