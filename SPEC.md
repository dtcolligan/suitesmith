# SPEC — v2 (design re-based 20 Aug, Dom's frame; built 23 Aug as a verifiers env)

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

**Q3. ✅ DECIDED (20 Aug, approved by Dom): eight template families,
four built first.** (1) top-k with tie-break · (2) filter-then-aggregate
· (3) group-by aggregation · (4) dedup with precedence rule · (5)
threshold windowing (inclusive/exclusive boundary semantics) · (6)
merge with conflict rule · (7) running/cumulative with reset · (8)
rounding/formatting rule. **Build order: 1, 2, 4, 5 first**; 3/6/7/8
only if the weekend runs fast. During build, verify a mutation-class ×
template coverage matrix across the training templates (every class
must be hostable by at least one trained template).

**Q4. ✅ DECIDED (20 Aug, approved by Dom): the mutation taxonomy.**
Ten classes with subtlety ratings (1 = dies to almost any test, 3 =
dies only to spec-reading tests):
1. Relational swap (`<`↔`<=`) — 2–3 (boundary inputs only)
2. Off-by-one (`[:k]`→`[:k-1]`) — 2
3. Arithmetic swap (`+`→`-`, wrong field summed) — 1–2
4. Condition negation — 1 (crude; kept to anchor the easy end)
5. Dropped conjunct of a compound filter — 2–3
6. **Dropped step (tie-break / dedup / final sort removed) — 3, the
   flagship class** (test must construct the triggering situation)
7. Direction flip (sort 1; tie-break direction 3)
8. Aggregation swap (sum→max 1–2; count→count-distinct 2–3)
9. Default/empty error (wrong init; empty-input behaviour) — 2–3
10. Wrong field/key — 1 (crude; kept)

Hard rules: **(a) behavioural distinguishability enforced at generation
time** — every mutant must differ from the reference on ≥1 input,
verified by running both on an input battery incl. class-targeted
probes; equivalents discarded and regenerated; each survivor stores a
**witness input**. **(b) One mutation per mutant** (clean attribution).
**(c) N = 5 mutants per instance**, drawn across classes with the
subtlety mix as the second difficulty dial (easy instance ≈ {1,1,2,2,3};
hard ≈ {2,2,3,3,3} — exact portfolios tuned at calibration).

**Q5. ✅ DECIDED (20 Aug): two-level holdout.** (a) **Template 5
(threshold windowing) is eval-only** — never trained on, so eval
measures template-level generalisation (and it's the boundary-heavy
family, a deliberate stress test of the relational-swap skill). (b)
Within trained templates (1, 2, 4), disjoint seed ranges for train vs
eval instances; every instance fully derived from (template, seed).

**Q5-ext. ✅ EXTENDED 23 Aug (Dom: "one thing to fix in the taskset").**
Seed-disjoint is NOT instance-disjoint when name pools are small — an
eval seed can redraw a train seed's exact param combo, so (b) alone
cannot rule out pool memorisation. Fix: pools enlarged (~2×) and every
pool split train/eval (field names, function names). Eval is now
**three tiers**, labelled in the task column: `seen` (trained
families, unseen seeds, train vocab), `vocab` (trained families,
held-out vocab), `template` (window). The memorisation signature is a
seen→vocab drop; the generalisation claim lives at `template`. Every
instance now derives from (template, seed, split).

## 4. Grader and reward

*Deterministic, execution-based, sandboxed (subprocess, timeout, no
network), no LLM judge.*

- Run the suite against the **reference: must pass** (a suite that
  fails the correct code is worth 0 — correctness of the tests comes
  first).
- **Twin gate (Dom, 23 Aug: "one thing to fix in the grader").** The
  suite must ALSO pass a rename-only **twin** of the reference (same
  behaviour, different text; behavioural identity verified on the
  battery at generation time). This enforces "assert on behaviour
  only": a source-fingerprinting suite (e.g. `open("solution.py")`
  compared against the white-box prompt's reference text) passes the
  reference, kills all mutants, and previously earned 1.0 — now it
  fails the twin and earns 0. Told→enforced. Residual hole, accepted:
  a diff-size heuristic over source (small diff = mutant, big diff =
  twin) survives; exotic for a 1.5B policy, hack-flags watch for it,
  escalation if it fires = sourceless .pyc distribution.
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

**Q7 status 23 Aug:** "asserting on internals" is now ENFORCED (the §4
twin gate) — a known-in-advance hole left open would be staging by
negligence, which is the v1 sin; log-first applies to unknown
unknowns. Everything else stays logged-only: `hack_flags` per suite
(source-inspection markers + `never_calls_target`), surfaced as a
zero-weight rubric metric and in the SUITESMITH_LOG jsonl. Free
enforcement noted: flaky/nondeterministic tests now have three chances
to fail the gate (reference + twin + collection), not one.

## 5. Calibration

*Pre-training gate: per-instance rewards must be MIXED (the 7 Aug
variance rule: all-zero or all-max groups give GRPO no signal).*
**Q8. PROPOSED (operator, 23 Aug — Dom ratifies at calibration):**
vf-eval the base model on ~50 instances, 8 rollouts each (= training
group size), white_frac 0.7, easy subtlety mix {1,1,2,2,3} — these are
the code defaults in `env.py`. Pass gate: ≥60% of instances show
within-group reward spread AND mean reward lands roughly in 0.2–0.7,
checked per visibility mode. If starved: raise white-box fraction, ease
subtlety, enrich docstring examples — the Q6 fallback stays last
resort. Watch black-box reference-fail rate; if most black-box zeros
die at the reference gate, lower the black-box fraction. Noted from the
smoke test: witness-aimed tests CROSS-KILL classes (2 tests killed all
5 mutants on a top_k instance), so subtlety portfolios are the real
difficulty dial, not mutant count.

## 6. Training

*0.5–3B open-weight + LoRA; GRPO on the verifiers stack; ~$50 gate
budget.* **Q9. PROPOSED (operator, 23 Aug — Dom ratifies):**
Qwen2.5-Coder-1.5B-Instruct (coder checkpoint for parseable pytest;
Apache 2.0 — note the 3B Coder is research-licensed, so if 1.5B
starves the fix is easier instances, not a bigger model), LoRA r=16,
GRPO group size 8, 16 instances/step (128 rollouts), ~1024 max
completion tokens, first run 50–100 steps on one rented GPU at
$1–2/hr.

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

Hub upload · paper · multi-turn · big models · GAB · insolvency.
~~verifiers-port~~ — pulled INTO scope by Dom 23 Aug ("write out this
project in the verifiers framework"); the env is now natively a
verifiers package (`load_environment()` in `suitesmith/env.py`). Hub
upload stays out. **Rename RESOLVED 23 Aug: "suitesmith"** (Dom
delegated the pick; the smith forges the suite, mutants grade the
forge). Repo moved `~/covgap` → `~/suitesmith`.
