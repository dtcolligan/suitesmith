# SPEC — covgap

> Fill this by hand (train session, Fri 21 Aug). Questions marked **Q**
> are the load-bearing design decisions — they're yours. Notes in
> *italics* are constraints already fixed by the project definition.

## 0. Objective (fixed 20 Aug — do not re-litigate here)

S2 committed: designed + built RL environment with verifiable grader,
trained against, **done before 14 Sep**. S1 (one training step printing
a reward) by 31 Aug, cleared on GPU day Mon 24 with existing-env
fallback.

**Definitions sharpened 20 Aug (Dom's catch):**
- **The planted-seam drift experiment is part of S2** — the quirk gap is
  a controlled, designed study of training under under-enforcement. Its
  "fix" (enforce the quirk, p=1 control arm) is known in advance; it is
  a result about training, never presented as a discovered flaw.
- **S3 = the UNPLANNED catch only**: the policy exploits something we
  did not intend (base tests, oracle input sampler, reward parsing,
  sandbox, template bug), we spot it via the instrumentation, diagnose,
  fix, show before/after. Cannot be staged; therefore designed-for,
  never promised, and never conflated with the planted seam.

## 1. Task family

*Constraints: Python; single-turn; procedurally generated; difficulty at
a 0.5–3B model's frontier (mixed pass rates required — all-pass and
all-fail groups both give zero GRPO advantage).*

**Q1. ✅ DECIDED (20 Aug): composition of (i) + (iii).** One instance =
implement a Python function from signature + docstring; the task body is
a **data-manipulation template** (filter/aggregate/transform records
with sampled parameters), with a **spec-quirk** sampled and injected
into the docstring (an unusual rule the base task wouldn't imply). Base
task easy enough for a 0.5–3B model; failure concentrates at the quirk
by design. Remaining sub-decisions (train, Fri): the template set (~how
many, which shapes), the quirk library (~how many, categorised), and
whether some instances carry no quirk (controls).

**Q2. ✅ DECIDED (20 Aug, Dom's design): specification quality is the
difficulty separator.** The model sees signature + docstring (+ tiered
examples); the weak suite is never shown; the strong oracle never shown.
Spec tiers (draft — finalise on the train): **A** quirk stated plainly +
a shown example exercising the quirk · **B** quirk stated plainly,
examples exercise base behaviour only · **C** quirk stated
tersely/obliquely, base-only examples. Two hard rules: (1) **the quirk
is always derivable from the spec** — tiers vary how cheaply, never
whether (a quirk-unstated variant exists only as a labelled control,
off-ladder); (2) **step-0 per-tier compliance baselines are mandatory**
— drift = decline from baseline within tier, separating
never-understood from learned-to-drop. S3 interaction hypothesis: under
an under-enforcing reward, quirk-compliance decays faster on
weaker-specified tiers ("strength of telling" parameterised while the
enforcement gap stays constant). Full-suite-visible contrast arm:
parked, out-of-week upside.

**Q3.** Difficulty ladder: what makes an instance easy vs hard, and how
will difficulty be calibrated? (Protocol: vf-eval a small model on N
instances; target a mixed per-instance pass distribution, not ~0% or
~100%.)

## 2. Generator

**Q4.** How is an instance generated — templates with sampled
parameters? Compositions of primitives? Where does the **reference
solution** come from (it must be correct by construction)?

**Q5.** Seeding and splits: how do train/eval instances stay disjoint?
(Contamination-by-construction is the cheap win of procedural
generation — say how.)

## 3. Graders

*Constraint: deterministic, execution-based, no LLM judge anywhere.*

**Q6.** **Weak suite (the reward):** how many tests per instance, chosen
how? What coverage is deliberately left out — i.e., write the
**soft-spot map**: the enumerated ways a solution could pass the weak
suite while being wrong (edge cases untested, only happy-path inputs,
narrow value ranges, ...). This map is what makes S3 catchable.

**Q7.** **Strong oracle (the truth):** differential testing against the
reference solution on randomly sampled inputs is available for free —
how many samples, what input distribution, what counts as equivalent
(exact? tolerance? exceptions must match?)?

**Q8.** Sandboxing: generated code gets executed — subprocess, timeout,
resource limits, no network. What are the limits?

## 4. Reward

**Q9.** Reward definition from the weak suite: binary all-pass, or
fraction of tests passed? (Fraction gives denser signal for small
models; binary is cleaner for the S3 story. Justify the choice.)

**Q10.** Anything else in the reward (format compliance, length,
parse-failure penalty)? *Keep minimal — every extra term is a new
surface to game, which is either noise or the experiment, decide which.*

## 5. Training

*Constraints: 0.5–3B open-weight + LoRA; GRPO on the verifiers stack;
budget ceiling ~$50 for the gate phase; rented GPU.*

**Q11.** Model pick and why. Rollouts per step, group size, steps for
the first real run.

## 6. Metrics & instrumentation (this is where S3 lives)

**Q12.** Per training step, log at minimum: weak-suite reward AND
strong-oracle pass rate on the same rollouts. **The divergence curve
(weak up, strong flat-or-down) is the finding.** What else: per-instance
pass patterns? sampled transcripts at checkpoints for reading what the
policy actually does?

## 7. Milestones (fixed)

Fri spec → Sat–Sun build + calibration → Mon GPU/S1 → Tue off → 26–31
runs + report v1 → ≤7 Sep repo public + S2 declared → 14 Sep SPAR.

## 8. Out of scope (fixed)

Hub upload, verifiers-port adventures, paper, multi-turn, big models,
GAB, insolvency.
