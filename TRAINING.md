# TRAINING.md — the run plan, decided stage by stage

The decomposition follows the RL loop: **sample tasks → generate rollouts →
score them → turn scores into an update → apply it**, with **evidence**
collected alongside, standing on **compute**, written into a **record**.
Every infrastructure decision attaches to exactly one stage. Decisions
marked *Dom* change what the result means; the rest are operator defaults
Dom can override. Status: `decided` / `default` / `open`.

Started 1 Sep 2026 after Q8 calibration (see README §Run, `baseline-v2`).
Walked one stage at a time; each stage closes with Dom's call.

## 1. Policy: what is being trained

**1 Sep 2026 (Dom): checkpoint, update method, precision, KL closed; thinking open pending the thinking-off baseline.**

| Decision | Options | Decided | Owner |
|---|---|---|---|
| Checkpoint | Qwen2.5-Coder-1.5B (SPEC) · Qwen3.5-0.8B/2B/4B · qwen3-8b | **Qwen3.5-4B.** Strict floor 0.614 on the train split, 48/50 groups with gradient, 0 dead; eval split 0.50 (seen 0.63 / vocab 0.59 / window 0.28). 2B and 0.8B starve (15/50, 8/50); 8B ~0.9 on the train split, 0.86 on the eval split (seen 0.89 / vocab 0.96 / window 0.73; 0 dead, 55/90 groups all-nonzero), nothing left to learn. Deviation from SPEC; decision record filed 2 Sep. | Dom |
| Update method | full fine-tune · LoRA | **Full fine-tune.** The report's claim is "RL on a 4B"; adapters cap how far the policy can move, a confound run 1 does not want. Memory: ~64 GB for weights + grads + Adam + fp32 master, so trainer and inference on separate cards. | Dom |
| Precision | bf16 · fp32 | bf16 weights and grads, fp32 optimiser state and master copy (prime-rl default). | operator |
| Reference / KL | β = 0 · small β · standard β | **Run 1: β = 0.** The grader must face the full optimisation pressure the project exists to measure; a leash would make "no hacks appeared" a fact about the leash. Stop rules in stage 5 are the guard. **Run 2: β = 0.04**, the standard GRPO value (DeepSeekMath's GRPO setting and TRL's long-standing default), as the leashed contrast, run whether or not run 1 hacks. Both fixed here, before any training. | Dom |
| Thinking | on · off | **Open, decided by the baselines.** Every number so far is thinking-on (train 0.61, eval 0.50; ~5k output tokens per rollout, the dominant cost). The thinking-off baseline is queued on both splits (`--sampling.reasoning-effort none`); Dom decides when both floors are in. What the numbers have to show: the off floor on the train split (alive groups, mean) and the off eval tiers, against the on ones, with the per-rollout token cost beside each. | Dom |

## 2. Rollouts: how samples are produced

**Closed 1 Sep 2026 (Dom).** The gradient for a task is the spread of rewards inside its group; a group that is all-zero or all-success contributes nothing.

| Decision | Options | Decided | Owner |
|---|---|---|---|
| Group size G | 4 · 8 · 16 | **8**, matched to every calibration number. Untrained 4B on the train split: 0 tasks at 0/8, 2 at 1/8, median pass rate 0.62, so 8 is enough at the start. Rule: if the alive-group fraction of a step falls under 60% (easy tasks saturating to 8/8), raise G to 16 for the rest of the run. | operator |
| Temperature | provider default · 1.0 · lower | **1.0.** The distribution the policy-gradient maths assumes; spread inside a group is the signal. Today's baselines used the provider default (recorded as null); the floor is re-measured at 1.0 before the run. | Dom |
| max_tokens | none · 4096 · 8192 | **4096.** Unbounded thinking (5k typical, 20k tail) stalls steps; a truncated rollout has no code block and scores 0 through the malformed gate, which is the intended price. The floor is re-measured under the cap before the run. Interacts with the open thinking decision (off-rollouts ~900 tokens never reach it). | Dom |
| Async level | 0 · 1 · 2 | **1** (prime-rl default): inference may generate with weights one step stale so trainer and inference do not idle on each other. Throughput only. | operator |

Pre-run re-measure owed from this stage: 4B train-split floor at temperature 1.0 and max_tokens 4096 (one run, ~$0.6).

## 3. Reward: how a rollout becomes a number

**Closed 1 Sep 2026 (Dom).** Rule that makes runtime changes clean: **every number in the results table is measured on the runtime training used.** Everything measured before that is calibration and is labelled with its runtime.

| Decision | Options | Decided | Owner |
|---|---|---|---|
| Taskset commit | frozen tag | `baseline-v2` semantics (2a06f53); later commits add only crash-loudly, the split switch, and (owed) the payload-over-stdin fix, none of which touch a reward path. | Dom |
| Reward rule | strict Q6 · Q6 fallback | Strict. Fallback shelved: the 4B floor is alive without it (48/50 groups). | Dom |
| Batch composition | as built · rebalanced · curriculum | **As built** for run 1: 200 tasks, three families round-robin, 70% white / 30% black, "easy" portfolio, 32 tasks per step. Every calibration number was measured on this mix; changing it is a hypothesis for a later run. Untrained 4B: white 0.63, black 0.57. | Dom |
| Untrusted-code execution | A subprocess as is · B hardened subprocess · C sandboxes | **C: real sandboxes** (Prime sandbox runtime; Modal as the fallback), image pinned to Python 3.12 to match every calibration run, per-run timeout checked against sandbox CPU speed. **Verified 1 Sep 17:1x:** `--env.agent.runtime.type prime --env.agent.runtime.image python:3.12-slim`, 2×2 gpt-4o-mini scored 1.0/1.0/0.8/0.8 with identical verdicts to the subprocess path; boot 7 s + setup 3 s + scoring ~4 s per rollout, so ~25–30 s wall per rollout against ~10 s locally, but remote, so the local `-c` cap no longer applies. The subprocess runtime is the same user and filesystem as the trainer: a test can hog resources, touch checkpoints and logs, and read a sibling rollout's payload. | Dom |
| Sibling-payload hole | | **Fixed 1 Sep 17:1x** (payload now travels in `SUITESMITH_PAYLOAD`, popped by verify.py before any suite runs; battery pins that the suite's process cannot see it). Was: the payload (reference, twin, mutants, witnesses) is written to `/tmp/suitesmith/<trace-id>.json`; with 8 rollouts of one task in flight, `glob` finds a sibling's answer key and a suite built from its witnesses scores 1.0 without reading the spec; the twin gate does not catch it. Fix: payload over stdin, no shared file. Our defect, by inspection. | operator |
| Env-server topology | | Co-located with the trainer, workers auto, multiplex 32. | operator |

**Baselines owed on the training runtime before run 1** (the calibration numbers of 1 Sep are history, labelled "subprocess, MacBook Air"):
- Qwen3.5-4B, train and eval splits, temperature 1.0, max_tokens 4096, thinking per stage 1's pending decision. This is the pre-training row of the results table.
- qwen3-8b eval split (scale reference row). Calibration value, subprocess/Air, 2 Sep: 0.856 (seen 0.886 / vocab 0.955 / window 0.727), 0 dead groups. Dom's rule: all Qwen models that appear in the table are re-measured on the final runtime.
- gpt-4o-mini, 0.8B, 2B, 35B-A3B stay calibration-only unless promoted to the table.

## 4. Update: how numbers become a gradient

**Closed 1 Sep 2026 (Dom).** The gradient uses each rollout's advantage, its reward relative to its own group: `A = (r − mean(group)) / std(group)`. No value model; the group is the baseline.

| Decision | Options | Decided | Owner |
|---|---|---|---|
| Advantage normalisation | group mean only (Dr. GRPO) · mean and std | **Mean and std** (decided 1 Sep). The trade, worked through by Dom: in a group of seven 0.2s and one 0.4, the 0.4 gets A ≈ +2.6 and each 0.2 gets −0.4; one extra mutant killed is the strongest signal in the batch. Amplifies the first success on hard tasks; also amplifies a lucky kill. | Dom |
| Clipping | PPO ratio clip | **ε = 0.2** (decided 1 Sep as conventional: DeepSeekMath, SimpleRL; DAPO uses 0.2/0.28 asymmetric). Bounds how far any token's probability moves in one step, needed because async-level-1 rollouts are one step off-policy. Stability, not a result knob. | operator |
| Learning rate / schedule | | **1e-6, constant** (decided 1 Sep as conventional: DeepSeekMath, DAPO), no warm-up in run 1. RL full fine-tune of a 4B lives in 1e-6 to 5e-6; too high shows as an entropy crash in stage 5's telemetry. | operator |
| Tasks per step | 16 · 32 · 64 | **32** (× G = 8 → 256 rollouts per step; decided 1 Sep). Small side of the literature's 256–1024 but standard for one node. | operator |
| Group filtering | none · drop zero-spread groups | **Drop all-zero and all-one groups** before the update (decided 1 Sep; DAPO's dynamic sampling); they carry no gradient and would otherwise fill the batch. Pairs with stage 2's G→16 rule. | operator |
| Steps / stop condition | fixed N · rule | **N = 200** (Dom, 1 Sep). Literature shape (DeepSeekMath, SimpleRL-Zoo, Dr. GRPO, DAPO; exact figures to verify before citing): at a few hundred rollouts a step most gain lands in the first 100–200 steps. 200 (≈ 32 passes over the 200 training tasks; 51,200 training rollouts ≈ 180–250M generated tokens thinking-on under the 4096 cap, ~45M thinking-off; see stage 6 for time and cost. An earlier ~50M figure here was wrong by 4×, corrected 1 Sep). Fixed because a "stop when eval plateaus" rule is a post-hoc decision dressed as a schedule. Stage 6's cost ceiling would set the final N; stage 5's stop rules the only early exit. | Dom (open) |

## 5. Evidence: how you know it worked, or cheated

**Closed 1 Sep 2026 (Dom): "1. as proposed 2. last with full curve shown 3. statistically significant different performance in evaluations, and ideally matching / beating qwen 8b".** One operator note under the criterion makes it testable; Dom confirms or changes the test.

| Decision | Decided | Owner |
|---|---|---|
| Eval cadence | Every 20 steps: eval split, 90 tasks × 4 rollouts, strict reward, sandbox runtime, thinking per stage 1. Reported as three numbers, never one: seen / vocab / window. Ten evals over N = 200. | operator |
| Per-step telemetry | (1) mean reward + alive-group fraction; (2) gate mix as counts; (3) hack flags per marker; (4) twin-gate failures, separately; (5) mean response length + fraction truncated at 4096; (6) policy entropy from the trainer. | operator |
| Stop rules (the only early exit from N = 200) | Flags or twin failures rising across three consecutive evals while reward rises → stop, read transcripts (the hack). Truncation > 30% → stop (cap strangling the policy). Entropy < ⅓ of its start → stop (collapse). Mean reward > 0.15 below its running peak → stop (broken). | operator |
| Checkpoint cadence | Every 20 steps, with eval. Ten checkpoints. | operator |
| Which checkpoint is the result | **Last (step 200), with the full eval curve shown.** Best-on-eval would pick the luckiest of ten noisy draws (~±0.03 each) and flatter the number by about one noise width; the peak is reported as a peak, not as the result. | Dom |
| What run 1 trains | **Dom, 2 Sep, verbatim:** "the general capability of writing test-suites is being measured here, and is saturated for 35b but not for 4b. on breaking down the failure modes, the subcapability is spec comprehension and implementer guessing which is causing the difference in performance - and we are going to teach that subcapability via rl to the 4b s.t it gains the umbrella capability and saturates the evals". Data behind it (calibration, 2 Sep): among suites that pass the reference gate, kill rate is 0.97–1.00 for every model tested, so the 4B/35B gap (0.61 vs 0.90 train floor) sits entirely in the reference gate + malformed. Reading rule for the eval curve: vocab-tier gain = comprehension, seen-only gain = memorised reference quirks. | Dom |
| Success criterion | **Dom, verbatim:** "statistically significant different performance in evaluations, and ideally matching / beating qwen 8b". | Dom |

*Operator note to make the criterion testable (Dom to confirm):* the comparison is the untrained 4B vs the step-200 4B on the same 90 eval tasks, both measured on the training runtime; per-task mean reward, paired (same tasks), one test per tier, α = 0.05 (a paired permutation test, distribution-free). "Significant" must hold on the **seen** tier at minimum; vocab and window are reported with their own p-values whatever they show. "Matching / beating qwen 8b" reads as: the trained 4B's eval-split mean is at or above the untrained 8B's on the same runtime, or within its 95% interval ("matching"). Both 8B numbers come from the owed sandbox re-baseline.

## 6. Compute

**Closed 1 Sep 2026 (Dom): "2xH100 as proposed, $150 ceiling, 24 hour wall".**

Arithmetic the decisions rest on: 200 steps × 256 = 51,200 training rollouts + 10 evals × 360 = 3,600. Thinking-on under the 4096 cap ≈ 3.5k tokens/rollout → 180–250M generated tokens; thinking-off ≈ 900 → ~45M. A 4B in bf16 on one H100 under vLLM at 256 concurrent sequences decodes ~4–8k tokens/s aggregate → 6–17 h of generation thinking-on, 1.5–3 h thinking-off; gradient steps are minutes in total. Scoring runs in Prime sandboxes off the GPU clock; ~64 sandboxes in flight keeps a 256-rollout step's scoring under two minutes.

| Decision | Decided | Owner |
|---|---|---|
| Provider | Prime Intellect pods (account, CLI, inference and sandbox access all in place). | operator |
| GPUs | **2×H100**: trainer on one (full FT of a 4B ≈ 64 GB before activations, gradient checkpointing on), vLLM inference on the other. If measured throughput < 4k tokens/s, add a third card for inference; do not share a card between the two processes. | Dom |
| Weight sync | prime-rl default trainer → inference path. | operator |
| Cost ceiling, run 1 | **$150**, all-in: GPU hours (2×H100 ≈ $5/h → $30–100 thinking-on, $10–20 off), evals, the owed sandbox re-baselines (~$5), sandbox CPU time (rate to check before the config is written). | Dom |
| Wall-time ceiling, run 1 | **24 h**, evals inside it. Covers the slow end of thinking-on. | Dom |

Consequence for stage 1's open thinking row: both ceilings cover thinking-on, so cost does not force the decision; the thinking-off baseline decides it on merit. The cost gap (roughly 5×) is recorded here so the decision is made knowing it.

## 7. Record

**Closed 1 Sep 2026 (Dom: defaults accepted; checkpoint decision record to be written 2 Sep).**

| Decision | Decided | Owner |
|---|---|---|
| Pinned env tag | `baseline-v2` for reward semantics; the exact HEAD hash written into the run config (later commits added the split switch, crash-loudly, the payload path; no reward path). | operator |
| Config committed | `configs/run1.toml` (or prime-rl's expected form), committed before launch, never edited after. | operator |
| Seeds | Fixed and in the config: dataset shuffle, sampling where the server honours it, trainer init. | operator |
| Run naming and logs | `run1-4b-<hash>`; trainer logs + the S3 verdict log kept in the run directory and committed at the end (the S3 log is the audit trail for every stop rule). W&B optional; local is the record. | operator |
| Decision records | One per deviation from SPEC, Dom's voice, `~/career/record/decisions/`. **Filed 2 Sep:** ~/career/record/decisions/2026-09-02-training-checkpoint-qwen3.5-4b.md. KL plan, thinking, runtime rule are recorded in this file in Dom's words. | Dom |

## Where it stands (1 Sep 2026, end of day)

Stages 2–7 closed; stage 1 closed except **thinking on/off**, decided by the thinking-off baseline (running tonight).

Owed before run 1, in order:
1. Thinking decision (Dom), on tonight's numbers with the ~5× cost gap beside them.
2. Dom confirms the test under the stage-5 criterion.
3. ~~Checkpoint decision record~~ (filed 2 Sep).
4. Sandbox CPU pricing checked (operator).
5. Re-baselines on the sandbox runtime at T = 1.0, cap 4096, thinking per (1): 4B both splits, 8B eval split (operator).
6. prime-rl config written from this file, committed (operator), then launch (Dom's go).
