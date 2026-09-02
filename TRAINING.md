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

**Closed 1–2 Sep 2026 (Dom).**

| Decision | Options | Decided | Owner |
|---|---|---|---|
| Checkpoint | Qwen2.5-Coder-1.5B (SPEC) · Qwen3.5-0.8B/2B/4B · qwen3-8b | **Qwen3.5-4B.** Train floor 0.614 with 48/50 groups carrying gradient; 2B and 0.8B starve (15/50, 8/50); 8B and 35B ~0.9, nothing left to learn (§Calibration). Deviation from SPEC; decision record filed 2 Sep. | Dom |
| Update method | full fine-tune · LoRA | **Full fine-tune.** The report's claim is "RL on a 4B"; adapters cap how far the policy can move, a confound run 1 does not want. Memory: ~64 GB for weights + grads + Adam + fp32 master, so trainer and inference on separate cards. | Dom |
| Precision | bf16 · fp32 | bf16 weights and grads, fp32 optimiser state and master copy (prime-rl default). | operator |
| Reference / KL | β = 0 · small β · standard β | **Run 1: β = 0.** The grader must face the full optimisation pressure the project exists to measure; a leash would make "no hacks appeared" a fact about the leash. Stop rules in stage 5 are the guard. **Run 2: β = 0.04**, the standard GRPO value (DeepSeekMath's GRPO setting and TRL's long-standing default), as the leashed contrast, run whether or not run 1 hacks. Both fixed here, before any training. | Dom |
| Thinking | on · off | **ON (Dom, 2 Sep).** Rule pre-registered before the off numbers: OFF if the off train floor sits in 0.2–0.7 with ≥ 40/50 mixed groups, the off eval window tier is > 0.10, and off seen/vocab are within 0.15 of on; otherwise ON. Off cleared the train floor (0.366, 48/50 mixed) and failed the eval split (window 0.039 vs 0.277; seen/vocab trail by 0.25/0.20). Cost: ~5k vs ~940 output tokens per rollout. Open consequence: the on baselines are uncapped, so the sandbox re-baseline runs the 4B at cap 4096 and 8192 and the run takes the smaller cap that keeps the window tier alive. | Dom |

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

**Baselines owed on the training runtime before run 1** (1–2 Sep numbers are calibration, subprocess on the Air; see §Calibration):
- Qwen3.5-4B, both splits, temperature 1.0, thinking on, at cap 4096 and at 8192 (the cap decision). The pre-training row of the results table.
- qwen3-8b eval split, thinking on (scale reference row). Every Qwen model in the table is re-measured on the final runtime.
- gpt-4o-mini, 0.8B, 2B, 35B-A3B stay calibration-only unless promoted.

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

**Closed 1 Sep 2026 (Dom): "1. as proposed 2. last with full curve shown 3. statistically significant different performance in evaluations, and ideally matching / beating qwen 8b".** Test wording confirmed 2 Sep.

| Decision | Decided | Owner |
|---|---|---|
| Eval cadence | Every 20 steps: eval split, 90 tasks × 4 rollouts, strict reward, sandbox runtime, thinking per stage 1. Reported as three numbers, never one: seen / vocab / window. Ten evals over N = 200. | operator |
| Per-step telemetry | (1) mean reward + alive-group fraction; (2) gate mix as counts; (3) hack flags per marker; (4) twin-gate failures, separately; (5) mean response length + fraction truncated at 4096; (6) policy entropy from the trainer. | operator |
| Stop rules (the only early exit from N = 200) | Flags or twin failures rising across three consecutive evals while reward rises → stop, read transcripts (the hack). Truncation > 30% → stop (cap strangling the policy). Entropy < ⅓ of its start → stop (collapse). Mean reward > 0.15 below its running peak → stop (broken). | operator |
| Checkpoint cadence | Every 20 steps, with eval. Ten checkpoints. | operator |
| Which checkpoint is the result | **Last (step 200), with the full eval curve shown.** Best-on-eval would pick the luckiest of ten noisy draws (~±0.03 each) and flatter the number by about one noise width; the peak is reported as a peak, not as the result. | Dom |
| What run 1 trains | **Dom, 2 Sep, verbatim:** "the general capability of writing test-suites is being measured here, and is saturated for 35b but not for 4b. on breaking down the failure modes, the subcapability is spec comprehension and implementer guessing which is causing the difference in performance - and we are going to teach that subcapability via rl to the 4b s.t it gains the umbrella capability and saturates the evals". Basis: kill-given-pass is 0.97–1.00 for every model, so the gap sits in the reference gate. Curve reading: vocab gain = comprehension, seen-only gain = memorised reference quirks. | Dom |
| Success criterion | **Dom, verbatim:** "statistically significant different performance in evaluations, and ideally matching / beating qwen 8b". | Dom |

*Test (confirmed by Dom, 2 Sep):* untrained vs step-200 4B on the 90 eval tasks, same runtime; per-task mean reward, paired permutation test per tier, α = 0.05; must hold on seen, vocab and window reported with their own p-values. "Matching the 8B" = trained 4B eval mean at or above the untrained 8B's on the same runtime, or inside its 95% interval.

## 6. Compute

**Closed 1 Sep 2026 (Dom): "2xH100 as proposed, $150 ceiling, 24 hour wall".**

Arithmetic the decisions rest on: 200 steps × 256 = 51,200 training rollouts + 10 evals × 360 = 3,600. Thinking-on under the 4096 cap ≈ 3.5k tokens/rollout → 180–250M generated tokens; thinking-off ≈ 900 → ~45M. A 4B in bf16 on one H100 under vLLM at 256 concurrent sequences decodes ~4–8k tokens/s aggregate → 6–17 h of generation thinking-on, 1.5–3 h thinking-off; gradient steps are minutes in total. Scoring runs in Prime sandboxes off the GPU clock; ~64 sandboxes in flight keeps a 256-rollout step's scoring under two minutes.

| Decision | Decided | Owner |
|---|---|---|
| Provider | Prime Intellect pods (account, CLI, inference and sandbox access all in place). | operator |
| GPUs | **2×H100**: trainer on one (full FT of a 4B ≈ 64 GB before activations, gradient checkpointing on), vLLM inference on the other. If measured throughput < 4k tokens/s, add a third card for inference; do not share a card between the two processes. | Dom |
| Weight sync | prime-rl default trainer → inference path. | operator |
| Cost ceiling, run 1 | **$150**, all-in. GPU: 2×H100 ≈ $5/h → $30–100 thinking-on. Sandboxes: $0.075/h each (1 core / 2 GB / 5 GB), one per rollout, billed by the second, measured $0.0003–0.0005 per rollout → ~$25–35 for 54,800 rollouts; re-baselines ~$1. Worst case ~$135; rechecked once the cap is set. | Dom |
| Wall-time ceiling, run 1 | **24 h**, evals inside it. Covers the slow end of thinking-on. | Dom |



## 7. Record

**Closed 1 Sep 2026 (Dom: defaults accepted; checkpoint decision record to be written 2 Sep).**

| Decision | Decided | Owner |
|---|---|---|
| Pinned env tag | `baseline-v2` for reward semantics; the exact HEAD hash written into the run config (later commits added the split switch, crash-loudly, the payload path; no reward path). | operator |
| Config committed | `configs/run1.toml` (or prime-rl's expected form), committed before launch, never edited after. | operator |
| Seeds | Fixed and in the config: dataset shuffle, sampling where the server honours it, trainer init. | operator |
| Run naming and logs | `run1-4b-<hash>`; trainer logs + the S3 verdict log kept in the run directory and committed at the end (the S3 log is the audit trail for every stop rule). W&B optional; local is the record. | operator |
| Decision records | One per deviation from SPEC, Dom's voice, `~/career/record/decisions/`. Filed 2 Sep: `2026-09-02-training-checkpoint-qwen3.5-4b.md`. | Dom |
| Launchers | One launcher per chain; a filed result jsonl is set read-only (2 Sep: a second launcher appended 13 rows to a filed file before it was killed). | operator |

## Calibration (subprocess runtime, MacBook Air, 1–2 Sep 2026)

Train split 50 tasks × 8, eval split 90 tasks × 8 (seen 30 / vocab 30 / window 30), strict reward, provider-default temperature, no token cap.

| Model | Train mean | Train groups dead / all-nonzero | Eval mean | seen / vocab / window | Eval groups dead |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0.022 | 8/50 carry gradient | | | |
| Qwen3.5-2B | 0.047 | 15/50 carry gradient | | | |
| **Qwen3.5-4B, thinking on** | **0.614** | 0 / 2 | **0.497** | 0.63 / 0.585 / 0.277 | 2 |
| Qwen3.5-4B, thinking off | 0.366 | 2 / 0 | 0.271 | 0.384 / 0.390 / 0.039 | 23 |
| qwen3-8b | 0.905 (332 rows) | 0 / 33 | 0.856 | 0.886 / 0.955 / 0.727 | 0 |
| Qwen3.5-35B-A3B | 0.904 | 0 / 34 | 0.878 | 0.928 / 0.896 / 0.809 | 0 |
| gpt-4o-mini | | | 0.693 | 0.82 / 0.90 / 0.35 | |

Among suites that pass the reference gate, kill rate is 0.97–1.00 for every model: the reward is the reference gate. Thinking on ≈ 5k output tokens per rollout, off ≈ 940.

## Where it stands (2 Sep 2026, 11:30)

Stages 1–7 closed. Owed before run 1, in order:
1. Wallet top-up for the re-baselines (Dom; ~$6 needed, $6.56 in it).
2. Re-baselines on the sandbox runtime, T = 1.0, thinking on: 4B at cap 4096 and 8192 on both splits, 8B eval split (operator).
3. Cap decision (Dom) and the stage 6 cost recheck.
4. prime-rl config written from this file, committed (operator), then launch (Dom's go).