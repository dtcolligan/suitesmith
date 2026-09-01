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
| Checkpoint | Qwen2.5-Coder-1.5B (SPEC) · Qwen3.5-0.8B/2B/4B · qwen3-8b | **Qwen3.5-4B.** Strict floor 0.614 on the train split, 48/50 groups with gradient, 0 dead; eval split 0.50 (seen 0.63 / vocab 0.59 / window 0.28). 2B and 0.8B starve (15/50, 8/50); 8B ~0.9, nothing left to learn. Deviation from SPEC; decision record owed. | Dom |
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
- qwen3-8b eval split (scale reference row). Dom's rule: all Qwen models that appear in the table are re-measured on the final runtime.
- gpt-4o-mini, 0.8B, 2B, 35B-A3B stay calibration-only unless promoted to the table.

## 4. Update: how numbers become a gradient

| Decision | Options | Recommendation | Owner | Status |
|---|---|---|---|---|
| Tasks per step | 16 · 32 · 64 | 32 (256 rollouts/step) | operator | default |
| Advantage normalisation | group mean only · group mean and std | prime-rl default | operator | default |
| Clipping | PPO-style ratio clip | prime-rl default | operator | default |
| Learning rate / schedule | | prime-rl default for a 4B; no warm-up games in run 1 | operator | default |
| Group filtering | none · drop all-zero and all-one groups | drop both | operator | default |
| Steps / stop condition | fixed N · eval plateau · budget | fixed N for run 1, sized by the cost ceiling | operator | default |

## 5. Evidence: how you know it worked, or cheated

| Decision | Options | Recommendation | Owner | Status |
|---|---|---|---|---|
| Eval cadence | every N steps | every 20 steps, eval split 90 tasks × 4 rollouts, strict reward, three tiers reported apart | operator | default |
| Per-step telemetry | | mean reward, gate mix, hack flags, twin failures, response length, entropy | operator | default |
| Stop rules | | flags or twin failures rising with reward; response length explosion; reward collapse | operator | default |
| Checkpoint cadence | | every 20 steps | operator | default |
| Which checkpoint is "the result" | last · best on eval · fixed step | open, decided before the run | Dom | open |
| Success criterion | | open, written before the run: what seen-tier strict and window must do for this to be a finding | Dom | open |

## 6. Compute

| Decision | Options | Recommendation | Owner | Status |
|---|---|---|---|---|
| Provider | Prime Intellect pods · other | Prime (account, CLI, credits in place) | operator | default |
| GPUs | 1×H100 · 2×H100 · 1×H200 | 2×H100: trainer on one, vLLM inference on the other | Dom | open |
| Weight sync | prime-rl default | default | operator | default |
| Cost ceiling / time ceiling | | open; first run ~6 h ≈ $30–40 | Dom | open |

## 7. Record

| Decision | Options | Recommendation | Owner | Status |
|---|---|---|---|---|
| Pinned env tag | | `baseline-v2` (+ HEAD hash in the config) | operator | default |
| Config committed | | yes, under `configs/` | operator | default |
| Seeds | | fixed and recorded | operator | default |
| Run naming / logs | local · W&B | local S3 logs + trainer logs in the run dir; W&B optional | operator | default |
| Decision records | | one per deviation from SPEC; the checkpoint change owes one (Dom writes) | Dom | open |

## Open decisions, in the order they get walked

1. Thinking on/off, decided when the thinking-off baseline lands (stage 1; everything else in stage 1 closed 1 Sep)
2. ~~Stage 2 closed 1 Sep~~ (pre-run floor re-measure at T=1.0, cap 4096 owed)
3. ~~Stage 3 closed 1 Sep~~ (sandbox runtime verified, payload fix shipped; owed: Qwen re-baselines on the sandbox once thinking is decided)
3. Which checkpoint is the result; success criterion (stage 5)
4. GPUs; cost and time ceilings (stage 6)
5. Decision record for the checkpoint change (stage 7)
