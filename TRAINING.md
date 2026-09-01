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

**Closed 1 Sep 2026 (Dom).**

| Decision | Options | Decided | Owner |
|---|---|---|---|
| Checkpoint | Qwen2.5-Coder-1.5B (SPEC) · Qwen3.5-0.8B/2B/4B · qwen3-8b | **Qwen3.5-4B.** Strict floor 0.614 on the train split, 48/50 groups with gradient, 0 dead; eval split 0.50 (seen 0.63 / vocab 0.59 / window 0.28). 2B and 0.8B starve (15/50, 8/50); 8B ~0.9, nothing left to learn. Deviation from SPEC; decision record owed. | Dom |
| Update method | full fine-tune · LoRA | **Full fine-tune.** The report's claim is "RL on a 4B"; adapters cap how far the policy can move, a confound run 1 does not want. Memory: ~64 GB for weights + grads + Adam + fp32 master, so trainer and inference on separate cards. | Dom |
| Precision | bf16 · fp32 | bf16 weights and grads, fp32 optimiser state and master copy (prime-rl default). | operator |
| Reference / KL | β = 0 · small β · standard β | **Run 1: β = 0.** The grader must face the full optimisation pressure the project exists to measure; a leash would make "no hacks appeared" a fact about the leash. Stop rules in stage 5 are the guard. **Run 2: small β (0.001–0.01)** as the contrast, run whether or not run 1 hacks. Both fixed here, before any training, so neither is post-hoc. | Dom |
| Thinking | on · off | **On for run 1.** Every baseline is thinking-on; the trained artefact is a model that reasons then writes tests. Cost: ~5k output tokens per rollout, the dominant cost. A thinking-off baseline is queued on both splits (`--sampling.reasoning-effort none`) for information; it reopens this row only if the off floor matches or beats the on floor on the train split, in which case the cheaper artefact is at least as good and the choice is remade with that number. | Dom |

Reading of Dom's KL wording ("run 1 with KL, run 2 with small KL", 1 Sep 16:5x) taken as the plan above (run 1 unleashed, run 2 leashed); correct here if the other reading was meant.

## 2. Rollouts: how samples are produced

| Decision | Options | Recommendation | Owner | Status |
|---|---|---|---|---|
| Group size (rollouts per task) | 4 · 8 · 16 | 8, matched to every calibration number | operator | default |
| Sampling temperature | model default · 1.0 · lower | 1.0 for training (diversity inside a group is the gradient) | operator | default |
| max_tokens | none · 4096 · 8192 | 4096 for run 1; **re-measure the floor under the cap before training** | operator | default |
| Async level (steps off-policy allowed) | 0 (sync) · 1 · 2 | prime-rl default (1); raise only if inference starves the trainer | operator | default |

## 3. Reward: how a rollout becomes a number

| Decision | Options | Recommendation | Owner | Status |
|---|---|---|---|---|
| Taskset commit | frozen tag | `baseline-v2` semantics (2a06f53); HEAD 7bd23cc adds only crash-loudly + the split switch, no reward path | Dom | decided 1 Sep |
| Reward rule | strict Q6 · Q6 fallback | strict; fallback shelved (4B floor is alive without it) | Dom | decided 1 Sep |
| Batch composition | train split as built (70% white, mix "easy") · tier/visibility rebalanced | as built for run 1 | Dom | open |
| Untrusted-code execution | subprocess on the pod · prime sandbox · modal | subprocess on the pod (what calibration ran); 5 s timeout; no isolation from the trainer box | Dom | open |
| Env-server topology | co-located with trainer · separate | co-located, workers = auto, multiplex 32 | operator | default |

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

1. ~~Stage 1 closed 1 Sep~~ (thinking row reopens only if the off floor ≥ the on floor)
2. Batch composition, isolation (stage 3)
3. Which checkpoint is the result; success criterion (stage 5)
4. GPUs; cost and time ceilings (stage 6)
5. Decision record for the checkpoint change (stage 7)
