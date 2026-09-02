# Training run 1

| | |
|---|---|
| Policy | Qwen3.5-4B, full fine-tune, bf16 weights, fp32 optimiser state, thinking on |
| Algorithm | GRPO. Group size 8, advantages standardised within the group, clip 0.2, learning rate 1e-6 constant, KL β = 0 |
| Batch | 32 tasks × 8 rollouts per step. Zero-spread groups dropped. Group size 16 if under 60% of groups are alive |
| Steps | 200 |
| Sampling | Temperature 1.0, token cap from [baselines.md](baselines.md), one turn |
| Reward | Strict, as in the README. Scored in Prime sandboxes, python:3.12-slim, one sandbox per rollout |
| Eval | Every 20 steps on the eval split, 90 tasks × 4 rollouts, reported as seen / vocab / window |
| Checkpoints | Every 20 steps. The result is step 200 with the full curve shown |
| Stop rules | Hack flags or twin failures rising over three evals while reward rises. Truncation above 30%. Entropy below a third of its start. Reward 0.15 below its running peak |
| Success | Paired permutation test per tier, untrained vs step-200 4B on the eval split, α = 0.05, must hold on seen. Matching the 8B: eval mean at or above the 8B's, or inside its 95% interval |
| Run 2 | Same, KL β = 0.04 |
| Compute | 2×H100. Ceiling $150 and 24 hours |
| Record | Environment tag baseline-v2. Config committed before launch. Seeds fixed. Verdict log kept |
