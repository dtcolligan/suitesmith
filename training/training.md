# Training run 1

| | |
|---|---|
| Policy | Qwen3.5-4B, full fine-tune, bf16 weights, fp32 optimiser state, thinking on |
| Algorithm | GRPO. Group size 8, learning rate 1e-6 constant, KL off. Advantage = reward minus the group mean, no division by the group std (Dom, 2 Sep: "mean-only"; prime-rl's GRPO offers nothing else). Trust region: a token whose probability moved more than eps = 0.1 is masked, prime-rl's default (Dom: "eps default"). Weight decay 0.01, gradient norm clipped at 1.0, both defaults |
| Batch | 32 tasks × 8 rollouts per step. Zero-spread groups dropped. Group size 16 if under 60% of groups are alive |
| Steps | 200 |
| Sampling | Temperature 1.0, token cap 8192, one turn. Sequence length 10240 |
| Reward | Strict, as in the README. Scored in Prime sandboxes, python:3.12-slim, one sandbox per rollout |
| Eval | Every 20 steps on the eval split, 90 tasks × 4 rollouts, reported as seen / vocab / window |
| Checkpoints | Every 20 steps. The result is step 200 with the full curve shown |
| Stop rules | Stop early if any of these happen. Suites flagged as hacks, or failing the twin check, rise for three evals in a row while reward rises. More than 30% of rollouts hit the token cap. The policy's entropy falls below a third of its starting value. Mean reward drops 0.15 below its best so far |
| Success | The trained model scores higher than the untrained one on the eval split. Test: for each tier, compare the two models' per-task scores on the same tasks with a paired permutation test at the 5% level. It must hold on the seen tier. Stretch goal: the trained 4B matches or beats the untrained 8B on the eval split, where matching means inside the 8B's 95% confidence interval |
| Run 2 | Same, KL β = 0.04 |
| Compute | 2×H100. Ceiling $150 and 24 hours |
| Record | Environment tag baseline-v2. Config [configs/run1.toml](../configs/run1.toml). Seeds fixed. Verdict log kept |
| Resilience | Whole-rollout retry, twice, on provider and sandbox errors. Task dispatch capped at 60 per minute. Sandbox idle fallback 10 minutes |
