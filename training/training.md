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
| Stop rules | Stop early if any of these happen. Suites flagged as hacks, or failing the twin check, rise for three evals in a row while reward rises. More than 30% of rollouts hit the token cap. The policy's entropy falls below a third of its starting value. Mean reward drops 0.15 below its best so far |
| Success | The trained model scores higher than the untrained one on the eval split. Test: for each tier, compare the two models' per-task scores on the same tasks with a paired permutation test at the 5% level. It must hold on the seen tier. Stretch goal: the trained 4B matches or beats the untrained 8B on the eval split, where matching means inside the 8B's 95% confidence interval |
| Run 2 | Same, KL β = 0.04 |
| Compute | 2×H100. Ceiling $150 and 24 hours |
| Record | Environment tag baseline-v2. Config committed before launch. Seeds fixed. Verdict log kept |
