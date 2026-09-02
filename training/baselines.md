# Baselines

Pre-training numbers on the training runtime: Prime sandboxes, temperature 1.0, thinking on, 2 September 2026. The 4B is trained; the 8B and 35B are the comparison. Train split 50 tasks × 8 rollouts, eval split 90 × 8.

| Model | Token cap | Train mean | Train dead groups | Eval mean | seen / vocab / window | Eval dead groups | Truncated |
|---|---|---|---|---|---|---|---|
| Qwen3.5-4B | 4096 | 0.090 | 26 of 50 | 0.070 | 0.099 / 0.099 / 0.013 | 57 of 90 | 74–76% |
| **Qwen3.5-4B** | **8192** | **0.521** | 1 of 50 | **0.406** | 0.479 / 0.502 / 0.237 | 3 of 90 | 12–16% |
| qwen3-8b | 8192 | | | pending | | | |
| Qwen3.5-35B-A3B | 8192 | | | 0.870 | 0.910 / 0.896 / 0.803 | 0 of 90 | 1% |

- Cap: 8192. At 4096 the 4B loses three quarters of its rollouts to truncation and half its groups go dead.
- Truncated = rollouts that hit the cap before emitting a code block, scored 0 as malformed.
- Every row was checked against the run's own trace file: same rollout count, same mean.
