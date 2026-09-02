# Calibration

Purpose: choose the checkpoint and the thinking mode. Subprocess runtime on a laptop, provider-default temperature, no token cap, 1–2 September 2026. Train split 50 tasks × 8 rollouts, eval split 90 × 8.

| Model | Train mean | Train groups dead / all-nonzero | Eval mean | seen / vocab / window | Eval groups dead |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0.022 | 8 of 50 carry gradient | | | |
| Qwen3.5-2B | 0.047 | 15 of 50 carry gradient | | | |
| **Qwen3.5-4B, thinking on** | **0.614** | 0 / 2 | **0.497** | 0.63 / 0.585 / 0.277 | 2 |
| Qwen3.5-4B, thinking off | 0.366 | 2 / 0 | 0.271 | 0.384 / 0.390 / 0.039 | 23 |
| qwen3-8b | 0.905 | 0 / 33 | 0.856 | 0.886 / 0.955 / 0.727 | 0 |
| Qwen3.5-35B-A3B | 0.904 | 0 / 34 | 0.878 | 0.928 / 0.896 / 0.809 | 0 |
| gpt-4o-mini | | | 0.693 | 0.82 / 0.90 / 0.35 | |

- Among suites that pass the reference gate, kill rate is 0.97 to 1.00 for every model. The reward is the reference gate.
- Checkpoint: Qwen3.5-4B. 0.8B and 2B starve. 8B and 35B have nothing left to learn.
- Thinking: on. Off is trainable on the train split but scores 0.04 on the window tier. On costs about 5k output tokens per rollout, off about 940.

Rule for thinking, fixed before the off numbers were in: off if its train floor is in 0.2–0.7 with at least 40 of 50 groups mixed, its window tier is above 0.10, and its seen and vocab tiers are within 0.15 of on. Otherwise on.
