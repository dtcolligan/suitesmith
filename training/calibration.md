# Calibration

Purpose: choose the model and the thinking mode.

| Model | Train mean | Train groups dead / all-nonzero | Eval mean | seen / vocab / window | Eval groups dead |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0.022 | 8 of 50 carry gradient | | | |
| Qwen3.5-2B | 0.047 | 15 of 50 carry gradient | | | |
| **Qwen3.5-4B, thinking on** | **0.614** | 0 / 2 | **0.497** | 0.63 / 0.585 / 0.277 | 2 |
| Qwen3.5-4B, thinking off | 0.366 | 2 / 0 | 0.271 | 0.384 / 0.390 / 0.039 | 23 |
| qwen3-8b | 0.905 | 0 / 33 | 0.856 | 0.886 / 0.955 / 0.727 | 0 |
| Qwen3.5-35B-A3B | 0.904 | 0 / 34 | 0.878 | 0.928 / 0.896 / 0.809 | 0 |
| gpt-4o-mini | | | 0.693 | 0.82 / 0.90 / 0.35 | |

- Subprocess runtime on a laptop
- Provider-default temperature, no token cap
- Train split 50 tasks × 8 rollouts, eval split 90 × 8
- 1–2 September 2026
