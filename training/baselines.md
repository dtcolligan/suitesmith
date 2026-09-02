# Baselines

Pre-training numbers on the training runtime: Prime sandboxes, temperature 1.0, thinking on. The 4B is trained; the 8B and 35B are the comparison. Not yet run.

| Model | Token cap | Train mean | Eval mean | seen / vocab / window |
|---|---|---|---|---|
| Qwen3.5-4B | 4096 | | | |
| Qwen3.5-4B | 8192 | | | |
| qwen3-8b | as chosen | | | |
| Qwen3.5-35B-A3B | as chosen | | | |

The cap is chosen here: the smaller of 4096 and 8192 that keeps the window tier alive.
