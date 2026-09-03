# Results

Run 1, 2 September 2026. Stopped at step 23 of 200 by Dom's call after pipeline stalls; the result of record is the step-20 checkpoint and its eval. Config [configs/run1.toml](../configs/run1.toml); 2× H100 on Prime Intellect, 5.5 h of pod, $46.

Eval split, 90 tasks × 4 rollouts, temperature 1.0, cap 8192, Prime sandboxes, the same settings as [baselines.md](baselines.md).

| | step 0 | step 20 | untrained 8B | untrained 35B |
|---|---|---|---|---|
| seen | 0.479 | 0.948 | 0.899 | 0.910 |
| vocab | 0.502 | 0.933 | 0.943 | 0.896 |
| window | 0.237 | 0.762 | 0.863 | 0.803 |
| mean | 0.406 | 0.881 | 0.902 | 0.870 |

Train reward per batch rose from 0.48 at step 1 to 0.85–0.96 by step 20. In the verdict log the gain sits in the reference gate: reference failures fell from 220 to 33 per 512 rollouts and truncations from 63 to 1. No hack flags and no twin-gate failures. Suites shrank from a median of 8 tests to 5, and the share of passing suites that leave a mutant alive rose from 1% to 24%, so the mutant term, at ceiling for every baseline model, carried gradient by step 20.

## Criterion

`scripts/criterion_test.py`: per task, mean reward over rollouts; per tier, a sign-flip permutation test on the trained-minus-untrained per-task differences, 20,000 flips. The untrained side is the 4B baseline row in [baselines.md](baselines.md) (90 × 8, cap 8192); the trained side is the step-20 eval (90 × 4). Same 90 tasks.

| tier | untrained | step 20 | diff | tasks up / same / down | p (one-sided) |
|---|---|---|---|---|---|
| seen | 0.479 | 0.948 | +0.469 | 28 / 1 / 1 | < 0.0001 |
| vocab | 0.502 | 0.933 | +0.431 | 29 / 0 / 1 | < 0.0001 |
| window | 0.237 | 0.762 | +0.524 | 30 / 0 / 0 | < 0.0001 |
| all | 0.406 | 0.881 | +0.475 | 87 / 1 / 2 | < 0.0001 |

The criterion holds on every tier. The trainer's own step-0 eval (90 × 4, pod vLLM) gives the same verdict from a different untrained sample: 0.587 / 0.548 / 0.187, mean 0.441, every tier p < 0.0001.

Stretch goal, against the untrained 8B (90 × 8, cap 8192): step 20's mean 0.881 is inside the 8B's 95% task-bootstrap interval [0.873, 0.928], so it matches by the stated definition. Per tier, seen and vocab are not distinguishable (two-sided p 0.14 and 0.72); on window the 8B is ahead, 0.863 against 0.762, 20 tasks down against 7, two-sided p 0.014.

Not yet done: the remaining 177 steps. Weights for step 20 are held locally, not published.
