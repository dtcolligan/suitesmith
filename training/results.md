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

Not yet done: the paired permutation test per tier on step 0 versus step 20; the remaining 177 steps. Weights for step 20 are held locally, not published.
