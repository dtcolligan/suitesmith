# suitesmith

An RL environment, written on the [verifiers](https://github.com/willccbb/verifiers) framework, that trains a small model to write test suites.

- Task: a generated function with a spec, a reference implementation and five mutants. The model writes a pytest suite.
- Reward: 0 unless the suite passes the reference and a behaviourally identical function, else fraction of mutant functions failed.
- Splits: train on three function families. Eval on seen (new instances of those families), vocab (renamed variants) and window (a family never trained on).
- Eval: `.venv/bin/eval suitesmith -m <model> -n 50 -r 8 -c 8 --env.agent.harness.id null --env.agent.runtime.type subprocess --no-push`. Set `SUITESMITH_SPLIT=eval` for the eval split.
- Training run: [training/](training/) — configs, the analysis scripts, and the calibration, baseline and results notes.
- Write-up: [dtcolligan.com/posts/suitesmith](https://dtcolligan.com/posts/suitesmith/)
- Weights: [dtcolligan/suitesmith-qwen3.5-4b-run1-step20](https://huggingface.co/dtcolligan/suitesmith-qwen3.5-4b-run1-step20)

Dominic Colligan, 2026.
