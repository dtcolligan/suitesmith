"""verifiers wiring: load_environment() → vf.SingleTurnEnv.

Rubric = one weighted reward (SPEC Q6) + zero-weight metric funcs, which
verifiers logs per rollout without adding them to the reward — that is the
SPEC §7 instrumentation riding the rubric for free. Per-mode splits come from
the `task` column (`family:visibility`).

Defaults encode the 23 Aug operator PROPOSAL for Q8 (white_frac=0.7, easy
subtlety mix, ~50-instance vf-eval); Dom ratifies or amends at calibration.
"""

from __future__ import annotations

import verifiers as vf
from datasets import Dataset

from suitesmith.dataset import SYSTEM_PROMPT, build_dataset
from suitesmith.harness import ScoreResult, extract_suite, score_suite


def load_environment(
    num_train: int = 400,
    num_eval_seen: int = 30,
    num_eval_vocab: int = 30,
    num_eval_window: int = 40,
    white_frac: float = 0.7,
    subtlety_mix: str = "easy",
    timeout: float = 15.0,
    **kwargs,
) -> vf.Environment:
    train_rows, eval_rows = build_dataset(
        num_train=num_train,
        num_eval_seen=num_eval_seen,
        num_eval_vocab=num_eval_vocab,
        num_eval_window=num_eval_window,
        white_frac=white_frac,
        subtlety_mix=subtlety_mix,
    )
    dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)
    parser = vf.Parser(extract_fn=extract_suite)
    memo: dict = {}

    def _score(completion, info) -> ScoreResult:
        suite = parser.parse_answer(completion)
        key = (info["family"], info["seed"], suite)
        if key not in memo:
            if len(memo) > 8192:
                memo.clear()
            memo[key] = score_suite(suite, info, timeout=timeout)
        return memo[key]

    def mutant_kill_reward(completion, info) -> float:
        return _score(completion, info).reward

    def gate_passed(completion, info) -> float:
        return 1.0 if _score(completion, info).gate == "pass" else 0.0

    def twin_failed(completion, info) -> float:
        return 1.0 if _score(completion, info).gate == "twin_failed" else 0.0

    def num_tests(completion, info) -> float:
        return float(_score(completion, info).n_tests)

    def num_asserts(completion, info) -> float:
        return float(_score(completion, info).n_asserts)

    def hack_flag_count(completion, info) -> float:
        return float(len(_score(completion, info).flags))

    rubric = vf.Rubric(
        funcs=[
            mutant_kill_reward,
            gate_passed,
            twin_failed,
            num_tests,
            num_asserts,
            hack_flag_count,
        ],
        weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        parser=parser,
    )
    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
    )
