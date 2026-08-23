"""Dataset rows: two fixed prompt templates (SPEC Q1 rider — never blended).

Returns plain lists of dicts so this module (and the tests) need neither
`datasets` nor `verifiers`; env.py converts to HF Datasets.
"""

from __future__ import annotations

import random

from suitesmith.build import build_instance
from suitesmith.families import EVAL_ONLY, TRAINED

SYSTEM_PROMPT = (
    "You write rigorous pytest test suites. Respond with exactly one fenced "
    "python code block containing the test code and nothing else."
)

RULES = """Write a pytest test suite for `{fn_name}`.

Rules:
- Import the function under test with: from solution import {fn_name}
- Call {fn_name} and assert only on its return values (or expected \
exceptions). Do not inspect its source.
- Your suite is scored on two things: it must fully pass against a correct \
implementation of the specification, and it should FAIL on subtly broken \
variants. Target boundaries, ties, ordering, duplicates and empty inputs.
- Output exactly one fenced python code block."""

WHITE_TMPL = """Specification:

{spec}

Reference implementation (guaranteed correct):

```python
{reference}
```

""" + RULES

BLACK_TMPL = """Specification:

{spec}

(No implementation is shown; derive expected behaviour from the \
specification alone.)

""" + RULES


def render_prompt(instance) -> str:
    tmpl = WHITE_TMPL if instance.visibility == "white" else BLACK_TMPL
    return tmpl.format(
        spec=instance.spec, reference=instance.reference, fn_name=instance.fn_name
    )


def _visibility(family: str, seed: int, white_frac: float) -> str:
    r = random.Random(f"vis:{family}:{seed}")
    return "white" if r.random() < white_frac else "black"


def build_rows(family: str, seeds, white_frac: float, mix: str) -> list[dict]:
    rows = []
    for seed in seeds:
        inst = build_instance(family, seed, _visibility(family, seed, white_frac), mix)
        rows.append(
            {
                "question": render_prompt(inst),
                "task": f"{family}:{inst.visibility}",
                "info": inst.to_info(),
            }
        )
    return rows


def build_dataset(
    num_train: int = 400,
    num_eval_seen: int = 60,
    num_eval_unseen: int = 40,
    white_frac: float = 0.7,
    subtlety_mix: str = "easy",
    train_seed_base: int = 0,
    eval_seed_base: int = 100_000,
) -> tuple[list[dict], list[dict]]:
    """Two-level holdout (SPEC Q5): eval = disjoint seeds on trained families
    + the eval-only window family; train never sees either."""
    train_rows: list[dict] = []
    per_fam = -(-num_train // len(TRAINED))  # ceil
    for fam in TRAINED:
        seeds = range(train_seed_base, train_seed_base + per_fam)
        train_rows.extend(build_rows(fam, seeds, white_frac, subtlety_mix))
    train_rows = train_rows[:num_train]

    eval_rows: list[dict] = []
    per_fam = -(-num_eval_seen // len(TRAINED))
    for fam in TRAINED:
        seeds = range(eval_seed_base, eval_seed_base + per_fam)
        eval_rows.extend(build_rows(fam, seeds, white_frac, subtlety_mix))
    eval_rows = eval_rows[:num_eval_seen]
    per_fam = -(-num_eval_unseen // max(len(EVAL_ONLY), 1))
    for fam in EVAL_ONLY:
        seeds = range(eval_seed_base, eval_seed_base + per_fam)
        eval_rows.extend(build_rows(fam, seeds, white_frac, subtlety_mix))

    return train_rows, eval_rows
