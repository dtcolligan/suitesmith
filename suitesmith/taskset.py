"""Task construction (prime-envs: the taskset side), on the verifiers.v1 API.

Structure per the humaneval/uuid_ctf conventions (Dom's study, 26 Aug):
constants → spec dataclasses → framework-free core (battery-pinned) →
v1 role classes (Data/Task/Taskset) wrapping the core.

TODO(Dom): every body below. The acceptance battery in tests/ pins the
names and behaviour of the framework-free core; the v1 classes follow
the humaneval pattern. Design authority: README §Design specification.
"""

import random
from dataclasses import dataclass
from pathlib import Path

import verifiers.v1 as vf

# ── constants ────────────────────────────────────────────────────────

TIMEOUT = 5.0
VERIFY = (Path(__file__).parent / "verify.py").read_bytes()  # code as cargo

TRAINED = ("top_k", "filter_agg", "dedup")
EVAL_ONLY = ("window",)

# TODO(Dom): Q5-ext — every pool split train/eval (field names, fn names)
POOLS: dict = {}

# TODO(Dom): Q4 — subtlety mixes, uuid_ctf's DIFFICULTY_PRESETS pattern
PORTFOLIOS: dict = {}

# TODO(Dom): Q1 — two fixed templates, never blended
WHITE_TEMPLATE = ""
BLACK_TEMPLATE = ""


# ── spec dataclasses (generated-artifact records) ────────────────────

@dataclass(frozen=True)
class MutantSpec:
    """One broken variant. Frozen: immutable once its witness is verified."""

    source: str        # full source of the mutated function
    mut_class: str     # Q4 taxonomy class name
    subtlety: int      # 1–3
    witness: tuple     # input demonstrating divergence from the reference


# ── framework-free core (names pinned by the battery) ────────────────

def load_fn(source, fn_name):
    """Exec `source`, return the callable named `fn_name`."""
    raise NotImplementedError  # TODO(Dom)


def run_source(source, fn_name, args):
    """Run one call against `source`; return its result (or the error)."""
    raise NotImplementedError  # TODO(Dom)


def build_top_k(rng: random.Random, split: str):
    """Family 1: top-k with tie-break. Returns the family's raw parts."""
    raise NotImplementedError  # TODO(Dom): Thursday opens here


def build_filter_agg(rng: random.Random, split: str):
    raise NotImplementedError  # TODO(Dom)


def build_dedup(rng: random.Random, split: str):
    raise NotImplementedError  # TODO(Dom)


def build_window(rng: random.Random, split: str):
    """Eval-only family (Q5): never trained on."""
    raise NotImplementedError  # TODO(Dom)


FAMILIES = {
    "top_k": build_top_k,
    "filter_agg": build_filter_agg,
    "dedup": build_dedup,
    "window": build_window,
}


def render_prompt(data, visibility: str) -> str:
    """Fill the (fixed) template for the given visibility mode."""
    raise NotImplementedError  # TODO(Dom)


def build_instance(family, seed, visibility, mix, split="train"):
    """Assemble one SuitesmithData: call the family, generate mutants,
    enforce witnesses (discard equivalents), build the twin, fill fields.
    Pure function of its arguments — determinism is the contract."""
    raise NotImplementedError  # TODO(Dom)


def build_dataset(*args, **kwargs):
    """Loop families × seeds → (train_rows, eval_rows), eval rows
    labelled by tier: seen / vocab / template (Q5-ext)."""
    raise NotImplementedError  # TODO(Dom)


# ── verifiers.v1 role classes (humaneval pattern) ────────────────────
# TODO(Dom): first experiment Thursday — subclass vf.TaskData, check its
# required fields and whether list[MutantSpec] nests cleanly (else make
# MutantSpec a pydantic model). Until then these stay commented so the
# module imports cleanly for the battery.
#
# class SuitesmithData(vf.TaskData):
#     family: str
#     seed: int
#     split: str
#     visibility: str    # "white" | "black"
#     tier: str          # "seen" | "vocab" | "template"
#     fn_name: str
#     spec: str
#     reference: str
#     twin: str
#     mutants: list[MutantSpec]
#
#     def to_info(self) -> dict: ...
#
#
# class SuitesmithTask(vf.Task[SuitesmithData]):
#     @vf.reward(weight=1.0)
#     async def suite_reward(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
#         # extract_suite → empty ⇒ 0.0 (malformed gate, before any sandbox)
#         # payload = suite + self.data.to_info() → runtime.write(...)
#         # runtime.run_uv_script(VERIFY, ...) → parse verdict → reward
#         # verifier crash ⇒ raise, never 0.0 (a reward is a verdict)
#         ...
#
#
# class SuitesmithTaskset(vf.Taskset[SuitesmithTask, vf.TasksetConfig]):
#     def load(self) -> list["SuitesmithTask"]:
#         # thin wrapper: stamp SuitesmithTask over build_dataset(...)
#         ...
#
#
# __all__ = ["SuitesmithTask", "SuitesmithTaskset"]
