"""Instance construction: mutant portfolio selection + distinguishability.

SPEC Q4 hard rules live here: every mutant must differ behaviourally from the
reference on at least one battery input (witness stored); equivalent renders
are replaced from the family's unused menu, and generation fails loudly if a
full portfolio cannot be assembled.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

from suitesmith.families import FAMILIES, MutationSpec, load_fn

PORTFOLIOS = {"easy": (1, 1, 2, 2, 3), "hard": (2, 2, 3, 3, 3)}
N_MUTANTS = 5


class EquivalentMutantError(RuntimeError):
    pass


@dataclass(frozen=True)
class Mutant:
    mclass: int
    name: str
    subtlety: int
    source: str
    witness: str  # repr of the args tuple where behaviour first differs


@dataclass(frozen=True)
class Instance:
    family: str
    seed: int
    visibility: str  # "white" | "black"
    split: str  # "train" | "eval" — which name pools params drew from
    fn_name: str
    spec: str
    reference: str
    twin: str  # rename-only reference variant; the gate requires passing it too
    mutants: tuple = field(default_factory=tuple)

    def to_info(self) -> dict:
        d = asdict(self)
        d["mutants"] = [asdict(m) for m in self.mutants]
        return d


def _outcome(fn, args):
    try:
        return ("ok", fn(*copy.deepcopy(list(args))))
    except Exception as e:  # noqa: BLE001 — any behavioural difference counts
        return ("raise", type(e).__name__)


def find_witness(ref_src: str, mut_src: str, fn_name: str, battery) -> str | None:
    ref_fn = load_fn(ref_src, fn_name)
    try:
        mut_fn = load_fn(mut_src, fn_name)
    except Exception:
        return repr(battery[0]) if battery else None
    for args in battery:
        if _outcome(ref_fn, args) != _outcome(mut_fn, args):
            return repr(args)
    return None


def _pick_portfolio(menu: list[MutationSpec], portfolio, rng) -> list[MutationSpec]:
    pool = list(menu)
    chosen = []
    for target in portfolio:
        m = min(pool, key=lambda x: (abs(x.subtlety - target), rng.random()))
        pool.remove(m)
        chosen.append(m)
    return chosen


def build_instance(
    family: str, seed: int, visibility: str, mix: str = "easy", split: str = "train"
) -> Instance:
    fam = FAMILIES[family]
    params = fam.sample_params(seed, split)
    fn_name = fam.fn_name(params)
    ref_src = fam.render(params)
    battery = fam.battery(params, seed)
    twin_src = fam.twin(params)
    if twin_src == ref_src:
        raise EquivalentMutantError(f"{family}:{seed}: twin is textually identical")
    if find_witness(ref_src, twin_src, fn_name, battery) is not None:
        raise EquivalentMutantError(f"{family}:{seed}: twin diverges behaviourally")
    menu = fam.mutations(params)
    if len(menu) < N_MUTANTS:
        raise EquivalentMutantError(f"{family}:{seed}: menu smaller than N={N_MUTANTS}")
    rng = fam.rng(seed, "mutants")
    chosen = _pick_portfolio(menu, PORTFOLIOS[mix], rng)
    unused = [m for m in menu if m not in chosen]

    mutants = []
    for m in chosen:
        src = fam.render(params, knob=m.knob)
        w = find_witness(ref_src, src, fn_name, battery)
        while w is None and unused:
            m = unused.pop(0)
            src = fam.render(params, knob=m.knob)
            w = find_witness(ref_src, src, fn_name, battery)
        if w is None:
            raise EquivalentMutantError(f"{family}:{seed}: knob {m.knob} equivalent on battery")
        mutants.append(Mutant(m.mclass, m.name, m.subtlety, src, w))

    return Instance(
        family=family,
        seed=seed,
        visibility=visibility,
        split=split,
        fn_name=fn_name,
        spec=fam.spec(params),
        reference=ref_src,
        twin=twin_src,
        mutants=tuple(mutants),
    )
