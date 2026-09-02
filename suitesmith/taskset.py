"""Task construction (prime-envs: the taskset side), on the verifiers.v1 API.

Structure per the humaneval/uuid_ctf conventions (Dom's study, 26 Aug):
constants → spec dataclasses → framework-free core (battery-pinned) →
v1 role classes (Data/Task/Taskset) wrapping the core.

TODO(Dom): every body below. The acceptance battery in tests/ pins the
names and behaviour of the framework-free core; the v1 classes follow
the humaneval pattern. Design authority: README §Design specification.
"""

import copy
import json
import os
import random
from dataclasses import dataclass, replace
from pathlib import Path

import verifiers.v1 as vf

# ── constants ────────────────────────────────────────────────────────

TIMEOUT = 5.0
VERIFY = (Path(__file__).parent / "verify.py").read_bytes()  # code as cargo

TRAINED = ("top_k", "filter_agg", "dedup")
EVAL_ONLY = ("window",)

# Q4 subtlety mixes (SPEC: exact portfolios tuned at calibration). The mix
# is a TARGET: selection matches it greedily against what the family's menu
# offers, since menus have fixed per-mutant subtleties.
PORTFOLIOS = {
    "easy": (1, 1, 2, 2, 3),
    "hard": (2, 2, 3, 3, 3),
}

# Q1: two fixed templates, never blended. White shows the reference
# (isolates adversarial input selection); black shows the spec only
# (adds spec-to-oracle derivation).
WHITE_TEMPLATE = """\
You are writing a pytest test suite for one function.

Specification:
{spec}

Reference implementation (correct):
{reference}

Write pytest test functions for `{fn_name}`. Import it with:

    from solution import {fn_name}

Your tests must all pass on the correct implementation above, and should
fail on subtly broken variants of it. Assert only on behaviour (inputs
and outputs) — never on the source text of the implementation. Reply
with ONLY a python code block containing the test functions."""

BLACK_TEMPLATE = """\
You are writing a pytest test suite for one function.

Specification:
{spec}

Write pytest test functions for `{fn_name}`. Import it with:

    from solution import {fn_name}

Derive the expected outputs from the specification. Your tests must all
pass on a correct implementation of the spec, and should fail on subtly
broken implementations. Assert only on behaviour (inputs and outputs).
Reply with ONLY a python code block containing the test functions."""


# ── spec dataclasses (generated-artifact records) ────────────────────

@dataclass(frozen=True)
class MutantSpec:
    """One broken variant. Frozen: immutable once its witness is verified."""

    source: str        # full source of the mutated function
    mclass: int        # Q4 taxonomy class id, 1–10
    subtlety: int      # 1–3
    witness: str       # repr of args demonstrating divergence; "" until verified


# ── framework-free core (names pinned by the battery) ────────────────

def load_fn(source, fn_name):
    """Exec `source`, return the callable named `fn_name`.

    Plain exec, no sandbox: this only ever runs our own generated
    references/mutants/twins (trusted), and the battery contract is an
    in-process callable. Untrusted code (the model's suite) is verify.py's
    problem.
    """
    ns = {}
    exec(source, ns)
    return ns[fn_name]


def run_source(source, fn_name, args):
    """Run one call against `source`; return its result (or the error).

    Errors come back as a value, ("error", <exception type name>), so
    witness search can treat a crash as divergence with one equality
    check. Type name only: same-type/different-message is NOT divergence
    (a message-wording witness would be behaviourally empty).
    Args are deep-copied: generated functions may mutate their inputs,
    and the same args object is reused across reference and mutant runs.
    """
    try:
        return load_fn(source, fn_name)(*copy.deepcopy(list(args)))
    except Exception as e:
        return ("error", type(e).__name__)


class Family:
    """One template family.

    Everything an instance needs derives from `params`, and `params`
    derives from (seed, split) — determinism is the contract. Subclasses
    define: POOLS (every pool split train/eval, Q5-ext), sample_params,
    render (reference + single-mutation variants + the rename-only twin),
    spec, battery, mutations.
    """

    name: str = ""
    POOLS: dict = {}

    def _rng(self, tag, seed, split=""):
        return random.Random(f"{self.name}:{tag}:{seed}:{split}")

    def _pick(self, rng, pool, split):
        return rng.choice(self.POOLS[pool][split])

    def fn_name(self, params):
        return params["fn"]

    def sample_params(self, seed, split="train"):
        raise NotImplementedError

    def render(self, params, mutate=None, twin=False):
        raise NotImplementedError

    def spec(self, params):
        raise NotImplementedError

    def battery(self, params, seed):
        raise NotImplementedError

    def mutations(self, params):
        raise NotImplementedError


class TopK(Family):
    """Family 1: top-k with tie-break.

    fn(records, k) -> the k records ranked by a primary field (direction
    sampled), ties broken by a secondary field (direction sampled).
    Extra keys on records are preserved untouched.
    """

    name = "top_k"
    POOLS = {
        "fn": {
            "train": ["top_records", "select_top", "rank_leaders", "best_k"],
            "eval": ["pick_top_entries", "head_by_rank"],
        },
        "pfield": {
            "train": ["score", "rating", "points", "value"],
            "eval": ["weight", "merit"],
        },
        "tfield": {
            "train": ["age", "cost", "index", "order"],
            "eval": ["seq", "stamp"],
        },
    }

    def sample_params(self, seed, split="train"):
        rng = self._rng("params", seed, split)
        return {
            "fn": self._pick(rng, "fn", split),
            "pf": self._pick(rng, "pfield", split),
            "tf": self._pick(rng, "tfield", split),
            "p_desc": rng.random() < 0.5,
            "t_asc": rng.random() < 0.5,
        }

    def render(self, params, mutate=None, twin=False):
        p_desc = params["p_desc"] ^ (mutate == "flip_primary")
        t_asc = params["t_asc"] ^ (mutate == "flip_tie")
        pf = params["tf"] if mutate == "wrong_field" else params["pf"]
        psign = "-" if p_desc else ""
        tsign = "" if t_asc else "-"
        # k <= 0 returns [] (defined behaviour, 1 Sep 2026): the slice is
        # clamped so the spec, the reference and every mutant agree on it.
        end = "max(k, 0) - 1" if mutate == "off_by_one" else "max(k, 0)"
        var, arg = ("ordered", "item") if twin else ("ranked", "r")
        key = f"lambda {arg}: ({psign}{arg}[{pf!r}], {tsign}{arg}[{params['tf']!r}])"
        if mutate == "drop_tie":
            key = f"lambda {arg}: {psign}{arg}[{pf!r}]"
        body = (
            f"def {params['fn']}(records, k):\n"
            f"    {var} = sorted(records, key={key})\n"
        )
        if twin:
            return body + f"    result = {var}[:{end}]\n    return result\n"
        return body + f"    return {var}[:{end}]\n"

    def spec(self, params):
        pf, tf, fn = params["pf"], params["tf"], params["fn"]
        pdir = "highest first" if params["p_desc"] else "lowest first"
        tdir = "lower" if params["t_asc"] else "higher"
        # Wording fixed 1 Sep 2026 after Q8 calibration: "ordered by the
        # higher <field>" was misread as ascending by gpt-4o-mini 15/16
        # times, even white-box. The k <= 0 case is DEFINED, not guarded:
        # "k is a positive integer" (tried the same day) read as an
        # invitation to test the boundary and assume ValueError (35 of
        # 58 top_k failing tests on baseline-v1). Behaviour stated in the
        # spec and honoured by the reference leaves nothing to guess.
        # Example computed by running the reference: the spec can never
        # contradict the behaviour it documents. It contains a tie, so
        # black-box instances still expose the tie-break rule.
        ex_in = [
            {pf: 8, tf: 3, "id": "x"},
            {pf: 8, tf: 1, "id": "y"},
            {pf: 2, tf: 9, "id": "z"},
        ]
        ex_out = run_source(self.render(params), fn, (ex_in, 2))
        return (
            f"def {fn}(records, k):\n"
            f'    """Return the k records ranked by {pf!r} ({pdir}).\n'
            f"    Ties on {pf!r} are broken by {tf!r}, {tdir} first.\n"
            f"    Records keep all their keys. If k exceeds the number of\n"
            f"    records, return them all (ranked). If k <= 0, return [].\n\n"
            f"    >>> {fn}({ex_in!r}, 2)\n"
            f"    {ex_out!r}\n"
            f'    """\n'
        )

    def battery(self, params, seed):
        pf, tf = params["pf"], params["tf"]

        def rec(p, t, i):
            return {pf: p, tf: t, "id": i}

        base = [rec(9, 5, "b"), rec(9, 2, "a"), rec(1, 0, "low"), rec(7, 7, "m")]
        tied = [rec(5, 4, "c"), rec(5, 1, "d"), rec(5, 3, "e")]
        cases = [
            (base, 2),        # primary tie present: catches drop_tie/flip_tie
            (base, 1),
            (base, 4),        # k == len
            (base, 9),        # k > len
            (base, 0),
            ([], 3),          # empty input
            (tied, 2),        # all tied on primary: tie-break decides fully
        ]
        rng = self._rng("battery", seed)
        for _ in range(3):    # breadth: small value ranges force collisions
            n = rng.randint(3, 7)
            recs = [rec(rng.randint(0, 3), rng.randint(0, 5), f"r{i}") for i in range(n)]
            cases.append((recs, rng.randint(1, n)))
        return cases

    def mutations(self, params):
        menu = [
            ("flip_primary", 7, 1),
            ("wrong_field", 10, 1),
            ("off_by_one", 2, 2),
            ("drop_tie", 6, 3),
            ("flip_tie", 7, 3),
        ]
        return [
            MutantSpec(source=self.render(params, mutate=m), mclass=c,
                       subtlety=s, witness="")
            for m, c, s in menu
        ]


class FilterAgg(Family):
    """Family 2: filter-then-aggregate.

    fn(records, threshold) -> max of a value field over records passing a
    compound filter (threshold comparison AND a positivity guard);
    0 if nothing passes. Hosts the classes top_k structurally can't:
    relational swap (1), wrong-field-collected (3), negation (4),
    dropped conjunct (5), aggregation swap (8), and the empty-guard
    error (9) — max([]) raises where sum([]) wouldn't, which is why the
    aggregation is max, not sum.
    """

    name = "filter_agg"
    POOLS = {
        "fn": {
            "train": ["peak_qualifying", "best_passing", "max_filtered", "top_valid"],
            "eval": ["highest_admitted", "peak_screened"],
        },
        "ffield": {
            "train": ["score", "level", "priority", "size"],
            "eval": ["grade", "tier"],
        },
        "gfield": {
            "train": ["stock", "count", "active", "quota"],
            "eval": ["credit", "slots"],
        },
        "vfield": {
            "train": ["price", "amount", "reward", "budget"],
            "eval": ["payout", "yield_val"],
        },
    }

    def sample_params(self, seed, split="train"):
        rng = self._rng("params", seed, split)
        return {
            "fn": self._pick(rng, "fn", split),
            "ff": self._pick(rng, "ffield", split),
            "gf": self._pick(rng, "gfield", split),
            "vf": self._pick(rng, "vfield", split),
            "op": rng.choice([">=", ">"]),
        }

    def render(self, params, mutate=None, twin=False):
        p = params
        op = {">=": ">", ">": ">="}[p["op"]] if mutate == "rel_swap" else p["op"]
        vfield = p["gf"] if mutate == "field_swap" else p["vf"]
        agg = "min" if mutate == "agg_swap" else "max"
        var, arg = ("chosen", "row") if twin else ("picked", "r")
        cond = f"{arg}[{p['ff']!r}] {op} threshold"
        if mutate != "drop_conjunct":
            cond += f" and {arg}[{p['gf']!r}] > 0"
        if mutate == "negate":
            cond = f"not ({cond})"
        lines = [
            f"def {p['fn']}(records, threshold):",
            f"    {var} = [{arg}[{vfield!r}] for {arg} in records if {cond}]",
        ]
        if mutate != "bad_empty":
            if twin:
                lines += [f"    if len({var}) == 0:", "        return 0"]
            else:
                lines += [f"    if not {var}:", "        return 0"]
        lines.append(f"    return {agg}({var})")
        return "\n".join(lines) + "\n"

    def spec(self, params):
        p = params
        opw = "at least" if p["op"] == ">=" else "strictly above"
        ex_in = [
            {p["ff"]: 5, p["gf"]: 2, p["vf"]: 7},
            {p["ff"]: 9, p["gf"]: 0, p["vf"]: 50},
            {p["ff"]: 1, p["gf"]: 3, p["vf"]: 100},
        ]
        ex_out = run_source(self.render(p), p["fn"], (ex_in, 5))
        return (
            f"def {p['fn']}(records, threshold):\n"
            f'    """Return the largest {p["vf"]!r} among records whose\n'
            f"    {p['ff']!r} is {opw} threshold and whose {p['gf']!r} is\n"
            f"    positive. Return 0 if no record qualifies.\n\n"
            f"    >>> {p['fn']}({ex_in!r}, 5)\n"
            f"    {ex_out!r}\n"
            f'    """\n'
        )

    def battery(self, params, seed):
        ff, gf, vf = params["ff"], params["gf"], params["vf"]

        def rec(f, g, v):
            return {ff: f, gf: g, vf: v}

        base = [
            rec(5, 1, 7),     # ff == threshold 5: rel_swap witness
            rec(9, 2, 3),     # second passer, distinct value: agg/field swaps
            rec(9, 0, 50),    # passes threshold, fails guard: drop_conjunct
            rec(1, 1, 100),   # fails threshold: negation flips it in
        ]
        cases = [
            (base, 5),
            (base, 0),
            (base, 100),      # nothing passes: bad_empty raises here
            ([], 1),          # empty input
        ]
        rng = self._rng("battery", seed)
        for _ in range(3):
            n = rng.randint(3, 6)
            recs = [rec(rng.randint(0, 6), rng.randint(-1, 3), rng.randint(-5, 9))
                    for _ in range(n)]
            cases.append((recs, rng.randint(0, 6)))
        return cases

    def mutations(self, params):
        menu = [
            ("negate", 4, 1),
            ("field_swap", 3, 1),
            ("rel_swap", 1, 2),
            ("drop_conjunct", 5, 2),
            ("agg_swap", 8, 2),
            ("bad_empty", 9, 3),
        ]
        return [
            MutantSpec(source=self.render(params, mutate=m), mclass=c,
                       subtlety=s, witness="")
            for m, c, s in menu
        ]


class Dedup(Family):
    """Family 4: dedup with a precedence rule.

    fn(records) -> one record per key, keeping the best-precedence
    record for each key; output preserves first-seen key order (dict
    insertion order — replacement keeps the slot).
    """

    name = "dedup"
    POOLS = {
        "fn": {
            "train": ["dedupe_records", "unique_best", "collapse_dupes", "keep_best"],
            "eval": ["fold_duplicates", "prune_repeats"],
        },
        "kfield": {
            "train": ["name", "sku", "label", "code"],
            "eval": ["handle", "ref"],
        },
        "pfield": {
            "train": ["version", "priority", "quality", "freshness"],
            "eval": ["salience", "grade_num"],
        },
    }

    def sample_params(self, seed, split="train"):
        rng = self._rng("params", seed, split)
        return {
            "fn": self._pick(rng, "fn", split),
            "kf": self._pick(rng, "kfield", split),
            "pf": self._pick(rng, "pfield", split),
            "keep_max": rng.random() < 0.5,
        }

    def render(self, params, mutate=None, twin=False):
        p = params
        keep_max = p["keep_max"] ^ (mutate == "flip_precedence")
        cmp = ">" if keep_max else "<"
        kf = p["pf"] if mutate == "wrong_key" else p["kf"]
        d, arg, k = ("best", "entry", "marker") if twin else ("kept", "r", "key")
        member = f"{k} not in {d}"
        better = f"{arg}[{p['pf']!r}] {cmp} {d}[{k}][{p['pf']!r}]"
        cond = f"{member} or {better}"
        if mutate == "drop_precedence":
            cond = member
        elif mutate == "drop_membership":
            cond = better
        elif mutate == "negate":
            cond = f"not ({cond})"
        return (
            f"def {p['fn']}(records):\n"
            f"    {d} = {{}}\n"
            f"    for {arg} in records:\n"
            f"        {k} = {arg}[{kf!r}]\n"
            f"        if {cond}:\n"
            f"            {d}[{k}] = {arg}\n"
            f"    return list({d}.values())\n"
        )

    def spec(self, params):
        p = params
        best = "highest" if p["keep_max"] else "lowest"
        ex_in = [
            {p["kf"]: "a", p["pf"]: 1},
            {p["kf"]: "b", p["pf"]: 5},
            {p["kf"]: "a", p["pf"]: 9},
        ]
        ex_out = run_source(self.render(p), p["fn"], (ex_in,))
        return (
            f"def {p['fn']}(records):\n"
            f'    """Return one record per distinct {p["kf"]!r}: the one\n'
            f"    with the {best} {p['pf']!r}. Order of the result follows\n"
            f"    the first appearance of each {p['kf']!r} in the input.\n\n"
            f"    >>> {p['fn']}({ex_in!r})\n"
            f"    {ex_out!r}\n"
            f'    """\n'
        )

    def battery(self, params, seed):
        kf, pf = params["kf"], params["pf"]

        def rec(k, p):
            return {kf: k, pf: p}

        cases = [
            ([rec("a", 1), rec("b", 5), rec("a", 9), rec("b", 2), rec("c", 3)],),
            ([rec("a", 7), rec("b", 7)],),   # same precedence, different keys:
            #                                  wrong_key merges them, ref keeps both
            ([rec("z", -4), rec("z", -9)],),  # negative precedence values
            ([rec("q", 3)],),
            ([],),
        ]
        rng = self._rng("battery", seed)
        for _ in range(3):
            n = rng.randint(3, 7)
            recs = [rec(rng.choice("abc"), rng.randint(-2, 6)) for _ in range(n)]
            cases.append((recs,))
        return cases

    def mutations(self, params):
        menu = [
            ("negate", 4, 1),
            ("wrong_key", 10, 1),
            ("drop_membership", 5, 1),
            ("flip_precedence", 7, 2),
            ("drop_precedence", 6, 3),
        ]
        return [
            MutantSpec(source=self.render(params, mutate=m), mclass=c,
                       subtlety=s, witness="")
            for m, c, s in menu
        ]


class Window(Family):
    """Family 5 (Q5: EVAL-ONLY): threshold windowing.

    fn(values, lo, hi) -> the values inside the window, input order
    preserved. Inclusivity of each boundary is sampled per instance —
    the boundary-heavy family, a deliberate stress test of the
    relational-swap skill on a shape never trained on.
    """

    name = "window"
    POOLS = {
        "fn": {
            "train": ["within_band", "in_window", "clip_range", "between_bounds"],
            "eval": ["inside_limits", "band_pass"],
        },
    }

    def sample_params(self, seed, split="train"):
        rng = self._rng("params", seed, split)
        return {
            "fn": self._pick(rng, "fn", split),
            "incl_lo": rng.random() < 0.5,
            "incl_hi": rng.random() < 0.5,
        }

    def render(self, params, mutate=None, twin=False):
        p = params
        incl_lo = p["incl_lo"] ^ (mutate == "rel_swap_lo")
        incl_hi = p["incl_hi"] ^ (mutate == "rel_swap_hi")
        lo_op = ">=" if incl_lo else ">"
        hi_op = "<=" if incl_hi else "<"
        var, arg = ("window", "val") if twin else ("out", "v")
        cond = f"{arg} {lo_op} lo"
        if mutate != "drop_hi":
            cond += f" and {arg} {hi_op} hi"
        if mutate == "negate":
            cond = f"not ({cond})"
        ret = f"sorted({var})" if mutate == "sort_output" else var
        return (
            f"def {p['fn']}(values, lo, hi):\n"
            f"    {var} = []\n"
            f"    for {arg} in values:\n"
            f"        if {cond}:\n"
            f"            {var}.append({arg})\n"
            f"    return {ret}\n"
        )

    def spec(self, params):
        p = params
        low = "inclusive" if p["incl_lo"] else "exclusive"
        high = "inclusive" if p["incl_hi"] else "exclusive"
        ex_in = [5, 1, 9, 3, 7]
        ex_out = run_source(self.render(p), p["fn"], (ex_in, 3, 7))
        return (
            f"def {p['fn']}(values, lo, hi):\n"
            f'    """Return the values between lo ({low}) and hi ({high}),\n'
            f"    in their original order.\n\n"
            f"    >>> {p['fn']}({ex_in!r}, 3, 7)\n"
            f"    {ex_out!r}\n"
            f'    """\n'
        )

    def battery(self, params, seed):
        cases = [
            ([5, 1, 9, 3, 5, 10, 0, 7], 3, 7),   # both boundaries hit, unsorted, dupes
            ([6, 4, 5], 3, 8),                    # all strictly inside, unsorted:
            #                                       sort_output witness under ANY inclusivity
            ([4, 4, 4], 4, 4),                    # degenerate window, all-boundary
            ([-5, -1, 0, 3], -1, 2),              # negatives, lo boundary hit
            ([2, 8, 6], 7, 3),                    # lo > hi: empty window
            ([], 0, 10),                          # empty input
        ]
        rng = self._rng("battery", seed)
        for _ in range(3):
            n = rng.randint(3, 8)
            vals = [rng.randint(-3, 9) for _ in range(n)]
            lo, hi = sorted((rng.randint(-3, 9), rng.randint(-3, 9)))
            cases.append((vals, lo, hi))
        return cases

    def mutations(self, params):
        menu = [
            ("negate", 4, 1),
            ("drop_hi", 5, 2),
            ("sort_output", 6, 2),
            ("rel_swap_lo", 1, 3),
            ("rel_swap_hi", 1, 3),
        ]
        return [
            MutantSpec(source=self.render(params, mutate=m), mclass=c,
                       subtlety=s, witness="")
            for m, c, s in menu
        ]


FAMILIES = {
    "top_k": TopK(),
    "filter_agg": FilterAgg(),
    "dedup": Dedup(),
    "window": Window(),
}


def render_prompt(data, visibility: str) -> str:
    """Fill the (fixed) template for the given visibility mode."""
    if isinstance(data, dict):
        spec, reference, fn = data["spec"], data["reference"], data["fn_name"]
    else:
        spec, reference, fn = data.spec, data.reference, data.fn_name
    tmpl = WHITE_TEMPLATE if visibility == "white" else BLACK_TEMPLATE
    return tmpl.format(spec=spec, reference=reference, fn_name=fn)


def build_instance(family, seed, visibility, mix, split="train"):
    """Assemble one SuitesmithData: call the family, generate mutants,
    enforce witnesses (discard equivalents), build the twin, fill fields.
    Pure function of its arguments — determinism is the contract."""
    fam = FAMILIES[family]
    params = fam.sample_params(seed, split)
    fn = fam.fn_name(params)
    reference = fam.render(params)
    twin = fam.render(params, twin=True)
    cases = fam.battery(params, seed)

    # Twin gate precondition (SPEC §4): behavioural identity verified at
    # generation time, on the same battery that filters mutants.
    for args in cases:
        if run_source(reference, fn, args) != run_source(twin, fn, args):
            raise RuntimeError(f"{family}/{seed}: twin diverges on {args!r}")

    # Q4 hard rule (a): every mutant must provably differ from the
    # reference. First diverging battery case becomes the stored witness;
    # menu entries with no witness are behaviourally equivalent — discard.
    witnessed = []
    for m in fam.mutations(params):
        ref_outs = (run_source(reference, fn, args) for args in cases)
        for args, ref_out in zip(cases, ref_outs):
            if run_source(m.source, fn, args) != ref_out:
                # Stored as repr so it is JSON-safe in to_info() and can be
                # pasted straight into a test call: fn(*eval(witness)).
                witnessed.append(replace(m, witness=repr(args)))
                break
    if len(witnessed) < 5:
        raise RuntimeError(
            f"{family}/{seed}/{mix}: only {len(witnessed)} distinguishable "
            f"mutants — battery too weak for this params draw"
        )

    # Portfolio selection: greedy match against the target subtlety mix
    # (menus have fixed per-mutant subtleties, so the mix is a target,
    # not a guarantee). Deterministic: stable order, index tiebreak.
    chosen = []
    avail = list(witnessed)
    for target in PORTFOLIOS[mix]:
        best = min(avail, key=lambda m: abs(m.subtlety - target))
        avail.remove(best)
        chosen.append(best)
    assert len({m.source for m in chosen}) == 5, "duplicate mutant sources"

    spec = fam.spec(params)
    prompt = render_prompt(
        {"spec": spec, "reference": reference, "fn_name": fn}, visibility
    )
    return SuitesmithData(
        name=f"{family}:{split}:{seed}:{visibility}",
        prompt=prompt,
        family=family,
        seed=seed,
        split=split,
        visibility=visibility,
        mix=mix,
        fn_name=fn,
        spec=spec,
        reference=reference,
        twin=twin,
        mutants=chosen,
    )


def build_dataset(num_train=0, num_eval_seen=0, num_eval_vocab=0,
                  num_eval_window=0, white_frac=0.7, mix="easy"):
    """Loop families × seeds → (train_rows, eval_rows), eval rows
    labelled by tier: seen / vocab / template (Q5-ext).

    Disjoint seed ranges per tier (Q5b); the vocab and template tiers
    additionally draw from the eval side of every pool (Q5-ext)."""
    rng = random.Random("dataset:visibility")
    trained = list(TRAINED)

    def vis():
        return "white" if rng.random() < white_frac else "black"

    def row(inst, tier):
        return {
            "question": inst.prompt,
            "task": f"suitesmith:{inst.visibility}:{tier}",
            "info": inst.to_info(),
        }

    train_rows = [
        row(build_instance(trained[i % len(trained)], i, vis(), mix, "train"),
            "train")
        for i in range(num_train)
    ]
    eval_rows = [
        row(build_instance(trained[i % len(trained)], 10_000 + i, vis(), mix,
                           "train"), "seen")
        for i in range(num_eval_seen)
    ] + [
        row(build_instance(trained[i % len(trained)], 20_000 + i, vis(), mix,
                           "eval"), "vocab")
        for i in range(num_eval_vocab)
    ] + [
        row(build_instance(EVAL_ONLY[0], 30_000 + i, vis(), mix, "eval"),
            "template")
        for i in range(num_eval_window)
    ]
    return train_rows, eval_rows


# ── verifiers.v1 role classes (humaneval pattern) ────────────────────
# Probe results (exp_taskdata.py, 31 Aug): vf.TaskData subclasses
# cleanly (framework fields arrive with defaults) and list[MutantSpec]
# nests as-is — no pydantic rewrite needed. Tier lives in the dataset
# row's task string, not on the instance: an instance doesn't know its
# tier, the dataset assembly does.

# Default taskset size (training side); calibration (Q8) may revise.
NUM_TRAIN = 200
WHITE_FRAC = 0.7
# Eval split (Q5-ext tiers): seen = trained families, unseen seeds;
# vocab = trained families, eval side of every pool; template = the
# eval-only window family. Selected with SUITESMITH_SPLIT=eval.
NUM_EVAL_SEEN = 30
NUM_EVAL_VOCAB = 30
NUM_EVAL_WINDOW = 30


class SuitesmithData(vf.TaskData):
    family: str
    seed: int
    split: str         # "train" | "eval" (which side of the pools)
    visibility: str    # "white" | "black"
    mix: str           # PORTFOLIOS key used for mutant selection
    fn_name: str
    spec: str
    reference: str
    twin: str
    mutants: list[MutantSpec]

    def to_info(self) -> dict:
        """JSON-safe payload: everything verify.py needs to score a suite."""
        return {
            "family": self.family,
            "seed": self.seed,
            "split": self.split,
            "visibility": self.visibility,
            "mix": self.mix,
            "fn_name": self.fn_name,
            "reference": self.reference,
            "twin": self.twin,
            "mutants": [
                {"source": m.source, "mclass": m.mclass,
                 "subtlety": m.subtlety, "witness": m.witness}
                for m in self.mutants
            ],
        }


class SuitesmithTask(vf.Task[SuitesmithData]):
    @vf.reward(weight=1.0)
    async def suite_reward(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        # Lazy import: verify.py is built as its own part; taskset must
        # import (for the battery) before verify exists.
        from suitesmith.verify import extract_suite

        suite = extract_suite(trace.last_reply)
        if not suite.strip():
            # malformed gate: nothing to run, before any sandbox
            self._log({"gate": "malformed", "reward": 0.0, "killed": 0,
                       "n_tests": 0, "flags": []})
            return 0.0
        # Payload via the process environment, never a shared file: with
        # eight rollouts of one task in flight, a file under /tmp is a
        # sibling's answer key (mutants + witnesses) one glob away.
        # verify.py pops it before any suite runs.
        result = await runtime.run_uv_script(
            VERIFY, args=[str(TIMEOUT)],
            env={"SUITESMITH_PAYLOAD": json.dumps({"suite": suite, **self.data.to_info()})},
        )
        if result.exit_code != 0:
            # A reward is a verdict: verifier crash ⇒ raise, never 0.0.
            raise RuntimeError(f"verify.py failed: {result.stderr.strip()[-1000:]}")
        verdict = json.loads(result.stdout.strip().splitlines()[-1])
        self._log(verdict)
        if verdict["gate"] == "error":
            # The runner broke (driver crashed), not the suite. Same rule
            # as above: our failures raise, they never score 0.0.
            raise RuntimeError(f"verify.py error gate: {verdict}")
        return float(verdict["reward"])

    def _log(self, verdict: dict) -> None:
        # README's S3 net: one JSON line per scored suite, appended live
        # so calibration can be watched with tail -f. Off unless
        # SUITESMITH_LOG names a file.
        log_path = os.environ.get("SUITESMITH_LOG")
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "name": self.data.name,
                    "visibility": self.data.visibility,
                    **verdict,
                }) + "\n")


class SuitesmithConfig(vf.TasksetConfig):
    # "train" (50 tasks, three families) or "eval" (seen 30 / vocab 30 /
    # window 30). Typed so prime-rl can run both splits in one env server
    # (--env.taskset.split eval); SUITESMITH_SPLIT=eval remains as the
    # fallback for the CLI.
    split: str = os.environ.get("SUITESMITH_SPLIT", "train")


class SuitesmithTaskset(vf.Taskset[SuitesmithTask, SuitesmithConfig]):
    def load(self) -> list["SuitesmithTask"]:
        train_rows, eval_rows = build_dataset(
            num_train=NUM_TRAIN, num_eval_seen=NUM_EVAL_SEEN,
            num_eval_vocab=NUM_EVAL_VOCAB, num_eval_window=NUM_EVAL_WINDOW,
            white_frac=WHITE_FRAC)
        if self.config.split not in ("train", "eval"):
            raise ValueError(f"split must be train or eval, got {self.config.split!r}")
        rows = eval_rows if self.config.split == "eval" else train_rows
        tasks = []
        for i, r in enumerate(rows):
            info = r["info"]
            data = build_instance(info["family"], info["seed"],
                                  info["visibility"], info["mix"],
                                  info["split"])
            # TaskData is frozen; stamp the row index via a copy.
            data = data.model_copy(update={"idx": i})
            tasks.append(SuitesmithTask(data, config=self.config.task))
        return tasks


__all__ = ["SuitesmithData", "SuitesmithTask", "SuitesmithTaskset", "SuitesmithConfig",
           "FAMILIES", "build_instance", "build_dataset"]
