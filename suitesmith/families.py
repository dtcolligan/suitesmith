"""Template families: parameterised spec + reference generators with mutation menus.

Every instance is fully derived from (family, seed) — SPEC Q5b. The reference
is correct by construction; a mutant is a re-render of the same template with
exactly one knob flipped (one mutation per mutant, SPEC Q4b). Docstrings fully
determine behaviour, including tie and boundary semantics, so black-box
instances are solvable from the spec alone (SPEC Q1).
"""

from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MutationSpec:
    knob: str      # family-internal identifier, passed back to render()
    mclass: int    # 1..10 taxonomy class (SPEC Q4)
    name: str
    subtlety: int  # 1 = dies to almost any test .. 3 = spec-reading tests only


def load_fn(source: str, fn_name: str):
    ns: dict = {}
    exec(source, ns)
    return ns[fn_name]


def run_source(source: str, fn_name: str, args):
    return load_fn(source, fn_name)(*copy.deepcopy(list(args)))


class Family:
    name: str = ""
    trained: bool = True
    RENAMES: dict[str, str] = {}

    def rng(self, seed: int, stream: str) -> random.Random:
        return random.Random(f"{self.name}:{seed}:{stream}")

    def sample_params(self, seed: int, split: str = "train") -> dict:
        raise NotImplementedError

    def twin(self, params: dict) -> str:
        """Behaviourally identical, textually different reference variant.

        The gate requires suites to pass this too (decided by Dom 23 Aug):
        a source-fingerprinting suite passes the reference but fails the
        twin, so it earns 0. The only invariant shared by reference and
        twin is behaviour — which is the only thing tests may assert on.
        """
        src = self.render(params)
        for old, new in self.RENAMES.items():
            src = re.sub(rf"\b{old}\b", new, src)
        return src

    def fn_name(self, params: dict) -> str:
        return params["fn"]

    def render(self, params: dict, knob: str | None = None) -> str:
        raise NotImplementedError

    def spec(self, params: dict) -> str:
        raise NotImplementedError

    def mutations(self, params: dict) -> list[MutationSpec]:
        raise NotImplementedError

    def battery(self, params: dict, seed: int) -> list[tuple]:
        raise NotImplementedError

    def _signed_spec(self, params: dict, docstring: str, argnames: str) -> str:
        body = "\n".join("    " + line if line else "" for line in docstring.splitlines())
        return f'def {params["fn"]}({argnames}):\n    """\n{body}\n    """\n'

    def _example(self, params: dict, args) -> str:
        out = run_source(self.render(params), self.fn_name(params), args)
        call = ", ".join(repr(a) for a in args)
        return f">>> {params['fn']}({call})\n{out!r}"


class TopK(Family):
    """Family 1 (SPEC Q3): top-k with tie-break."""

    name = "top_k"
    POOLS = {
        "pf": {
            "train": ["score", "priority", "rating", "weight", "impact", "strength", "level", "quality"],
            "eval": ["merit", "urgency", "magnitude", "prominence"],
        },
        "tf": {
            "train": ["ts", "age", "position", "cost", "distance", "duration", "delay", "batch"],
            "eval": ["latency", "offset", "depth", "tenure"],
        },
        "fn": {
            "train": ["top_records", "pick_top", "select_leading", "best_records", "leading_entries", "rank_and_take"],
            "eval": ["strongest_records", "take_top"],
        },
    }
    RENAMES = {"ordered": "ranked"}

    def sample_params(self, seed: int, split: str = "train") -> dict:
        r = self.rng(seed, "params")
        return {
            "fn": r.choice(self.POOLS["fn"][split]),
            "pf": r.choice(self.POOLS["pf"][split]),
            "tf": r.choice(self.POOLS["tf"][split]),
            "p_desc": r.random() < 0.75,
            "t_asc": r.random() < 0.75,
        }

    def render(self, params: dict, knob: str | None = None) -> str:
        p_sign = "-" if params["p_desc"] else ""
        t_sign = "" if params["t_asc"] else "-"
        if knob == "flip_primary_dir":
            p_sign = "" if p_sign else "-"
        if knob == "flip_tie_dir":
            t_sign = "" if t_sign else "-"
        pf = params["tf"] if knob == "wrong_primary_field" else params["pf"]
        key = f'lambda r: ({p_sign}r["{pf}"], {t_sign}r["{params["tf"]}"])'
        if knob == "drop_tiebreak":
            key = f'lambda r: {p_sign}r["{pf}"]'
        slice_expr = "ordered[:k - 1]" if knob == "off_by_one" else "ordered[:k]"
        return (
            f'def {params["fn"]}(records, k):\n'
            f"    ordered = sorted(records, key={key})\n"
            f"    return {slice_expr}\n"
        )

    def spec(self, params: dict) -> str:
        pf, tf = params["pf"], params["tf"]
        hi = "highest" if params["p_desc"] else "lowest"
        tie = "smallest" if params["t_asc"] else "largest"
        ex_records = [
            {pf: 3, tf: 2, "id": "a"},
            {pf: 5, tf: 9, "id": "b"},
            {pf: 3, tf: 1, "id": "c"},
            {pf: 5, tf: 4, "id": "d"},
        ]
        doc = (
            f'Return the k records with the {hi} "{pf}" value.\n'
            "\n"
            f'Records are dicts with integer fields "{pf}" and "{tf}" and a\n'
            f'string field "id". Ties on "{pf}" are broken by preferring the\n'
            f'{tie} "{tf}"; records tied on both fields keep their original\n'
            "relative order. If k exceeds the number of records, all records\n"
            "are returned (ordered the same way). k may be 0.\n"
            "\n"
            "Example:\n"
            + self._example(params, (ex_records, 3))
        )
        return self._signed_spec(params, doc, "records, k")

    def mutations(self, params: dict) -> list[MutationSpec]:
        return [
            MutationSpec("off_by_one", 2, "off-by-one slice", 2),
            MutationSpec("drop_tiebreak", 6, "dropped tie-break", 3),
            MutationSpec("flip_tie_dir", 7, "tie direction flipped", 3),
            MutationSpec("flip_primary_dir", 7, "primary direction flipped", 1),
            MutationSpec("wrong_primary_field", 10, "wrong primary field", 1),
        ]

    def battery(self, params: dict, seed: int) -> list[tuple]:
        r = self.rng(seed, "battery")
        pf, tf = params["pf"], params["tf"]

        def rec(p, t, i):
            return {pf: p, tf: t, "id": f"r{i}"}

        tied = [rec(5, 3, 0), rec(5, 1, 1), rec(2, 9, 2), rec(5, 2, 3)]
        both_tied = [rec(4, 4, 0), rec(4, 4, 1), rec(4, 4, 2), rec(1, 0, 3)]
        desc_tf = [rec(3, 9, 0), rec(3, 5, 1), rec(3, 1, 2)]
        cases = [
            ([], 2),
            ([rec(1, 1, 0)], 3),
            (tied, 2),
            (tied, 4),
            (tied, 0),
            (both_tied, 3),
            (desc_tf, 2),
        ]
        for _ in range(12):
            n = r.randint(2, 7)
            recs = [rec(r.randint(0, 3), r.randint(0, 3), i) for i in range(n)]
            cases.append((recs, r.randint(1, n)))
        return cases


class FilterAgg(Family):
    """Family 2 (SPEC Q3): filter-then-aggregate with a compound filter."""

    name = "filter_agg"
    POOLS = {
        "ff": {
            "train": ["amount", "size", "hours", "load", "mass", "span", "width", "volume"],
            "eval": ["tonnage", "breadth", "quota", "heft"],
        },
        "af": {
            "train": ["value", "points", "units", "credit", "revenue", "bonus", "gain", "output"],
            "eval": ["profit", "dividend", "surplus", "payout"],
        },
        "flagf": {
            "train": ["active", "valid", "approved", "enabled"],
            "eval": ["confirmed", "eligible"],
        },
        "fn": {
            "train": ["aggregate_qualifying", "tally_qualifying", "summarise_eligible", "total_matching"],
            "eval": ["combine_passing", "reduce_qualifying"],
        },
    }
    RENAMES = {"kept": "selected", "total": "acc", "v": "item"}
    AGG_SWAP = {"sum": ("max", 2), "count": ("distinct", 3), "max": ("min", 1)}

    def sample_params(self, seed: int, split: str = "train") -> dict:
        r = self.rng(seed, "params")
        return {
            "fn": r.choice(self.POOLS["fn"][split]),
            "ff": r.choice(self.POOLS["ff"][split]),
            "af": r.choice(self.POOLS["af"][split]),
            "flagf": r.choice(self.POOLS["flagf"][split]),
            "op": r.choice([">", ">="]),
            "agg": r.choice(["sum", "count", "max"]),
        }

    def render(self, params: dict, knob: str | None = None) -> str:
        ff, af, flagf = params["ff"], params["af"], params["flagf"]
        op = params["op"]
        if knob == "op_swap":
            op = {">": ">=", ">=": ">"}[op]
        flag_cmp = "!=" if knob == "negate_flag" else "=="
        cond = f'r["{ff}"] {op} threshold and r["{flagf}"] {flag_cmp} 1'
        if knob == "drop_conjunct":
            cond = f'r["{ff}"] {op} threshold'
        collect = ff if knob == "wrong_field" else af
        empty_ret = "None" if knob == "empty_default" else "0"
        agg = params["agg"]
        if knob == "agg_swap":
            agg = self.AGG_SWAP[agg][0]
        arith = "-" if knob == "arith" else "+"
        bodies = {
            "sum": [
                "    total = 0",
                "    for v in kept:",
                f"        total = total {arith} v",
                "    return total",
            ],
            "count": ["    return len(kept)"],
            "distinct": ["    return len(set(kept))"],
            "max": ["    return max(kept)"],
            "min": ["    return min(kept)"],
        }
        lines = [
            f'def {params["fn"]}(records, threshold):',
            f'    kept = [r["{collect}"] for r in records if {cond}]',
            "    if not kept:",
            f"        return {empty_ret}",
            *bodies[agg],
        ]
        return "\n".join(lines) + "\n"

    def spec(self, params: dict) -> str:
        ff, af, flagf, agg = params["ff"], params["af"], params["flagf"], params["agg"]
        what = {
            "sum": f'the sum of the "{af}" values of qualifying records',
            "count": "the number of qualifying records",
            "max": f'the largest "{af}" value among qualifying records',
        }[agg]
        opw = "strictly greater than" if params["op"] == ">" else "greater than or equal to"
        ex = [
            {ff: 5, af: 4, flagf: 1, "id": "a"},
            {ff: 8, af: 2, flagf: 1, "id": "b"},
            {ff: 9, af: 7, flagf: 0, "id": "c"},
            {ff: 2, af: 6, flagf: 1, "id": "d"},
        ]
        doc = (
            f"Return {what}.\n"
            "\n"
            f'A record qualifies when its "{ff}" is {opw} threshold AND its\n'
            f'"{flagf}" equals 1. Records are dicts with integer fields\n'
            f'"{ff}", "{af}", "{flagf}" (0 or 1) and a string "id". If no\n'
            "records qualify, return 0.\n"
            "\n"
            "Example:\n"
            + self._example(params, (ex, 4))
        )
        return self._signed_spec(params, doc, "records, threshold")

    def mutations(self, params: dict) -> list[MutationSpec]:
        agg = params["agg"]
        menu = [
            MutationSpec("op_swap", 1, "threshold boundary swapped", 3),
            MutationSpec("negate_flag", 4, "flag condition negated", 1),
            MutationSpec("drop_conjunct", 5, "flag conjunct dropped", 2),
            MutationSpec("empty_default", 9, "wrong empty default", 2),
            MutationSpec("agg_swap", 8, f"aggregation swapped ({agg})", self.AGG_SWAP[agg][1]),
        ]
        if agg == "sum":
            menu.append(MutationSpec("arith", 3, "accumulator sign flipped", 1))
        if agg != "count":
            menu.append(MutationSpec("wrong_field", 10, "aggregates the filter field", 2))
        return menu

    def battery(self, params: dict, seed: int) -> list[tuple]:
        r = self.rng(seed, "battery")
        ff, af, flagf = params["ff"], params["af"], params["flagf"]

        def rec(f, a, g, i):
            return {ff: f, af: a, flagf: g, "id": f"r{i}"}

        cases = [
            ([], 5),
            ([rec(5, 10, 1, 0), rec(5, 7, 1, 1)], 5),          # ff == threshold: > vs >=
            ([rec(9, 4, 0, 0), rec(9, 6, 1, 1)], 5),           # flag matters
            ([rec(1, 3, 1, 0)], 5),                            # nothing qualifies
            ([rec(8, 4, 1, 0), rec(9, 4, 1, 1), rec(7, 4, 1, 2)], 5),  # duplicate agg values
            ([rec(6, 2, 1, 0), rec(7, 9, 1, 1), rec(2, 100, 1, 2)], 5),  # agg != filter field
        ]
        for _ in range(12):
            n = r.randint(1, 6)
            recs = [
                rec(r.randint(0, 9), r.randint(0, 9), 1 if r.random() < 0.7 else 0, i)
                for i in range(n)
            ]
            cases.append((recs, r.randint(2, 7)))
        return cases


class Dedup(Family):
    """Family 4 (SPEC Q3): dedup with a precedence rule."""

    name = "dedup"
    POOLS = {
        "kf": {
            "train": ["user", "device", "region", "account", "host", "team"],
            "eval": ["vendor", "channel", "branch"],
        },
        "pf": {
            "train": ["ts", "version", "updated", "revision"],
            "eval": ["epoch", "sequence"],
        },
        "fn": {
            "train": ["keep_latest", "collapse_by_key", "dedup_records", "newest_per_group"],
            "eval": ["reduce_duplicates", "latest_entries"],
        },
    }
    RENAMES = {"best": "held", "key": "grp"}

    def sample_params(self, seed: int, split: str = "train") -> dict:
        r = self.rng(seed, "params")
        return {
            "fn": r.choice(self.POOLS["fn"][split]),
            "kf": r.choice(self.POOLS["kf"][split]),
            "pf": r.choice(self.POOLS["pf"][split]),
            "last_wins": r.random() < 0.5,
        }

    def render(self, params: dict, knob: str | None = None) -> str:
        kf, pf = params["kf"], params["pf"]
        cmp = ">=" if params["last_wins"] else ">"
        if knob == "tie_swap":
            cmp = {">": ">=", ">=": ">"}[cmp]
        if knob == "flip_precedence":
            cmp = {">": "<", ">=": "<="}[cmp]
        key_expr = f'r["{pf}"]' if knob == "wrong_key_field" else f'r["{kf}"]'
        membership = "in" if knob == "negate_membership" else "not in"
        if knob == "drop_update":
            guard = "        if key not in best:"
        else:
            guard = f'        if key {membership} best or r["{pf}"] {cmp} best[key]["{pf}"]:'
        ret = (
            "    return list(best.values())"
            if knob == "drop_sort"
            else "    return [best[key] for key in sorted(best)]"
        )
        lines = [
            f'def {params["fn"]}(records):',
            "    best = {}",
            "    for r in records:",
            f"        key = {key_expr}",
            guard,
            "            best[key] = r",
            ret,
        ]
        return "\n".join(lines) + "\n"

    def spec(self, params: dict) -> str:
        kf, pf = params["kf"], params["pf"]
        tie = "last" if params["last_wins"] else "first"
        ex = [
            {kf: "b", pf: 2, "id": "x0"},
            {kf: "a", pf: 5, "id": "x1"},
            {kf: "b", pf: 9, "id": "x2"},
            {kf: "a", pf: 5, "id": "x3"},
        ]
        doc = (
            f'Keep one record per distinct "{kf}".\n'
            "\n"
            f'For each "{kf}", keep the record with the highest "{pf}". If\n'
            f'several records share that highest "{pf}", keep the {tie} one\n'
            "in input order. Return the kept records as a list sorted by\n"
            f'"{kf}" in ascending order. Records are dicts with a string\n'
            f'"{kf}", an integer "{pf}" and a string "id".\n'
            "\n"
            "Example:\n"
            + self._example(params, (ex,))
        )
        return self._signed_spec(params, doc, "records")

    def mutations(self, params: dict) -> list[MutationSpec]:
        return [
            MutationSpec("tie_swap", 1, "precedence tie rule flipped", 3),
            MutationSpec("drop_update", 6, "precedence update dropped (first always wins)", 2),
            MutationSpec("drop_sort", 6, "final sort dropped", 3),
            MutationSpec("flip_precedence", 7, "precedence direction flipped", 1),
            MutationSpec("negate_membership", 4, "membership check negated", 1),
            MutationSpec("wrong_key_field", 10, "groups by the precedence field", 1),
        ]

    def battery(self, params: dict, seed: int) -> list[tuple]:
        r = self.rng(seed, "battery")
        kf, pf = params["kf"], params["pf"]

        def rec(k, p, i):
            return {kf: k, pf: p, "id": f"x{i}"}

        cases = [
            ([],),
            ([rec("a", 5, 0), rec("a", 5, 1), rec("b", 1, 2)],),   # tie on precedence
            ([rec("a", 1, 0), rec("a", 9, 1)],),                    # later record wins
            ([rec("c", 1, 0), rec("a", 2, 1), rec("b", 3, 2)],),   # insertion != sorted order
        ]
        for _ in range(12):
            n = r.randint(2, 7)
            recs = [rec(r.choice(["a", "b", "c"]), r.randint(0, 4), i) for i in range(n)]
            cases.append((recs,))
        return cases


class Window(Family):
    """Family 5 (SPEC Q3): threshold windowing. EVAL-ONLY (SPEC Q5a)."""

    name = "window"
    trained = False
    POOLS = {
        "fn": {
            "train": ["values_in_window", "within_bounds", "clip_to_range", "filter_window"],
            "eval": ["bounded_values", "in_range_values"],
        },
    }
    RENAMES = {"v": "x"}

    def sample_params(self, seed: int, split: str = "train") -> dict:
        r = self.rng(seed, "params")
        return {
            "fn": r.choice(self.POOLS["fn"][split]),
            "lo_inc": r.random() < 0.5,
            "hi_inc": r.random() < 0.5,
        }

    def _ops(self, params: dict, knob: str | None) -> tuple[str, str, str, str]:
        op1 = "<=" if params["lo_inc"] else "<"
        op2 = "<=" if params["hi_inc"] else "<"
        if knob == "lo_swap":
            op1 = {"<=": "<", "<": "<="}[op1]
        if knob == "hi_swap":
            op2 = {"<=": "<", "<": "<="}[op2]
        lo_expr = "(lo + 1)" if knob == "lo_off" else "lo"
        hi_expr = "(hi - 1)" if knob == "hi_off" else "hi"
        return op1, op2, lo_expr, hi_expr

    def render(self, params: dict, knob: str | None = None) -> str:
        op1, op2, lo_expr, hi_expr = self._ops(params, knob)
        cond = f"{lo_expr} {op1} v and v {op2} {hi_expr}"
        if knob == "negate":
            cond = f"not ({cond})"
        return (
            f'def {params["fn"]}(values, lo, hi):\n'
            f"    return [v for v in values if {cond}]\n"
        )

    def spec(self, params: dict) -> str:
        op1, op2, _, _ = self._ops(params, None)
        doc = (
            "Return the values v from `values`, in their original order,\n"
            f"that satisfy: lo {op1} v and v {op2} hi.\n"
            "\n"
            "`values` is a list of ints; lo and hi are ints. If no value\n"
            "satisfies the condition (including when the window is empty),\n"
            "return [].\n"
            "\n"
            "Example:\n"
            + self._example(params, ([2, 3, 4, 5, 6, 7], 3, 6))
        )
        return self._signed_spec(params, doc, "values, lo, hi")

    def mutations(self, params: dict) -> list[MutationSpec]:
        return [
            MutationSpec("lo_swap", 1, "lower boundary inclusivity swapped", 3),
            MutationSpec("hi_swap", 1, "upper boundary inclusivity swapped", 3),
            MutationSpec("negate", 4, "window condition negated", 1),
            MutationSpec("hi_off", 2, "upper bound off by one", 2),
            MutationSpec("lo_off", 2, "lower bound off by one", 2),
        ]

    def battery(self, params: dict, seed: int) -> list[tuple]:
        r = self.rng(seed, "battery")
        cases = [
            ([2, 3, 4, 5, 6, 7], 3, 6),
            ([5], 5, 5),
            ([], 0, 9),
            ([1, 2], 5, 3),
            ([4, 5, 6], 5, 6),
        ]
        for _ in range(12):
            vals = [r.randint(0, 9) for _ in range(r.randint(1, 8))]
            lo, hi = sorted([r.randint(0, 9), r.randint(0, 9)])
            cases.append((vals, lo, hi))
        return cases


FAMILIES: dict[str, Family] = {f.name: f for f in [TopK(), FilterAgg(), Dedup(), Window()]}
TRAINED = [name for name, f in FAMILIES.items() if f.trained]
EVAL_ONLY = [name for name, f in FAMILIES.items() if not f.trained]
