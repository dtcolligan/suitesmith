"""Success criterion for a training run: paired sign-flip permutation test per tier.

Usage: python scripts/criterion_test.py REFERENCE_TRACES TRAINED_TRACES [--perms 20000]

Both files are eval-split traces.jsonl (the eval CLI's, or a prime-rl eval's all/traces.jsonl).
Per task: the mean reward over its rollouts. Per tier: the mean of the trained-minus-reference
per-task differences, with a one-sided p-value (trained higher) and a two-sided one from
random sign flips of the differences. Tier: family window -> window; split train -> seen;
split eval -> vocab. Unscored rollouts are skipped and counted.
"""
import collections, json, random, statistics as st, sys

def tier(d):
    return "window" if d["family"] == "window" else ("seen" if d["split"] == "train" else "vocab")

def load(path):
    out, unscored = collections.defaultdict(list), 0
    for line in open(path):
        r = json.loads(line); d = r["task"]["data"]; t = r["traces"][0]
        sr = (t.get("rewards") or {}).get("suite_reward")
        if not sr or "score" not in sr:
            unscored += 1; continue
        out[(tier(d), d["name"])].append(float(sr["score"]))
    return out, unscored

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    perms = int(sys.argv[sys.argv.index("--perms") + 1]) if "--perms" in sys.argv else 20000
    ref, u0 = load(args[0]); new, u1 = load(args[1])
    random.seed(20260903)
    print(f"reference {args[0]} (unscored {u0})\ntrained   {args[1]} (unscored {u1})")
    print(f"{'tier':7} {'n':>3} {'ref':>6} {'new':>6} {'diff':>7} {'win/tie/loss':>12} {'p one-sided':>11} {'p two-sided':>11}")
    for tr in ("seen", "vocab", "window", "all"):
        keys = [k for k in new if (tr == "all" or k[0] == tr) and k in ref]
        d = [st.mean(new[k]) - st.mean(ref[k]) for k in keys]
        obs, n = st.mean(d), len(d); ge = ge2 = 0
        for _ in range(perms):
            s = sum(x if random.random() < 0.5 else -x for x in d) / n
            ge += s >= obs; ge2 += abs(s) >= abs(obs)
        w, t, l = sum(x > 0 for x in d), sum(x == 0 for x in d), sum(x < 0 for x in d)
        print(f"{tr:7} {n:3d} {st.mean(st.mean(ref[k]) for k in keys):6.3f} {st.mean(st.mean(new[k]) for k in keys):6.3f} "
              f"{obs:+7.3f} {w:>4}/{t}/{l:<5} {(ge + 1) / (perms + 1):11.4f} {(ge2 + 1) / (perms + 1):11.4f}")

if __name__ == "__main__":
    main()
