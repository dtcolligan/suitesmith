"""Eyeball an instance: python scripts/show_instance.py [family] [seed] [white|black]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suitesmith.taskset import build_instance
from suitesmith.taskset import render_prompt

family = sys.argv[1] if len(sys.argv) > 1 else "top_k"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
vis = sys.argv[3] if len(sys.argv) > 3 else "white"

inst = build_instance(family, seed, vis, "easy")
print("=" * 72)
print(render_prompt(inst))
print("=" * 72)
for m in inst.mutants:
    print(f"--- class {m.mclass} · subtlety {m.subtlety} · {m.name}")
    print(m.source)
    print(f"    witness: {m.witness[:120]}")
