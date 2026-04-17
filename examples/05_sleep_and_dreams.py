"""
Example 05 — Sleep, watch the Hebbian graph grow.

Run:  python examples/05_sleep_and_dreams.py

Seeds BODHI with 20 varied sentences, runs 5 sleep cycles, and shows:
  - Connections gained per sleep
  - Triangles inferred (A-B strong + B-C strong -> infer A-C)
  - Dreams generated
  - Top connections after sleeps
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bodhi import BODHI

b = BODHI(load_llm_flag=False)

prompts = [
    "Fire burns wood", "Fire makes heat", "Fire is danger",
    "Heat comes from fire", "Heat burns skin",
    "Water cools fire", "Water is safe",
    "Ocean has water", "Ocean is big",
    "Snake is danger", "Snake bites", "Snake is in forest",
    "Forest has trees", "Forest is home to animals",
    "Mountain is big", "Mountain has stones",
    "Stones are heavy", "Music is sound",
    "Love is warm", "Warm is good",
]

for _ in range(2):
    for p in prompts:
        b.think(p)

print("after %d turns: %d Hebbian connections"
      % (len(prompts) * 2, b.hebbian.total_connections()))
print()

for i in range(5):
    s = b.do_sleep()
    print("SLEEP %d  replayed=%-3d  inferred=%-3d  strengthened=%-3d  pruned=%-2d  dreams=%d  total=%d"
          % (i + 1, s["replayed"], s["inferred"], s["strengthened"],
             s["pruned"], s["dreams"], b.hebbian.total_connections()))

print()
print("top 10 Hebbian connections:")
for (a, c), w in b.hebbian.strongest(10):
    print("  %-20s <-> %-20s  weight=%d" % (a, c, w))
