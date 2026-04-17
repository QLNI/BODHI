"""
Example 01 — Chat basics.

Run:  python examples/01_chat_basics.py

Shows how to talk to BODHI programmatically.
Each turn returns (response_text, state_dict).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bodhi import BODHI

b = BODHI(load_llm_flag=False)

prompts = [
    "What is fire?",
    "Tell me about the ocean",
    "I see a snake",
    "Who are you?",
    "What is xyzzy?",           # unknown — BODHI says so
]

for q in prompts:
    resp, st = b.think(q)
    print()
    print("You  :", q)
    print("BODHI:", resp)
    print("       [emotion=%s  reflex=%s(%d)  source=%s]" %
          (st.get("emotion"), st.get("reflex"),
           st.get("worm_confidence", 0), st.get("source")))
