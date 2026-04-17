"""
Example 04 — Look inside BODHI's brain after a single question.

Run:  python examples/04_inspect_brain_state.py

Prints the full brain_result dict so you can see exactly which bands,
regions, worm neurons, and drives fire when BODHI processes an input.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bodhi import BODHI

b = BODHI(load_llm_flag=False)

# Replicate what think() does internally, but keep the raw brain_result.
text = "What is fire?"
matched = b.match_concepts(text)
fp_idx = b.fp_index["img_name_to_idx"].get(matched[0]) if matched else None
fp = b.img_data[fp_idx] if fp_idx is not None else None
emotion = b.concept_emotions.get(matched[0], "neutral") if matched else "neutral"
intensity = 350 if emotion in ("fear", "anger", "anxiety") else 250

brain_result = b.brain.process(fp, emotion, intensity)

print("matched concepts :", matched)
print("primary emotion  :", emotion)
print()
print("WHT frequency bands:")
for band, val in sorted(brain_result["bands"].items(),
                        key=lambda x: -x[1] if x[0] != "_overall" else 0):
    print("  %-10s = %d" % (band, val))
print()
print("worm reflex      :", brain_result["worm"]["reflex"],
      "at confidence", brain_result["worm"]["confidence"])
print("worm circuit     :", brain_result["worm"]["circuit"])
print("worm neurons fired:", ", ".join(sorted(brain_result["worm"]["fired"])))
print()
print("top brain regions:")
for name, v in brain_result["top_regions"][:8]:
    print("  %-12s %d" % (name, v))
print()
print("region group totals:")
for g, v in sorted(brain_result["group_activation"].items(),
                   key=lambda x: -x[1]):
    print("  %-12s %d" % (g, v))
