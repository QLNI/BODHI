"""
Example 02 — Teach BODHI a new concept from an image.

Run:  python examples/02_teach_from_image.py path/to/image.jpg my_concept_name

BODHI computes the WHT fingerprint, encrypts it with AES-256-GCM, saves it
to data/learned/, and then reasons about the new concept using its actual
brain pipeline.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bodhi import BODHI

if len(sys.argv) < 3:
    print("Usage: python 02_teach_from_image.py <image_path> <concept_name>")
    sys.exit(1)

img_path = sys.argv[1]
concept = sys.argv[2]

b = BODHI(load_llm_flag=False)
info = b.teacher.teach_image(concept, img_path, emotion="neutral")
print("Taught '%s'. Fingerprint length=%d, index=%d." %
      (info["concept"], info["fingerprint_length"], info["index"]))
print()

resp, st = b.think("Tell me about " + concept)
print("You  : Tell me about", concept)
print("BODHI:", resp)
print("       [matched=%s  emotion=%s  reflex=%s(%d)]" %
      (st.get("concepts"), st.get("emotion"),
       st.get("reflex"), st.get("worm_confidence", 0)))
