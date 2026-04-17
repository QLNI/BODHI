#!/usr/bin/env python3
"""
Wipe BODHI's memory — clean slate for re-teaching.

Clears:
  - conversations / dreams / sleep_logs / self_descriptions / goals (SQLite)
  - data/brain_state/hebbian.json
  - data/brain_state/emotional.json
  - data/brain_state/meta.json
  - data/brain_state/lora_adapter.pt (if present — keeps the .bak)

Does NOT touch the base LLM weights, concept fingerprints, engrams, or the
trained SmallGPT in bodhi_llm/. Only BODHI's lived experience is wiped.
"""

import os
import sqlite3

ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT, "data", "bodhi_memory.db")
SAVE_DIR = os.path.join(ROOT, "data", "brain_state")


def clear_db():
    if not os.path.exists(DB_PATH):
        print("[clear] no db at", DB_PATH)
        return
    db = sqlite3.connect(DB_PATH)
    for t in ("conversations", "dreams", "sleep_logs", "self_descriptions", "goals"):
        try:
            n = db.execute("SELECT COUNT(*) FROM %s" % t).fetchone()[0]
        except Exception:
            continue
        db.execute("DELETE FROM %s" % t)
        print("[clear] %-20s cleared (%d rows)" % (t, n))
    db.commit()
    db.close()


def clear_state():
    for fname in ("hebbian.json", "emotional.json", "meta.json", "lora_adapter.pt"):
        path = os.path.join(SAVE_DIR, fname)
        if os.path.exists(path):
            os.remove(path)
            print("[clear] removed", path)


if __name__ == "__main__":
    clear_db()
    clear_state()
    print("[clear] done. BODHI has no memory of past conversations, no emotional "
          "history, no Hebbian wiring, no LoRA adapter. Clean slate.")
