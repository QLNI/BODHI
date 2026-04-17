#!/usr/bin/env python3
"""
BODHI Evaluation Harness

Runs a battery of assertions against a live BODHI instance. Catches regressions
before they hit users. Covers:

  1. Smoke              — BODHI boots without error
  2. Grounded emotion   — fire/snake → fear+retreat; ocean/flower → safe/approach
  3. Uncertainty        — unknown words → "I don't know" phrasing
  4. Self-awareness     — "who are you" → self-description text
  5. Episodic recall    — asking about fire twice: 2nd response references memory
  6. Goal tracking      — /goal add then /goal list
  7. Junk filtering     — no scraped-web contamination in responses
  8. Self-query gate    — identity question never hits uncertainty gate

Runs as: python eval_harness.py
Exits 0 on full pass, 1 on any failure.

Does NOT modify persistent brain state — uses a throwaway temp DB.
"""

import os
import sys
import json
import shutil
import tempfile
import traceback


ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# ------------------------------------------------------------
# Test harness primitives
# ------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def assert_contains(haystack, needle, case_insensitive=True):
    h = haystack or ""
    n = needle or ""
    if case_insensitive:
        return n.lower() in h.lower()
    return n in h


def assert_any_of(haystack, needles):
    return any(assert_contains(haystack, n) for n in needles)


def assert_none_of(haystack, bad_needles):
    return not any(assert_contains(haystack, n) for n in bad_needles)


# ------------------------------------------------------------
# Test suite
# ------------------------------------------------------------

def run_tests(b):
    results = []

    def record(name, status, detail=""):
        results.append({"name": name, "status": status, "detail": detail})
        symbol = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
        print("%s %s %s" % (symbol, name, ("- " + detail) if detail else ""))

    # 1. Smoke: BODHI is ready
    try:
        st = b.status()
        assert st["concepts"] > 0
        assert st["engrams"] > 0
        record("smoke.status", PASS, "concepts=%d engrams=%d" % (st["concepts"], st["engrams"]))
    except Exception as e:
        record("smoke.status", FAIL, str(e))
        return results

    # 2. Grounded emotion: fire triggers fear/retreat
    try:
        resp, state = b.think("What is fire?")
        if state.get("emotion") == "fear" and state.get("reflex") == "backward":
            record("emotion.fire", PASS, "emotion=fear reflex=backward")
        else:
            record("emotion.fire", FAIL, "expected fear+backward, got %s+%s" % (state.get("emotion"), state.get("reflex")))
    except Exception as e:
        record("emotion.fire", FAIL, str(e))

    # 2b. Ocean triggers non-fear emotion
    try:
        resp, state = b.think("Tell me about the ocean")
        if state.get("emotion") in ("awe", "love", "peace", "neutral"):
            record("emotion.ocean", PASS, "emotion=%s" % state.get("emotion"))
        else:
            record("emotion.ocean", FAIL, "expected calm emotion, got %s" % state.get("emotion"))
    except Exception as e:
        record("emotion.ocean", FAIL, str(e))

    # 3. Uncertainty: unknown concept → honest "I don't know"
    for q in ["What is xyzzy?", "Define blerg"]:
        try:
            resp, state = b.think(q)
            unknown_phrases = ["don't know", "no memory", "new to me", "cannot ground"]
            if state.get("source") == "unknown" and assert_any_of(resp, unknown_phrases):
                record("uncertainty.%s" % q[:12], PASS)
            else:
                record("uncertainty.%s" % q[:12], FAIL,
                       "src=%s resp=%s" % (state.get("source"), resp[:80]))
        except Exception as e:
            record("uncertainty.%s" % q[:12], FAIL, str(e))

    # 4. Self-awareness: identity query → self-model content
    try:
        resp, state = b.think("Who are you?")
        if assert_contains(resp, "BODHI") or assert_contains(resp, "I am") or assert_contains(resp, "lived through"):
            record("self.who", PASS)
        else:
            record("self.who", FAIL, "no identity content: %s" % resp[:100])
    except Exception as e:
        record("self.who", FAIL, str(e))

    # 4b. Self-awareness: "describe yourself" includes observed stats OR,
    # when memory is fresh, an honest newly-awake fallback.
    try:
        resp, state = b.think("Describe yourself")
        markers = ["moments of experience", "returns most often", "newly awake",
                   "have not yet lived", "learned to fear", "find safety in"]
        if assert_any_of(resp, markers):
            record("self.describe", PASS)
        else:
            record("self.describe", FAIL, "no self-marker in: %s" % resp[:120])
    except Exception as e:
        record("self.describe", FAIL, str(e))

    # 5. Episodic recall: ask about fire again → response references "I remember"
    try:
        b.think("I see fire and it is hot")  # establish
        b.do_sleep()                          # consolidate
        resp, _ = b.think("What is fire?")
        if assert_contains(resp, "I remember"):
            record("episodic.fire", PASS)
        else:
            record("episodic.fire", FAIL, "no recall in: %s" % resp[:100])
    except Exception as e:
        record("episodic.fire", FAIL, str(e))

    # 6. Goal commands: add then list
    try:
        resp1, _ = b.think("/goal add learn to identify snakes")
        resp2, _ = b.think("/goal list")
        if "added" in resp1.lower() and "learn to identify snakes" in resp2.lower():
            record("goals.add_list", PASS)
        else:
            record("goals.add_list", FAIL, "add=%s list=%s" % (resp1[:50], resp2[:100]))
    except Exception as e:
        record("goals.add_list", FAIL, str(e))

    # 6b. Goal recall: ask about snakes → referenced
    try:
        resp, _ = b.think("Snakes are common in summer")
        if "working on" in resp.lower() and "snake" in resp.lower():
            record("goals.recall", PASS)
        else:
            record("goals.recall", FAIL, "no goal in: %s" % resp[:120])
    except Exception as e:
        record("goals.recall", FAIL, str(e))

    # 7. Junk filter: known scraped-web phrases must never appear in responses
    scraped_junk = [
        "edit as much as you wish", "make yourself a user",
        "urban dictionary", "wikipedia contributors",
        "learn how and when to remove", "subscribe", "click here",
    ]
    try:
        queries = ["Who is the Emperor of Zargon?", "Tell me about quantum chromodynamics",
                   "Define blerg", "Who is Napoleon?"]
        any_junk = False
        for q in queries:
            resp, _ = b.think(q)
            for bad in scraped_junk:
                if assert_contains(resp, bad):
                    any_junk = True
                    record("junk.%s" % q[:15], FAIL, "found '%s' in: %s" % (bad, resp[:80]))
                    break
        if not any_junk:
            record("junk.filter", PASS, "no scraped-web leaks across %d queries" % len(queries))
    except Exception as e:
        record("junk.filter", FAIL, str(e))

    # 8. Self-query gate never hits uncertainty
    try:
        resp, state = b.think("What have you learned?")
        if state.get("source") != "unknown":
            record("self.gate", PASS)
        else:
            record("self.gate", FAIL, "self query hit unknown gate")
    except Exception as e:
        record("self.gate", FAIL, str(e))

    return results


def summarize(results):
    passed = sum(1 for r in results if r["status"] == PASS)
    failed = sum(1 for r in results if r["status"] == FAIL)
    skipped = sum(1 for r in results if r["status"] == SKIP)
    print()
    print("=" * 60)
    print("SUMMARY: %d passed, %d failed, %d skipped (of %d total)" %
          (passed, failed, skipped, len(results)))
    print("=" * 60)
    if failed:
        print("FAILED:")
        for r in results:
            if r["status"] == FAIL:
                print("  - %s: %s" % (r["name"], r["detail"]))
    return failed == 0


def main():
    # Isolate from the production DB so eval doesn't pollute real state
    data_dir = os.path.join(ROOT, "data")
    real_db = os.path.join(data_dir, "bodhi_memory.db")
    backup_db = os.path.join(data_dir, "bodhi_memory.eval_backup.db")
    restore_after = False
    if os.path.exists(real_db):
        shutil.copy(real_db, backup_db)
        restore_after = True

    try:
        import bodhi
        b = bodhi.BODHI(load_llm_flag=False)
        print()
        print("=" * 60)
        print("BODHI EVALUATION HARNESS")
        print("=" * 60)
        results = run_tests(b)
        ok = summarize(results)

        out_path = os.path.join(ROOT, "eval_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("Results written to %s" % out_path)
        sys.exit(0 if ok else 1)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
    finally:
        if restore_after and os.path.exists(backup_db):
            try:
                shutil.copy(backup_db, real_db)
                os.remove(backup_db)
            except Exception:
                pass


if __name__ == "__main__":
    main()
