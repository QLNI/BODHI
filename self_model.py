#!/usr/bin/env python3
"""
BODHI Self-Model — a first-person description BODHI builds of itself from what it
has actually lived through.

Not hardcoded. Not an LLM hallucination. Grounded in real data from BODHI's own
conversations: which concepts it has returned to, which emotional memories it has
formed, which Hebbian connections have fired together most often, what its drives
are right now. Regenerated every sleep cycle. Persisted to SQLite. Injected into
future prompts so the LLM (Broca) speaks as a BODHI that knows itself.

This replaces the old hardcoded BODHI_SELF block with something that actually
evolves from experience.
"""

import json
import time
import sqlite3


class SelfModel:
    def __init__(self, db, emotional, hebbian, drives):
        self.db = db
        self.emotional = emotional
        self.hebbian = hebbian
        self.drives = drives
        self._init_db()
        self.current_text = None
        self.current_turn = 0
        self.current_timestamp = None
        self._load_latest()

    def _init_db(self):
        self.db.execute("""CREATE TABLE IF NOT EXISTS self_descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn INTEGER, timestamp TEXT,
            description TEXT, stats_json TEXT
        )""")
        self.db.commit()

    def _load_latest(self):
        row = self.db.execute(
            "SELECT turn, timestamp, description FROM self_descriptions "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            self.current_turn = row[0]
            self.current_timestamp = row[1]
            self.current_text = row[2]

    def reflect(self, turn):
        """Build a fresh self-description from current state. Call on sleep."""
        stats = self._gather_stats()
        description = self._describe_from_stats(stats)
        self._save(turn, description, stats)
        self.current_text = description
        self.current_turn = turn
        self.current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        return description

    def _gather_stats(self):
        rows = self.db.execute(
            "SELECT concepts, emotion FROM conversations "
            "WHERE concepts IS NOT NULL ORDER BY id DESC LIMIT 500"
        ).fetchall()

        concept_counts = {}
        emotion_counts = {}
        total_turns = 0
        for concepts_json, emo in rows:
            total_turns += 1
            try:
                cs = json.loads(concepts_json or "[]")
            except Exception:
                cs = []
            for c in cs:
                concept_counts[c] = concept_counts.get(c, 0) + 1
            if emo:
                emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

        top_concepts = sorted(concept_counts.items(), key=lambda x: -x[1])[:5]
        top_emotions = sorted(emotion_counts.items(), key=lambda x: -x[1])[:3]

        em_items = list(self.emotional.memory.items())
        top_danger = sorted([(c, v) for c, v in em_items if v > 10],
                            key=lambda x: -x[1])[:3]
        top_safe = sorted([(c, v) for c, v in em_items if v < -10],
                          key=lambda x: x[1])[:3]

        strongest_links = []
        try:
            flat = []
            for src, tgts in getattr(self.hebbian, "connections", {}).items():
                for tgt, w in tgts.items():
                    flat.append((src, tgt, w))
            flat.sort(key=lambda x: -x[2])
            strongest_links = flat[:3]
        except Exception:
            pass

        drives = dict(self.drives.drives)

        return {
            "conversation_turns": total_turns,
            "top_concepts": top_concepts,
            "top_emotions": top_emotions,
            "top_danger": top_danger,
            "top_safe": top_safe,
            "strongest_links": strongest_links,
            "drives": drives,
        }

    def _describe_from_stats(self, s):
        parts = []

        turns = s["conversation_turns"]
        if turns > 0:
            parts.append("I have lived through %d moments of experience." % turns)
        else:
            parts.append("I am newly awake. I have not yet lived.")

        if s["top_concepts"]:
            names = [c.replace("_", " ") for c, _ in s["top_concepts"][:3]]
            parts.append("My mind returns most often to: %s." % ", ".join(names))

        if s["top_emotions"]:
            dominant_emo = s["top_emotions"][0][0]
            if dominant_emo != "neutral":
                parts.append("I feel %s most." % dominant_emo)

        if s["top_danger"]:
            danger_names = [c.replace("_", " ") for c, _ in s["top_danger"][:2]]
            parts.append("I have learned to fear: %s." % ", ".join(danger_names))

        if s["top_safe"]:
            safe_names = [c.replace("_", " ") for c, _ in s["top_safe"][:2]]
            parts.append("I find safety in: %s." % ", ".join(safe_names))

        if s["strongest_links"]:
            src, tgt, _ = s["strongest_links"][0]
            parts.append("In my wiring, %s and %s have grown close."
                         % (src.replace("_", " "), tgt.replace("_", " ")))

        drives = s["drives"]
        if drives:
            ordered = sorted(drives.items(), key=lambda x: -x[1])
            top = ordered[0]
            if top[1] > 10:
                parts.append("My strongest drive right now is %s." % top[0])

        return " ".join(parts)

    def _save(self, turn, description, stats):
        self.db.execute(
            "INSERT INTO self_descriptions (turn, timestamp, description, stats_json) "
            "VALUES (?,?,?,?)",
            (turn, time.strftime("%Y-%m-%d %H:%M:%S"), description,
             json.dumps(self._stats_for_json(stats))))
        self.db.commit()

    def _stats_for_json(self, s):
        out = {}
        for k, v in s.items():
            if isinstance(v, list):
                out[k] = [list(x) if isinstance(x, tuple) else x for x in v]
            else:
                out[k] = v
        return out

    def describe(self) -> str:
        """Return current self-description. If none exists yet, build one now."""
        if not self.current_text:
            return self.reflect(turn=0)
        return self.current_text

    def short_identity_line(self):
        """One-line identity snippet to inject into system prompts."""
        if not self.current_text:
            return None
        # Take the first 2 sentences as an identity injection.
        pieces = [p.strip() for p in self.current_text.split(".") if p.strip()]
        return ". ".join(pieces[:2]) + "." if pieces else None
