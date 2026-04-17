#!/usr/bin/env python3
"""
BODHI Episodic Memory — recall past conversations by concept overlap.

BODHI-native retrieval: uses the concept IDs already extracted per turn, expanded
through the Hebbian associative network, scored by recency. No embeddings, no
external dependencies, integer/integer-ratio math only.

"I remember 3 days ago you asked X and I felt Y" becomes possible.
"""

import json
import time
import sqlite3


class EpisodicMemory:
    def __init__(self, db):
        self.db = db

    def recall(self, query_concepts, hebbian, current_turn=None,
               top_k=3, min_overlap=1, exclude_recent=5, hebbian_threshold=10,
               max_rows=500):
        if not query_concepts:
            return []

        expanded = set(query_concepts)
        for c in query_concepts:
            try:
                for assoc, weight in hebbian.get_associates(c):
                    if weight >= hebbian_threshold:
                        expanded.add(assoc)
            except Exception:
                pass

        # exclude_recent is applied by row id (unique across sessions),
        # NOT by turn number — turn numbers overlap across sessions so a
        # turn-based filter incorrectly excludes today's new rows while
        # keeping yesterday's rows with matching turn numbers.
        sql = "SELECT id, turn, timestamp, user_text, response, emotion, concepts FROM conversations "
        params = []
        if exclude_recent and exclude_recent > 0:
            last_id_row = self.db.execute("SELECT MAX(id) FROM conversations").fetchone()
            last_id = last_id_row[0] if last_id_row and last_id_row[0] else 0
            sql += "WHERE id <= ? "
            params.append(last_id - exclude_recent)
        sql += "ORDER BY id DESC LIMIT ?"
        params.append(max_rows)

        rows = self.db.execute(sql, params).fetchall()

        now = time.time()
        scored = []
        seen_texts = set()
        for row in rows:
            id_, turn, ts, user_text, response, emotion, concepts_json = row
            if not user_text:
                continue
            key = (user_text or "").strip().lower()[:80]
            if key in seen_texts:
                continue
            try:
                past_concepts = set(json.loads(concepts_json or "[]"))
            except Exception:
                continue
            overlap = past_concepts & expanded
            if len(overlap) < min_overlap:
                continue
            try:
                past_t = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
                age_hours = max(1.0, (now - past_t) / 3600.0)
                # Aggressive recency decay: halves every 3 hours.
                # At 1h ago -> 0.95, 3h -> 0.73, 6h -> 0.50, 12h -> 0.25, 24h -> 0.09.
                # Old but concept-rich rows no longer dominate freshly-relevant ones.
                recency = 0.5 ** (age_hours / 3.0)
            except Exception:
                recency = 0.05

            # Multiplicative with overlap so relevance still matters, but
            # recency can dominate for large age gaps.
            score = (1 + len(overlap)) * recency
            seen_texts.add(key)
            scored.append({
                "score": score,
                "id": id_,
                "turn": turn,
                "timestamp": ts,
                "user_text": user_text,
                "response": response,
                "emotion": emotion or "neutral",
                "overlap": list(overlap),
            })

        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]

    def format_memory(self, row, style="natural"):
        ago = self._time_ago(row.get("timestamp", ""))
        ut = (row.get("user_text") or "").strip()
        if len(ut) > 70:
            ut = ut[:67] + "..."
        emo = row.get("emotion", "neutral")
        overlap = row.get("overlap", [])
        if not overlap:
            return None

        if style == "short":
            head = overlap[0]
            return "I remember %s we spoke of %s." % (ago, head)

        return "I remember %s you said '%s' and I felt %s." % (ago, ut, emo)

    def _time_ago(self, ts):
        try:
            past_t = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
            diff = time.time() - past_t
        except Exception:
            return "once"
        if diff < 120:
            return "just now"
        if diff < 3600:
            return "%d minutes ago" % int(diff / 60)
        if diff < 86400:
            hrs = int(diff / 3600)
            return "%d hour%s ago" % (hrs, "" if hrs == 1 else "s")
        days = int(diff / 86400)
        if days == 1:
            return "yesterday"
        if days < 7:
            return "%d days ago" % days
        weeks = days // 7
        if weeks < 5:
            return "%d week%s ago" % (weeks, "" if weeks == 1 else "s")
        months = days // 30
        if months < 12:
            return "%d month%s ago" % (months, "" if months == 1 else "s")
        return "long ago"

    def count(self):
        try:
            row = self.db.execute("SELECT COUNT(*) FROM conversations").fetchone()
            return row[0] if row else 0
        except Exception:
            return 0
