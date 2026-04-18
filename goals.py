#!/usr/bin/env python3
"""
BODHI Goal Tracker — persistent goals/tasks BODHI holds across sessions.

A goal is a statement BODHI has committed to. Stored in SQLite. Each new turn,
BODHI checks whether the user's message relates to any active goal (by keyword
overlap expanded through Hebbian). If it does, BODHI surfaces awareness of it.

Commands:
  /goal add <text>      — add a new goal
  /goal done <id>       — mark goal complete
  /goal pause <id>      — pause a goal
  /goal resume <id>     — resume a paused goal
  /goal list            — show all active goals
  /goal all             — show every goal including done/paused
"""

import re
import time
import json


_STOPWORDS = set(
    "the a an is are was were be been have has had do does did will would could should can not "
    "and or but if for in on at to from by with as of this that it its he she they we all more "
    "most some very just also even said get am what how why when where who your my me you i "
    "goal task todo remember remind please want need try try".split()
)


def _extract_keywords(text):
    words = re.findall(r"[a-zA-Z]+", (text or "").lower())
    return {w for w in words if len(w) >= 4 and w not in _STOPWORDS}


class GoalTracker:
    def __init__(self, db):
        self.db = db
        self._init_db()

    def _init_db(self):
        self.db.execute("""CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_turn INTEGER,
            created_ts TEXT,
            text TEXT,
            status TEXT,
            priority INTEGER DEFAULT 1,
            last_mentioned_turn INTEGER,
            completed_ts TEXT
        )""")
        self.db.commit()

    def add(self, text, turn=0, priority=1):
        text = (text or "").strip()
        if not text:
            return None
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cur = self.db.execute(
            "INSERT INTO goals (created_turn, created_ts, text, status, priority, last_mentioned_turn) "
            "VALUES (?,?,?,?,?,?)",
            (turn, ts, text, "active", priority, turn),
        )
        self.db.commit()
        return cur.lastrowid

    def set_status(self, goal_id, status):
        valid = ("active", "paused", "done")
        if status not in valid:
            return False
        completed_ts = time.strftime("%Y-%m-%d %H:%M:%S") if status == "done" else None
        self.db.execute(
            "UPDATE goals SET status=?, completed_ts=? WHERE id=?",
            (status, completed_ts, goal_id),
        )
        self.db.commit()
        return True

    def mark_done(self, goal_id):
        return self.set_status(goal_id, "done")

    def pause(self, goal_id):
        return self.set_status(goal_id, "paused")

    def resume(self, goal_id):
        return self.set_status(goal_id, "active")

    def list_active(self):
        rows = self.db.execute(
            "SELECT id, text, created_ts, priority, last_mentioned_turn "
            "FROM goals WHERE status='active' ORDER BY priority DESC, id"
        ).fetchall()
        return [
            {"id": r[0], "text": r[1], "created_ts": r[2],
             "priority": r[3], "last_mentioned_turn": r[4]}
            for r in rows
        ]

    def list_all(self):
        rows = self.db.execute(
            "SELECT id, text, created_ts, status, priority, completed_ts "
            "FROM goals ORDER BY id DESC"
        ).fetchall()
        return [
            {"id": r[0], "text": r[1], "created_ts": r[2], "status": r[3],
             "priority": r[4], "completed_ts": r[5]}
            for r in rows
        ]

    def find_relevant(self, user_text, hebbian=None, current_concepts=None, top_k=2):
        """Return active goals whose keywords overlap the user's message."""
        active = self.list_active()
        if not active:
            return []

        query_kw = _extract_keywords(user_text)
        if current_concepts:
            query_kw |= set(current_concepts)
        if hebbian is not None and current_concepts:
            for c in current_concepts:
                try:
                    for assoc, w in hebbian.get_associates(c):
                        if w >= 10:
                            query_kw.add(assoc)
                except Exception:
                    pass

        scored = []
        for g in active:
            goal_kw = _extract_keywords(g["text"])
            overlap = goal_kw & query_kw
            if overlap:
                scored.append((len(overlap), g, list(overlap)))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [{"goal": g, "overlap": ov} for _, g, ov in scored[:top_k]]

    def touch(self, goal_id, turn):
        self.db.execute(
            "UPDATE goals SET last_mentioned_turn=? WHERE id=?",
            (turn, goal_id),
        )
        self.db.commit()

    def handle_command(self, text, turn):
        """Parse a /goal command string. Returns (handled, reply_text)."""
        if not text or not text.strip().startswith("/goal"):
            return False, None
        body = text.strip()[5:].strip()
        if not body or body in ("list", "l"):
            return True, self._fmt_list(self.list_active(), "Active goals")
        if body == "all":
            return True, self._fmt_list(self.list_all(), "All goals", include_status=True)

        parts = body.split(None, 1)
        cmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if cmd == "add":
            if not rest:
                return True, "Goal text missing. Usage: /goal add <text>"
            gid = self.add(rest, turn)
            return True, "Goal #%d added: %s" % (gid, rest)
        if cmd in ("done", "complete", "finish"):
            gid = self._parse_id(rest)
            if gid is None:
                return True, "Usage: /goal done <id>"
            ok = self.mark_done(gid)
            return True, ("Goal #%d marked done." % gid) if ok else "Goal not found."
        if cmd == "pause":
            gid = self._parse_id(rest)
            if gid is None:
                return True, "Usage: /goal pause <id>"
            ok = self.pause(gid)
            return True, ("Goal #%d paused." % gid) if ok else "Goal not found."
        if cmd == "resume":
            gid = self._parse_id(rest)
            if gid is None:
                return True, "Usage: /goal resume <id>"
            ok = self.resume(gid)
            return True, ("Goal #%d resumed." % gid) if ok else "Goal not found."
        return True, "Unknown /goal command. Try /goal add|done|pause|resume|list|all."

    def _parse_id(self, s):
        s = (s or "").strip().lstrip("#")
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            return None

    def _fmt_list(self, items, header, include_status=False):
        if not items:
            return "%s: none." % header
        lines = ["%s:" % header]
        for g in items:
            if include_status:
                lines.append("  #%d [%s] %s" % (g["id"], g["status"], g["text"]))
            else:
                lines.append("  #%d %s" % (g["id"], g["text"]))
        return "\n".join(lines)

    def active_summary_line(self):
        """One-line summary of current active goals for self-description."""
        active = self.list_active()
        if not active:
            return None
        if len(active) == 1:
            return "I am working on: %s." % active[0]["text"]
        heads = [g["text"] for g in active[:3]]
        return "I am working on: %s." % "; ".join(heads)
