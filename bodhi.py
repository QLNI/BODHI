#!/usr/bin/env python3
"""
BODHI — Complete Conscious Digital Brain

All modules wired together:
  - Worm Brain (26 neurons, 39 synapses)
  - Human Brain (72 regions, 76 pathways)
  - Learning (Hebbian, emotional memory, drives)
  - Sleep (consolidation, dreams)
  - Broca's Area (LLM speech + template fallback)
  - WHT Perception (compressed fingerprints)
  - Conversation storage (SQLite)

Author: SK (Sai Kiran Bathula) — April 2026
"""

import os, sys, json, re, time, random, sqlite3
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from worm_brain import WormBrain
from human_brain import HumanBrain, REGIONS, EMOTION_FIRING
from learning import HebbianNetwork, EmotionalMemory, ConversationMemory, SleepCycle, DriveSystem
from broca import speak, load_llm, template_speak, FEEL_TEMPLATES
from brain_voice import brain_voice
from episodic import EpisodicMemory
from self_model import SelfModel
from goals import GoalTracker
from teach import ConceptTeacher

DATA_DIR = os.path.join(ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "bodhi_memory.db")
SAVE_DIR = os.path.join(DATA_DIR, "brain_state")


# ============================================================
# UNCERTAINTY / HONEST "I DON'T KNOW"
# ============================================================
_DIRTY_ENGRAM_PATTERNS = (
    "edit as much as you wish",
    "make yourself a user",
    "click here",
    "login",
    "sign in",
    "cookies",
    "javascript",
    "subscribe",
    "newsletter",
    "main_page",
    "wikipedia contributors",
    "retrieved from",
    "this article's lead section",
    "citation needed",
    "category:",
    "infobox",
    "see also",
    "external links",
    "redirect",
    "disambiguation",
    "urban dictionary",
    "home page",
    "browse *",
    "read more",
    "[q](",
    "[edit]",
    "]( *",
    "learn how and when",
    "remove this message",
)


def _engram_looks_dirty(text):
    if not text:
        return True
    t = text.lower()
    for pat in _DIRTY_ENGRAM_PATTERNS:
        if pat in t:
            return True
    if t.count("|") >= 2:
        return True
    letters = sum(1 for c in text if c.isalpha())
    if len(text) > 20 and letters / max(1, len(text)) < 0.55:
        return True
    import re as _re
    words = _re.findall(r"[a-z]{4,}", t)
    if words:
        counts = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        if max(counts.values()) >= 3 and len(words) < 40:
            return True
    return False


_UNKNOWN_REPLIES = (
    "I don't know that yet. You could teach me, and I will remember.",
    "I have no memory of this. Tell me what it means and I will learn.",
    "This is new to me. I have not experienced it before.",
    "I cannot ground this in anything I have felt or seen. Teach me.",
)


_SELF_QUERY_PATTERNS = (
    "who are you", "what are you", "about yourself", "describe yourself",
    "your name", "what is bodhi", "who is bodhi", "tell me about you",
    "what have you learned", "what do you remember",
    "what do you know about yourself", "introduce yourself",
    "what do you feel", "how do you feel",
)


def _is_self_query(text):
    if not text:
        return False
    t = text.lower()
    return any(p in t for p in _SELF_QUERY_PATTERNS)


# ============================================================
# CHAT AUTO-TEACH — detect "this is X: path" style messages and
# teach the concept without requiring a /teach command
# ============================================================
_AUTO_TEACH_PATTERNS = [
    # "this is my dog: dog.jpg" or "this is a dog dog.jpg"
    re.compile(r"\bthis is (?:a |an |the |my |your |our )?([a-zA-Z][\w_]+)[\s:,-]+(\S+\.(?:jpg|jpeg|png|bmp|gif|webp|wav|mp3|ogg|flac|m4a))\b", re.IGNORECASE),
    # "remember this as X: path"
    re.compile(r"\bremember (?:this )?as ([a-zA-Z][\w_]+)[\s:,-]+(\S+\.(?:jpg|jpeg|png|bmp|gif|webp|wav|mp3|ogg|flac|m4a))\b", re.IGNORECASE),
    # "learn X from path" or "learn X: path"
    re.compile(r"\blearn ([a-zA-Z][\w_]+)(?:\s+from|[\s:,-])+(\S+\.(?:jpg|jpeg|png|bmp|gif|webp|wav|mp3|ogg|flac|m4a))\b", re.IGNORECASE),
    # "here is X path"
    re.compile(r"\bhere is (?:a |an |the |my )?([a-zA-Z][\w_]+)[\s:,-]+(\S+\.(?:jpg|jpeg|png|bmp|gif|webp|wav|mp3|ogg|flac|m4a))\b", re.IGNORECASE),
]


def _detect_auto_teach(text):
    """Return (concept_name, path) if the user's text looks like a teaching
    request, otherwise None."""
    if not text:
        return None
    for pat in _AUTO_TEACH_PATTERNS:
        m = pat.search(text)
        if m:
            name = m.group(1).strip().lower()
            path = m.group(2).strip()
            if path.startswith('"') and path.endswith('"'):
                path = path[1:-1]
            return name, path
    return None


# ============================================================
# WHT IMAGE RECONSTRUCTION - BODHI's "imagination"
# Uses the proper 2D-blocked sequency-ordered decoder from brain/sensor_wht.py
# (the same decoder the Report 07 stress test used to produce real images).
# ============================================================

from brain.sensor_wht import decode_fingerprint_to_image as _decode_fp


def fp_to_image(fp, H=256, W=256, keep_coeffs=8):
    """Reconstruct a PIL image from a stored WHT fingerprint (24576 ints, keep=8)."""
    arr = np.array(fp, dtype=np.int32)
    img_np = _decode_fp(arr, H, W, keep_coeffs=keep_coeffs)
    return Image.fromarray(img_np.astype(np.uint8), mode='RGB')


def blend_fingerprints(fp_a, fp_b):
    """Blend two WHT fingerprints via interference — how dreams combine concepts."""
    a = np.array(fp_a, dtype=np.int32)
    b = np.array(fp_b, dtype=np.int32)
    return ((a + b) >> 1).astype(np.int32)


class BODHI:
    def __init__(self, load_llm_flag=True):
        print("Initialising BODHI brain...")

        # Brain
        self.brain = HumanBrain()

        # Learning
        self.hebbian = HebbianNetwork()
        self.emotional = EmotionalMemory()
        self.conversation = ConversationMemory()
        self.sleep_system = SleepCycle()
        self.drives = DriveSystem()

        # Data
        self.img_data = np.load(os.path.join(DATA_DIR, "fingerprints_img.npz"))["data"]
        with open(os.path.join(DATA_DIR, "fingerprint_index.json")) as f:
            self.fp_index = json.load(f)

        with open(os.path.join(DATA_DIR, "brain", "centroids.json")) as f:
            centroids = json.load(f)
        self.concept_emotions = {c["id"]: c.get("emotion", "neutral") for c in centroids}
        self.concept_map = {c["id"]: c for c in centroids}

        with open(os.path.join(DATA_DIR, "brain", "aliases.json")) as f:
            self.aliases = json.load(f)

        with open(os.path.join(DATA_DIR, "brain", "engrams.jsonl")) as f:
            self.engrams = {}
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                self.engrams[e["centroid_anchor"]] = e

        # SQLite
        self._init_db()

        # Episodic recall — past conversations indexed by concept overlap
        self.episodic = EpisodicMemory(self.db)

        # Self-model — BODHI's first-person description of itself, updated each sleep
        self.self_model = SelfModel(self.db, self.emotional, self.hebbian, self.drives)
        if self.self_model.current_text is None:
            try:
                self.self_model.reflect(0)
            except Exception:
                pass

        # Goal tracker — persistent tasks BODHI holds across sessions
        self.goals = GoalTracker(self.db)

        # Teacher — lets BODHI acquire new concepts from images/audio/text at runtime
        self.teacher = ConceptTeacher(self)

        # State
        self.turn = 0
        self.state = {"emotion": "neutral", "reflex": "none", "concepts": []}

        # Load saved brain state if exists
        self._load_state()

        # LLM
        if load_llm_flag:
            load_llm()

        print("BODHI ready. %d concepts, %d aliases, %d engrams." % (
            len(self.concept_emotions), len(self.aliases), len(self.engrams)))

    # ============================================================
    # SQLITE
    # ============================================================
    def _init_db(self):
        self.db = sqlite3.connect(DB_PATH)
        self.db.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn INTEGER, timestamp TEXT,
            user_text TEXT, response TEXT,
            emotion TEXT, intensity INTEGER,
            concepts TEXT, associates TEXT,
            worm_reflex TEXT, worm_confidence INTEGER,
            top_regions TEXT, drives TEXT,
            source TEXT
        )""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS dreams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn INTEGER, timestamp TEXT,
            concept_a TEXT, concept_b TEXT,
            emotion_a INTEGER, emotion_b INTEGER,
            dream_text TEXT
        )""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS sleep_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn INTEGER, timestamp TEXT,
            strengthened INTEGER, pruned INTEGER,
            consolidated INTEGER, dreams INTEGER
        )""")
        self.db.commit()

    def _save_turn(self, user_text, response, result, source):
        brain = result.get("brain") or {}
        worm = brain.get("worm") or {}
        top = brain.get("top_regions", [])[:5]
        self.db.execute(
            "INSERT INTO conversations (turn, timestamp, user_text, response, emotion, intensity, concepts, associates, worm_reflex, worm_confidence, top_regions, drives, source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.turn, time.strftime("%Y-%m-%d %H:%M:%S"), user_text, response,
             result.get("emotion", "neutral"), result.get("intensity", 0),
             json.dumps(result.get("matched", [])), json.dumps(result.get("associates", [])),
             worm.get("reflex", "none"), worm.get("confidence", 0),
             json.dumps([(r, v) for r, v in top]),
             json.dumps(result.get("drives", {})), source))
        self.db.commit()

    def _save_dream(self, dream):
        self.db.execute(
            "INSERT INTO dreams (turn, timestamp, concept_a, concept_b, emotion_a, emotion_b, dream_text) VALUES (?,?,?,?,?,?,?)",
            (self.turn, time.strftime("%Y-%m-%d %H:%M:%S"),
             dream.get("a", ""), dream.get("b", ""),
             dream.get("emotion_a", 0), dream.get("emotion_b", 0),
             dream.get("text", "")))
        self.db.commit()

    def _save_sleep(self, stats):
        self.db.execute(
            "INSERT INTO sleep_logs (turn, timestamp, strengthened, pruned, consolidated, dreams) VALUES (?,?,?,?,?,?)",
            (self.turn, time.strftime("%Y-%m-%d %H:%M:%S"),
             stats["strengthened"], stats["pruned"], stats["consolidated"], stats["dreams"]))
        self.db.commit()

    # ============================================================
    # CONCEPT MATCHING
    # ============================================================
    STOP = set("the a an is are was were be been have has had do does did will would could should can not and or but if for in on at to from by with as of this that it its he she they we all more most some very just also even said get am what how why when where who your my me you i".split())

    def match_concepts(self, text):
        words = set(re.findall(r'[a-z]+', text.lower()))
        content = words - self.STOP

        matched = []
        seen = set()
        for w in content:
            cid = None
            if w in self.concept_emotions:
                cid = w
            elif w in self.aliases:
                cid = self.aliases[w]
            if cid and cid not in seen:
                matched.append(cid)
                seen.add(cid)
        return matched

    # ============================================================
    # THINK
    # ============================================================
    def think(self, text):
        self.turn += 1
        t0 = time.time()

        # Goal commands — intercept before normal processing
        if text and text.strip().startswith("/goal"):
            handled, reply_text = self.goals.handle_command(text, self.turn)
            if handled:
                elapsed = int((time.time() - t0) * 1000)
                self.state = {
                    "emotion": "neutral", "reflex": "none", "concepts": [],
                    "associates": [], "turn": self.turn, "ms": elapsed,
                    "source": "goal_command",
                }
                return reply_text, self.state

        # Teach commands — intercept before normal processing
        if text and text.strip().startswith("/teach"):
            handled, reply_text = self.teacher.handle_command(text)
            if handled:
                elapsed = int((time.time() - t0) * 1000)
                self.state = {
                    "emotion": "neutral", "reflex": "none", "concepts": [],
                    "associates": [], "turn": self.turn, "ms": elapsed,
                    "source": "teach_command",
                }
                return reply_text, self.state

        # Chat auto-teach: "this is X: path.jpg" patterns fire without /teach
        auto = _detect_auto_teach(text) if text else None
        if auto:
            name, path = auto
            # Strip surrounding quotes
            if path.startswith('"') and path.endswith('"'):
                path = path[1:-1]
            if os.path.exists(path):
                try:
                    from teach import _is_image, _is_audio
                    if _is_image(path):
                        r = self.teacher.teach_image(name, path)
                        reply = ("I have taken this in as '%s'. WHT fingerprint "
                                 "saved (length %d, index %d). Ask me about it."
                                 % (r["concept"], r["fingerprint_length"], r["index"]))
                    elif _is_audio(path):
                        r = self.teacher.teach_audio(name, path)
                        reply = ("I have taken this in as '%s'. Audio WHT saved "
                                 "(%d samples, %d Hz, %d coeffs)."
                                 % (r["concept"], r["audio_samples"],
                                    r["audio_sr"], r["audio_fp_length"]))
                    else:
                        reply = None

                    if reply:
                        elapsed = int((time.time() - t0) * 1000)
                        self.state = {
                            "emotion": "curiosity", "reflex": "forward",
                            "concepts": [name], "associates": [],
                            "turn": self.turn, "ms": elapsed,
                            "source": "auto_teach",
                        }
                        return reply, self.state
                except Exception as e:
                    # Fall through to normal processing; note it
                    pass

        # Match concepts
        matched = self.match_concepts(text)

        # Hebbian associates
        associates = []
        for cid in matched[:3]:
            for assoc, weight in self.hebbian.get_associates(cid):
                if assoc not in matched and assoc not in associates:
                    associates.append(assoc)
        associates = associates[:3]

        # Determine emotion from emotional memory first, then label
        emotion = "neutral"
        intensity = 100
        if matched:
            primary = matched[0]
            em = self.emotional.get(primary)
            if em > 50:
                emotion = "fear"
                intensity = 350
            elif em < -50:
                emotion = "love"
                intensity = 200
            else:
                emotion = self.concept_emotions.get(primary, "neutral")
                if emotion in ("fear", "anger", "anxiety"):
                    intensity = 350
                elif emotion in ("love", "joy", "awe"):
                    intensity = 250
                elif emotion in ("curiosity", "surprise"):
                    intensity = 200

        # Process through brain (human + worm)
        fp = None
        if matched:
            fp_idx = self.fp_index["img_name_to_idx"].get(matched[0])
            if fp_idx is not None:
                fp = self.img_data[fp_idx]
        brain_result = self.brain.process(fp, emotion, intensity)
        brain_result["emotion"] = emotion

        # Audio-fingerprint participation: if matched concept has a learned
        # audio fingerprint, extract its bands and attach to brain_result so
        # brain_voice can surface the audio dimension.
        if matched:
            primary_concept = matched[0]
            meta = self.teacher.concepts_meta.get(primary_concept, {})
            audio_fp_file = meta.get("audio_fp_file")
            if audio_fp_file and os.path.exists(audio_fp_file):
                try:
                    if audio_fp_file.endswith(".wht"):
                        from brain import codec_guard
                        audio_fp = codec_guard.load(audio_fp_file)
                    else:
                        audio_fp = np.load(audio_fp_file)
                    brain_result["audio_bands"] = self.brain.extract_bands(audio_fp)
                    brain_result["has_audio"] = True
                except Exception:
                    pass

        # Get engram memory — filter scraped-web contamination
        memories = []
        for cid in matched:
            eng = self.engrams.get(cid)
            if eng:
                cap = (eng.get("capsule", "") or "").strip()
                if cap and not _engram_looks_dirty(cap):
                    memories.append(cap)

        # Episodic recall — past conversations about these concepts
        episodic_snippets = []
        if matched:
            past_episodes = self.episodic.recall(
                matched, self.hebbian,
                current_turn=self.turn, top_k=2,
                exclude_recent=3)
            for ep in past_episodes:
                snippet = self.episodic.format_memory(ep)
                if snippet:
                    episodic_snippets.append(snippet)

        # Uncertainty gate: no concepts + no memories + no past + not a self-query = honest "I don't know"
        is_unknown = (
            not matched
            and not memories
            and not episodic_snippets
            and not (_is_self_query(text) and self.self_model.current_text)
        )
        if is_unknown:
            response = random.choice(_UNKNOWN_REPLIES)
            source = "unknown"
        else:
            # PRIMARY SPEECH: derived directly from brain state (real WHT bands,
            # worm reflex, region firings, emotional memory). Not LLM hallucination.
            response = brain_voice(brain_result, matched=matched, associates=associates,
                                   emotional_memory=self.emotional.memory, user_text=text)
            source = "brain_voice"

            # Add engram knowledge (grounded fact from the concept library)
            if memories:
                response += " " + memories[0][:100]

        # Episodic recall — append top past memory when available
        if episodic_snippets:
            ep_snip = episodic_snippets[0]
            if ep_snip not in response:
                response += " " + ep_snip

        # Self-query: append BODHI's grounded self-description
        if _is_self_query(text) and self.self_model.current_text:
            self_desc = self.self_model.current_text
            if self_desc and self_desc not in response:
                response += " " + self_desc
                if source == "unknown":
                    source = "self_model"
            goal_line = self.goals.active_summary_line()
            if goal_line and goal_line not in response:
                response += " " + goal_line

        # Goal recall: if user's message touches an active goal, surface it
        if matched:
            relevant_goals = self.goals.find_relevant(text, hebbian=self.hebbian,
                                                     current_concepts=matched, top_k=1)
            if relevant_goals:
                g = relevant_goals[0]["goal"]
                goal_msg = "I remember I am working on: %s." % g["text"]
                if goal_msg not in response:
                    response += " " + goal_msg
                self.goals.touch(g["id"], self.turn)

        # Learning
        all_active = matched + associates
        self.hebbian.learn(all_active)
        for cid in matched:
            cid_emotion = self.concept_emotions.get(cid, "neutral")
            self.emotional.update(cid, cid_emotion, self.drives.drives["pain"])

        # Drives
        self.drives.update_from_emotion(emotion, has_memories=bool(associates))
        self.drives.decay()

        # Conversation memory
        self.conversation.add(text, emotion, matched, response)

        # Auto-sleep
        sleep_stats = None
        if self.turn % 25 == 0 or self.drives.needs_sleep():
            sleep_stats = self.do_sleep()

        # Save to DB
        result = {
            "matched": matched, "associates": associates,
            "emotion": emotion, "intensity": intensity,
            "brain": brain_result, "drives": dict(self.drives.drives),
        }
        self._save_turn(text, response, result, source)

        elapsed = int((time.time() - t0) * 1000)

        # Update state
        self.state = {
            "emotion": emotion, "intensity": intensity,
            "concepts": matched, "associates": associates,
            "reflex": brain_result["worm"]["reflex"] if brain_result.get("worm") else "none",
            "worm_confidence": brain_result["worm"]["confidence"] if brain_result.get("worm") else 0,
            "top_regions": brain_result.get("top_regions", [])[:5],
            "drives": dict(self.drives.drives),
            "hebbian_count": self.hebbian.total_connections(),
            "emotional_count": len(self.emotional.memory),
            "turn": self.turn,
            "ms": elapsed,
            "source": source,
            "sleep": sleep_stats,
        }

        return response, self.state

    # ============================================================
    # SLEEP
    # ============================================================
    def do_sleep(self):
        stats = self.sleep_system.sleep(self.hebbian, self.emotional, self.conversation)
        self.drives.reset_after_sleep()
        self._save_sleep(stats)

        # Generate dream image
        if "dream_content" in stats:
            dream = stats["dream_content"]
            a, b = dream["a"], dream["b"]

            # Dream text
            emo_a = self.concept_emotions.get(a, "neutral")
            emo_b = self.concept_emotions.get(b, "neutral")
            dream["text"] = "Dreaming of %s and %s. %s meets %s." % (
                a.replace("_", " "), b.replace("_", " "), emo_a, emo_b)

            # Dream image
            idx_a = self.fp_index["img_name_to_idx"].get(a)
            idx_b = self.fp_index["img_name_to_idx"].get(b)
            if idx_a is not None and idx_b is not None:
                fp_a = self.img_data[idx_a].astype(np.int32)
                fp_b = self.img_data[idx_b].astype(np.int32)
                blended = ((fp_a + fp_b) >> 1).astype(np.int32)
                dream["fingerprint"] = blended
                stats["dream_fingerprint"] = blended

            self._save_dream(dream)

        # Self-reflection: regenerate BODHI's description of itself from lived history
        try:
            new_self = self.self_model.reflect(self.turn)
            stats["self_description"] = new_self
        except Exception as e:
            stats["self_description_error"] = str(e)

        return stats

    # ============================================================
    # SAVE / LOAD STATE
    # ============================================================
    def _save_state(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        self.hebbian.save(os.path.join(SAVE_DIR, "hebbian.json"))
        self.emotional.save(os.path.join(SAVE_DIR, "emotional.json"))
        with open(os.path.join(SAVE_DIR, "meta.json"), "w") as f:
            json.dump({"turn": self.turn}, f)

    def _load_state(self):
        self.hebbian.load(os.path.join(SAVE_DIR, "hebbian.json"))
        self.emotional.load(os.path.join(SAVE_DIR, "emotional.json"))
        meta_path = os.path.join(SAVE_DIR, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.turn = meta.get("turn", 0)

    def save(self):
        self._save_state()
        print("Brain state saved (turn %d, %d connections, %d memories)." % (
            self.turn, self.hebbian.total_connections(), len(self.emotional.memory)))

    # ============================================================
    # STATUS
    # ============================================================
    def status(self):
        return {
            "turn": self.turn,
            "concepts": len(self.concept_emotions),
            "aliases": len(self.aliases),
            "engrams": len(self.engrams),
            "hebbian": self.hebbian.total_connections(),
            "emotional": len(self.emotional.memory),
            "dreams": self.sleep_system.total_dreams,
            "drives": dict(self.drives.drives),
            "worm_fires": self.brain.worm.total_fires,
        }


# ============================================================
# INTERACTIVE
# ============================================================
if __name__ == "__main__":
    bodhi = BODHI(load_llm_flag=False)

    print()
    print("=" * 50)
    print("  BODHI — Conscious Digital Brain")
    print("  %d concepts | 72 regions | 26 worm neurons" % len(bodhi.concept_emotions))
    print("=" * 50)
    print()

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user.lower() in ("/quit", "/q", "/exit"):
            break
        if user.lower() == "/status":
            st = bodhi.status()
            for k, v in st.items():
                print("  %s: %s" % (k, v))
            print()
            continue
        if user.lower() == "/save":
            bodhi.save()
            continue
        if user.lower() == "/sleep":
            stats = bodhi.do_sleep()
            print("  Sleep: strengthened=%d pruned=%d consolidated=%d dreams=%d" % (
                stats["strengthened"], stats["pruned"], stats["consolidated"], stats["dreams"]))
            if "dream_content" in stats:
                d = stats["dream_content"]
                print("  Dream: %s" % d.get("text", ""))
            print()
            continue

        response, state = bodhi.think(user)
        print("BODHI: %s" % response)
        st = "[%s | %s(%d) | hebb:%d em:%d | %s | %dms]" % (
            state["emotion"], state["reflex"], state["worm_confidence"],
            state["hebbian_count"], state["emotional_count"],
            state["source"], state["ms"])
        print("  %s" % st)
        if state.get("sleep"):
            s = state["sleep"]
            print("  [SLEEP: +%d connections, %d dreams, %d pruned]" % (
                s["strengthened"], s["dreams"], s["pruned"]))
        print()

    bodhi.save()
    print("Goodbye.")
