#!/usr/bin/env python3
"""
BODHI Learning System — Hebbian Wiring, Emotional Memory, Sleep, Dreams

The brain from Reports 02-04 can perceive and react.
This module makes it LEARN from experience and REMEMBER.

Four learning mechanisms:
  1. Hebbian Learning — "neurons that fire together wire together"
  2. Emotional Memory — permanent feelings about concepts (fire=danger)
  3. Sleep Consolidation — strengthen strong memories, prune weak ones
  4. Dream Generation — blend recent experiences into new combinations

All integer math. All permanent (survives restart via save/load).

Author: SK (Sai Kiran Bathula) — April 2026
"""

import os
import sys
import json
import random
import time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from human_brain import HumanBrain


# ============================================================
# HEBBIAN LEARNING
# "Neurons that fire together wire together"
# When two concepts are active at the same time, the connection
# between them strengthens. Over time, thinking about fire
# automatically activates danger, burns, heat.
# ============================================================

class HebbianNetwork:
    """Integer-only associative memory.

    Stores (concept_a, concept_b) -> weight (int, 0-255).
    Weight increases when both concepts are active together.
    Weight decays slowly over time (forgetting).
    """

    def __init__(self):
        self.connections = {}  # (cid_a, cid_b) -> weight

    def learn(self, active_concepts, strength=3):
        """Strengthen connections between all co-active concepts."""
        for i in range(len(active_concepts)):
            for j in range(i + 1, min(len(active_concepts), i + 5)):
                pair = tuple(sorted([active_concepts[i], active_concepts[j]]))
                old = self.connections.get(pair, 0)
                self.connections[pair] = min(255, old + strength)

    def get_associates(self, concept, min_weight=5, max_results=5):
        """Find concepts associated with this one through learned wiring."""
        associates = []
        for (a, b), weight in self.connections.items():
            if weight < min_weight:
                continue
            if a == concept:
                associates.append((b, weight))
            elif b == concept:
                associates.append((a, weight))
        associates.sort(key=lambda x: -x[1])
        return associates[:max_results]

    def decay(self, amount=1):
        """Global decay — weak connections fade, strong ones persist."""
        to_delete = []
        for pair in self.connections:
            self.connections[pair] = max(0, self.connections[pair] - amount)
            if self.connections[pair] == 0:
                to_delete.append(pair)
        for pair in to_delete:
            del self.connections[pair]

    def consolidate(self):
        """Sleep consolidation — strengthen strong, prune weak.
        Connections > 20 get +2 (consolidate).
        Connections < 5 get deleted (prune)."""
        strengthened = 0
        pruned = 0
        to_delete = []
        for pair, weight in self.connections.items():
            if weight >= 20:
                self.connections[pair] = min(255, weight + 2)
                strengthened += 1
            elif weight < 5:
                to_delete.append(pair)
                pruned += 1
        for pair in to_delete:
            del self.connections[pair]
        return strengthened, pruned

    def total_connections(self):
        return len(self.connections)

    def strongest(self, n=10):
        """Return the N strongest connections."""
        return sorted(self.connections.items(), key=lambda x: -x[1])[:n]

    def save(self, path):
        data = {"%s|%s" % (a, b): w for (a, b), w in self.connections.items()}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.connections = {}
        for key, w in data.items():
            a, b = key.split("|")
            self.connections[(a, b)] = w


# ============================================================
# EMOTIONAL MEMORY
# Permanent feelings about concepts.
# Positive = danger (fire=+255 means maximum learned danger).
# Negative = safety (ocean=-200 means learned safety).
# These survive restarts. Touch fire once, remember forever.
# ============================================================

class EmotionalMemory:
    """Integer-only permanent emotional associations.

    Each concept gets a value from -255 (maximum safety) to +255 (maximum danger).
    Updated from experience, consolidated during sleep.
    """

    DANGER_EMOTIONS = ("fear", "anxiety", "anger", "disgust", "shame", "contempt")
    SAFETY_EMOTIONS = ("love", "joy", "peace", "trust", "awe", "pride")
    APPROACH_EMOTIONS = ("curiosity", "surprise")

    def __init__(self):
        self.memory = {}  # concept_id -> int (-255 to +255)

    def update(self, concept, emotion, pain_level=0):
        """Update emotional memory for a concept based on experienced emotion.
        Uses the CONCEPT'S emotion, not the overall conversation emotion.
        This prevents contamination (asking about 'mother vs fire' shouldn't
        make mother dangerous)."""
        old = self.memory.get(concept, 0)

        if emotion in self.DANGER_EMOTIONS:
            self.memory[concept] = min(255, old + 15)
        elif emotion in self.SAFETY_EMOTIONS:
            self.memory[concept] = max(-255, old - 10)
        elif emotion in self.APPROACH_EMOTIONS:
            self.memory[concept] = max(-255, old - 3)

        # Pain spike = stronger danger memory
        if pain_level > 50 and emotion in ("fear", "anger"):
            self.memory[concept] = min(255, self.memory.get(concept, 0) + 25)

    def get(self, concept):
        """Get emotional value. Positive=danger, negative=safe, 0=unknown."""
        return self.memory.get(concept, 0)

    def is_dangerous(self, concept, threshold=50):
        return self.memory.get(concept, 0) > threshold

    def is_safe(self, concept, threshold=-50):
        return self.memory.get(concept, 0) < threshold

    def consolidate(self):
        """Sleep consolidation — strong memories get stronger, weak fade."""
        consolidated = 0
        to_delete = []
        for cid, val in self.memory.items():
            if abs(val) > 30:
                # Strong memory — consolidate
                if val > 0:
                    self.memory[cid] = min(255, val + 5)
                else:
                    self.memory[cid] = max(-255, val - 3)
                consolidated += 1
            elif abs(val) < 5:
                to_delete.append(cid)
        for cid in to_delete:
            del self.memory[cid]
        return consolidated

    def top_dangers(self, n=10):
        return sorted([(c, v) for c, v in self.memory.items() if v > 0],
                      key=lambda x: -x[1])[:n]

    def top_safe(self, n=10):
        return sorted([(c, v) for c, v in self.memory.items() if v < 0],
                      key=lambda x: x[1])[:n]

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.memory, f)

    def load(self, path):
        if not os.path.exists(path):
            return
        with open(path) as f:
            self.memory = json.load(f)


# ============================================================
# CONVERSATION MEMORY
# Short-term: last 50 turns.
# Each turn stores: user input, emotion, concepts, response.
# ============================================================

class ConversationMemory:
    def __init__(self, max_turns=50):
        self.turns = []
        self.max_turns = max_turns

    def add(self, user_text, emotion, concepts, response):
        self.turns.append({
            "turn": len(self.turns) + 1,
            "user": user_text,
            "emotion": emotion,
            "concepts": concepts[:5],
            "response": response,
            "time": int(time.time()),
        })
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def recent_concepts(self, n=10):
        """Get concepts from last N turns (flattened + deduped)."""
        concepts = []
        for turn in self.turns[-n:]:
            concepts.extend(turn.get("concepts", []))
        return list(set(concepts))

    def recent_concept_sets(self, n=10):
        """Get the per-turn concept lists from the last N turns.
        Used for sleep replay — reinforces patterns that actually co-occurred."""
        return [turn.get("concepts", []) for turn in self.turns[-n:]]

    def last_emotion(self):
        if self.turns:
            return self.turns[-1].get("emotion", "neutral")
        return "neutral"


# ============================================================
# SLEEP CYCLE
# Consolidate memories, prune weak connections, dream.
# ============================================================

class SleepCycle:
    """Brain sleeps. Memories consolidate. Dreams happen.

    1. Hebbian consolidation (strengthen strong, prune weak)
    2. Emotional memory consolidation (strong feelings deepen)
    3. Dream generation (blend 2 recent concepts)
    4. Drive reset (fatigue drops, alertness resets)
    """

    def __init__(self):
        self.total_dreams = 0
        self.dream_log = []

    def sleep(self, hebbian, emotional, conversation,
              replay_turns=15, num_dreams=3,
              triangle_threshold=15, replay_strength=2,
              max_inferred=200):
        """Execute one sleep cycle with real Hebbian wiring. Returns stats dict.

        Phases:
          1. Replay — re-run hebbian.learn on the last N turns' concept sets
             with reduced strength. Reinforces patterns the user actually
             cared about, not one random dream.
          2. Triangle completion — for every strong A-B and B-C where A-C is
             missing, infer A-C at half the minimum link strength. This is how
             brains generalise.
          3. Consolidate — strengthen already-strong links, prune weak ones.
          4. Emotional consolidation — strong valences deepen, weak fade.
          5. Multiple dreams — blend N pairs of recent concepts (not one),
             each pair gets +5 and becomes a dream log entry. Primary dream
             (first one) is used for image reconstruction.
        """
        stats = {"strengthened": 0, "pruned": 0, "consolidated": 0,
                 "dreams": 0, "replayed": 0, "inferred": 0}

        # 1. REPLAY — re-run Hebbian learn on each recent turn's concepts
        turn_sets = conversation.recent_concept_sets(replay_turns)
        for concepts in turn_sets:
            if concepts and len(concepts) >= 2:
                hebbian.learn(concepts, strength=replay_strength)
                stats["replayed"] += 1

        # 2. TRIANGLE COMPLETION — infer A-C from A-B and B-C
        # Snapshot current connections so we don't iterate a mutating dict
        existing = dict(hebbian.connections)
        neighbors = {}
        for (a, b), w in existing.items():
            neighbors.setdefault(a, {})[b] = w
            neighbors.setdefault(b, {})[a] = w

        inferred = 0
        for a in list(neighbors.keys()):
            if inferred >= max_inferred:
                break
            a_neigh = neighbors[a]
            for b, wab in a_neigh.items():
                if wab < triangle_threshold:
                    continue
                b_neigh = neighbors.get(b, {})
                for c, wbc in b_neigh.items():
                    if c == a:
                        continue
                    if wbc < triangle_threshold:
                        continue
                    pair = tuple(sorted([a, c]))
                    if pair in hebbian.connections:
                        continue  # direct link already exists
                    inferred_w = min(wab, wbc) // 2
                    if inferred_w >= 5:
                        hebbian.connections[pair] = inferred_w
                        inferred += 1
                        if inferred >= max_inferred:
                            break
                if inferred >= max_inferred:
                    break
        stats["inferred"] = inferred

        # 3. Standard consolidation
        s, p = hebbian.consolidate()
        stats["strengthened"] = s
        stats["pruned"] = p

        # 4. Emotional consolidation
        stats["consolidated"] = emotional.consolidate()

        # 5. MULTIPLE DREAMS — N pairs instead of 1
        recent = conversation.recent_concepts(10)
        dream_list = []
        seen_pairs = set()
        tries = 0
        while len(dream_list) < num_dreams and len(recent) >= 2 and tries < num_dreams * 4:
            tries += 1
            a = random.choice(recent)
            others = [c for c in recent if c != a]
            if not others:
                break
            b = random.choice(others)
            pair = tuple(sorted([a, b]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            old = hebbian.connections.get(pair, 0)
            hebbian.connections[pair] = min(255, old + 5)
            dream = {"a": a, "b": b,
                     "emotion_a": emotional.get(a),
                     "emotion_b": emotional.get(b)}
            dream_list.append(dream)
            self.dream_log.append(dream)
            self.total_dreams += 1

        stats["dreams"] = len(dream_list)
        if dream_list:
            stats["dream_content"] = dream_list[0]
            stats["all_dreams"] = dream_list

        return stats

    def dream_image(self, concept_a, concept_b, img_data, index):
        """Generate a dream image by blending two concept fingerprints."""
        idx_a = index["img_name_to_idx"].get(concept_a)
        idx_b = index["img_name_to_idx"].get(concept_b)
        if idx_a is None or idx_b is None:
            return None
        fp_a = img_data[idx_a].astype(np.int32)
        fp_b = img_data[idx_b].astype(np.int32)
        # Blend: average the two fingerprints
        blended = ((fp_a + fp_b) >> 1).astype(np.int16)
        return blended


# ============================================================
# DRIVES — 7 internal motivations
# ============================================================

class DriveSystem:
    """7 drives that motivate BODHI's behavior.
    Each drive: 0-255 integer.
    Drives influence what BODHI pays attention to and how it responds."""

    DRIVE_NAMES = ["curiosity", "satisfaction", "confusion", "pain",
                   "alertness", "fatigue", "attachment"]

    def __init__(self):
        self.drives = {d: 0 for d in self.DRIVE_NAMES}

    def update_from_emotion(self, emotion, has_memories=False):
        """Update drives based on experienced emotion."""
        if emotion in ("fear", "anxiety", "anger"):
            self.drives["alertness"] = min(255, self.drives["alertness"] + 60)
            self.drives["confusion"] = min(255, self.drives["confusion"] + 30)
        elif emotion in ("love", "joy"):
            self.drives["satisfaction"] = min(255, self.drives["satisfaction"] + 50)
            self.drives["attachment"] = min(255, self.drives["attachment"] + 20)
        elif emotion in ("curiosity", "awe"):
            self.drives["curiosity"] = min(255, self.drives["curiosity"] + 60)
        else:
            self.drives["curiosity"] = min(255, self.drives["curiosity"] + 10)

        if has_memories:
            self.drives["satisfaction"] = min(255, self.drives["satisfaction"] + 20)
        else:
            self.drives["confusion"] = min(255, self.drives["confusion"] + 15)

    def decay(self):
        """Natural decay each turn."""
        for d in self.DRIVE_NAMES:
            if d == "fatigue":
                self.drives[d] = min(255, self.drives[d] + 1)
            else:
                self.drives[d] = max(0, self.drives[d] - 3)

    def needs_sleep(self):
        return self.drives["fatigue"] > 200

    def dominant(self):
        """Return the strongest drive."""
        top = max(self.drives.items(), key=lambda x: x[1])
        return top if top[1] > 30 else ("calm", 0)

    def reset_after_sleep(self):
        self.drives["fatigue"] = max(0, self.drives["fatigue"] - 50)
        self.drives["alertness"] = max(0, self.drives["alertness"] - 80)
        self.drives["confusion"] = max(0, self.drives["confusion"] - 40)
        self.drives["pain"] = max(0, self.drives["pain"] - 30)


# ============================================================
# LEARNING BRAIN — integrates everything
# ============================================================

class LearningBrain:
    """Complete BODHI brain with perception + learning.

    Combines:
      - HumanBrain (72 regions + worm brain)
      - HebbianNetwork (associative wiring)
      - EmotionalMemory (permanent feelings)
      - ConversationMemory (short-term)
      - SleepCycle (consolidation + dreams)
      - DriveSystem (7 motivations)
    """

    def __init__(self):
        self.brain = HumanBrain()
        self.hebbian = HebbianNetwork()
        self.emotional = EmotionalMemory()
        self.conversation = ConversationMemory()
        self.sleep_system = SleepCycle()
        self.drives = DriveSystem()
        self.turn_count = 0

        # Load fingerprint data
        data_dir = os.path.join(ROOT, "data")
        self.img_data = np.load(os.path.join(data_dir, "fingerprints_img.npz"))["data"]
        with open(os.path.join(data_dir, "fingerprint_index.json")) as f:
            self.fp_index = json.load(f)

        # Load centroids for concept lookup
        with open(os.path.join(data_dir, "brain", "centroids.json")) as f:
            centroids = json.load(f)
        self.concept_emotions = {c["id"]: c.get("emotion", "neutral") for c in centroids}
        self.concept_domains = {c["id"]: c.get("domain", "default") for c in centroids}

        # Aliases for word matching
        with open(os.path.join(data_dir, "brain", "aliases.json")) as f:
            self.aliases = json.load(f)

        print("LearningBrain: %d concepts, %d aliases, %d image fps" % (
            len(self.concept_emotions), len(self.aliases), self.img_data.shape[0]))

    def match_concepts(self, text):
        """Match words in text to known concepts."""
        import re
        words = set(re.findall(r'[a-z]+', text.lower()))
        stop = set("the a an is are was were be been have has had do does did will would could should can not and or but if for in on at to from by with as of this that it its he she they we all more most some very just also even said get am".split())
        content = words - stop

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

    def process(self, text):
        """Full brain cycle with learning.

        1. Match concepts from text
        2. Get Hebbian associates
        3. Process through human+worm brain with fingerprints
        4. Learn from experience
        5. Update drives
        6. Auto-sleep if fatigued
        """
        self.turn_count += 1

        # 1. Match concepts
        matched = self.match_concepts(text)

        # 2. Hebbian spread — activate learned associations
        associates = []
        for cid in matched[:3]:
            for assoc, weight in self.hebbian.get_associates(cid):
                if assoc not in matched and assoc not in associates:
                    associates.append(assoc)
        associates = associates[:3]

        # 3. Process primary concept through brain
        emotion = "neutral"
        intensity = 0
        brain_result = None

        if matched:
            primary = matched[0]
            # Check emotional memory first
            em = self.emotional.get(primary)
            if em > 50:
                emotion = "fear"
            elif em < -50:
                emotion = "love"
            else:
                emotion = self.concept_emotions.get(primary, "neutral")

            # Get fingerprint
            fp_idx = self.fp_index["img_name_to_idx"].get(primary)
            fp = self.img_data[fp_idx] if fp_idx is not None else None

            # Compute intensity from emotion
            intensity = 200
            if emotion in ("fear", "anger"):
                intensity = 350
            elif emotion in ("love", "joy"):
                intensity = 200
            elif emotion in ("curiosity", "awe"):
                intensity = 250

            brain_result = self.brain.process(fp, emotion, intensity)

        # 4. Learning
        all_active = matched + associates
        self.hebbian.learn(all_active)

        for cid in matched:
            cid_emotion = self.concept_emotions.get(cid, "neutral")
            self.emotional.update(cid, cid_emotion, self.drives.drives["pain"])

        # 5. Update drives
        self.drives.update_from_emotion(emotion, has_memories=bool(associates))
        self.drives.decay()

        # 6. Auto-sleep every 25 turns
        sleep_stats = None
        if self.turn_count % 25 == 0 or self.drives.needs_sleep():
            sleep_stats = self.sleep_system.sleep(
                self.hebbian, self.emotional, self.conversation)
            self.drives.reset_after_sleep()

        return {
            "matched": matched,
            "associates": associates,
            "emotion": emotion,
            "intensity": intensity,
            "brain": brain_result,
            "drives": dict(self.drives.drives),
            "dominant_drive": self.drives.dominant(),
            "hebbian_connections": self.hebbian.total_connections(),
            "emotional_memories": len(self.emotional.memory),
            "turn": self.turn_count,
            "sleep": sleep_stats,
        }

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.hebbian.save(os.path.join(directory, "hebbian.json"))
        self.emotional.save(os.path.join(directory, "emotional.json"))
        print("Saved brain state to %s" % directory)

    def load(self, directory):
        self.hebbian.load(os.path.join(directory, "hebbian.json"))
        self.emotional.load(os.path.join(directory, "emotional.json"))
        print("Loaded brain state from %s" % directory)


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    brain = LearningBrain()

    print("=" * 60)
    print("  BODHI Learning Brain Test")
    print("=" * 60)
    print()

    # Phase 1: Teach fire = danger
    print("--- Phase 1: Learning fire = danger (10 exposures) ---")
    for i in range(10):
        brain.process("fire is dangerous and burns everything")
        brain.process("fire causes pain and destruction")

    fire_em = brain.emotional.get("fire")
    fire_assoc = brain.hebbian.get_associates("fire")
    print("  fire emotional memory: %d (%s)" % (fire_em, "DANGER" if fire_em > 50 else "not strong"))
    print("  fire associates: %s" % [(c, w) for c, w in fire_assoc[:5]])
    print()

    # Phase 2: Teach ocean = safe
    print("--- Phase 2: Learning ocean = safe (10 exposures) ---")
    for i in range(10):
        brain.process("the ocean is peaceful and calm")
        brain.process("ocean waves are beautiful and soothing")

    ocean_em = brain.emotional.get("ocean")
    ocean_assoc = brain.hebbian.get_associates("ocean")
    print("  ocean emotional memory: %d (%s)" % (ocean_em, "SAFE" if ocean_em < -50 else "not strong"))
    print("  ocean associates: %s" % [(c, w) for c, w in ocean_assoc[:5]])
    print()

    # Phase 3: Sleep
    print("--- Phase 3: Sleep cycle ---")
    stats = brain.sleep_system.sleep(brain.hebbian, brain.emotional, brain.conversation)
    print("  Strengthened: %d, Pruned: %d, Consolidated: %d, Dreams: %d" % (
        stats["strengthened"], stats["pruned"], stats["consolidated"], stats["dreams"]))
    if "dream_content" in stats:
        d = stats["dream_content"]
        print("  Dream: %s + %s (emotions: %d, %d)" % (d["a"], d["b"], d["emotion_a"], d["emotion_b"]))
    print()

    # Phase 4: Test recall
    print("--- Phase 4: Does learning persist? ---")
    result = brain.process("tell me about fire")
    print("  fire matched: %s" % result["matched"])
    print("  fire associates (Hebbian): %s" % result["associates"])
    print("  emotion: %s (from emotional memory: %d)" % (result["emotion"], brain.emotional.get("fire")))
    if result["brain"]:
        w = result["brain"]["worm"]
        top = result["brain"]["top_regions"][:3]
        print("  worm: %s(%d)" % (w["reflex"], w["confidence"]))
        print("  top regions: %s" % [(r, v) for r, v in top])
    print()

    result = brain.process("tell me about ocean")
    print("  ocean matched: %s" % result["matched"])
    print("  ocean associates: %s" % result["associates"])
    print("  emotion: %s (emotional memory: %d)" % (result["emotion"], brain.emotional.get("ocean")))
    if result["brain"]:
        w = result["brain"]["worm"]
        print("  worm: %s(%d)" % (w["reflex"], w["confidence"]))
    print()

    # Phase 5: Multiple sleep cycles
    print("--- Phase 5: Multiple sleep cycles ---")
    for cycle in range(5):
        brain.process("fire destroyed the village")
        brain.process("ocean healed my wounds")
        stats = brain.sleep_system.sleep(brain.hebbian, brain.emotional, brain.conversation)
        print("  Cycle %d: fire=%d ocean=%d connections=%d" % (
            cycle+1, brain.emotional.get("fire"), brain.emotional.get("ocean"),
            brain.hebbian.total_connections()))
    print()

    # Phase 6: Test with unknown concept
    print("--- Phase 6: Unknown concept ---")
    result = brain.process("what is quantum physics")
    print("  matched: %s" % result["matched"])
    print("  associates: %s" % result["associates"])
    print("  emotion: %s" % result["emotion"])
    print()

    # Summary
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("  Turns: %d" % brain.turn_count)
    print("  Hebbian connections: %d" % brain.hebbian.total_connections())
    print("  Emotional memories: %d" % len(brain.emotional.memory))
    print("  Dreams: %d" % brain.sleep_system.total_dreams)
    print("  Top dangers: %s" % brain.emotional.top_dangers(5))
    print("  Top safe: %s" % brain.emotional.top_safe(5))
    print("  Strongest wiring: %s" % brain.hebbian.strongest(5))
    print("  Drives: %s" % brain.drives.drives)
