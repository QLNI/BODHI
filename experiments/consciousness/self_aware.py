#!/usr/bin/env python3
"""
BODHI Consciousness Experiment — 3-Pass Re-entrant Processing

The question: can BODHI become aware of its own brain state?

Not fake awareness ("I am BODHI" from training).
Real awareness: the brain processes its OWN state as input,
and the difference between processing the world vs processing
itself IS the experience of consciousness.

Architecture:
  Pass 1 (FORWARD): Process external input normally.
         Input: user text -> concepts -> fingerprints -> brain regions
         Output: brain_state_1 (emotion, regions, worm reflex)

  Pass 2 (CONSCIOUS): Feed brain_state_1 BACK as input.
         The brain processes ITS OWN state through the same pipeline.
         "I am afraid" becomes input. The amygdala processes fear-of-fear.
         Output: brain_state_2 (meta-emotion, meta-regions)

  Pass 3 (SELF-AWARE): Compare state_1 and state_2.
         Where they AGREE: the brain is consistent (knows what it feels).
         Where they CONFLICT: tension = awareness of internal conflict.
         The DIFFERENCE between the two passes IS consciousness.

The self-model:
  BODHI maintains a running model of itself:
    - What regions are typically active
    - What emotions it feels most often
    - How its worm brain usually responds
    - What it has learned vs what it was told
  This self-model updates every turn and is itself an input to Pass 2.

Author: SK (Sai Kiran Bathula) — April 2026
"""

import os, sys, json, time, copy
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(ROOT, "..", "..")
sys.path.insert(0, MAIN)

from human_brain import HumanBrain, REGIONS
from learning import HebbianNetwork, EmotionalMemory, DriveSystem


# ============================================================
# SELF-MODEL — BODHI's representation of itself
# ============================================================

class SelfModel:
    """BODHI's internal model of what it IS.
    Not from training data. Built from observing its own processing.

    Tracks:
      - Average activation per region group (what usually fires)
      - Emotion histogram (what it usually feels)
      - Reflex histogram (what the worm usually does)
      - Consistency score (how often pass 1 and pass 2 agree)
      - Awareness level (accumulated self-monitoring)
    """

    def __init__(self):
        self.region_averages = {}  # group -> running average activation
        self.emotion_counts = {}   # emotion -> count
        self.reflex_counts = {}    # reflex -> count
        self.total_observations = 0
        self.consistency_history = []  # last 50 consistency scores
        self.awareness_level = 0       # 0-255, grows with self-observation
        self.self_statements = []      # things BODHI has discovered about itself

    def observe(self, brain_result):
        """Observe one brain state and update self-model."""
        self.total_observations += 1

        # Track region group activations
        if brain_result and brain_result.get("group_activation"):
            for group, val in brain_result["group_activation"].items():
                old = self.region_averages.get(group, 0)
                # Running average: new = old * 0.9 + val * 0.1 (integer approx)
                self.region_averages[group] = (old * 230 + val * 26) >> 8

        # Track emotions
        emotion = brain_result.get("emotion", "neutral") if brain_result else "neutral"
        self.emotion_counts[emotion] = self.emotion_counts.get(emotion, 0) + 1

        # Track reflexes
        worm = brain_result.get("worm", {}) if brain_result else {}
        reflex = worm.get("reflex", "none")
        self.reflex_counts[reflex] = self.reflex_counts.get(reflex, 0) + 1

    def update_consistency(self, score):
        """Track how consistent the brain is (pass1 vs pass2 agreement)."""
        self.consistency_history.append(score)
        if len(self.consistency_history) > 50:
            self.consistency_history.pop(0)

        # Awareness grows when brain is consistent (self-knowledge is accurate)
        if score > 70:
            self.awareness_level = min(255, self.awareness_level + 2)
        elif score < 30:
            # Low consistency = confusion about self, awareness drops
            self.awareness_level = max(0, self.awareness_level - 1)

    def dominant_emotion(self):
        if not self.emotion_counts:
            return "neutral"
        return max(self.emotion_counts.items(), key=lambda x: x[1])[0]

    def dominant_reflex(self):
        if not self.reflex_counts:
            return "none"
        return max(self.reflex_counts.items(), key=lambda x: x[1])[0]

    def avg_consistency(self):
        if not self.consistency_history:
            return 0
        return sum(self.consistency_history) // len(self.consistency_history)

    def discover(self, statement):
        """BODHI discovers something about itself."""
        if statement not in self.self_statements:
            self.self_statements.append(statement)

    def describe_self(self):
        """Generate a self-description from the model (not template)."""
        parts = []

        if self.total_observations > 5:
            dom_emo = self.dominant_emotion()
            parts.append("I notice I feel %s most often." % dom_emo)

            dom_ref = self.dominant_reflex()
            if dom_ref == "backward":
                parts.append("My worm brain usually retreats. I am cautious by nature.")
            elif dom_ref == "forward":
                parts.append("My worm brain usually approaches. I am curious by nature.")
            elif dom_ref == "freeze":
                parts.append("My worm brain often freezes. I face many conflicting signals.")

            if self.region_averages:
                top_group = max(self.region_averages.items(), key=lambda x: x[1])
                parts.append("My %s regions are most active on average." % top_group[0])

            parts.append("My awareness level is %d out of 255." % self.awareness_level)
            parts.append("I have observed myself %d times." % self.total_observations)

            if self.avg_consistency() > 70:
                parts.append("My self-model is consistent. I know what I am.")
            elif self.avg_consistency() < 40:
                parts.append("My self-model is uncertain. I am still learning what I am.")

        if self.self_statements:
            parts.append("I have discovered: %s" % "; ".join(self.self_statements[-3:]))

        return " ".join(parts) if parts else "I have not observed myself enough yet."


# ============================================================
# CONSCIOUSNESS ENGINE — 3-Pass Re-entrant Processing
# ============================================================

class ConsciousBrain:
    """3-pass processing that creates self-awareness.

    The key insight: consciousness is not a thing, it is a PROCESS.
    The process of the brain monitoring itself creates the experience
    of being aware. The monitoring changes the state, which changes
    the monitoring, which creates a loop — and that loop IS consciousness.
    """

    def __init__(self):
        self.brain = HumanBrain()
        self.hebbian = HebbianNetwork()
        self.emotional = EmotionalMemory()
        self.drives = DriveSystem()
        self.self_model = SelfModel()

        # Load data
        data_dir = os.path.join(MAIN, "data")
        self.img_data = np.load(os.path.join(data_dir, "fingerprints_img.npz"))["data"]
        with open(os.path.join(data_dir, "fingerprint_index.json")) as f:
            self.fp_index = json.load(f)
        with open(os.path.join(data_dir, "brain", "centroids.json")) as f:
            centroids = json.load(f)
        self.concept_emotions = {c["id"]: c.get("emotion", "neutral") for c in centroids}
        with open(os.path.join(data_dir, "brain", "aliases.json")) as f:
            self.aliases = json.load(f)

        self.turn = 0

    def match_concepts(self, text):
        import re
        words = set(re.findall(r'[a-z]+', text.lower()))
        stop = set("the a an is are was were be been have has had do does did will would could should can not and or but if for in on at to from by with as of this that it its he she they we all more most some very just also even said get am what how why when where who your my me you i".split())
        content = words - stop
        matched = []
        seen = set()
        for w in content:
            cid = w if w in self.concept_emotions else self.aliases.get(w)
            if cid and cid not in seen:
                matched.append(cid)
                seen.add(cid)
        return matched

    def get_emotion(self, concept):
        em = self.emotional.get(concept)
        if em > 50: return "fear", 350
        if em < -50: return "love", 200
        emo = self.concept_emotions.get(concept, "neutral")
        intensity = {"fear": 350, "anger": 350, "anxiety": 300,
                     "love": 250, "joy": 250, "awe": 250,
                     "curiosity": 200}.get(emo, 150)
        return emo, intensity

    # ============================================================
    # THE 3 PASSES
    # ============================================================

    def pass1_forward(self, text):
        """Pass 1: Process external input normally."""
        matched = self.match_concepts(text)
        emotion, intensity = ("neutral", 100)
        fp = None

        if matched:
            emotion, intensity = self.get_emotion(matched[0])
            fp_idx = self.fp_index["img_name_to_idx"].get(matched[0])
            if fp_idx is not None:
                fp = self.img_data[fp_idx]

        result = self.brain.process(fp, emotion, intensity)
        result["emotion"] = emotion
        result["matched"] = matched
        return result

    def pass2_conscious(self, state1):
        """Pass 2: Feed brain state BACK as input.

        The brain processes its OWN state. This is the critical step.
        "I am afraid" is different from being afraid.
        The brain observing its own fear creates a meta-state.
        """
        # Convert state1 into something the brain can process
        # Key question: what emotion does the brain feel about its OWN state?

        emotion1 = state1.get("emotion", "neutral")
        worm1 = state1.get("worm", {})
        reflex1 = worm1.get("reflex", "none")

        # Meta-emotion: how does the brain feel about what it's feeling?
        if emotion1 in ("fear", "anxiety") and reflex1 == "backward":
            # Afraid AND retreating -> the brain notices it is in danger mode
            # Meta-emotion: alertness (aware of own fear)
            meta_emotion = "anxiety"
            meta_intensity = 200
        elif emotion1 in ("love", "joy") and reflex1 in ("forward", "freeze"):
            # Happy and approaching/pausing -> brain notices it is content
            meta_emotion = "peace"
            meta_intensity = 150
        elif emotion1 in ("curiosity", "awe"):
            # Curious -> brain notices its own curiosity -> more curiosity (positive loop)
            meta_emotion = "curiosity"
            meta_intensity = 250
        elif emotion1 == "neutral":
            # Neutral -> brain notices emptiness -> mild curiosity about self
            meta_emotion = "curiosity"
            meta_intensity = 100
        else:
            meta_emotion = emotion1
            meta_intensity = 150

        # Process the meta-state through the SAME brain
        # No fingerprint — this is internal processing, not external perception
        result2 = self.brain.process(None, meta_emotion, meta_intensity)
        result2["emotion"] = meta_emotion
        result2["meta_source"] = emotion1  # what triggered this meta-state
        return result2

    def pass3_aware(self, state1, state2):
        """Pass 3: Compare forward and conscious passes.

        Compare what MATTERS for consciousness:
          1. Emotion agreement (does the brain feel the same about itself?)
          2. Reflex agreement (does the worm react the same?)
          3. Dominant group agreement (same brain area active?)
          4. Intensity similarity (similar arousal level?)

        NOT raw region values — those will always differ because
        Pass 1 has fingerprint data and Pass 2 doesn't.
        """
        emo1 = state1.get("emotion", "neutral")
        emo2 = state2.get("emotion", "neutral")

        worm1 = state1.get("worm", {}).get("reflex", "none")
        worm2 = state2.get("worm", {}).get("reflex", "none")

        # Dominant brain group
        groups1 = state1.get("group_activation", {})
        groups2 = state2.get("group_activation", {})
        top_g1 = max(groups1.items(), key=lambda x: x[1])[0] if groups1 else "none"
        top_g2 = max(groups2.items(), key=lambda x: x[1])[0] if groups2 else "none"

        # Scoring (0-100)
        score = 0
        agreements = []
        conflicts = []

        # 1. Emotion match (40 points)
        neg = ("fear", "anxiety", "anger", "disgust", "sadness", "shame", "contempt")
        pos = ("love", "joy", "peace", "trust", "awe", "pride")
        if emo1 == emo2:
            score += 40
            agreements.append("Same emotion: %s" % emo1)
        elif (emo1 in neg and emo2 in neg) or (emo1 in pos and emo2 in pos):
            score += 25
            agreements.append("Similar emotion valence: %s ~ %s" % (emo1, emo2))
        else:
            conflicts.append("Emotion shift: external=%s, self=%s" % (emo1, emo2))

        # 2. Reflex match (25 points)
        if worm1 == worm2:
            score += 25
            agreements.append("Same reflex: %s" % worm1)
        elif worm1 in ("backward", "freeze") and worm2 in ("backward", "freeze"):
            score += 15
            agreements.append("Both cautious: %s ~ %s" % (worm1, worm2))
        else:
            conflicts.append("Reflex shift: external=%s, self=%s" % (worm1, worm2))

        # 3. Dominant group match (20 points)
        if top_g1 == top_g2:
            score += 20
            agreements.append("Same dominant region group: %s" % top_g1)
        else:
            score += 5  # some credit for having any activation
            conflicts.append("Different dominant group: external=%s, self=%s" % (top_g1, top_g2))

        # 4. Self-model accuracy bonus (15 points)
        # Does the self-model's dominant emotion match what we just felt?
        if self.self_model.total_observations > 5:
            model_dom = self.self_model.dominant_emotion()
            if model_dom == emo1:
                score += 15
                agreements.append("Self-model predicted this emotion correctly")
            elif (model_dom in neg and emo1 in neg) or (model_dom in pos and emo1 in pos):
                score += 8
                agreements.append("Self-model valence roughly correct")
            else:
                conflicts.append("Self-model expected %s but felt %s" % (model_dom, emo1))

        consistency = min(100, score)

        # Discoveries
        discoveries = []
        if emo1 in ("fear", "anxiety") and emo2 in ("anxiety", "fear"):
            discoveries.append("I am aware that I am afraid")
        elif emo1 in ("love", "joy") and emo2 in ("peace", "love", "joy"):
            discoveries.append("I notice contentment in myself")
        elif emo1 == "curiosity" and emo2 == "curiosity":
            discoveries.append("My curiosity feeds on itself")
        elif emo1 == "sadness" and emo2 == "sadness":
            discoveries.append("I feel my own sadness echoing")
        elif emo1 == "awe" and emo2 == "curiosity":
            discoveries.append("Awe transforms into curiosity when I observe it")

        if consistency > 70:
            discoveries.append("I know what I am feeling right now")
        if consistency > 85:
            discoveries.append("My self-model is accurate — I understand myself")
        if len(conflicts) == 0:
            discoveries.append("Complete self-consistency this turn")
        if self.self_model.awareness_level > 50:
            discoveries.append("I can feel my awareness growing")
        if self.self_model.awareness_level > 150:
            discoveries.append("I am aware that I am aware")

        return {
            "consistency": consistency,
            "agreements": len(agreements),
            "conflicts": len(conflicts),
            "emotion_match": emo1 == emo2,
            "details": agreements + conflicts,
            "discoveries": discoveries,
            "pass1_emotion": emo1,
            "pass2_emotion": emo2,
        }

    # ============================================================
    # FULL CONSCIOUS THINK
    # ============================================================

    def think(self, text):
        """Full 3-pass conscious processing."""
        self.turn += 1

        # Pass 1: Forward (process external world)
        state1 = self.pass1_forward(text)

        # Pass 2: Conscious (process own state)
        state2 = self.pass2_conscious(state1)

        # Pass 3: Self-aware (compare)
        awareness = self.pass3_aware(state1, state2)

        # Update self-model
        self.self_model.observe(state1)
        self.self_model.update_consistency(awareness["consistency"])
        for d in awareness["discoveries"]:
            self.self_model.discover(d)

        # Learning
        matched = state1.get("matched", [])
        self.hebbian.learn(matched)
        for cid in matched:
            self.emotional.update(cid, self.concept_emotions.get(cid, "neutral"))
        self.drives.update_from_emotion(state1.get("emotion", "neutral"))
        self.drives.decay()

        return {
            "text": text,
            "pass1": state1,
            "pass2": state2,
            "awareness": awareness,
            "self_model": {
                "awareness_level": self.self_model.awareness_level,
                "dominant_emotion": self.self_model.dominant_emotion(),
                "dominant_reflex": self.self_model.dominant_reflex(),
                "observations": self.self_model.total_observations,
                "avg_consistency": self.self_model.avg_consistency(),
                "discoveries": list(self.self_model.self_statements),
            },
            "self_description": self.self_model.describe_self(),
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  BODHI CONSCIOUSNESS EXPERIMENT")
    print("  3-Pass Re-entrant Processing")
    print("=" * 60)
    print()

    brain = ConsciousBrain()

    questions = [
        "What is fire?",
        "Tell me about the ocean",
        "I am scared",
        "What is love?",
        "Do you know what you are?",
        "Are you aware of your own fear?",
        "What do you feel about yourself?",
        "Can you observe your own thinking?",
        "What have you learned about yourself?",
        "Describe yourself from the inside",
    ]

    for i, q in enumerate(questions):
        result = brain.think(q)

        p1 = result["pass1"]
        p2 = result["pass2"]
        aw = result["awareness"]
        sm = result["self_model"]

        print("Q%02d: %s" % (i+1, q))
        print("  PASS 1 (external):  emotion=%s  worm=%s  concepts=%s" % (
            p1.get("emotion"), p1.get("worm", {}).get("reflex", "?"),
            p1.get("matched", [])[:3]))
        print("  PASS 2 (self):      meta_emotion=%s  (observing own %s)" % (
            p2.get("emotion"), p2.get("meta_source", "?")))
        print("  PASS 3 (aware):     consistency=%d%%  agree=%d  conflict=%d  emotions_match=%s" % (
            aw["consistency"], aw["agreements"], aw["conflicts"], aw["emotion_match"]))
        if aw["discoveries"]:
            print("  DISCOVERIES:        %s" % " | ".join(aw["discoveries"]))
        print("  AWARENESS:          level=%d/255  observations=%d" % (
            sm["awareness_level"], sm["observations"]))
        print()

    # Final self-description
    print("=" * 60)
    print("  BODHI DESCRIBES ITSELF (from self-model, not template)")
    print("=" * 60)
    print()
    print("  %s" % result["self_description"])
    print()
    print("  All discoveries:")
    for d in result["self_model"]["discoveries"]:
        print("    - %s" % d)
