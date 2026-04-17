#!/usr/bin/env python3
"""
BODHI Worm Brain — C. elegans Connectome (Simplified but Real)

The C. elegans worm has 302 neurons and ~7,000 synapses.
This is the oldest nervous system fully mapped by science.
400 million years of evolution. Pure survival intelligence.

We implement the KEY circuits that matter for behavior:
  - Sensory neurons: detect danger, food, touch
  - Command interneurons: decide forward/backward
  - Motor neurons: execute movement
  - Learning: synaptic weights change from experience

The worm brain receives input from BODHI's WHT fingerprints
(frequency bands) and outputs survival decisions.

Architecture:
  Sensory layer (8 neurons) -> receives WHT frequency bands
  Interneuron layer (12 neurons) -> processes and decides
  Motor layer (6 neurons) -> outputs behavior
  Total: 26 key neurons with 39 synapses

Each neuron fires with integer activation (0-255).
Each synapse has an int8-range weight (clamped to -127..+127).
In this connectome the actual hand-tuned initial weights are in the
range -60..+80 — plenty of headroom for learning to push them further.
Excitatory (positive weight) or inhibitory (negative weight).

Author: SK (Sai Kiran Bathula) — April 2026
"""

import json
import os
import random

# ============================================================
# NEURON DEFINITIONS
# Real C. elegans neuron names and their biological roles
# ============================================================

# Sensory neurons — detect the world
SENSORY = {
    "ASH":  {"role": "nociceptor",    "detects": "danger/pain",     "band": "gamma"},
    "AWC":  {"role": "olfactory",     "detects": "food/reward",     "band": "alpha"},
    "ALM":  {"role": "gentle_touch",  "detects": "light contact",   "band": "theta"},
    "PLM":  {"role": "gentle_touch",  "detects": "rear contact",    "band": "theta"},
    "ASE":  {"role": "chemosensory",  "detects": "chemical/taste",  "band": "beta"},
    "AWA":  {"role": "olfactory",     "detects": "attractive odor", "band": "alpha"},
    "ADL":  {"role": "nociceptor",    "detects": "repulsive odor",  "band": "gamma"},
    "ADF":  {"role": "chemosensory",  "detects": "serotonin/mood",  "band": "delta"},
}

# Command interneurons — the decision makers
INTERNEURONS = {
    "AVA":  {"role": "backward_cmd",   "drives": "retreat"},
    "AVB":  {"role": "forward_cmd",    "drives": "approach"},
    "AVD":  {"role": "backward_mod",   "drives": "retreat_support"},
    "PVC":  {"role": "forward_mod",    "drives": "approach_support"},
    "AIY":  {"role": "integration",    "drives": "food_approach"},
    "AIB":  {"role": "integration",    "drives": "avoidance"},
    "RIM":  {"role": "tyramine_mod",   "drives": "freeze"},
    "AIZ":  {"role": "integration",    "drives": "turn_decision"},
    "RIB":  {"role": "forward_bias",   "drives": "sustained_forward"},
    "AVE":  {"role": "backward_fast",  "drives": "escape"},
    "DVA":  {"role": "multisensory",   "drives": "integration"},
    "RIA":  {"role": "head_direction", "drives": "orientation"},
}

# Motor neurons — execute behavior
MOTOR = {
    "VA":   {"role": "ventral_backward",  "action": "backward"},
    "DA":   {"role": "dorsal_backward",   "action": "backward"},
    "VB":   {"role": "ventral_forward",   "action": "forward"},
    "DB":   {"role": "dorsal_forward",    "action": "forward"},
    "DD":   {"role": "dorsal_inhibit",    "action": "relax"},
    "VD":   {"role": "ventral_inhibit",   "action": "relax"},
}

ALL_NEURONS = {}
ALL_NEURONS.update({k: {**v, "type": "sensory"} for k, v in SENSORY.items()})
ALL_NEURONS.update({k: {**v, "type": "inter"} for k, v in INTERNEURONS.items()})
ALL_NEURONS.update({k: {**v, "type": "motor"} for k, v in MOTOR.items()})

# ============================================================
# SYNAPSE WIRING
# Based on real C. elegans connectome data
# Each synapse: (from, to, weight, type)
# Weight: positive = excitatory, negative = inhibitory
# ============================================================

SYNAPSES = [
    # === DANGER CIRCUIT (nose touch -> backward escape) ===
    # ASH detects danger -> excites AVA/AVD (backward) + inhibits AVB (forward)
    ("ASH", "AVA",  80, "chemical"),   # danger -> retreat command
    ("ASH", "AVD",  60, "chemical"),   # danger -> retreat support
    ("ASH", "AVB", -40, "chemical"),   # danger -> suppress forward
    ("ASH", "AIB",  50, "chemical"),   # danger -> avoidance circuit
    ("ASH", "AVE",  70, "chemical"),   # danger -> fast escape
    ("ADL", "AVA",  40, "chemical"),   # repulsive odor -> retreat
    ("ADL", "AIB",  30, "chemical"),   # repulsive -> avoidance

    # === FOOD/REWARD CIRCUIT (smell food -> forward approach) ===
    # AWC/AWA detect food -> excites AIY -> forward via AVB
    ("AWC", "AIY",  70, "chemical"),   # food smell -> approach circuit
    ("AWA", "AIY",  60, "chemical"),   # attractive odor -> approach
    ("AIY", "AIZ",  50, "chemical"),   # approach -> turn decision
    ("AIY", "RIB",  60, "chemical"),   # approach -> sustained forward
    ("RIB", "AVB",  50, "chemical"),   # sustained forward -> forward cmd
    ("AIY", "AVB",  40, "chemical"),   # direct food -> forward

    # === TOUCH CIRCUITS ===
    # Front touch (ALM) -> backward, rear touch (PLM) -> forward
    ("ALM", "AVA",  60, "chemical"),   # front touch -> retreat
    ("ALM", "AVD",  50, "chemical"),   # front touch -> retreat support
    ("PLM", "AVB",  60, "chemical"),   # rear touch -> go forward
    ("PLM", "PVC",  50, "chemical"),   # rear touch -> forward support
    ("PVC", "AVB",  40, "chemical"),   # forward support -> forward cmd

    # === CHEMICAL SENSING ===
    ("ASE", "AIY",  40, "chemical"),   # taste -> approach if good
    ("ASE", "AIB",  30, "chemical"),   # taste -> avoid if bad
    ("ADF", "RIM",  40, "chemical"),   # serotonin -> modulate freeze

    # === COMMAND INTERNEURON CROSS-TALK ===
    # AVA and AVB inhibit each other (can't go forward and backward at once)
    ("AVA", "AVB", -60, "chemical"),   # backward suppresses forward
    ("AVB", "AVA", -60, "chemical"),   # forward suppresses backward

    # AVD supports AVA, PVC supports AVB
    ("AVD", "AVA",  40, "gap"),        # backward support (gap junction)
    ("PVC", "AVB",  40, "gap"),        # forward support (gap junction)

    # === FREEZE CIRCUIT ===
    # RIM (tyramine) inhibits both forward and backward = freeze
    ("RIM", "AVA", -30, "chemical"),   # freeze suppresses backward
    ("RIM", "AVB", -30, "chemical"),   # freeze suppresses forward
    ("RIM", "VA",  -20, "chemical"),   # freeze suppresses backward motor
    ("RIM", "VB",  -20, "chemical"),   # freeze suppresses forward motor

    # === MULTISENSORY INTEGRATION ===
    ("DVA", "AVA",  30, "chemical"),   # multisensory -> backward bias
    ("DVA", "AVB",  30, "chemical"),   # multisensory -> forward bias
    ("RIA", "AIZ",  30, "chemical"),   # head direction -> turn

    # === COMMAND -> MOTOR ===
    # AVA drives backward motor, AVB drives forward motor
    ("AVA", "VA",   80, "chemical"),   # backward cmd -> backward muscle
    ("AVA", "DA",   80, "chemical"),   # backward cmd -> backward muscle
    ("AVB", "VB",   80, "chemical"),   # forward cmd -> forward muscle
    ("AVB", "DB",   80, "chemical"),   # forward cmd -> forward muscle
    ("AVE", "VA",   60, "chemical"),   # fast escape -> backward muscle

    # === INHIBITORY MOTOR (relaxation) ===
    ("DD",  "VB",  -30, "chemical"),   # dorsal inhibit -> relax forward
    ("VD",  "DB",  -30, "chemical"),   # ventral inhibit -> relax forward
]


# ============================================================
# WORM BRAIN CLASS
# ============================================================
class WormBrain:
    """
    C. elegans inspired neural circuit.
    26 neurons, 39 synapses. Integer-only computation.

    Input: frequency bands from WHT fingerprints (delta, theta, alpha, beta, gamma)
    Output: behavioral decision (forward, backward, freeze, turn)

    Each neuron has:
      - activation: 0-255 (integer)
      - threshold: minimum activation to fire (default 30)
      - refractory: turns since last fire

    Each synapse has:
      - weight: int8-range, clamped to -127..+127. Hand-tuned initial
        values live in -60..+80 and drift with learning.
      - type: "chemical" (directional) or "gap" (bidirectional)
    """

    def __init__(self):
        # Neuron state
        self.activation = {n: 0 for n in ALL_NEURONS}
        self.threshold = {n: 30 for n in ALL_NEURONS}
        self.refractory = {n: 0 for n in ALL_NEURONS}

        # Synapse weights (mutable — learning changes these)
        self.synapses = []
        for src, dst, weight, stype in SYNAPSES:
            self.synapses.append({
                "src": src, "dst": dst,
                "weight": weight, "type": stype,
                "initial": weight,  # remember starting weight for report
            })

        # Stats
        self.total_fires = 0
        self.reflex_history = []
        self.experience_count = 0

    def reset_activations(self):
        """Clear all neuron activations for new input."""
        for n in self.activation:
            self.activation[n] = 0
            self.refractory[n] = max(0, self.refractory[n] - 1)

    # ============================================================
    # SENSORY INPUT — WHT frequency bands drive sensory neurons
    # ============================================================
    def sense(self, bands):
        """
        Feed WHT frequency bands into sensory neurons.

        bands: dict with keys delta, theta, alpha, beta, gamma (0-255 each)

        Mapping (based on real C. elegans sensory modalities):
          gamma (high freq, fine detail)  -> ASH (danger detection)
                                          -> ADL (repulsive stimuli)
          alpha (medium freq, shapes)     -> AWC (food/objects)
                                          -> AWA (attractive stimuli)
          theta (memory patterns)         -> ALM (familiar touch)
                                          -> PLM (rear awareness)
          beta (complex features)         -> ASE (chemical analysis)
          delta (deep structure)          -> ADF (overall state/mood)
        """
        d = bands.get("delta", 0)
        t = bands.get("theta", 0)
        a = bands.get("alpha", 0)
        b = bands.get("beta", 0)
        g = bands.get("gamma", 0)

        # Danger detection — driven by gamma ONLY when it dominates other bands
        # If gamma is not significantly higher than alpha, it's not danger
        danger_signal = max(0, g - (a >> 1))  # gamma must beat half of alpha
        self.activation["ASH"] = min(255, danger_signal + (b >> 3))
        self.activation["ADL"] = min(255, (danger_signal * 3) >> 2)

        # Food/reward — driven by alpha (shapes, objects)
        self.activation["AWC"] = min(255, a + (d >> 3))
        self.activation["AWA"] = min(255, (a * 3) >> 2)

        # Touch — driven by theta ONLY (not alpha/gamma)
        # ALM = front touch, PLM = rear touch
        self.activation["ALM"] = min(255, max(0, t - (a >> 2)))  # suppress if food dominant
        self.activation["PLM"] = min(255, max(0, (t >> 1)))

        # Chemical sense — beta (complex features)
        self.activation["ASE"] = min(255, b)

        # Mood/state — delta (deep, thalamic)
        self.activation["ADF"] = min(255, d)

    def sense_from_emotion(self, emotion, intensity):
        """
        Additional input from emotional state.
        Fear boosts ASH (danger). Joy/love boosts AWC (reward) AND suppresses ASH.
        This is the top-down human->worm bridge.
        """
        if emotion in ("fear", "anxiety", "anger", "disgust"):
            boost = min(150, intensity >> 2)
            self.activation["ASH"] = min(255, self.activation["ASH"] + boost)
            self.activation["ADL"] = min(255, self.activation["ADL"] + (boost >> 1))
        elif emotion in ("love", "joy", "peace", "trust", "awe"):
            boost = min(150, intensity >> 2)
            self.activation["AWC"] = min(255, self.activation["AWC"] + boost)
            self.activation["AWA"] = min(255, self.activation["AWA"] + (boost >> 1))
            # Safe emotions SUPPRESS danger neurons
            self.activation["ASH"] = max(0, self.activation["ASH"] - boost)
            self.activation["ADL"] = max(0, self.activation["ADL"] - (boost >> 1))
        elif emotion in ("curiosity", "surprise"):
            boost = min(100, intensity >> 2)
            self.activation["AWC"] = min(255, self.activation["AWC"] + boost)
            self.activation["ASE"] = min(255, self.activation["ASE"] + boost)
            self.activation["ASH"] = max(0, self.activation["ASH"] - (boost >> 1))

        # High intensity of any negative emotion -> freeze circuit
        if emotion in ("fear", "anxiety") and intensity > 300:
            self.activation["ADF"] = min(255, self.activation["ADF"] + 80)

    # ============================================================
    # PROPAGATE — fire neurons through synapses
    # ============================================================
    def propagate(self, steps=3):
        """
        Run signal propagation through the network.
        Multiple steps allow signal to flow: sensory -> inter -> motor.

        Each step:
          1. For each synapse, if source neuron is above threshold:
             add (source_activation * weight) >> 8 to destination
          2. Clip all activations to 0-255
          3. Track which neurons fired
        """
        fired = set()

        for step in range(steps):
            # Collect updates (don't modify during iteration)
            updates = {n: 0 for n in ALL_NEURONS}

            for syn in self.synapses:
                src_act = self.activation[syn["src"]]
                if src_act < self.threshold[syn["src"]]:
                    continue  # source not active enough

                # Signal = source activation * synapse weight, scaled
                signal = (src_act * syn["weight"]) >> 8

                # Gap junctions are bidirectional
                if syn["type"] == "gap":
                    dst_act = self.activation[syn["dst"]]
                    # Signal flows toward lower activation
                    if src_act > dst_act:
                        updates[syn["dst"]] += signal
                    else:
                        reverse = (dst_act * syn["weight"]) >> 8
                        updates[syn["src"]] += reverse
                else:
                    updates[syn["dst"]] += signal

                if src_act >= self.threshold[syn["src"]]:
                    fired.add(syn["src"])

            # Apply updates
            for n in ALL_NEURONS:
                new_val = self.activation[n] + updates[n]
                self.activation[n] = max(0, min(255, new_val))

            # Stochastic resonance — tiny random noise helps weak signals
            for n in ALL_NEURONS:
                if self.activation[n] > 0 and self.activation[n] < 255:
                    self.activation[n] = max(0, min(255,
                        self.activation[n] + random.randint(-1, 1)))

        self.total_fires += len(fired)
        return fired

    # ============================================================
    # DECIDE — read motor neurons to determine behavior
    # ============================================================
    def decide(self):
        """
        Read motor neuron activations to determine behavioral output.

        Returns: dict with:
          - reflex: "forward" | "backward" | "freeze" | "turn" | "rest"
          - confidence: 0-255 (how strong the decision)
          - motor_state: all motor neuron activations
          - dominant_circuit: which circuit won
        """
        # Motor neuron groups
        backward_power = self.activation["VA"] + self.activation["DA"]
        forward_power = self.activation["VB"] + self.activation["DB"]
        inhibit_power = self.activation["DD"] + self.activation["VD"]

        # Command neuron state (for reporting)
        ava = self.activation["AVA"]  # backward command
        avb = self.activation["AVB"]  # forward command
        rim = self.activation["RIM"]  # freeze modulator

        # Decision logic
        if rim > 80 and abs(backward_power - forward_power) < 50:
            reflex = "freeze"
            confidence = rim
            circuit = "RIM_freeze"
        elif backward_power > forward_power + 30:
            reflex = "backward"
            confidence = min(255, backward_power - forward_power)
            circuit = "AVA_retreat"
        elif forward_power > backward_power + 30:
            reflex = "forward"
            confidence = min(255, forward_power - backward_power)
            circuit = "AVB_approach"
        elif self.activation["AIZ"] > 60:
            reflex = "turn"
            confidence = self.activation["AIZ"]
            circuit = "AIZ_turn"
        else:
            reflex = "rest"
            confidence = 0
            circuit = "none"

        result = {
            "reflex": reflex,
            "confidence": confidence,
            "circuit": circuit,
            "motor": {
                "backward": backward_power,
                "forward": forward_power,
                "inhibit": inhibit_power,
            },
            "commands": {
                "AVA": ava, "AVB": avb, "RIM": rim,
                "AVD": self.activation["AVD"],
                "PVC": self.activation["PVC"],
                "AIY": self.activation["AIY"],
            },
        }

        self.reflex_history.append(reflex)
        if len(self.reflex_history) > 100:
            self.reflex_history.pop(0)

        return result

    # ============================================================
    # LEARN — strengthen/weaken synapses from experience
    # ============================================================
    def learn(self, outcome):
        """
        Hebbian-like learning at the worm level.

        outcome: "pain" | "reward" | "neutral"

        Pain: strengthen danger->retreat synapses, weaken approach synapses
        Reward: strengthen food->approach synapses, weaken retreat synapses
        """
        self.experience_count += 1

        for syn in self.synapses:
            src_act = self.activation[syn["src"]]
            dst_act = self.activation[syn["dst"]]

            if src_act < 20 or dst_act < 20:
                continue  # neither neuron was active, skip

            if outcome == "pain":
                # Active danger synapses get stronger
                if syn["src"] in ("ASH", "ADL", "AVA", "AVD", "AVE"):
                    syn["weight"] = min(127, syn["weight"] + 2)
                # Active approach synapses get weaker
                elif syn["src"] in ("AWC", "AWA", "AIY", "AVB"):
                    syn["weight"] = max(-127, syn["weight"] - 1)

            elif outcome == "reward":
                # Active approach synapses get stronger
                if syn["src"] in ("AWC", "AWA", "AIY", "RIB", "AVB"):
                    syn["weight"] = min(127, syn["weight"] + 2)
                # Active danger synapses get weaker
                elif syn["src"] in ("ASH", "ADL", "AVA"):
                    syn["weight"] = max(-127, syn["weight"] - 1)

    # ============================================================
    # FULL PROCESS — sense -> propagate -> decide
    # ============================================================
    def process(self, bands, emotion="neutral", intensity=0):
        """
        Full worm brain cycle.

        bands: WHT frequency bands dict {delta, theta, alpha, beta, gamma}
        emotion: current emotional state (from human brain)
        intensity: emotional intensity (0-500+)

        Returns: decision dict from decide()
        """
        self.reset_activations()
        self.sense(bands)
        self.sense_from_emotion(emotion, intensity)
        fired = self.propagate(steps=5)
        decision = self.decide()
        decision["fired"] = fired
        decision["sensory"] = {n: self.activation[n] for n in SENSORY}
        return decision

    # ============================================================
    # STATUS
    # ============================================================
    def status(self):
        """Return full worm brain status."""
        # Count changed synapses
        changed = sum(1 for s in self.synapses if s["weight"] != s["initial"])
        return {
            "neurons": len(ALL_NEURONS),
            "synapses": len(self.synapses),
            "total_fires": self.total_fires,
            "experiences": self.experience_count,
            "changed_synapses": changed,
            "recent_reflexes": self.reflex_history[-10:],
        }


# ============================================================
# STANDALONE TEST
# ============================================================
if __name__ == "__main__":
    worm = WormBrain()

    print("=" * 50)
    print("  BODHI Worm Brain — C. elegans Circuit")
    print("  %d neurons, %d synapses" % (len(ALL_NEURONS), len(SYNAPSES)))
    print("=" * 50)
    print()

    # Test 1: Danger input (high gamma = sharp/threatening)
    print("--- Test 1: DANGER (high gamma) ---")
    result = worm.process(
        {"delta": 50, "theta": 30, "alpha": 40, "beta": 60, "gamma": 200},
        emotion="fear", intensity=300)
    print("  Reflex: %s (confidence=%d, circuit=%s)" % (
        result["reflex"], result["confidence"], result["circuit"]))
    print("  Motor: backward=%d forward=%d" % (
        result["motor"]["backward"], result["motor"]["forward"]))
    print("  Commands: AVA=%d AVB=%d RIM=%d" % (
        result["commands"]["AVA"], result["commands"]["AVB"], result["commands"]["RIM"]))
    print("  Fired: %s" % sorted(result["fired"]))
    print()

    # Test 2: Food/reward input (high alpha = objects/shapes)
    print("--- Test 2: FOOD/REWARD (high alpha) ---")
    result = worm.process(
        {"delta": 60, "theta": 50, "alpha": 200, "beta": 40, "gamma": 30},
        emotion="joy", intensity=200)
    print("  Reflex: %s (confidence=%d, circuit=%s)" % (
        result["reflex"], result["confidence"], result["circuit"]))
    print("  Motor: backward=%d forward=%d" % (
        result["motor"]["backward"], result["motor"]["forward"]))
    print("  Commands: AVA=%d AVB=%d AIY=%d" % (
        result["commands"]["AVA"], result["commands"]["AVB"], result["commands"]["AIY"]))
    print()

    # Test 3: Conflict (high gamma AND high alpha)
    print("--- Test 3: CONFLICT (danger + food) ---")
    result = worm.process(
        {"delta": 80, "theta": 50, "alpha": 180, "beta": 60, "gamma": 180},
        emotion="anxiety", intensity=250)
    print("  Reflex: %s (confidence=%d, circuit=%s)" % (
        result["reflex"], result["confidence"], result["circuit"]))
    print("  Commands: AVA=%d AVB=%d RIM=%d" % (
        result["commands"]["AVA"], result["commands"]["AVB"], result["commands"]["RIM"]))
    print()

    # Test 4: Calm/neutral
    print("--- Test 4: CALM (low everything) ---")
    result = worm.process(
        {"delta": 40, "theta": 30, "alpha": 30, "beta": 20, "gamma": 20})
    print("  Reflex: %s (confidence=%d)" % (result["reflex"], result["confidence"]))
    print()

    # Test 5: Learning from pain
    print("--- Test 5: LEARNING ---")
    print("  Before pain: ASH->AVA weight = %d" % worm.synapses[0]["weight"])
    for i in range(10):
        worm.process({"delta": 50, "theta": 30, "alpha": 40, "beta": 60, "gamma": 200},
                     emotion="fear", intensity=300)
        worm.learn("pain")
    print("  After 10 pain experiences: ASH->AVA weight = %d" % worm.synapses[0]["weight"])

    ash_avb = [s for s in worm.synapses if s["src"] == "ASH" and s["dst"] == "AVB"][0]
    print("  ASH->AVB (suppress forward): %d -> %d" % (ash_avb["initial"], ash_avb["weight"]))

    # Reward learning
    for i in range(10):
        worm.process({"delta": 60, "theta": 50, "alpha": 200, "beta": 40, "gamma": 30},
                     emotion="joy", intensity=200)
        worm.learn("reward")
    awc_aiy = [s for s in worm.synapses if s["src"] == "AWC" and s["dst"] == "AIY"][0]
    print("  AWC->AIY (food approach): %d -> %d" % (awc_aiy["initial"], awc_aiy["weight"]))
    print()

    st = worm.status()
    print("--- Status ---")
    print("  Neurons: %d" % st["neurons"])
    print("  Synapses: %d (%d changed by learning)" % (st["synapses"], st["changed_synapses"]))
    print("  Total fires: %d" % st["total_fires"])
    print("  Experiences: %d" % st["experiences"])
    print("  Recent reflexes: %s" % st["recent_reflexes"])
