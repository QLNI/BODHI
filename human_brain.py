#!/usr/bin/env python3
"""
BODHI Human Brain — 76 Regions from Desikan-Killiany Atlas

Built on top of the worm brain (Report 02).
Worm = fast survival reflexes (400M years old).
Human = slow deep cognition (emotion, memory, planning, consciousness).

Each region:
  - Has a role (what it does)
  - Receives input from WHT frequency bands
  - Receives input from emotions
  - Connects to other regions via pathways
  - Fires with integer activation (0-255)

The worm-human bridge:
  - Fear (amygdala) -> worm retreat (AVA)
  - Curiosity (PFC) -> worm approach (AVB)
  - Worm retreat -> amygdala boost
  - Worm approach -> PFC boost

Author: SK (Sai Kiran Bathula) — April 2026
"""

import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from worm_brain import WormBrain

# ============================================================
# 76 BRAIN REGIONS — Desikan-Killiany Atlas
# Grouped by function, left (l) and right (r) hemispheres
# ============================================================

REGIONS = {
    # --- VISUAL CORTEX ---
    "rv1":    {"name": "Right V1 (Primary Visual)", "group": "visual", "band": "alpha"},
    "lv1":    {"name": "Left V1",                   "group": "visual", "band": "alpha"},
    "rv2":    {"name": "Right V2 (Secondary Visual)","group": "visual", "band": "beta"},
    "lv2":    {"name": "Left V2",                   "group": "visual", "band": "beta"},
    "rcun":   {"name": "Right Cuneus",              "group": "visual", "band": "alpha"},
    "lcun":   {"name": "Left Cuneus",               "group": "visual", "band": "alpha"},
    "rling":  {"name": "Right Lingual Gyrus",       "group": "visual", "band": "alpha"},
    "lling":  {"name": "Left Lingual Gyrus",        "group": "visual", "band": "alpha"},
    "rlat_occ":{"name":"Right Lateral Occipital",   "group": "visual", "band": "beta"},
    "llat_occ":{"name":"Left Lateral Occipital",    "group": "visual", "band": "beta"},

    # --- AUDITORY CORTEX ---
    "ra1":    {"name": "Right A1 (Primary Auditory)","group": "auditory","band": "theta"},
    "la1":    {"name": "Left A1",                   "group": "auditory","band": "theta"},
    "rsup_temp":{"name":"Right Superior Temporal",  "group": "auditory","band": "theta"},
    "lsup_temp":{"name":"Left Superior Temporal",   "group": "auditory","band": "theta"},
    "rmid_temp":{"name":"Right Middle Temporal",    "group": "temporal","band": "theta"},
    "lmid_temp":{"name":"Left Middle Temporal",     "group": "temporal","band": "theta"},
    "rinf_temp":{"name":"Right Inferior Temporal",  "group": "temporal","band": "beta"},
    "linf_temp":{"name":"Left Inferior Temporal",   "group": "temporal","band": "beta"},

    # --- LIMBIC (EMOTION) ---
    "ramyg":  {"name": "Right Amygdala",            "group": "limbic",  "band": "gamma"},
    "lamyg":  {"name": "Left Amygdala",             "group": "limbic",  "band": "gamma"},
    "rcca":   {"name": "Right Anterior Cingulate",  "group": "limbic",  "band": "gamma"},
    "lcca":   {"name": "Left Anterior Cingulate",   "group": "limbic",  "band": "gamma"},
    "rpcc":   {"name": "Right Posterior Cingulate",  "group": "limbic",  "band": "delta"},
    "lpcc":   {"name": "Left Posterior Cingulate",   "group": "limbic",  "band": "delta"},
    "rins":   {"name": "Right Insula",              "group": "limbic",  "band": "gamma"},
    "lins":   {"name": "Left Insula",               "group": "limbic",  "band": "gamma"},
    "racc":   {"name": "Right Accumbens",           "group": "limbic",  "band": "beta"},
    "lacc":   {"name": "Left Accumbens",            "group": "limbic",  "band": "beta"},

    # --- MEMORY (HIPPOCAMPAL) ---
    "rhc":    {"name": "Right Hippocampus",         "group": "memory",  "band": "theta"},
    "lhc":    {"name": "Left Hippocampus",          "group": "memory",  "band": "theta"},
    "rphc":   {"name": "Right Parahippocampal",     "group": "memory",  "band": "theta"},
    "lphc":   {"name": "Left Parahippocampal",      "group": "memory",  "band": "theta"},
    "rent":   {"name": "Right Entorhinal",          "group": "memory",  "band": "theta"},
    "lent":   {"name": "Left Entorhinal",           "group": "memory",  "band": "theta"},

    # --- PREFRONTAL (PLANNING/DECISION) ---
    "rpfcdl": {"name": "Right Dorsolateral PFC",    "group": "prefrontal","band": "beta"},
    "lpfcdl": {"name": "Left Dorsolateral PFC",     "group": "prefrontal","band": "beta"},
    "rpfcm":  {"name": "Right Medial PFC",          "group": "prefrontal","band": "beta"},
    "lpfcm":  {"name": "Left Medial PFC",           "group": "prefrontal","band": "beta"},
    "rpfco":  {"name": "Right Orbitofrontal",       "group": "prefrontal","band": "beta"},
    "lpfco":  {"name": "Left Orbitofrontal",        "group": "prefrontal","band": "beta"},
    "rsup_fr":{"name": "Right Superior Frontal",    "group": "prefrontal","band": "beta"},
    "lsup_fr":{"name": "Left Superior Frontal",     "group": "prefrontal","band": "beta"},
    "rmid_fr":{"name": "Right Middle Frontal",      "group": "frontal",  "band": "beta"},
    "lmid_fr":{"name": "Left Middle Frontal",       "group": "frontal",  "band": "beta"},
    "rinf_fr":{"name": "Right Inferior Frontal",    "group": "frontal",  "band": "beta"},
    "linf_fr":{"name": "Left Inferior Frontal",     "group": "frontal",  "band": "beta"},

    # --- MOTOR ---
    "rm1":    {"name": "Right Primary Motor",       "group": "motor",   "band": "beta"},
    "lm1":    {"name": "Left Primary Motor",        "group": "motor",   "band": "beta"},
    "rsma":   {"name": "Right Supplementary Motor", "group": "motor",   "band": "beta"},
    "lsma":   {"name": "Left Supplementary Motor",  "group": "motor",   "band": "beta"},

    # --- SOMATOSENSORY ---
    "rs1":    {"name": "Right Primary Somatosensory","group":"sensory",  "band": "alpha"},
    "ls1":    {"name": "Left Primary Somatosensory", "group":"sensory",  "band": "alpha"},
    "rsup_par":{"name":"Right Superior Parietal",   "group": "parietal","band": "alpha"},
    "lsup_par":{"name":"Left Superior Parietal",    "group": "parietal","band": "alpha"},
    "rinf_par":{"name":"Right Inferior Parietal",   "group": "parietal","band": "beta"},
    "linf_par":{"name":"Left Inferior Parietal",    "group": "parietal","band": "beta"},
    "rprec":  {"name": "Right Precuneus",           "group": "parietal","band": "alpha"},
    "lprec":  {"name": "Left Precuneus",            "group": "parietal","band": "alpha"},
    "rsup_mg":{"name": "Right Supramarginal",       "group": "parietal","band": "beta"},
    "lsup_mg":{"name": "Left Supramarginal",        "group": "parietal","band": "beta"},

    # --- THALAMUS & BASAL GANGLIA ---
    "rtha":   {"name": "Right Thalamus",            "group": "subcortical","band": "delta"},
    "ltha":   {"name": "Left Thalamus",             "group": "subcortical","band": "delta"},
    "rcau":   {"name": "Right Caudate",             "group": "subcortical","band": "delta"},
    "lcau":   {"name": "Left Caudate",              "group": "subcortical","band": "delta"},
    "rput":   {"name": "Right Putamen",             "group": "subcortical","band": "delta"},
    "lput":   {"name": "Left Putamen",              "group": "subcortical","band": "delta"},
    "rpal":   {"name": "Right Pallidum",            "group": "subcortical","band": "delta"},
    "lpal":   {"name": "Left Pallidum",             "group": "subcortical","band": "delta"},

    # --- LANGUAGE ---
    "rbroca": {"name": "Right Broca's Area",        "group": "language", "band": "beta"},
    "lbroca": {"name": "Left Broca's Area",         "group": "language", "band": "beta"},
    "rwern":  {"name": "Right Wernicke's Area",     "group": "language", "band": "theta"},
    "lwern":  {"name": "Left Wernicke's Area",      "group": "language", "band": "theta"},
}


# ============================================================
# REGION-TO-REGION PATHWAYS (major white matter tracts)
# Each: (from, to, strength)
# ============================================================

PATHWAYS = [
    # Visual stream: V1 -> V2 -> temporal (ventral "what" stream)
    ("rv1", "rv2", 80), ("lv1", "lv2", 80),
    ("rv2", "rinf_temp", 60), ("lv2", "linf_temp", 60),
    ("rv2", "rlat_occ", 70), ("lv2", "llat_occ", 70),

    # Visual -> parietal (dorsal "where" stream)
    ("rv1", "rsup_par", 50), ("lv1", "lsup_par", 50),
    ("rv2", "rinf_par", 50), ("lv2", "linf_par", 50),

    # Auditory stream
    ("ra1", "rsup_temp", 80), ("la1", "lsup_temp", 80),
    ("rsup_temp", "rmid_temp", 60), ("lsup_temp", "lmid_temp", 60),
    ("lsup_temp", "lwern", 70),  # auditory -> Wernicke's

    # Limbic circuits
    ("ramyg", "rhc", 70), ("lamyg", "lhc", 70),     # amygdala -> hippocampus
    ("ramyg", "rcca", 60), ("lamyg", "lcca", 60),    # amygdala -> cingulate
    ("ramyg", "rins", 50), ("lamyg", "lins", 50),    # amygdala -> insula
    ("ramyg", "rpfcm", 40), ("lamyg", "lpfcm", 40),  # amygdala -> medial PFC
    ("rcca", "rpfcdl", 50), ("lcca", "lpfcdl", 50),  # cingulate -> dorsolateral PFC
    ("rins", "rcca", 40), ("lins", "lcca", 40),      # insula -> cingulate

    # Memory circuits
    ("rhc", "rent", 70), ("lhc", "lent", 70),        # hippocampus -> entorhinal
    ("rent", "rphc", 60), ("lent", "lphc", 60),      # entorhinal -> parahippocampal
    ("rhc", "rpfcm", 50), ("lhc", "lpfcm", 50),      # hippocampus -> medial PFC
    ("rhc", "rpcc", 40), ("lhc", "lpcc", 40),        # hippocampus -> posterior cingulate

    # Prefrontal connections
    ("rpfcdl", "rpfcm", 50), ("lpfcdl", "lpfcm", 50),
    ("rpfcdl", "rmid_fr", 60), ("lpfcdl", "lmid_fr", 60),
    ("rpfcm", "rpfco", 40), ("lpfcm", "lpfco", 40),
    ("rpfcdl", "rcau", 40), ("lpfcdl", "lcau", 40),  # PFC -> caudate (planning)

    # Motor planning
    ("rpfcdl", "rsma", 50), ("lpfcdl", "lsma", 50),  # PFC -> supplementary motor
    ("rsma", "rm1", 70), ("lsma", "lm1", 70),        # SMA -> primary motor

    # Thalamus hub (relays everything)
    ("rtha", "rv1", 60), ("ltha", "lv1", 60),        # thalamus -> visual
    ("rtha", "ra1", 50), ("ltha", "la1", 50),        # thalamus -> auditory
    ("rtha", "rpfcdl", 40), ("ltha", "lpfcdl", 40),  # thalamus -> PFC
    ("rtha", "rs1", 50), ("ltha", "ls1", 50),        # thalamus -> somatosensory
    ("rtha", "ramyg", 30), ("ltha", "lamyg", 30),    # thalamus -> amygdala

    # Basal ganglia loop
    ("rcau", "rput", 50), ("lcau", "lput", 50),
    ("rput", "rpal", 50), ("lput", "lpal", 50),
    ("rpal", "rtha", 40), ("lpal", "ltha", 40),      # pallidum -> thalamus (feedback)

    # Language
    ("lwern", "lbroca", 70),  # Wernicke -> Broca (arcuate fasciculus)
    ("lbroca", "lm1", 50),    # Broca -> motor (speech production)
    ("lbroca", "linf_fr", 60),# Broca -> inferior frontal

    # Reward
    ("racc", "rpfcm", 50), ("lacc", "lpfcm", 50),  # accumbens -> medial PFC
    ("racc", "ramyg", 30), ("lacc", "lamyg", 30),   # accumbens -> amygdala

    # Cross-hemisphere (corpus callosum — simplified)
    ("rv1", "lv1", 30), ("ramyg", "lamyg", 30),
    ("rhc", "lhc", 30), ("rpfcdl", "lpfcdl", 30),
    ("rtha", "ltha", 20), ("rcca", "lcca", 30),
]

# ============================================================
# EMOTION -> REGION FIRING MAP
# Which regions fire for each emotion
# ============================================================

EMOTION_FIRING = {
    "fear":      {"ramyg": 220, "lamyg": 220, "rpfcdl": 150, "lpfcdl": 150, "rm1": 100, "rins": 120, "lins": 120},
    "anxiety":   {"ramyg": 160, "lamyg": 160, "rcca": 200, "lcca": 200, "rins": 150, "lins": 150},
    "anger":     {"ramyg": 200, "lamyg": 200, "rm1": 180, "lm1": 180, "rpfcdl": 130},
    "disgust":   {"ramyg": 120, "lamyg": 120, "rins": 180, "lins": 180, "rcca": 100},
    "sadness":   {"ramyg": 150, "lamyg": 150, "rhc": 160, "lhc": 160, "rpcc": 120, "lpcc": 120},
    "shame":     {"rpfcm": 200, "lpfcm": 200, "rcca": 180, "lcca": 180, "rins": 100},
    "love":      {"rpfcm": 200, "lpfcm": 200, "rhc": 150, "lhc": 150, "racc": 180, "lacc": 180},
    "joy":       {"rpfcm": 150, "lpfcm": 150, "racc": 200, "lacc": 200, "rm1": 120},
    "trust":     {"rpfcm": 160, "lpfcm": 160, "rpfcdl": 140, "rpfco": 120},
    "surprise":  {"ramyg": 180, "lamyg": 180, "rpfcdl": 200, "rtha": 150, "ltha": 150},
    "curiosity": {"rpfcdl": 200, "lpfcdl": 200, "rhc": 150, "lhc": 150, "rcau": 120, "lcau": 120},
    "awe":       {"rpfcdl": 220, "lpfcdl": 220, "rv1": 200, "lv1": 200, "rprec": 150},
    "peace":     {"rpfcm": 150, "lpfcm": 150, "rpcc": 130, "lpcc": 130},
    "pride":     {"rpfcm": 180, "lpfcm": 180, "rm1": 100, "racc": 130},
    "nostalgia": {"rhc": 220, "lhc": 220, "ramyg": 80, "rpcc": 150, "lpcc": 150},
    "contempt":  {"rpfcdl": 180, "lpfcdl": 180, "ramyg": 100, "rins": 120},
}


# ============================================================
# HUMAN BRAIN CLASS
# ============================================================
class HumanBrain:
    """
    76 brain regions. ~100 pathways. Integer-only.

    Input: WHT frequency bands + emotion label
    Processing: activate regions -> propagate through pathways
    Output: full activation map of all 76 regions

    Wired bidirectionally to WormBrain.
    """

    def __init__(self):
        self.activation = {r: 0 for r in REGIONS}
        self.worm = WormBrain()

    def reset(self):
        for r in self.activation:
            self.activation[r] = 0

    # ============================================================
    # EXTRACT FREQUENCY BANDS FROM WHT FINGERPRINT
    # ============================================================
    def extract_bands(self, fingerprint):
        """Extract 5 frequency bands from a WHT fingerprint (int16/int32 array).
        Returns dict {delta, theta, alpha, beta, gamma} each 0-200."""
        fp = fingerprint
        n = min(len(fp), 200 * 8 * 3)
        if n == 0:
            return {"delta": 0, "theta": 0, "alpha": 0, "beta": 0, "gamma": 0}

        band_energy = [0, 0, 0, 0, 0]
        count = [0, 0, 0, 0, 0]
        for i in range(0, n, 8):
            for j in range(min(8, n - i)):
                val = abs(int(fp[i + j]))
                if j == 0:   band_energy[0] += val; count[0] += 1
                elif j < 3:  band_energy[1] += val; count[1] += 1
                elif j < 5:  band_energy[2] += val; count[2] += 1
                elif j < 7:  band_energy[3] += val; count[3] += 1
                else:        band_energy[4] += val; count[4] += 1

        band_avg = [band_energy[k] // max(1, count[k]) for k in range(5)]
        total_avg = sum(band_avg) // 5
        if total_avg == 0:
            total_avg = 1

        # Overall intensity
        overall = min(200, total_avg >> 3)

        # Proportional activation (relative to mean)
        proportions = [(avg * 1000) // total_avg for avg in band_avg]
        names = ["delta", "theta", "alpha", "beta", "gamma"]
        bands = {}
        for k in range(5):
            deviation = (proportions[k] - 1000) * 10
            bands[names[k]] = max(10, min(200, 100 + (deviation >> 3)))

        bands["_overall"] = overall
        return bands

    # ============================================================
    # ACTIVATE REGIONS FROM FREQUENCY BANDS
    # ============================================================
    def activate_from_bands(self, bands):
        """Activate brain regions based on frequency band values.
        Band values are 10-200. Scale to 0-120 for base activation
        so emotion and propagation have room to add on top."""
        overall = bands.get("_overall", 50)
        base = max(5, overall >> 2)  # 0-50 base

        for rid, info in REGIONS.items():
            preferred = info["band"]
            val = bands.get(preferred, 0)
            # Scale band value: 10-200 -> 5-100
            scaled = (val * 100) >> 8  # divide by ~2.5
            self.activation[rid] = min(150, base + scaled)

    # ============================================================
    # ACTIVATE FROM EMOTION
    # ============================================================
    def activate_from_emotion(self, emotion):
        """Boost regions based on emotional state.
        Emotion adds to existing activation, doesn't replace."""
        firing = EMOTION_FIRING.get(emotion, {})
        for rid, strength in firing.items():
            if rid in self.activation:
                # Add half the emotion strength on top of band activation
                boost = strength >> 1
                self.activation[rid] = min(220, self.activation[rid] + boost)

    # ============================================================
    # PROPAGATE THROUGH PATHWAYS
    # ============================================================
    def propagate(self, steps=2):
        """Signal flows through white matter pathways.
        Propagation adds small boosts — it connects regions, not floods them."""
        for step in range(steps):
            updates = {r: 0 for r in REGIONS}
            for src, dst, strength in PATHWAYS:
                if src not in self.activation or dst not in self.activation:
                    continue
                if self.activation[src] < 40:
                    continue
                # Small signal: src * strength >> 10 (divide by 1024)
                signal = (self.activation[src] * strength) >> 10
                updates[dst] += signal

            for r in REGIONS:
                new_val = self.activation[r] + updates[r]
                self.activation[r] = max(0, min(230, new_val))

    # ============================================================
    # WORM-HUMAN BRIDGE
    # ============================================================
    def bridge_to_worm(self, bands, emotion, intensity):
        """Send human brain state down to worm brain.
        Amygdala fear -> worm retreat. PFC curiosity -> worm approach."""
        decision = self.worm.process(bands, emotion, intensity)

        # Worm result feeds back to human brain
        if decision["reflex"] == "backward":
            # Worm retreating -> boost amygdala further
            self.activation["ramyg"] = min(255, self.activation["ramyg"] + 30)
            self.activation["lamyg"] = min(255, self.activation["lamyg"] + 30)
            self.activation["rm1"] = min(255, self.activation["rm1"] + 40)
        elif decision["reflex"] == "forward":
            # Worm approaching -> boost PFC
            self.activation["rpfcdl"] = min(255, self.activation["rpfcdl"] + 20)
            self.activation["lpfcdl"] = min(255, self.activation["lpfcdl"] + 20)
            self.activation["racc"] = min(255, self.activation["racc"] + 30)
        elif decision["reflex"] == "freeze":
            # Freeze -> boost cingulate (conflict monitoring)
            self.activation["rcca"] = min(255, self.activation["rcca"] + 40)
            self.activation["lcca"] = min(255, self.activation["lcca"] + 40)

        return decision

    # ============================================================
    # FULL PROCESS
    # ============================================================
    def process(self, fingerprint=None, emotion="neutral", intensity=0):
        """
        Full brain cycle:
        1. Reset
        2. Extract frequency bands from fingerprint
        3. Activate regions from bands
        4. Activate from emotion
        5. Bridge to worm brain (bidirectional)
        6. Propagate through pathways
        7. Return full state
        """
        self.reset()

        # Extract bands
        if fingerprint is not None:
            bands = self.extract_bands(fingerprint)
        else:
            bands = {"delta": 50, "theta": 50, "alpha": 50, "beta": 50, "gamma": 50, "_overall": 50}

        # Activate from frequency bands
        self.activate_from_bands(bands)

        # Activate from emotion
        self.activate_from_emotion(emotion)

        # Worm-human bridge
        worm_decision = self.bridge_to_worm(bands, emotion, intensity)

        # Propagate through pathways
        self.propagate(steps=2)

        # Collect results
        top_regions = sorted(self.activation.items(), key=lambda x: -x[1])[:10]
        groups = {}
        for rid, val in self.activation.items():
            g = REGIONS[rid]["group"]
            if g not in groups:
                groups[g] = 0
            groups[g] = max(groups[g], val)

        return {
            "bands": bands,
            "emotion": emotion,
            "worm": worm_decision,
            "top_regions": top_regions,
            "group_activation": groups,
            "all_regions": dict(self.activation),
            "total_activation": sum(self.activation.values()),
        }

    # ============================================================
    # STATUS
    # ============================================================
    def status(self):
        return {
            "regions": len(REGIONS),
            "pathways": len(PATHWAYS),
            "worm_neurons": len(self.worm.activation),
            "worm_synapses": len(self.worm.synapses),
            "worm_fires": self.worm.total_fires,
            "worm_experiences": self.worm.experience_count,
        }


# ============================================================
# STANDALONE TEST
# ============================================================
if __name__ == "__main__":
    brain = HumanBrain()
    print("=" * 60)
    print("  BODHI Human Brain - %d regions, %d pathways" % (len(REGIONS), len(PATHWAYS)))
    print("  + Worm Brain: %d neurons, %d synapses" % (
        len(brain.worm.activation), len(brain.worm.synapses)))
    print("=" * 60)
    print()

    # Load compressed fingerprints
    data_dir = os.path.join(ROOT, "data")
    img_data = np.load(os.path.join(data_dir, "fingerprints_img.npz"))["data"]
    with open(os.path.join(data_dir, "fingerprint_index.json")) as f:
        index = json.load(f)

    # Test with real concepts
    test_concepts = [
        ("fire",     "fear",     300),
        ("ocean",    "awe",      200),
        ("snake",    "fear",     350),
        ("flower",   "love",     150),
        ("music",    "joy",      200),
        ("mountain", "awe",      250),
        ("mother",   "love",     200),
        ("storm",    "fear",     250),
    ]

    for cid, emotion, intensity in test_concepts:
        idx = index["img_name_to_idx"].get(cid)
        if idx is None:
            print("%s: NOT FOUND" % cid)
            continue

        fp = img_data[idx]
        result = brain.process(fp, emotion, intensity)

        # Top 5 regions
        top5 = result["top_regions"][:5]
        top_str = ", ".join("%s=%d" % (r, v) for r, v in top5)

        # Group summary
        groups = result["group_activation"]
        group_str = " ".join("%s=%d" % (g, v) for g, v in sorted(groups.items(), key=lambda x: -x[1])[:4])

        # Worm
        w = result["worm"]

        print("%10s [%s]: worm=%s(%d)  top=[%s]  groups=[%s]" % (
            cid, emotion, w["reflex"], w["confidence"], top_str, group_str))

    print()
    st = brain.status()
    print("Status: %d regions, %d pathways, worm=%d neurons/%d synapses" % (
        st["regions"], st["pathways"], st["worm_neurons"], st["worm_synapses"]))
