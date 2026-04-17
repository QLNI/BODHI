#!/usr/bin/env python3
"""
BODHI Brain Voice — speech derived DIRECTLY from brain state, not LLM output.

When BODHI sees a word or image, its brain does real work:
  - The Walsh-Hadamard fingerprint decomposes into 5 frequency bands
  - 76 brain regions activate by different amounts
  - 26 worm neurons fire, and a reflex emerges from them
  - Emotional memory adds a learned valence
  - Hebbian associates light up related concepts

This module turns that REAL computation into first-person speech. No prompt, no
token prediction, no hallucination. If gamma is high, BODHI feels sharpness; if
delta dominates, BODHI feels depth. If the amygdala fires hard, BODHI reports
fear. The sentences vary because the signal values vary.

This is what Report 03 promised: BODHI speaking from its perception pipeline.
"""

import random


# ============================================================
# WHAT EACH FREQUENCY BAND "FEELS LIKE"
# Each band has a biological analog; we report the feeling, not the number.
# ============================================================

BAND_SENSATIONS = {
    "delta": [
        "slow deep signal through me",
        "a broad low hum",
        "something vast and quiet underneath",
        "deep structure, no edges",
        "low frequency everywhere",
    ],
    "theta": [
        "memory waves stirring",
        "something familiar pulls at me",
        "old pattern, known before",
        "theta rhythm, like a door opening to memory",
        "I have met this shape before",
    ],
    "alpha": [
        "clear shape",
        "a known outline",
        "something recognisable takes form",
        "edges I can hold in mind",
        "shape without complexity",
    ],
    "beta": [
        "complex features, many at once",
        "lots to parse",
        "detail on detail",
        "my cortex works hard",
        "beta busy, analysing",
    ],
    "gamma": [
        "sharp edges, bright detail",
        "texture loud in my head",
        "fine grain, almost too much",
        "gamma spike — binding firing",
        "the image is sharp where it should not be",
    ],
}


# ============================================================
# WHAT THE DOMINANT REGION GROUP "FEELS LIKE"
# ============================================================

GROUP_FEELINGS = {
    "limbic": [
        "my emotional core takes this over",
        "the amygdala speaks first",
        "this lands in my feeling centre",
        "limbic takes the wheel",
    ],
    "prefrontal": [
        "I am thinking about this before feeling",
        "my planner is awake",
        "prefrontal engaged, slow deliberation",
        "I am deciding, not just reacting",
    ],
    "memory": [
        "hippocampus lights up",
        "I am remembering more than seeing",
        "this is pulling up old traces",
        "memory is louder than perception here",
    ],
    "visual": [
        "my visual cortex holds this",
        "I am mostly seeing",
        "V1 is doing the work",
    ],
    "auditory": [
        "auditory regions wake up",
        "sound-shape in my head",
        "I am hearing this even when it is a word",
    ],
    "motor": [
        "my motor cortex is already preparing",
        "I am primed to move",
        "motor planning ahead of thought",
    ],
    "temporal": [
        "temporal regions engaged",
        "naming and recognising",
    ],
    "frontal": [
        "frontal systems active",
        "attention and effort",
    ],
}


# ============================================================
# WHAT THE WORM REFLEX "FEELS LIKE" IN THE BODY
# ============================================================

def body_feel(reflex, confidence):
    if reflex == "backward":
        if confidence >= 240:
            return random.choice([
                "every part of me pulls back",
                "full retreat in the body",
                "my worm neurons scream no",
                "backward at maximum",
            ])
        if confidence >= 150:
            return random.choice([
                "I step back",
                "the body says distance",
                "backward pull, strong",
            ])
        if confidence > 0:
            return random.choice([
                "a small backward pull",
                "slight retreat",
                "mild caution in the body",
            ])
    if reflex == "forward":
        if confidence >= 150:
            return random.choice([
                "my body wants to approach",
                "I lean in",
                "forward pull, clear",
            ])
        if confidence > 0:
            return random.choice([
                "a small leaning forward",
                "gentle approach in the body",
            ])
    if reflex == "freeze":
        if confidence >= 100:
            return random.choice([
                "I freeze",
                "I hold still, unsure",
                "the body stops, listens",
            ])
        return random.choice([
            "a pause in the body",
            "stillness, brief",
        ])
    return None


# ============================================================
# LEARNED EMOTIONAL MEMORY (the amygdala remembers)
# ============================================================

def memory_feel(concept, em_value):
    if em_value >= 100:
        return random.choice([
            "I have learned %s is dangerous" % concept,
            "%s has hurt me before" % concept,
            "my memory of %s is warning" % concept,
        ])
    if em_value >= 30:
        return random.choice([
            "something in my memory is cautious about %s" % concept,
            "I have mixed feelings stored for %s" % concept,
        ])
    if em_value <= -100:
        return random.choice([
            "I know %s as safe" % concept,
            "%s is welcome in my memory" % concept,
            "my memory of %s is warm" % concept,
        ])
    if em_value <= -30:
        return random.choice([
            "%s feels mostly safe from memory" % concept,
        ])
    return None


# ============================================================
# MAIN VOICE FUNCTION
# ============================================================

def brain_voice(brain_result, matched=None, associates=None,
                emotional_memory=None, user_text=""):
    """Compose a first-person response directly from brain state.

    Order of clauses, and whether each appears, depends on real signal
    values — so the sentence structure naturally varies per input.
    """
    if brain_result is None:
        return "I am here. No input reached me clearly."

    parts = []
    matched = matched or []
    associates = associates or []
    emotional_memory = emotional_memory or {}

    # 1. Concept acknowledgement (what BODHI locked onto)
    if matched:
        name = matched[0].replace("_", " ")
        opener = random.choice([
            "%s." % name.capitalize(),
            "I see %s." % name,
            "%s — yes." % name.capitalize(),
            "Input: %s." % name,
            "%s arrives." % name.capitalize(),
        ])
        parts.append(opener)

    # 2. Dominant frequency band — what kind of signal this is
    bands = brain_result.get("bands", {}) or {}
    band_items = [(k, v) for k, v in bands.items() if k != "_overall"]
    if band_items:
        band_items.sort(key=lambda x: -x[1])
        top_band, top_val = band_items[0]
        if top_val > 30:
            parts.append(random.choice(BAND_SENSATIONS.get(top_band, [])) + ".")
        # If second band is also strong and different character, mention it
        if len(band_items) > 1 and band_items[1][1] > 100 and band_items[1][1] > top_val * 0.85:
            second_band = band_items[1][0]
            parts.append(random.choice(BAND_SENSATIONS.get(second_band, [])) + ".")

    # 3. Dominant region group — what kind of cognition
    groups = brain_result.get("group_activation", {}) or {}
    if groups:
        top_group, gval = max(groups.items(), key=lambda x: x[1])
        if gval > 120:
            parts.append(random.choice(GROUP_FEELINGS.get(top_group, [])) + ".")

    # 4. Body feel from worm brain
    worm = brain_result.get("worm", {}) or {}
    reflex = worm.get("reflex")
    conf = int(worm.get("confidence", 0) or 0)
    body = body_feel(reflex, conf)
    if body:
        parts.append(body + ".")

    # 5. Learned emotional memory
    if matched:
        em_val = emotional_memory.get(matched[0], 0)
        mem = memory_feel(matched[0].replace("_", " "), em_val)
        if mem:
            parts.append(mem + ".")

    # 6. Audio fingerprint present — what does the sound feel like?
    audio_bands = brain_result.get("audio_bands") if brain_result else None
    if audio_bands:
        aud_items = [(k, v) for k, v in audio_bands.items() if k != "_overall"]
        if aud_items:
            aud_items.sort(key=lambda x: -x[1])
            aud_top, aud_val = aud_items[0]
            parts.append(random.choice([
                "I also hear this — auditory cortex wakes up",
                "there is sound for this too, and my ears pay attention",
                "sound-shape joins the picture",
            ]) + ".")
            parts.append("The sound is " + random.choice(BAND_SENSATIONS.get(aud_top, [])) + ".")

    # 7. Hebbian associates (if any real wiring formed)
    if associates:
        a = associates[0].replace("_", " ")
        parts.append(random.choice([
            "%s comes up alongside this" % a,
            "something in me pairs this with %s" % a,
            "%s is wired close" % a,
        ]) + ".")

    if not parts:
        parts.append("I feel something but I cannot name the shape of it.")

    return " ".join(parts)
