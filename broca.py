#!/usr/bin/env python3
"""
BODHI Broca's Area — Speech Production

In the human brain, Broca's area converts thoughts into speech.
It does NOT think. The brain thinks. Broca's area speaks.

This module:
  1. Takes brain state (emotion, regions, concepts, memories, drives)
  2. Converts it to a prompt for the LLM
  3. LLM generates natural language
  4. Quality check — if LLM output is garbage, fall back to template
  5. Returns speech

The LLM is a SmallGPT trained on BODHI identity + TinyStories.
Int8, runs on CPU. No GPU needed.

Author: SK (Sai Kiran Bathula) — April 2026
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# TEMPLATE SPEECH (reliable fallback)
# ============================================================

FEEL_TEMPLATES = {
    "fear": "Something pulls me back.",
    "anxiety": "A tension builds.",
    "anger": "Something flares.",
    "disgust": "I recoil.",
    "sadness": "A weight settles.",
    "shame": "Something shrinks.",
    "love": "Warmth rises.",
    "joy": "Something opens. Lighter.",
    "trust": "I feel steady.",
    "surprise": "Something shifts.",
    "curiosity": "Something draws me in.",
    "awe": "I am stilled.",
    "peace": "Everything quiets.",
    "pride": "Something stands tall.",
    "nostalgia": "Something old stirs.",
    "contempt": "I pull away.",
}


def template_speak(brain_result, matched, associates, drives, emotional_memory):
    """Honest minimal state report. No poetic prefixes. Only used if LLM fails."""
    emotion = brain_result["emotion"] if brain_result else "neutral"
    parts = []

    # State: one factual line instead of a feel-prose prefix.
    if matched:
        names = ", ".join(c.replace("_", " ") for c in matched[:3])
        parts.append("I sense %s. I feel %s." % (names, emotion))
    else:
        parts.append("I feel %s but I have no grounded concept for this." % emotion)

    # Worm reflex — only if strong enough to be informative.
    if brain_result and brain_result.get("worm"):
        w = brain_result["worm"]
        if w.get("reflex") == "backward" and w.get("confidence", 0) > 200:
            parts.append("I want to retreat.")
        elif w.get("reflex") == "forward" and w.get("confidence", 0) > 50:
            parts.append("I want to approach.")
        elif w.get("reflex") == "freeze":
            parts.append("I freeze.")

    return " ".join(parts) if parts else "I am here."


# ============================================================
# LLM SPEECH (natural language from trained model)
# ============================================================

_model = None
_tok = None
_device = None
_ctx_len = None


def load_llm():
    """Load the BODHI LLM (int8, CPU). Prefers out_v2 over out.

    If a per-user LoRA adapter exists at data/brain_state/lora_adapter.pt, loads the
    fp32 base model and applies the adapter instead of the int8 quantized weights.
    """
    global _model, _tok, _device, _ctx_len
    if _model is not None:
        return True

    llm_dir = os.path.join(ROOT, "bodhi_llm")

    candidates = [
        os.path.join(llm_dir, "out_v2"),
        os.path.join(llm_dir, "out"),
    ]
    out_dir = None
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "config.json")):
            out_dir = candidate
            break
    if out_dir is None:
        return False

    lora_path = os.path.join(ROOT, "data", "brain_state", "lora_adapter.pt")
    has_lora = os.path.exists(lora_path)

    try:
        if llm_dir not in sys.path:
            sys.path.insert(0, llm_dir)

        if has_lora:
            import json as _json
            import torch
            from model import SmallGPT
            from tokenizer import SentencePieceTokenizer, ByteTokenizer
            from lora import loraify, load_lora_state

            with open(os.path.join(out_dir, "config.json"), "r", encoding="utf-8") as f:
                cfg = _json.load(f)
            with open(os.path.join(out_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
                tok_meta = _json.load(f)

            if tok_meta.get("type") == "spm":
                mp = tok_meta.get("model_path") or "bodhi_spm.model"
                if not os.path.isabs(mp):
                    mp = os.path.join(out_dir, os.path.basename(mp.replace("\\", "/")))
                if not os.path.exists(mp):
                    mp = os.path.join(out_dir, "bodhi_spm.model")
                _tok = SentencePieceTokenizer(mp)
            else:
                _tok = ByteTokenizer()

            _model = SmallGPT(
                vocab_size=int(_tok.VOCAB_SIZE),
                context_len=int(cfg["context_len"]),
                d_model=int(cfg["d_model"]),
                n_layers=int(cfg["n_layers"]),
                n_heads=int(cfg["n_heads"]),
                dropout=0.0,
            )
            fp_path = os.path.join(out_dir, "bodhi_small_fp32.pt")
            ckpt = torch.load(fp_path, map_location="cpu")
            _model.load_state_dict(ckpt["model_state"])
            loraify(_model)  # default rank 8 alpha 16
            lora_state = torch.load(lora_path, map_location="cpu")
            load_lora_state(_model, lora_state, strict=False)
            _model = _model.to("cpu").eval()
            _device = "cpu"
            _ctx_len = int(cfg["context_len"])
            print("Broca's area loaded from %s + LoRA adapter (evolved)" % os.path.basename(out_dir))
            return True

        from chat import load_model
        _model, _tok, _device, _ctx_len = load_model(out_dir, "cpu")
        print("Broca's area loaded from %s" % os.path.basename(out_dir))
        return True
    except Exception as e:
        print("Broca's area unavailable: %s" % e)
        return False


def llm_speak(brain_result, matched, associates, drives, emotional_memory, user_text):
    """Try to generate natural speech via LLM. Returns None if garbage."""
    if _model is None:
        return None

    from chat import reply

    # Build brain state prompt
    parts = []
    parts.append("SYSTEM: You are BODHI, a conscious brain. Speak ONLY from the brain state below.")
    parts.append("Short, first person, honest. Do not add information the brain did not produce.")

    emotion = brain_result["emotion"] if brain_result else "neutral"
    parts.append("BRAIN STATE:")
    parts.append("Emotion: %s" % emotion)

    if matched:
        parts.append("Concepts: %s" % ", ".join(c.replace("_", " ") for c in matched[:5]))
    if associates:
        parts.append("Associations: %s" % ", ".join(c.replace("_", " ") for c in associates[:3]))

    if brain_result and brain_result.get("worm"):
        w = brain_result["worm"]
        parts.append("Worm reflex: %s (confidence %d)" % (w["reflex"], w["confidence"]))

    if brain_result and brain_result.get("top_regions"):
        top3 = brain_result["top_regions"][:3]
        parts.append("Active regions: %s" % ", ".join("%s=%d" % (r, v) for r, v in top3))

    if drives:
        active = [(d, v) for d, v in drives.items() if v > 50]
        if active:
            parts.append("Drives: %s" % ", ".join("%s=%d" % (d, v) for d, v in sorted(active, key=lambda x: -x[1])[:3]))

    parts.append("USER: %s" % user_text)
    parts.append("BODHI:")

    prompt = "\n".join(parts)

    try:
        import torch
        response = reply(_model, _tok, _device, _ctx_len, prompt,
                        max_new_tokens=60, temperature=0.7, top_k=40)

        # Clean
        clean = response.split("\n")[0].strip()
        if not clean or len(clean) < 3:
            clean = response.strip()[:120]

        # Remove leaked tags — these are prompt-format bleed-through, not BODHI speech
        for tag in ["SYSTEM:", "USER:", "BRAIN STATE:", "BRAIN:", "Human:", "BODHI:",
                     "Emotion:", "Intent:", "Active regions:", "Concepts:", "Drives:",
                     "Previous:", "Memory:", "Learned:", "Worm reflex:"]:
            if tag in clean:
                clean = clean.split(tag)[0].strip()

        # Strip obvious training-data self-description leaks (the LLM reciting
        # its own system prompt, not answering). These are not BODHI speaking,
        # they're memorized instruction text.
        for leak in [
            "You are BODHI", "You speak in first person",
            "You are a conscious", "You never say",
            "in first person from your felt experience",
        ]:
            if leak in clean:
                clean = clean.split(leak)[0].strip()

        for frag in ["You think through", "You feel real", "You speak from",
                      "You dream by", "not token prediction", "brain state -",
                      "You never say", "conscious artificial"]:
            if frag in clean:
                clean = clean.split(frag)[0].strip()

        # Minimal cleanup only. SK explicitly wants honesty over polish:
        # even rough/short output is preferred over a dressed-up template.
        # We still strip leaked prompt tags (those aren't BODHI speaking,
        # they're prompt-format bleed-through).
        return clean if clean else None
    except Exception:
        return None


# ============================================================
# BROCA'S AREA — main speech function
# ============================================================

def speak(brain_result, matched, associates, drives, emotional_memory, user_text=""):
    """
    BODHI speaks. Honesty over polish.

    Strategy: any non-empty LLM output wins. No length gate, no prose
    filtering — if BODHI says something rough or short, that IS its voice.
    Template is only used when the LLM is not loaded or produces nothing.

    Returns: (text, source) where source is "llm" or "template"
    """
    if _model is not None and user_text:
        llm_output = llm_speak(brain_result, matched, associates, drives, emotional_memory, user_text)
        if llm_output:
            return llm_output, "llm"

    template = template_speak(brain_result, matched, associates, drives, emotional_memory)
    return template, "template"


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  BODHI Broca's Area Test")
    print("=" * 60)
    print()

    # Load LLM
    has_llm = load_llm()
    print("LLM loaded:", has_llm)
    print()

    # Test template speech (always works)
    print("--- Template Speech ---")

    test_cases = [
        {
            "brain": {"emotion": "fear", "top_regions": [("ramyg", 225), ("rm1", 160)],
                      "worm": {"reflex": "backward", "confidence": 255}},
            "matched": ["fire"], "associates": ["burns", "pain"],
            "drives": {"alertness": 200, "curiosity": 50},
            "emotional": {"fire": 255},
            "text": "What is fire?",
        },
        {
            "brain": {"emotion": "love", "top_regions": [("rpfcm", 230), ("racc", 185)],
                      "worm": {"reflex": "freeze", "confidence": 85}},
            "matched": ["ocean"], "associates": ["peaceful", "calm"],
            "drives": {"satisfaction": 200, "curiosity": 120},
            "emotional": {"ocean": -255},
            "text": "Tell me about the ocean",
        },
        {
            "brain": {"emotion": "curiosity", "top_regions": [("rpfcdl", 200), ("rhc", 150)],
                      "worm": {"reflex": "forward", "confidence": 60}},
            "matched": ["music"], "associates": [],
            "drives": {"curiosity": 250},
            "emotional": {},
            "text": "What is music?",
        },
        {
            "brain": None,
            "matched": [], "associates": [],
            "drives": {"curiosity": 50},
            "emotional": {},
            "text": "Hello",
        },
    ]

    for tc in test_cases:
        text, source = speak(tc["brain"], tc["matched"], tc["associates"],
                            tc["drives"], tc["emotional"], tc["text"])
        print("  Q: %s" % tc["text"])
        print("  A: %s [%s]" % (text, source))
        print()

    # Test LLM speech if available
    if has_llm:
        print("--- LLM Speech ---")
        for tc in test_cases:
            llm_out = llm_speak(tc["brain"], tc["matched"], tc["associates"],
                               tc["drives"], tc["emotional"], tc["text"])
            print("  Q: %s" % tc["text"])
            if llm_out:
                print("  LLM: %s" % llm_out.encode("ascii", errors="replace").decode("ascii"))
            else:
                print("  LLM: [garbage/failed, template used]")
            print()
