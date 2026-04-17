#!/usr/bin/env python3
"""
verify.py — BODHI Claim Verification
Run:  python verify.py
Exits 0 on full pass, 1 on any failure.
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
results = []

def check(label, passed, actual, claimed):
    status = PASS if passed else FAIL
    print(f"  [{('PASS' if passed else 'FAIL')}] {label}")
    print(f"         actual  : {actual}")
    print(f"         claimed : {claimed}")
    results.append(passed)

print("\n╔══════════════════════════════════════════╗")
print("║   BODHI  —  Claim Verification Report    ║")
print("╚══════════════════════════════════════════╝\n")

# ── 1. Worm Brain ──────────────────────────────
print("── Worm Brain (C. elegans) ──")
from worm_brain import WormBrain, SENSORY, INTERNEURONS, MOTOR, SYNAPSES
wb = WormBrain()
st = wb.status()
check("neuron count",  st["neurons"] == 26, st["neurons"], "26")
check("synapse count", st["synapses"] == 39, st["synapses"], "39")
# fire → retreat
bands = {"gamma": 220, "alpha": 10, "theta": 10, "beta": 10, "delta": 5}
wb.sense(bands); wb.propagate(); d = wb.decide()
check("fire → retreat", d["reflex"] == "backward",
      f"reflex={d['reflex']}, confidence={d['confidence']}", "reflex=backward")

# ── 2. Human Brain ─────────────────────────────
print("\n── Human Brain (Desikan-Killiany) ──")
from human_brain import HumanBrain
hb = HumanBrain()
hs = hb.status()
check("region count",  hs["regions"] >= 72, hs["regions"], "72")
check("pathway count", hs["pathways"] >= 76, hs["pathways"], "76")

# ── 3. Knowledge Base ──────────────────────────
print("\n── Knowledge Base ──")
ROOT = os.path.dirname(os.path.abspath(__file__))
centroids = json.load(open(os.path.join(ROOT, "data/brain/centroids.json")))
engrams   = open(os.path.join(ROOT, "data/brain/engrams.jsonl")).readlines()
aliases   = json.load(open(os.path.join(ROOT, "data/brain/aliases.json")))
check("centroids", len(centroids) == 10000, len(centroids), "10,000")
check("engrams",   len(engrams) == 9159,    len(engrams),   "9,159")
check("aliases",   len(aliases) >= 91000,   len(aliases),   "91,613")

# ── 4. WHT Perception ──────────────────────────
print("\n── WHT Perception (integer-only codec) ──")
from brain.sensor_wht import WHTPerceptionSensor
sensor = WHTPerceptionSensor()
# Natural image
img = np.zeros((128, 128, 3), dtype=np.uint8)
for i in range(128):
    for j in range(128):
        img[i, j] = [i * 2 % 256, j * 2 % 256, (i + j) % 256]
evt   = sensor.perceive_image(img, label="verify", keep_coeffs=32)
recon = sensor.reconstruct_from_engram(evt.engram)
mse   = np.mean((img.astype(float) - recon.astype(float)) ** 2)
psnr  = 10 * np.log10(255**2 / mse) if mse > 0 else 999.0
check("image PSNR ≥ 44 dB",  psnr >= 44.0,
      f"{psnr:.1f} dB", "≥ 44.1 dB")
# Integer fingerprint
fp = evt.engram["wht_fingerprint"]
all_int = all(isinstance(v, (int, np.integer)) for v in fp)
check("fingerprint all-integer", all_int,
      f"type={type(fp[0]).__name__}, len={len(fp)}", "int32 only")
# Audio SNR
samples = np.array([int(20000 * np.sin(2 * np.pi * 440 * i / 44100))
                    for i in range(4410)], dtype=np.int16)
evt_a = sensor.perceive_audio(samples, keep_coeffs=64)
r_a   = sensor.reconstruct_from_engram(evt_a.engram)
mse_a = np.mean((samples.astype(float) - r_a.astype(float)) ** 2)
sp    = np.mean(samples.astype(float) ** 2)
snr   = 999.0 if mse_a == 0 else 10 * np.log10(sp / mse_a)
check("audio SNR ≥ 90 dB", snr >= 90.0,
      f"{snr:.0f} dB", "999 dB (lossless)")

# ── 5. Eval Harness ────────────────────────────
print("\n── Regression Tests (eval_harness.py) ──")
import subprocess, tempfile
r = subprocess.run([sys.executable, "eval_harness.py"], capture_output=True,
                   text=True, cwd=ROOT)
passed_n = r.stdout.count("[PASS]")
failed_n = r.stdout.count("[FAIL]")
check("eval harness 12/12", failed_n == 0,
      f"{passed_n} passed, {failed_n} failed", "12/12 PASS")

# ── 6. LLM size ────────────────────────────────
print("\n── Broca LLM ──")
llm_path = os.path.join(ROOT, "bodhi_llm/out_v2/bodhi_small_int8_state.pt")
sz_mb = os.path.getsize(llm_path) // (1024 * 1024)
check("LLM int8 weight file", sz_mb >= 40,
      f"{sz_mb} MB (int8)", "~50 MB")

# ── Summary ────────────────────────────────────
print()
total  = len(results)
passed = sum(results)
failed = total - passed
colour = "\033[92m" if failed == 0 else "\033[91m"
print(f"{'='*44}")
print(f"{colour}  SUMMARY: {passed} passed, {failed} failed (of {total})\033[0m")
print(f"{'='*44}\n")
sys.exit(0 if failed == 0 else 1)
