"""
BODHI WHT Perception Sensor — PUBLIC SHIM.

The actual Walsh–Hadamard Transform implementation (image + audio + video codec,
IP Australia 2026901656 / 2026901657) lives in the compiled binary
``_wht_core.cpython-<py-version>-<platform>.so`` (Linux / macOS) or
``_wht_core.cp<py-version>-<platform>.pyd`` (Windows) sitting next to this file.

This file is a thin wrapper that re-exports the public API. Nothing else in
BODHI has to care whether the implementation is Python, Cython, or C.

WHY LOCKED
----------
The WHT codec is the prize of the Final Match (see ROADMAP.md, Phase 4). For
Phase 1–3 the algorithm ships only in compiled form — you can USE BODHI with
full fidelity, but you cannot READ the algorithm. The patent (AU 2026901656 /
2026901657) provides the legal moat; this compilation provides a convenience
lock.

REBUILDING
----------
If the precompiled binary does not match your platform and you have a
privileged copy of ``_wht_core.pyx``, run::

    python -m bodhi.brain.build_wht

This will produce the native binary for your Python + OS + CPU.

USAGE
-----
Use it exactly as before::

    from bodhi.brain.sensor_wht import WHTPerceptionSensor
    sensor = WHTPerceptionSensor()
    event = sensor.perceive_image(img_np)         # encode
    reconstructed = sensor.reconstruct(event)     # decode
"""
from __future__ import annotations

# --- Runtime check -------------------------------------------------------
# BODHI gets run two ways:
#   (a) ``python bodhi.py`` from inside the bodhi/ directory → top-level import
#       path looks like ``from brain.sensor_wht import X``.
#   (b) ``python -m bodhi.bodhi`` from the repo root → path looks like
#       ``from bodhi.brain.sensor_wht import X``.
# The compiled binary sits right next to this file, so we also add the
# directory of this file to sys.path as a last-resort fallback. This makes the
# shim resilient to whatever import style the caller chose.
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))

_core = None
_err = None
for _modname in ("brain._wht_core", "bodhi.brain._wht_core", "_wht_core"):
    try:
        _core = __import__(_modname, fromlist=["_wht_core"])
        break
    except ImportError as _e:
        _err = _e

if _core is None:
    # Last resort: add this directory to sys.path and retry.
    if _HERE not in _sys.path:
        _sys.path.insert(0, _HERE)
    try:
        import _wht_core as _core  # type: ignore
    except ImportError as _e:  # pragma: no cover
        raise ImportError(
            "BODHI WHT codec binary not found. Expected "
            "_wht_core.cpython-<ver>-<platform>.so (Linux/macOS) or "
            "_wht_core.cp<ver>-<platform>.pyd (Windows) next to sensor_wht.py.\n"
            "If you have access to the .pyx source, rebuild with:\n"
            "    python -m bodhi.brain.build_wht\n"
            "Otherwise contact the repository maintainer for a prebuilt binary "
            "matching your platform.\n"
            f"Original error: {_err}"
        ) from _e

# --- Public API — re-export exactly what the pure-Python version exposed -
BLOCK                       = _core.BLOCK
AUDIO_BLOCK                 = _core.AUDIO_BLOCK

# Image codec
encode_image_to_fingerprint = _core.encode_image_to_fingerprint
decode_fingerprint_to_image = _core.decode_fingerprint_to_image
image_fingerprint_to_engram = _core.image_fingerprint_to_engram

# Audio codec
encode_audio_to_fingerprint = _core.encode_audio_to_fingerprint
decode_fingerprint_to_audio = _core.decode_fingerprint_to_audio
audio_fingerprint_to_engram = _core.audio_fingerprint_to_engram

# Video codec
encode_video_to_fingerprint = _core.encode_video_to_fingerprint
decode_fingerprint_to_video = _core.decode_fingerprint_to_video
video_fingerprint_to_engram = _core.video_fingerprint_to_engram

# Unified sensor
WHTPerceptEvent             = _core.WHTPerceptEvent
WHTPerceptionSensor         = _core.WHTPerceptionSensor

# Quality metrics (handy for tests & dream evaluations)
compute_image_psnr          = _core.compute_image_psnr
compute_audio_snr           = _core.compute_audio_snr


__all__ = [
    "BLOCK", "AUDIO_BLOCK",
    "encode_image_to_fingerprint", "decode_fingerprint_to_image", "image_fingerprint_to_engram",
    "encode_audio_to_fingerprint", "decode_fingerprint_to_audio", "audio_fingerprint_to_engram",
    "encode_video_to_fingerprint", "decode_fingerprint_to_video", "video_fingerprint_to_engram",
    "WHTPerceptEvent", "WHTPerceptionSensor",
    "compute_image_psnr", "compute_audio_snr",
]
