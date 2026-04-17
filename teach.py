#!/usr/bin/env python3
"""
BODHI Concept Teacher — learn new concepts from images / audio / text at runtime.

Before this, BODHI only knew the 10,000 pre-loaded concepts that shipped with the
system. The promise of Report 01 was that BODHI learns from experience. This
module makes that literal: when the user provides an image or audio clip with a
name, BODHI immediately computes the WHT fingerprint, appends it to the
fingerprint store, registers the concept, and from that moment forward treats
the new concept exactly like any pre-loaded one — it can be matched, retrieved,
imagined, emotionally bonded, and woven into Hebbian wiring.

Persisted under data/learned/ so new concepts survive restart.

API (on a BODHI instance, after teacher is attached):
  bodhi.teacher.teach_image(concept_name, image_path, emotion="neutral", description="")
  bodhi.teacher.teach_audio(concept_name, audio_path, emotion="neutral", description="")
  bodhi.teacher.teach_text(concept_name, description, emotion="neutral")

Slash command (intercepted by bodhi.think):
  /teach <concept_name> <path>       — auto-detects image vs audio by extension
  /teach <concept_name> --text <description>
  /teach list                         — show learned concepts
"""

import os
import json
import time
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

try:
    from brain.sensor_wht import (
        encode_image_to_fingerprint,
        encode_audio_to_fingerprint,
    )
    HAS_SENSOR = True
except Exception:
    HAS_SENSOR = False

try:
    from brain import codec_guard
    HAS_GUARD = True
except Exception:
    HAS_GUARD = False


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
AUD_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".m4a")


def _ext(path):
    return os.path.splitext(path)[1].lower()


def _is_image(path):
    return _ext(path) in IMG_EXTS


def _is_audio(path):
    return _ext(path) in AUD_EXTS


def _load_audio_samples(path):
    """Best-effort audio loader. Returns int16 mono array at native sample rate,
    or raises if no audio backend is available."""
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="int16")
        if data.ndim == 2:
            data = data.mean(axis=1).astype(np.int16)
        return data, sr
    except Exception:
        pass
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        if data.dtype != np.int16:
            if np.issubdtype(data.dtype, np.floating):
                data = (data * 32767).astype(np.int16)
            else:
                data = data.astype(np.int16)
        if data.ndim == 2:
            data = data.mean(axis=1).astype(np.int16)
        return data, sr
    except Exception:
        pass
    raise RuntimeError(
        "No audio backend. Install soundfile or scipy to enable audio teaching.")


class ConceptTeacher:
    """Attached to a BODHI instance after init. Handles new concept acquisition."""

    def __init__(self, bodhi):
        self.bodhi = bodhi
        root = os.path.dirname(os.path.abspath(__file__))
        self.dir = os.path.join(root, "data", "learned")
        self.audio_dir = os.path.join(self.dir, "audio")
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)

        # New path uses scrambled format (.wht) if codec_guard is available,
        # else fall back to plain .npz for compatibility.
        self.img_store_path_scrambled = os.path.join(self.dir, "fingerprints_img_learned.wht")
        self.img_store_path_plain = os.path.join(self.dir, "fingerprints_img_learned.npz")
        self.img_store_path = (self.img_store_path_scrambled
                               if HAS_GUARD else self.img_store_path_plain)
        self.concepts_path = os.path.join(self.dir, "concepts.json")

        # On-disk shape:
        #  - img_store: npz with 'data' = [N, 24576] int16, 'names' = list of strings
        #  - concepts.json: {name: {emotion, source, description, created, img_idx|aud_file}}
        self.concepts_meta = {}
        self._load()

    # ----------------------------------------------------------
    # Load / boot
    # ----------------------------------------------------------
    def _load(self):
        # Concepts metadata
        if os.path.exists(self.concepts_path):
            try:
                with open(self.concepts_path, "r", encoding="utf-8") as f:
                    self.concepts_meta = json.load(f)
            except Exception:
                self.concepts_meta = {}

        # Image fingerprints: scrambled preferred, plain as fallback
        data, names = None, []
        if HAS_GUARD and os.path.exists(self.img_store_path_scrambled):
            try:
                data, names = codec_guard.load_named(self.img_store_path_scrambled)
            except Exception as e:
                print("[teach] could not load scrambled image store:", e)
        elif os.path.exists(self.img_store_path_plain):
            try:
                z = np.load(self.img_store_path_plain, allow_pickle=True)
                data = z["data"]
                names = list(z["names"])
            except Exception as e:
                print("[teach] could not load plain image store:", e)

        if data is not None and len(data) > 0:
            base_count = len(self.bodhi.img_data)
            self.bodhi.img_data = np.vstack([self.bodhi.img_data, data])
            for i, n in enumerate(names):
                self.bodhi.fp_index["img_name_to_idx"][n] = base_count + i
            print("[teach] loaded %d learned image concepts%s" %
                  (len(names), " (scrambled)" if HAS_GUARD and os.path.exists(self.img_store_path_scrambled) else ""))

        # Concept emotions + aliases + engrams for each learned concept
        for name, meta in self.concepts_meta.items():
            self.bodhi.concept_emotions[name] = meta.get("emotion", "neutral")
            # Treat the lowercase name as an alias for itself (so match_concepts finds it)
            self.bodhi.aliases.setdefault(name.lower(), name)
            # If description present, make a minimal engram so template speech can
            # append a definition clause like for built-in concepts.
            desc = meta.get("description", "").strip()
            if desc and name not in self.bodhi.engrams:
                self.bodhi.engrams[name] = {"capsule": desc, "centroid_anchor": name}

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------
    def _save_concepts(self):
        tmp = self.concepts_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.concepts_meta, f, indent=2)
        os.replace(tmp, self.concepts_path)

    def _append_image_fingerprint(self, fp, name):
        """Append one fingerprint to the learned image store (scrambled if available)."""
        fp16 = fp.astype(np.int16)

        # Load existing (scrambled preferred, plain fallback for migration)
        old_data, old_names = None, []
        if HAS_GUARD and os.path.exists(self.img_store_path_scrambled):
            old_data, old_names = codec_guard.load_named(self.img_store_path_scrambled)
        elif os.path.exists(self.img_store_path_plain):
            z = np.load(self.img_store_path_plain, allow_pickle=True)
            old_data = z["data"]
            old_names = list(z["names"])

        if old_data is not None and len(old_data) > 0:
            if fp16.shape[0] != old_data.shape[1]:
                raise ValueError("Fingerprint length mismatch: new=%d existing=%d"
                                 % (fp16.shape[0], old_data.shape[1]))
            data = np.vstack([old_data, fp16[np.newaxis, :]])
            names = old_names + [name]
        else:
            data = fp16[np.newaxis, :]
            names = [name]

        # Save — prefer scrambled
        if HAS_GUARD:
            codec_guard.save_named(data, names, self.img_store_path_scrambled)
            # If a plain version still exists alongside, delete it (migration)
            if os.path.exists(self.img_store_path_plain):
                try:
                    os.remove(self.img_store_path_plain)
                except Exception:
                    pass
        else:
            np.savez_compressed(self.img_store_path_plain, data=data, names=np.array(names))
        return len(names) - 1

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def teach_image(self, concept_name, image_path, emotion="neutral",
                    description="", size=(256, 256)):
        if not HAS_PIL:
            raise RuntimeError("PIL not available. Cannot load images.")
        if not HAS_SENSOR:
            raise RuntimeError("brain.sensor_wht not available. Cannot encode.")
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        concept_name = concept_name.strip().lower().replace(" ", "_")
        if not concept_name:
            raise ValueError("concept_name required")

        img = Image.open(image_path).convert("RGB").resize(size, Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        fp = encode_image_to_fingerprint(arr, keep_coeffs=8).astype(np.int32)

        learned_idx = self._append_image_fingerprint(fp, concept_name)
        base_count_before_reload = len(self.bodhi.img_data)  # current full count

        # Live update in-memory arrays too so this concept is immediately queryable
        if concept_name in self.bodhi.fp_index["img_name_to_idx"]:
            # Overwrite existing — rare, but handle gracefully
            existing_idx = self.bodhi.fp_index["img_name_to_idx"][concept_name]
            self.bodhi.img_data[existing_idx] = fp.astype(self.bodhi.img_data.dtype)
        else:
            self.bodhi.img_data = np.vstack(
                [self.bodhi.img_data, fp.astype(self.bodhi.img_data.dtype)[np.newaxis, :]])
            self.bodhi.fp_index["img_name_to_idx"][concept_name] = base_count_before_reload

        # Metadata
        self.concepts_meta[concept_name] = {
            "emotion": emotion,
            "source": "learned_image",
            "description": description,
            "image_path": os.path.abspath(image_path),
            "img_learned_idx": learned_idx,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_concepts()

        self.bodhi.concept_emotions[concept_name] = emotion
        self.bodhi.aliases.setdefault(concept_name, concept_name)
        if description:
            self.bodhi.engrams[concept_name] = {"capsule": description,
                                                "centroid_anchor": concept_name}

        return {
            "concept": concept_name,
            "fingerprint_length": int(len(fp)),
            "index": int(self.bodhi.fp_index["img_name_to_idx"][concept_name]),
            "emotion": emotion,
            "path": os.path.abspath(image_path),
        }

    def teach_audio(self, concept_name, audio_path, emotion="neutral",
                    description=""):
        if not HAS_SENSOR:
            raise RuntimeError("brain.sensor_wht not available. Cannot encode.")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(audio_path)

        concept_name = concept_name.strip().lower().replace(" ", "_")
        samples, sr = _load_audio_samples(audio_path)
        fp = encode_audio_to_fingerprint(samples, keep_coeffs=64)

        # Store audio fingerprint per-concept (variable length — one file each)
        if HAS_GUARD:
            out_path = os.path.join(self.audio_dir, concept_name + ".wht")
            codec_guard.save(fp.astype(np.int64), out_path)
        else:
            out_path = os.path.join(self.audio_dir, concept_name + ".npy")
            np.save(out_path, fp.astype(np.int64))

        self.concepts_meta[concept_name] = {
            "emotion": emotion,
            "source": "learned_audio",
            "description": description,
            "audio_path": os.path.abspath(audio_path),
            "audio_fp_file": out_path,
            "audio_samples": int(len(samples)),
            "audio_sr": int(sr),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_concepts()

        self.bodhi.concept_emotions[concept_name] = emotion
        self.bodhi.aliases.setdefault(concept_name, concept_name)
        if description:
            self.bodhi.engrams[concept_name] = {"capsule": description,
                                                "centroid_anchor": concept_name}

        return {
            "concept": concept_name,
            "audio_samples": int(len(samples)),
            "audio_sr": int(sr),
            "audio_fp_length": int(len(fp)),
            "emotion": emotion,
        }

    def teach_text(self, concept_name, description, emotion="neutral"):
        """Register a concept with only a text description — no fingerprint."""
        concept_name = concept_name.strip().lower().replace(" ", "_")
        if not concept_name:
            raise ValueError("concept_name required")
        self.concepts_meta[concept_name] = {
            "emotion": emotion,
            "source": "learned_text",
            "description": description,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_concepts()

        self.bodhi.concept_emotions[concept_name] = emotion
        self.bodhi.aliases.setdefault(concept_name, concept_name)
        if description:
            self.bodhi.engrams[concept_name] = {"capsule": description,
                                                "centroid_anchor": concept_name}
        return {"concept": concept_name, "emotion": emotion}

    def list_learned(self):
        return [
            {"name": n, **meta}
            for n, meta in sorted(self.concepts_meta.items())
        ]

    # ----------------------------------------------------------
    # Slash command handler
    # ----------------------------------------------------------
    def handle_command(self, text):
        """Parse a /teach command. Returns (handled, reply_text)."""
        if not text or not text.strip().startswith("/teach"):
            return False, None

        body = text.strip()[6:].strip()
        if not body or body == "list":
            learned = self.list_learned()
            if not learned:
                return True, "No learned concepts yet."
            lines = ["Learned concepts:"]
            for item in learned:
                src = item.get("source", "?")
                emo = item.get("emotion", "neutral")
                lines.append("  %s [%s, %s]" % (item["name"], src, emo))
            return True, "\n".join(lines)

        # Parse: <name> <path_or_flag> ...
        parts = body.split(None, 2)
        if len(parts) < 2:
            return True, ("Usage: /teach <name> <path>\n"
                          "       /teach <name> --text <description>\n"
                          "       /teach list")

        name = parts[0]
        arg1 = parts[1]
        rest = parts[2] if len(parts) > 2 else ""

        try:
            if arg1 == "--text":
                desc = rest if rest else ""
                r = self.teach_text(name, desc)
                return True, "Learned '%s' as a text concept." % r["concept"]
            # Else treat arg1 as a path
            path = arg1
            # Allow quotation marks
            if path.startswith('"') and path.endswith('"'):
                path = path[1:-1]

            if _is_image(path):
                r = self.teach_image(name, path)
                return True, ("Learned '%s' from image. WHT fingerprint saved "
                              "(length %d, index %d). Ask me about it."
                              % (r["concept"], r["fingerprint_length"], r["index"]))
            if _is_audio(path):
                r = self.teach_audio(name, path)
                return True, ("Learned '%s' from audio. WHT fingerprint saved "
                              "(%d samples at %d Hz, %d coeffs). Ask me about it."
                              % (r["concept"], r["audio_samples"], r["audio_sr"],
                                 r["audio_fp_length"]))
            return True, ("Unknown file type: %s. Supported: %s (image), %s (audio)."
                          % (_ext(path), IMG_EXTS, AUD_EXTS))
        except FileNotFoundError as e:
            return True, "File not found: %s" % e
        except Exception as e:
            return True, "Teach failed: %s" % e
