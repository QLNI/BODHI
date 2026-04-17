#!/usr/bin/env python3
"""
BODHI Codec Guard — AES-256-GCM encryption for user-generated fingerprint files.

Users of BODHI should not be able to open each other's learned fingerprint
files. A trained BODHI's fingerprint library represents hours of the user's
teaching; stealing that file should not be trivial.

Files under data/learned/*.wht are encrypted with AES-256-GCM:
  - 32-byte key per-install (data/brain_state/_aes_key.bin, auto-generated)
  - 12-byte random nonce per file
  - 16-byte authentication tag (integrity check)
  - File on disk: magic(8) | nonce(12) | tag(16) | metadata-length(4) | metadata | ciphertext

Wrong key → AEAD authentication fails → InvalidTag exception, file rejected.
Anyone copying a .wht file to another machine without the key cannot read it.

NOTE: This is data-at-rest protection. Any process running on the user's own
machine with access to _aes_key.bin can still decrypt. That's expected —
BODHI needs the key to use its own fingerprints.
"""

import os
import json
import struct
import numpy as np

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


_HERE = os.path.dirname(os.path.abspath(__file__))
# Key lives in data/brain_state (per-install, per-user) — NOT next to the code,
# so the code can be open-source while the key stays private to the install.
_KEY_PATH = os.path.join(os.path.dirname(_HERE), "data", "brain_state", "_aes_key.bin")
_MAGIC = b"WHT_AES1"  # 8 bytes


def _key_bytes():
    """Load or generate the 32-byte AES key for this install."""
    if not os.path.exists(_KEY_PATH):
        os.makedirs(os.path.dirname(_KEY_PATH), exist_ok=True)
        with open(_KEY_PATH, "wb") as f:
            f.write(AESGCM.generate_key(bit_length=256))
    with open(_KEY_PATH, "rb") as f:
        key = f.read()
    if len(key) != 32:
        raise RuntimeError("AES key at %s has wrong length (%d bytes, expected 32)"
                           % (_KEY_PATH, len(key)))
    return key


def _encrypt(plaintext: bytes, associated_data: bytes = b"") -> tuple:
    aes = AESGCM(_key_bytes())
    nonce = os.urandom(12)
    ciphertext = aes.encrypt(nonce, plaintext, associated_data)
    # AESGCM.encrypt returns ciphertext||tag (tag is last 16 bytes)
    ct, tag = ciphertext[:-16], ciphertext[-16:]
    return nonce, tag, ct


def _decrypt(nonce: bytes, tag: bytes, ciphertext: bytes, associated_data: bytes = b"") -> bytes:
    aes = AESGCM(_key_bytes())
    return aes.decrypt(nonce, ciphertext + tag, associated_data)


# ============================================================
# Single-array files
# ============================================================

def save(arr: np.ndarray, path: str):
    """Save a single numpy array encrypted with AES-256-GCM."""
    meta = {"dtype": str(arr.dtype), "shape": list(arr.shape)}
    meta_bytes = json.dumps(meta).encode("utf-8")
    plaintext = arr.tobytes()
    # Bind the metadata as additional authenticated data so tampering breaks decrypt
    nonce, tag, ct = _encrypt(plaintext, associated_data=meta_bytes)
    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(nonce)
        f.write(tag)
        f.write(struct.pack("<I", len(meta_bytes)))
        f.write(meta_bytes)
        f.write(ct)


def load(path: str) -> np.ndarray:
    """Load and decrypt a single-array file saved by save()."""
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(_MAGIC):
        raise ValueError("Not an AES fingerprint file: %s" % path)
    off = len(_MAGIC)
    nonce = data[off:off + 12]; off += 12
    tag = data[off:off + 16]; off += 16
    meta_len = struct.unpack("<I", data[off:off + 4])[0]; off += 4
    meta_bytes = data[off:off + meta_len]; off += meta_len
    ciphertext = data[off:]
    plaintext = _decrypt(nonce, tag, ciphertext, associated_data=meta_bytes)
    meta = json.loads(meta_bytes.decode("utf-8"))
    arr = np.frombuffer(plaintext, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"]).copy()
    return arr


# ============================================================
# Multi-array files with per-row names (for learned image store)
# ============================================================

def save_named(arr: np.ndarray, names: list, path: str):
    """Save a [N, D] array + parallel names list, all encrypted."""
    meta = {"dtype": str(arr.dtype), "shape": list(arr.shape), "names": list(names)}
    meta_bytes = json.dumps(meta).encode("utf-8")
    plaintext = arr.tobytes()
    nonce, tag, ct = _encrypt(plaintext, associated_data=meta_bytes)
    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(nonce)
        f.write(tag)
        f.write(struct.pack("<I", len(meta_bytes)))
        f.write(meta_bytes)
        f.write(ct)


def load_named(path: str):
    """Load a multi-array+names file saved by save_named()."""
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(_MAGIC):
        raise ValueError("Not an AES fingerprint file: %s" % path)
    off = len(_MAGIC)
    nonce = data[off:off + 12]; off += 12
    tag = data[off:off + 16]; off += 16
    meta_len = struct.unpack("<I", data[off:off + 4])[0]; off += 4
    meta_bytes = data[off:off + meta_len]; off += meta_len
    ciphertext = data[off:]
    plaintext = _decrypt(nonce, tag, ciphertext, associated_data=meta_bytes)
    meta = json.loads(meta_bytes.decode("utf-8"))
    arr = np.frombuffer(plaintext, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"]).copy()
    names = list(meta.get("names", []))
    return arr, names


def is_encrypted_file(path):
    try:
        with open(path, "rb") as f:
            return f.read(len(_MAGIC)) == _MAGIC
    except Exception:
        return False


# Legacy XOR compatibility — detects old magic so existing learned files can migrate
_LEGACY_XOR_MAGIC = b"WHT_G_01"


def is_legacy_xor_file(path):
    try:
        with open(path, "rb") as f:
            return f.read(len(_LEGACY_XOR_MAGIC)) == _LEGACY_XOR_MAGIC
    except Exception:
        return False
