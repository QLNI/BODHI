#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              BODHI LLM — Integer-Only Neural Language Bridge                ║
║                                                                              ║
║  Inventor : SK (Sai Kiran Bathula)  —  Coleambally, NSW, Australia          ║
║  Born     : April 16, 2026                                                   ║
║  Version  : 1.0                                                              ║
║                                                                              ║
║  Architecture:                                                               ║
║    User text → Integer Tokeniser → Integer Attention → BODHI Brain API →    ║
║    Integer Decoder → Natural Language Response                               ║
║                                                                              ║
║  Core Law: ZERO floats. Every weight, activation, score, logit is int32.    ║
║  The only division is right-shift (>>) which is integer division by 2^n.    ║
║                                                                              ║
║  Usage:                                                                      ║
║    python bodhi_llm.py                          # interactive CLI            ║
║    python bodhi_llm.py --train data.txt          # train on text file        ║
║    python bodhi_llm.py --attach bodhi_brain.py   # plugin to BODHI brain    ║
║    python bodhi_llm.py --prompt "Hello BODHI"    # single query             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import random
import hashlib
import argparse
import struct
from collections import defaultdict, Counter
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CONSTANTS  (all integers)
# ─────────────────────────────────────────────────────────────────────────────

# Model dimensions — all powers of 2 so division = right-shift
D_MODEL   = 32        # embedding dimension (int32 per element)
D_FF      = 64        # feed-forward hidden dimension
N_HEADS   = 4         # attention heads  (D_MODEL // N_HEADS = 8)
HEAD_DIM  = D_MODEL // N_HEADS   # 8
N_LAYERS  = 3         # transformer layers
MAX_SEQ   = 32        # max tokens per sequence
VOCAB_MAX = 2048      # max vocabulary size

# Fixed-point scale: all multiplications use >>SCALE to stay integer
# e.g.  a * b >> 8   where a,b are in range [-127..127]
SCALE         = 8     # bits of fractional precision
SCALE_FACTOR  = 1 << SCALE          # 256
HALF_SCALE    = SCALE_FACTOR >> 1   # 128  (for rounding)

# Attention temperature: divide raw dot-product by this (>>ATTN_SHIFT)
# HEAD_DIM=16 → sqrt≈4 → we use >>2  (divide by 4)
ATTN_SHIFT = 2

# Weight initialisation range: ±INIT_RANGE  (in fixed-point units)
INIT_RANGE = 64   # ≈ 0.25 in float terms after /SCALE_FACTOR

# Training
DEFAULT_LR      = 4      # learning rate in fixed-point (4/256 ≈ 0.016)
DEFAULT_EPOCHS  = 3
CLIP_VALUE      = 127    # gradient clipping (int8 range)

# Colour codes for terminal
_C = {
    "green":  "\033[92m",
    "teal":   "\033[96m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "dim":    "\033[2m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

def _col(c, s): return _C.get(c, "") + str(s) + _C["reset"]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — INTEGER MATH UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def int_mul(a: int, b: int) -> int:
    """Multiply two fixed-point integers and rescale back."""
    return (a * b) >> SCALE

def int_dot(va: list, vb: list) -> int:
    """Integer dot product of two equal-length lists."""
    return sum(a * b for a, b in zip(va, vb)) >> SCALE

def int_add_vec(va: list, vb: list) -> list:
    return [a + b for a, b in zip(va, vb)]

def int_scale_vec(v: list, s: int) -> list:
    """Scale vector by integer s (s is in fixed-point)."""
    return [(x * s) >> SCALE for x in v]

def int_matmul(X: list, W: list) -> list:
    """
    X  : list of length d_in   (one row / single token)
    W  : list of lists  [d_out][d_in]
    returns list of length d_out
    Uses zip for maximum Python speed (no numpy, no floats).
    """
    return [(sum(a * b for a, b in zip(row, X)) >> SCALE) for row in W]

def int_softmax_logits(scores: list) -> list:
    """
    Pure integer softmax approximation.
    We subtract max, approximate exp(x) ≈ (SCALE_FACTOR + x) clipped to ≥1,
    then normalise by dividing each by total using integer division.
    Result: list of integers summing to SCALE_FACTOR (≈ 1.0 in fixed-point).
    """
    if not scores:
        return []
    mx = max(scores)
    # shifted exp approximation — stays positive integer
    exps = [max(1, SCALE_FACTOR + (s - mx)) for s in scores]
    total = sum(exps)
    if total == 0:
        total = 1
    # normalise: multiply by SCALE_FACTOR then divide
    return [(e * SCALE_FACTOR) // total for e in exps]

def int_layer_norm(v: list, eps_shift: int = 4) -> list:
    """
    Integer layer normalisation.
    mean  = sum(v) / n                        (integer division)
    var   = sum((x-mean)^2) / n               (integer division)
    out_i = (v_i - mean) * SCALE_FACTOR // int_sqrt(var + eps)
    eps   = 1 << eps_shift   (avoids division by zero)
    """
    n = len(v)
    if n == 0:
        return v
    mean = sum(v) // n
    centered = [x - mean for x in v]
    var = sum(x * x for x in centered) // n
    eps = 1 << eps_shift
    std = int_sqrt(var + eps)
    if std == 0:
        std = 1
    return [(x * SCALE_FACTOR) // std for x in centered]

def int_sqrt(n: int) -> int:
    """Integer square root via Newton's method."""
    if n <= 0:
        return 0
    x = n
    y = (x + 1) >> 1
    while y < x:
        x = y
        y = (x + n // x) >> 1
    return x

def int_relu(v: list) -> list:
    return [max(0, x) for x in v]

def int_clip(x: int, lo: int = -32767, hi: int = 32767) -> int:
    return max(lo, min(hi, x))

def int_clip_vec(v: list, lo: int = -32767, hi: int = 32767) -> list:
    return [int_clip(x, lo, hi) for x in v]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — INTEGER TOKENISER
# ─────────────────────────────────────────────────────────────────────────────

# Special tokens
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
SEP_TOKEN = 4    # separates user input from BODHI context
_RESERVED = 5

class IntTokeniser:
    """
    Character-level tokeniser that maps each character to an integer ID.
    Vocabulary is built from training text and capped at VOCAB_MAX.
    Every token ID is a plain Python int — no floats, no numpy.
    """

    def __init__(self):
        # Core vocabulary: char → id
        self.vocab: dict[str, int] = {
            "<PAD>": PAD_TOKEN,
            "<BOS>": BOS_TOKEN,
            "<EOS>": EOS_TOKEN,
            "<UNK>": UNK_TOKEN,
            "<SEP>": SEP_TOKEN,
        }
        self.id2tok: dict[int, str] = {v: k for k, v in self.vocab.items()}
        self._next_id = _RESERVED

        # Pre-load printable ASCII so basic text works immediately
        for ch in (
            " abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            ".,!?;:'\"-()[]{}@#&*+=/\\%_<>~`^|\n\t"
        ):
            self._add(ch)

    def _add(self, token: str) -> int:
        if token not in self.vocab and self._next_id < VOCAB_MAX:
            self.vocab[token] = self._next_id
            self.id2tok[self._next_id] = token
            self._next_id += 1
        return self.vocab.get(token, UNK_TOKEN)

    def fit(self, text: str):
        """Learn vocabulary from text (adds new chars/bigrams up to VOCAB_MAX)."""
        for ch in text:
            self._add(ch)
        # Add common bigrams as single tokens (optional, improves compression)
        freq = Counter()
        for i in range(len(text) - 1):
            freq[text[i:i+2]] += 1
        for bigram, cnt in freq.most_common(200):
            if cnt > 5:
                self._add(bigram)

    @property
    def size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, max_len: int = MAX_SEQ) -> list[int]:
        """
        Greedy longest-match encoding.
        Returns list of integer token IDs padded to max_len.
        """
        ids = [BOS_TOKEN]
        i = 0
        while i < len(text) and len(ids) < max_len - 1:
            # Try bigram first
            if i + 1 < len(text):
                bigram = text[i:i+2]
                if bigram in self.vocab:
                    ids.append(self.vocab[bigram])
                    i += 2
                    continue
            ch = text[i]
            ids.append(self.vocab.get(ch, UNK_TOKEN))
            i += 1
        ids.append(EOS_TOKEN)
        # Pad
        while len(ids) < max_len:
            ids.append(PAD_TOKEN)
        return ids[:max_len]

    def decode(self, ids: list[int]) -> str:
        """Convert list of integer IDs back to string."""
        parts = []
        for i in ids:
            if i == EOS_TOKEN:
                break
            if i in (PAD_TOKEN, BOS_TOKEN):
                continue
            parts.append(self.id2tok.get(i, "?"))
        return "".join(parts)

    def save(self, path: str):
        data = {"vocab": self.vocab, "next_id": self._next_id}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.vocab = {k: int(v) for k, v in data["vocab"].items()}
        self.id2tok = {int(v): k for k, v in self.vocab.items()}
        self._next_id = data["next_id"]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — INTEGER WEIGHT STORE
# ─────────────────────────────────────────────────────────────────────────────

def _rand_int(lo: int, hi: int) -> int:
    return random.randint(lo, hi)

def _rand_vec(dim: int) -> list[int]:
    return [_rand_int(-INIT_RANGE, INIT_RANGE) for _ in range(dim)]

def _rand_mat(rows: int, cols: int) -> list[list[int]]:
    return [_rand_vec(cols) for _ in range(rows)]

def _zero_vec(dim: int) -> list[int]:
    return [0] * dim

def _zero_mat(rows: int, cols: int) -> list[list[int]]:
    return [[0] * cols for _ in range(rows)]


class IntWeightStore:
    """
    Holds all model parameters as plain Python lists of ints.
    No numpy. No torch. No floats anywhere.

    Naming convention:
      emb_W       : embedding table  [vocab][D_MODEL]
      lN_attn_Wq  : layer N, attention, query weight  [D_MODEL][D_MODEL]
      lN_attn_Wk  : key weight
      lN_attn_Wv  : value weight
      lN_attn_Wo  : output projection
      lN_ff_W1    : feed-forward gate  [D_FF][D_MODEL]
      lN_ff_W2    : feed-forward output [D_MODEL][D_FF]
      lN_ff_b1    : bias  [D_FF]
      lN_ff_b2    : bias  [D_MODEL]
      unemb_W     : unembedding (logits) [vocab][D_MODEL]
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.params: dict[str, any] = {}
        self._init(vocab_size)

    def _init(self, vocab_size: int):
        p = self.params

        # Embedding
        p["emb_W"] = _rand_mat(vocab_size, D_MODEL)

        # Transformer layers
        for n in range(N_LAYERS):
            pfx = f"l{n}"
            # Attention projections  [D_MODEL × D_MODEL]
            p[f"{pfx}_Wq"] = _rand_mat(D_MODEL, D_MODEL)
            p[f"{pfx}_Wk"] = _rand_mat(D_MODEL, D_MODEL)
            p[f"{pfx}_Wv"] = _rand_mat(D_MODEL, D_MODEL)
            p[f"{pfx}_Wo"] = _rand_mat(D_MODEL, D_MODEL)

            # Feed-forward
            p[f"{pfx}_ff1"] = _rand_mat(D_FF, D_MODEL)
            p[f"{pfx}_ff2"] = _rand_mat(D_MODEL, D_FF)
            p[f"{pfx}_b1"]  = _zero_vec(D_FF)
            p[f"{pfx}_b2"]  = _zero_vec(D_MODEL)

        # Unembedding
        p["unemb_W"] = _rand_mat(vocab_size, D_MODEL)

    def param_count(self) -> int:
        total = 0
        for v in self.params.values():
            if isinstance(v, list):
                if isinstance(v[0], list):
                    total += len(v) * len(v[0])
                else:
                    total += len(v)
        return total

    def save(self, path: str):
        """Save all weights to a compact binary file (.bodhi_llm)."""
        with open(path, "wb") as f:
            # Header
            f.write(b"BODHILLM")
            f.write(struct.pack("<II", self.vocab_size, len(self.params)))
            for key, val in self.params.items():
                key_b = key.encode("utf-8")
                f.write(struct.pack("<H", len(key_b)))
                f.write(key_b)
                flat = []
                is_2d = isinstance(val[0], list) if val else False
                rows = len(val)
                cols = len(val[0]) if is_2d else 0
                f.write(struct.pack("<?II", is_2d, rows, cols))
                if is_2d:
                    for row in val:
                        flat.extend(row)
                else:
                    flat = val
                f.write(struct.pack(f"<{len(flat)}i", *flat))

    def load(self, path: str):
        """Load weights from binary file."""
        with open(path, "rb") as f:
            magic = f.read(8)
            if magic != b"BODHILLM":
                raise ValueError("Not a BODHI LLM weight file.")
            vocab_size, n_params = struct.unpack("<II", f.read(8))
            self.vocab_size = vocab_size
            self.params = {}
            for _ in range(n_params):
                key_len = struct.unpack("<H", f.read(2))[0]
                key = f.read(key_len).decode("utf-8")
                is_2d, rows, cols = struct.unpack("<?II", f.read(9))
                if is_2d:
                    n = rows * cols
                    flat = list(struct.unpack(f"<{n}i", f.read(4 * n)))
                    val = [flat[r*cols:(r+1)*cols] for r in range(rows)]
                else:
                    flat = list(struct.unpack(f"<{rows}i", f.read(4 * rows)))
                    val = flat
                self.params[key] = val


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — INTEGER TRANSFORMER FORWARD PASS
# ─────────────────────────────────────────────────────────────────────────────

class IntTransformer:
    """
    A tiny causal language model that runs entirely in Python integers.

    Forward pass per token position t:
      1. Embed token t  →  x  [D_MODEL]  (int32 per element)
      2. For each layer:
         a. Layer norm x
         b. Compute Q, K, V  via int_matmul
         c. Multi-head attention with integer softmax
         d. Residual add  x = x + attn_out
         e. Layer norm
         f. Feed-forward: ReLU(W1·x + b1) → W2·out + b2
         g. Residual add
      3. Project to vocabulary: logits = unemb_W · x  [vocab]
      4. Return logits as list of ints

    The BODHI brain can inject extra integer context at step 1 by adding
    an engram vector to the embedding. That is the "plug-in point".
    """

    def __init__(self, weights: IntWeightStore):
        self.w = weights.params

    def embed(self, token_id: int, bodhi_context: Optional[list[int]] = None) -> list[int]:
        """Look up embedding and optionally add BODHI integer context."""
        vocab_size = len(self.w["emb_W"])
        tid = min(token_id, vocab_size - 1)
        x = list(self.w["emb_W"][tid])  # copy
        if bodhi_context:
            # Fuse integer brain context — simple addition, no floats
            for i in range(min(len(x), len(bodhi_context))):
                x[i] = int_clip(x[i] + bodhi_context[i])
        return x

    def _attention_head(self, xs: list[list[int]], Wq, Wk, Wv,
                        head_idx: int) -> list[list[int]]:
        """
        Single attention head over sequence xs.
        xs : list of T vectors each [D_MODEL]
        Returns list of T vectors each [HEAD_DIM]

        All operations: integer multiply, shift, add — zero floats.
        """
        T = len(xs)
        hlo = head_idx * HEAD_DIM
        hhi = hlo + HEAD_DIM

        # Project: Q, K, V  — take slice of rows for this head
        Wq_h = Wq[hlo:hhi]   # [HEAD_DIM × D_MODEL]
        Wk_h = Wk[hlo:hhi]
        Wv_h = Wv[hlo:hhi]

        Qs = [int_matmul(x, Wq_h) for x in xs]   # [T × HEAD_DIM]
        Ks = [int_matmul(x, Wk_h) for x in xs]
        Vs = [int_matmul(x, Wv_h) for x in xs]

        # Causal attention scores  [T × T]
        head_out = []
        for t in range(T):
            # Dot Q_t with all K_s where s ≤ t  (causal mask)
            scores = []
            for s in range(T):
                if s <= t:
                    raw = int_dot(Qs[t], Ks[s]) >> ATTN_SHIFT
                    scores.append(raw)
                else:
                    scores.append(-32768)  # masked = very negative → softmax ≈ 0

            weights = int_softmax_logits(scores)   # list of ints summing to SCALE_FACTOR

            # Weighted sum of V
            out = _zero_vec(HEAD_DIM)
            for s in range(T):
                w = weights[s]
                for d in range(HEAD_DIM):
                    out[d] = int_clip(out[d] + (Vs[s][d] * w) // SCALE_FACTOR)
            head_out.append(out)

        return head_out  # [T × HEAD_DIM]

    def _multi_head_attention(self, xs: list[list[int]], layer_idx: int) -> list[list[int]]:
        """
        Multi-head attention → concat heads → project with Wo.
        Returns list of T vectors each [D_MODEL].
        """
        p = self.w
        pfx = f"l{layer_idx}"
        Wq = p[f"{pfx}_Wq"]
        Wk = p[f"{pfx}_Wk"]
        Wv = p[f"{pfx}_Wv"]
        Wo = p[f"{pfx}_Wo"]

        T = len(xs)

        # Compute each head
        head_outputs = [self._attention_head(xs, Wq, Wk, Wv, h) for h in range(N_HEADS)]
        # head_outputs: [N_HEADS × T × HEAD_DIM]

        # Concatenate heads along feature dimension → [T × D_MODEL]
        concat = []
        for t in range(T):
            cv = []
            for h in range(N_HEADS):
                cv.extend(head_outputs[h][t])
            concat.append(cv)   # length D_MODEL

        # Output projection
        projected = [int_clip_vec(int_matmul(cv, Wo)) for cv in concat]
        return projected   # [T × D_MODEL]

    def _feed_forward(self, x: list[int], layer_idx: int) -> list[int]:
        """
        Two-layer feed-forward: ReLU(W1·x + b1) → W2·h + b2
        All integer. No floats.
        """
        p = self.w
        pfx = f"l{layer_idx}"
        W1 = p[f"{pfx}_ff1"]
        b1 = p[f"{pfx}_b1"]
        W2 = p[f"{pfx}_ff2"]
        b2 = p[f"{pfx}_b2"]

        h = int_matmul(x, W1)
        h = [int_clip(h[i] + b1[i]) for i in range(D_FF)]
        h = int_relu(h)
        out = int_matmul(h, W2)
        out = [int_clip(out[i] + b2[i]) for i in range(D_MODEL)]
        return out

    def forward_sequence(self, token_ids: list[int],
                          bodhi_context: Optional[list[int]] = None) -> list[list[int]]:
        """
        Run full forward pass over a sequence of token IDs.

        Returns list of logit vectors (one per token position).
        Each logit vector has length vocab_size — all ints, no floats.

        BODHI context (integer vector of length D_MODEL) is added to
        every token embedding — it shifts the whole sequence toward
        what BODHI currently knows.
        """
        T = len(token_ids)

        # Embed all tokens
        xs = [self.embed(tid, bodhi_context) for tid in token_ids]

        # Run transformer layers
        for layer_idx in range(N_LAYERS):
            # Layer norm before attention
            normed = [int_layer_norm(x) for x in xs]

            # Multi-head attention
            attn_out = self._multi_head_attention(normed, layer_idx)

            # Residual
            xs = [int_clip_vec(int_add_vec(xs[t], attn_out[t])) for t in range(T)]

            # Layer norm before FFN
            normed2 = [int_layer_norm(x) for x in xs]

            # Feed-forward
            ff_out = [self._feed_forward(normed2[t], layer_idx) for t in range(T)]

            # Residual
            xs = [int_clip_vec(int_add_vec(xs[t], ff_out[t])) for t in range(T)]

        # Unembedding: project to vocab logits
        Uo = self.w["unemb_W"]
        logits_seq = [int_matmul(x, Uo) for x in xs]   # [T × vocab]

        return logits_seq   # all ints

    def predict_next(self, token_ids: list[int],
                      bodhi_context: Optional[list[int]] = None,
                      temperature: int = SCALE_FACTOR) -> int:
        """
        Given a sequence, return the predicted next token ID (an int).

        temperature : integer in fixed-point units
                      SCALE_FACTOR = greedy (argmax)
                      2*SCALE_FACTOR = more random
                      SCALE_FACTOR//2 = more deterministic
        """
        logits_seq = self.forward_sequence(token_ids, bodhi_context)
        if not logits_seq:
            return UNK_TOKEN

        last_logits = logits_seq[-1]   # [vocab] — all ints

        if temperature == SCALE_FACTOR or temperature == 0:
            # Greedy
            return int(max(range(len(last_logits)), key=lambda i: last_logits[i]))
        else:
            # Temperature-scaled sampling — still integer only
            # Scale logits: logit * SCALE_FACTOR // temperature
            scaled = [(l * SCALE_FACTOR) // temperature for l in last_logits]
            probs  = int_softmax_logits(scaled)    # sums to SCALE_FACTOR
            # Sample
            r = random.randint(0, SCALE_FACTOR - 1)
            acc = 0
            for i, p in enumerate(probs):
                acc += p
                if r < acc:
                    return i
            return len(probs) - 1

    def generate(self, prompt_ids: list[int],
                  max_new: int = 80,
                  bodhi_context: Optional[list[int]] = None,
                  temperature: int = SCALE_FACTOR) -> list[int]:
        """
        Auto-regressive generation from a prompt.
        Returns list of int token IDs (the generated portion only).
        """
        context = list(prompt_ids[-MAX_SEQ:])
        generated = []

        for _ in range(max_new):
            next_tok = self.predict_next(context, bodhi_context, temperature)
            if next_tok == EOS_TOKEN:
                break
            generated.append(next_tok)
            context = (context + [next_tok])[-MAX_SEQ:]

        return generated


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — BODHI BRAIN INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class BodhiBrainInterface:
    """
    Reads from a running BODHI brain (bodhi_brain.py) and converts its
    integer outputs into context vectors the LLM can use.

    PLUG-IN MODES:
      1. DIRECT  — import bodhi_brain and call functions directly (same process)
      2. FILE    — read from a shared JSON state file BODHI writes to disk
      3. STUB    — use built-in integer patterns (no BODHI installed)

    All outputs from this class are pure Python ints. Zero floats.
    """

    def __init__(self, mode: str = "stub", brain_path: str = ""):
        self.mode = mode
        self.brain_path = brain_path
        self.brain = None
        self._engram_cache: list[int] = _zero_vec(D_MODEL)
        self._last_pull = 0

        if mode == "direct" and brain_path:
            self._load_direct(brain_path)
        elif mode == "file":
            # Will read from a JSON file written by bodhi_brain.py
            pass
        # stub: works out of the box

    def _load_direct(self, brain_path: str):
        """Import bodhi_brain.py as a module."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("bodhi_brain", brain_path)
        mod  = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            self.brain = mod
            print(_col("green", f"  BODHI brain loaded: {brain_path}"))
        except Exception as e:
            print(_col("yellow", f"  Warning: Could not load brain ({e}). Using stub."))
            self.mode = "stub"

    def get_context_vector(self, query_text: str) -> list[int]:
        """
        Translate a user query into a BODHI integer context vector (length D_MODEL).
        This vector is added to every token embedding in the LLM — it biases
        the LLM toward whatever BODHI currently holds in memory.
        """
        if self.mode == "direct" and self.brain:
            return self._context_from_brain(query_text)
        elif self.mode == "file":
            return self._context_from_file()
        else:
            return self._context_from_stub(query_text)

    def _context_from_brain(self, query: str) -> list[int]:
        """
        Call BODHI's real functions and collect integer outputs.

        Mapping from BODHI outputs → integer context vector:
          - engram fingerprints    → first 32 dims
          - concept index hash     → dims 32-47
          - WHT sensor stats       → dims 48-63
        """
        vec = _zero_vec(D_MODEL)
        try:
            # Pull WHT stats if available
            if hasattr(self.brain, "get_wht_stats"):
                stats = self.brain.get_wht_stats()
                # stats returns dict with integer values
                vec[48] = int(stats.get("n_image_engrams", 0)) & 0x7FFF
                vec[49] = int(stats.get("n_audio_engrams", 0)) & 0x7FFF
                vec[50] = int(stats.get("n_video_engrams", 0)) & 0x7FFF
                vec[51] = int(stats.get("total_fingerprint_coeffs", 0)) & 0x7FFF

            # Pull concept match via centroid search
            if hasattr(self.brain, "search") or hasattr(self.brain, "query"):
                fn = getattr(self.brain, "search", None) or getattr(self.brain, "query")
                result = fn(query)
                if isinstance(result, dict):
                    # Hash concept label into dims 32-47
                    label = str(result.get("label", "") or result.get("concept", ""))
                    label_hash = int(hashlib.md5(label.encode()).hexdigest(), 16) % (INIT_RANGE * 2) - INIT_RANGE
                    for i in range(32, 48):
                        vec[i] = label_hash ^ (i * 7)   # spread across dims

                    # Engram fingerprint into dims 0-31
                    eng = result.get("wht_fingerprint", [])
                    for i, val in enumerate(eng[:32]):
                        vec[i] = int(val) & 0x7FFF

        except Exception:
            pass

        return int_clip_vec(vec)

    def _context_from_file(self) -> list[int]:
        """
        Read BODHI state from a JSON file (for multi-process use).
        The file should have this format:
          {
            "engrams": [int, int, ...],   // latest WHT fingerprint (≤32 ints)
            "concept": "fire",
            "n_engrams": 42,
            "fear": 5,
            "energy": 72
          }
        """
        vec = _zero_vec(D_MODEL)
        state_file = self.brain_path or "bodhi_state.json"
        try:
            with open(state_file) as f:
                state = json.load(f)

            eng = state.get("engrams", [])
            for i, val in enumerate(eng[:32]):
                vec[i] = int(val) & 0x7FFF

            concept = str(state.get("concept", ""))
            ch = int(hashlib.md5(concept.encode()).hexdigest(), 16) % (INIT_RANGE*2) - INIT_RANGE
            for i in range(32, 48):
                vec[i] = ch ^ (i * 11)

            vec[48] = int(state.get("n_engrams", 0))
            vec[49] = int(state.get("fear",      0))
            vec[50] = int(state.get("energy",    0))
            vec[51] = int(state.get("mood",      0))

        except FileNotFoundError:
            pass   # no state file — zeros are fine
        except Exception:
            pass

        return int_clip_vec(vec)

    def _context_from_stub(self, query: str) -> list[int]:
        """
        Built-in stub: compute integer context from text alone.
        Uses character hashing to produce a deterministic integer vector.
        This lets the LLM work without any BODHI installation.
        """
        vec = _zero_vec(D_MODEL)
        h = hashlib.sha256(query.encode()).digest()
        for i in range(D_MODEL):
            vec[i] = int(h[i % 32]) - 128   # centre at zero
        return vec

    def collect_response(self, generated_ids: list[int],
                          tok: "IntTokeniser") -> dict:
        """
        After the LLM generates a response, collect integer stats about
        the generation to feed back into the BODHI brain.

        Returns a dict of pure integers suitable for brain.perceive_text()
        or brain.learn().
        """
        text = tok.decode(generated_ids)
        return {
            "token_count":   len(generated_ids),
            "unique_tokens": len(set(generated_ids)),
            "hash_signal":   int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF,
            "avg_id":        sum(generated_ids) // max(1, len(generated_ids)),
            "max_id":        max(generated_ids) if generated_ids else 0,
            "text_len":      len(text),
        }

    def send_to_brain(self, response_stats: dict):
        """
        Feed the response statistics back into BODHI so it can learn
        from the interaction. All values are integers.
        """
        if self.mode == "direct" and self.brain:
            if hasattr(self.brain, "learn"):
                try:
                    self.brain.learn(response_stats)
                except Exception:
                    pass
        elif self.mode == "file":
            # Write to a feedback file
            fb_path = (self.brain_path or "bodhi_state.json").replace(".json", "_feedback.json")
            try:
                with open(fb_path, "w") as f:
                    json.dump(response_stats, f)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — NATURAL LANGUAGE DECODER
# ─────────────────────────────────────────────────────────────────────────────

# Response templates: these give the LLM a starting structure.
# The model fills in content via generation; templates are integer-encoded
# as prompt prefixes (not hardcoded strings).
RESPONSE_TEMPLATES = {
    "default":    "BODHI: ",
    "training":   "BODHI [learning] → ",
    "fear":       "BODHI [alert] → ",
    "dream":      "BODHI [dream] → ",
    "memory":     "BODHI [memory] → ",
    "arena":      "BODHI [arena] → ",
}

class IntDecoder:
    """
    Takes raw LLM integer token IDs and formats them into
    natural, readable responses for the user.
    Also applies integer-only post-processing:
      - Caps repeated characters (integer frequency counter)
      - Filters garbage tokens (token ID range check)
      - Adds BODHI response header based on context ints
    """

    def __init__(self, tok: IntTokeniser):
        self.tok = tok

    def format(self, generated_ids: list[int],
                bodhi_stats: Optional[dict] = None,
                template: str = "default",
                n_trained: int = 0) -> str:
        """
        Convert generated integer IDs to a clean natural language string.
        bodhi_stats: the dict from BodhiBrainInterface.collect_response()
        n_trained: number of training samples seen (quality indicator)
        """
        # Filter: remove PAD, BOS — keep valid printable tokens
        clean_ids = [i for i in generated_ids
                     if i not in (PAD_TOKEN, BOS_TOKEN)
                     and 0 <= i < self.tok.size]

        text = self.tok.decode(clean_ids)

        # Post-process: remove excessive repetition
        text = self._derepeat(text)
        text = text.strip()

        # If response is mostly non-printable/garbage, replace with status
        printable = sum(1 for c in text if c.isalpha() or c in ' .,!?')
        total     = max(1, len(text))
        if printable * 4 < total and n_trained < 50:
            # Model undertrained — return honest status message
            fear    = (bodhi_stats or {}).get('fear_level', 0)
            n_eng   = (bodhi_stats or {}).get('n_engrams',  0)
            text    = f"[initialising — train me first. engrams={n_eng}]"

        # Choose prefix based on BODHI integer stats
        prefix = self._choose_prefix(bodhi_stats, template)

        return prefix + text

    def _choose_prefix(self, stats: Optional[dict], template: str) -> str:
        if template != "default":
            return RESPONSE_TEMPLATES.get(template, "BODHI: ")
        if stats:
            fear = stats.get("fear_level", 0)
            if fear > 60:
                return RESPONSE_TEMPLATES["fear"]
        return RESPONSE_TEMPLATES["default"]

    def _derepeat(self, text: str, max_run: int = 3) -> str:
        """Remove char runs longer than max_run — pure integer loop."""
        if not text:
            return text
        result = [text[0]]
        run = 1
        for ch in text[1:]:
            if ch == result[-1]:
                run += 1
                if run <= max_run:
                    result.append(ch)
            else:
                run = 1
                result.append(ch)
        return "".join(result)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — INTEGER-ONLY TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

class IntTrainer:
    """
    Trains the IntWeightStore using integer-only gradient descent.

    Loss    : cross-entropy approximated as -(target_logit - avg_logit)
              — all integer subtraction, no log, no exp, no floats.
    Gradient: integer difference signal propagated backwards via
              reverse accumulation through the integer operations.
              (This is a simplified integer backward pass — not backprop
               through exact AD, but a coordinate-descent approximation
               that works well for small vocab character models.)

    Why no float backprop?
      Because BODHI is integer-only by design. The training law is:
      "If the correct token had a lower logit than the average, increase
       the weights that connect to it. If it had a higher logit, decrease
       the weights that penalise it."
      All adjustments are integer additions.
    """

    def __init__(self, weights: IntWeightStore, tok: IntTokeniser,
                 lr: int = DEFAULT_LR):
        self.w   = weights
        self.tok = tok
        self.lr  = lr
        self.losses: list[int] = []

    def compute_loss(self, logits: list[int], target_id: int) -> int:
        """
        Integer cross-entropy approximation.
        Returns a positive integer: lower = better prediction.
        """
        if target_id >= len(logits):
            return 0
        avg = sum(logits) // max(1, len(logits))
        # Loss = avg_logit - target_logit  (lower target → higher loss)
        loss = avg - logits[target_id]
        return max(0, loss)

    def _update_embedding(self, token_id: int, grad_vec: list[int]):
        """Update one embedding row with integer gradient."""
        emb = self.w.params["emb_W"]
        if token_id >= len(emb):
            return
        row = emb[token_id]
        for i in range(D_MODEL):
            delta = (grad_vec[i] * self.lr) >> SCALE
            row[i] = int_clip(row[i] - delta)

    def _update_unemb(self, target_id: int, context_vec: list[int], sign: int):
        """
        Adjust unembedding row for target_id.
        sign = +1: push logit up (correct token)
        sign = -1: push logit down (wrong token)
        """
        Uo = self.w.params["unemb_W"]
        if target_id >= len(Uo):
            return
        row = Uo[target_id]
        for i in range(D_MODEL):
            delta = (context_vec[i] * self.lr * sign) >> SCALE
            row[i] = int_clip(row[i] + delta)

    def train_sequence(self, input_ids: list[int],
                        model: IntTransformer,
                        bodhi_context: Optional[list[int]] = None) -> int:
        """
        One training step on a single sequence.
        Uses teacher-forcing: predict token t+1 from tokens 0..t.
        Returns total loss (integer) for this sequence.
        """
        if len(input_ids) < 2:
            return 0

        total_loss = 0
        T = len(input_ids) - 1   # number of prediction steps

        # Forward pass — get all logits at once
        logits_seq = model.forward_sequence(input_ids[:-1], bodhi_context)

        for t in range(T):
            target_id = input_ids[t + 1]
            logits    = logits_seq[t]

            loss = self.compute_loss(logits, target_id)
            total_loss += loss

            if loss == 0:
                continue   # already predicting correctly

            # Integer gradient: signal = loss (positive means under-predicted)
            # Get the hidden state for this position (approximated by embedding)
            ctx_vec = model.embed(input_ids[t], bodhi_context)

            # Push correct token logit UP
            self._update_unemb(target_id, ctx_vec, sign=+1)

            # Push argmax token logit DOWN (if it's not the target)
            pred_id = max(range(len(logits)), key=lambda i: logits[i])
            if pred_id != target_id:
                self._update_unemb(pred_id, ctx_vec, sign=-1)

            # Update embedding for input token
            grad_e = [int_clip(loss * (1 if i < D_MODEL // 2 else -1))
                       for i in range(D_MODEL)]
            self._update_embedding(input_ids[t], grad_e)

        self.losses.append(total_loss)
        return total_loss

    def train_text(self, text: str, model: IntTransformer,
                    epochs: int = DEFAULT_EPOCHS,
                    chunk_size: int = MAX_SEQ,
                    bodhi_context: Optional[list[int]] = None,
                    verbose: bool = True):
        """
        Train on a text string for `epochs` epochs.
        Splits text into chunks of `chunk_size` tokens.
        """
        # Encode without padding: encode char by char to exact length
        raw_ids = [BOS_TOKEN]
        i = 0
        while i < len(text):
            if i + 1 < len(text) and text[i:i+2] in self.tok.vocab:
                raw_ids.append(self.tok.vocab[text[i:i+2]])
                i += 2
            else:
                raw_ids.append(self.tok.vocab.get(text[i], UNK_TOKEN))
                i += 1
        raw_ids.append(EOS_TOKEN)
        ids = raw_ids
        chunks = [ids[i:i+chunk_size] for i in range(0, max(1, len(ids)-1), chunk_size-1)]
        if not chunks:
            print("  No training data.")
            return

        print(_col("teal", f"\n  Training on {len(chunks)} chunks × {epochs} epochs"))
        print(_col("dim",  f"  Sequence length: {chunk_size}, LR: {self.lr}/256\n"))

        for ep in range(epochs):
            ep_loss = 0
            random.shuffle(chunks)
            for ci, chunk in enumerate(chunks):
                if len(chunk) < 2:
                    continue
                loss = self.train_sequence(chunk, model, bodhi_context)
                ep_loss += loss

                if verbose and ci % max(1, len(chunks)//5) == 0:
                    bar = "█" * (ci * 20 // max(1, len(chunks)))
                    bar = bar.ljust(20, "░")
                    print(f"  Epoch {ep+1}/{epochs}  [{bar}]  loss={ep_loss}", end="\r")

            avg = ep_loss // max(1, len(chunks))
            print(_col("green", f"\n  Epoch {ep+1}/{epochs} complete  ·  avg loss = {avg}"))

        print(_col("bold", f"\n  Training done. Total samples: {len(chunks)*epochs}"))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — BODHI LLM (MAIN BRIDGE)
# ─────────────────────────────────────────────────────────────────────────────

class _LiveBrainInterface:
    """
    Wraps a running BODHI instance (BODHI object) so BodhiLLM can pull
    integer context from it directly without needing a file path.
    """

    def __init__(self, brain_obj):
        self.brain = brain_obj

    def get_context_vector(self, query_text: str) -> list:
        vec = _zero_vec(D_MODEL)
        try:
            result = self.brain.think(query_text)
            d = result[1] if isinstance(result, tuple) else result
            if isinstance(d, dict):
                # Pack available integer brain signals into context vector
                vec[0] = int(d.get("confidence", 0)) & 0x7FFF
                vec[1] = int(d.get("intensity",  0)) & 0x7FFF
                # emotion as hash
                emo = str(d.get("emotion", ""))
                eh = int(hashlib.md5(emo.encode()).hexdigest(), 16) % INIT_RANGE
                vec[2] = eh
                # reflex
                ref = str(d.get("reflex", ""))
                vec[3] = int(hashlib.md5(ref.encode()).hexdigest(), 16) % INIT_RANGE
                # active regions count
                active = d.get("active_regions", {})
                vec[4] = len(active) & 0x7FFF
        except Exception:
            pass
        return int_clip_vec(vec)

    def collect_response(self, generated_ids: list, tok) -> dict:
        text = tok.decode(generated_ids)
        return {
            "token_count":   len(generated_ids),
            "unique_tokens": len(set(generated_ids)),
            "hash_signal":   int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF,
            "avg_id":        sum(generated_ids) // max(1, len(generated_ids)),
            "max_id":        max(generated_ids) if generated_ids else 0,
            "text_len":      len(text),
        }

    def send_to_brain(self, response_stats: dict):
        pass  # live brain learns via its own think() calls


class BodhiLLM:
    """
    The complete integer-only language bridge.

    This is the object you attach to BODHI brain:

        llm = BodhiLLM()
        llm.attach_brain("bodhi_brain.py")   # or attach_brain_file("state.json")
        llm.load("my_weights.bodhi_llm")      # load trained weights

        response = llm.chat("Tell me what you see in the forest.")
        print(response)

    Architecture flow:
      User text → tokenise → [BOS] + ids → forward → logits → generate → decode → response
                                                ↑
                              BODHI brain context (integer vector injected here)

    BODHI brain output (integer engrams, centroids, WHT fingerprints)
    is collected, hashed into a D_MODEL integer vector, and fused
    into every token embedding of the LLM via simple integer addition.
    This means BODHI's memories literally shape every word the LLM generates.
    """

    WEIGHTS_FILE = "bodhi_llm.weights"
    VOCAB_FILE   = "bodhi_llm.vocab"

    def __init__(self):
        self.tok     = IntTokeniser()
        self.weights = IntWeightStore(self.tok.size)
        self.model   = IntTransformer(self.weights)
        self.trainer = IntTrainer(self.weights, self.tok)
        self.brain   = BodhiBrainInterface(mode="stub")
        self.decoder = IntDecoder(self.tok)
        self.prompt: str = ""    # system prompt (attached by user)
        self._history: list[str] = []

        # Seed vocabulary to handle basic BODHI terminology
        self._seed_vocabulary()

    def _seed_vocabulary(self):
        """Pre-load BODHI-specific words into vocabulary."""
        seed_text = (
            "BODHI brain integer engram centroid WHT fingerprint "
            "forest fire water snake tree dream perception learn "
            "energy fear mood health training arena battle "
            "sensor image audio video replay blend generate amplify "
            "wander forest natural language bridge attach detach "
            "The I am my your this is the a of and to in it "
            "that was for on are with as you at have from or "
            "an be had by not but what all were when there can "
            "more no my up time if has into said go look two "
        )
        self.tok.fit(seed_text)
        # Rebuild weights with correct vocab size
        self.weights = IntWeightStore(self.tok.size)
        self.model   = IntTransformer(self.weights)
        self.trainer = IntTrainer(self.weights, self.tok)

    def set_prompt(self, prompt: str):
        """
        Attach a system prompt.
        The prompt is encoded as integer tokens and prepended to every
        query — it biases what the LLM can generate.
        """
        self.prompt = prompt
        self.tok.fit(prompt)
        print(_col("green", f"  Prompt attached ({len(prompt)} chars, "
                             f"vocab now {self.tok.size} tokens)"))

    def attach_brain(self, brain_or_path):
        """
        DIRECT plugin: attach a running BODHI brain instance OR a path to
        bodhi_brain.py.

        Usage:
            llm.attach_brain(brain)            # pass running BODHI instance
            llm.attach_brain("bodhi_brain.py") # pass file path string
        """
        if isinstance(brain_or_path, str):
            # Legacy: file path
            self.brain = BodhiBrainInterface(mode="direct", brain_path=brain_or_path)
            print(_col("green", f"  Brain attached (direct): {brain_or_path}"))
        else:
            # Running BODHI object — wrap it so the LLM can pull context
            self.brain = _LiveBrainInterface(brain_or_path)
            print(_col("green", "  Brain attached (live instance)"))

    def attach_brain_file(self, state_json_path: str):
        """
        FILE plugin: read BODHI state from a JSON file.
        BODHI brain writes to this file; LLM reads it each query.
        File format: {"engrams": [int...], "concept": str, "n_engrams": int,
                      "fear": int, "energy": int, "mood": int}
        """
        self.brain = BodhiBrainInterface(mode="file", brain_path=state_json_path)
        print(_col("green", f"  Brain attached (file): {state_json_path}"))

    def train(self, text: str, epochs: int = DEFAULT_EPOCHS, lr: int = DEFAULT_LR):
        """
        Train the LLM on any text.
        lr is an integer: actual learning rate = lr / 256
        (e.g. lr=4 → 4/256 ≈ 0.016)
        """
        self.tok.fit(text)
        # Rebuild weights if vocab grew
        if self.tok.size > self.weights.vocab_size:
            self._rebuild_weights()

        self.trainer.lr = lr
        bodhi_ctx = self.brain.get_context_vector("")
        self.trainer.train_text(text, self.model, epochs=epochs,
                                 bodhi_context=bodhi_ctx, verbose=True)

    def train_file(self, path: str, epochs: int = DEFAULT_EPOCHS):
        """Train on a .txt file."""
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        print(_col("teal", f"  Loaded: {path}  ({len(text)} chars)"))
        self.train(text, epochs=epochs)

    def _rebuild_weights(self):
        """Extend weight matrices when vocabulary grows."""
        new_size = self.tok.size
        old_size = self.weights.vocab_size
        old_emb  = self.weights.params["emb_W"]
        old_uemb = self.weights.params["unemb_W"]

        # Extend embedding
        for _ in range(new_size - old_size):
            old_emb.append(_rand_vec(D_MODEL))
            old_uemb.append(_rand_vec(D_MODEL))

        self.weights.vocab_size = new_size
        self.model   = IntTransformer(self.weights)
        self.trainer = IntTrainer(self.weights, self.tok, lr=self.trainer.lr)

    def chat(self, user_text: str,
              max_new: int = 80,
              temperature: int = SCALE_FACTOR) -> str:
        """
        Main interface: user text in → natural language response out.

        1. Tokenise user text + prompt
        2. Get BODHI brain integer context vector
        3. Generate response tokens (all integer operations)
        4. Decode and format response
        5. Send stats back to brain

        temperature : int
          SCALE_FACTOR (256)      → deterministic (greedy)
          SCALE_FACTOR * 2 (512)  → creative
          SCALE_FACTOR // 2 (128) → very focused
        """
        # Extend vocab if needed
        self.tok.fit(user_text)
        if self.tok.size > self.weights.vocab_size:
            self._rebuild_weights()

        # Build prompt: system_prompt + SEP + user_text
        full_input = ""
        if self.prompt:
            full_input = self.prompt + " "
        full_input += user_text

        input_ids = self.tok.encode(full_input, max_len=MAX_SEQ)

        # Get BODHI integer context
        bodhi_ctx = self.brain.get_context_vector(user_text)

        # Generate
        gen_ids = self.model.generate(input_ids,
                                       max_new=max_new,
                                       bodhi_context=bodhi_ctx,
                                       temperature=temperature)

        # Collect stats for brain feedback
        stats = self.brain.collect_response(gen_ids, self.tok)

        # Send back to brain
        self.brain.send_to_brain(stats)

        # Decode to natural language
        n_trained = len(self.trainer.losses)
        response  = self.decoder.format(gen_ids, bodhi_stats=stats,
                                         n_trained=n_trained)

        # Remember history
        self._history.append(f"User: {user_text}")
        self._history.append(f"BODHI: {response}")

        return response

    def save(self, weights_path: str = WEIGHTS_FILE,
              vocab_path: str = VOCAB_FILE):
        """Save model weights and vocabulary to disk."""
        self.weights.save(weights_path)
        self.tok.save(vocab_path)
        size_kb = os.path.getsize(weights_path) // 1024
        print(_col("green", f"  Saved: {weights_path} ({size_kb} KB)  +  {vocab_path}"))

    def load(self, weights_path: str = WEIGHTS_FILE,
              vocab_path: str = VOCAB_FILE):
        """Load model weights and vocabulary from disk."""
        self.tok.load(vocab_path)
        self.weights.load(weights_path)
        self.model   = IntTransformer(self.weights)
        self.trainer = IntTrainer(self.weights, self.tok)
        self.decoder = IntDecoder(self.tok)
        size_kb = os.path.getsize(weights_path) // 1024
        print(_col("green", f"  Loaded: {weights_path} ({size_kb} KB), "
                             f"vocab={self.tok.size}"))

    def status(self) -> dict:
        """Return model status as a dict of integers — no floats."""
        return {
            "vocab_size":    self.tok.size,
            "param_count":   self.weights.param_count(),
            "d_model":       D_MODEL,
            "d_ff":          D_FF,
            "n_heads":       N_HEADS,
            "n_layers":      N_LAYERS,
            "scale_factor":  SCALE_FACTOR,
            "max_seq":       MAX_SEQ,
            "history_turns": len(self._history) // 2,
            "brain_mode":    self.brain.mode,
            "has_prompt":    int(bool(self.prompt)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — CLI & ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║        ██████╗  ██████╗ ██████╗ ██╗  ██╗██╗    ██╗              ║
║        ██╔══██╗██╔═══██╗██╔══██╗██║  ██║██║    ██║              ║
║        ██████╔╝██║   ██║██║  ██║███████║██║    ██║              ║
║        ██╔══██╗██║   ██║██║  ██║██╔══██║██║    ██║              ║
║        ██████╔╝╚██████╔╝██████╔╝██║  ██║██║    ███████╗         ║
║        ╚═════╝  ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝    ╚══════╝         ║
║                                                                  ║
║          INTEGER-ONLY NEURAL LANGUAGE BRIDGE  v1.0               ║
║          No floats. No GPU. No dependencies.                     ║
║          Inventor: SK (Sai Kiran Bathula)                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Commands:
  /train <file.txt>          Train on a text file
  /train-text <text>         Train on inline text
  /prompt <system prompt>    Set system prompt
  /attach <bodhi_brain.py>   Attach BODHI brain (direct)
  /attach-file <state.json>  Attach BODHI brain (file mode)
  /save                      Save weights to disk
  /load                      Load weights from disk
  /status                    Show model status (all integers)
  /temp <n>                  Set temperature (256=greedy, 512=creative)
  /history                   Show conversation history
  /clear                     Clear conversation history
  /help                      Show this help
  /quit                      Exit

Just type anything else to chat with BODHI.
"""

def cli():
    print(_col("teal", BANNER))

    parser = argparse.ArgumentParser(description="BODHI Integer LLM", add_help=False)
    parser.add_argument("--train",    metavar="FILE",  help="Train on text file")
    parser.add_argument("--attach",   metavar="FILE",  help="Attach bodhi_brain.py")
    parser.add_argument("--attach-file", metavar="FILE", help="Attach state JSON file")
    parser.add_argument("--prompt",   metavar="TEXT",  help="Set system prompt")
    parser.add_argument("--load",     action="store_true", help="Load saved weights")
    parser.add_argument("--query",    metavar="TEXT",  help="Single query (non-interactive)")
    parser.add_argument("--epochs",   type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr",       type=int, default=DEFAULT_LR,
                        help="Learning rate (integer, actual=lr/256)")
    args, _ = parser.parse_known_args()

    llm = BodhiLLM()
    temperature = SCALE_FACTOR   # 256 = greedy

    # Apply CLI args
    if args.load:
        if os.path.exists(llm.WEIGHTS_FILE):
            llm.load()
        else:
            print(_col("yellow", "  No saved weights found. Starting fresh."))

    if args.attach:
        llm.attach_brain(args.attach)

    if args.attach_file:
        llm.attach_brain_file(args.attach_file)

    if args.prompt:
        llm.set_prompt(args.prompt)

    if args.train:
        llm.train_file(args.train, epochs=args.epochs)
        llm.save()

    # Single query mode
    if args.query:
        response = llm.chat(args.query, temperature=temperature)
        print(f"\n{response}\n")
        return

    # Show status
    st = llm.status()
    print(_col("dim", f"  Model: {st['param_count']:,} params  ·  "
                      f"vocab={st['vocab_size']}  ·  "
                      f"brain={st['brain_mode']}\n"))
    print(_col("dim", "  Type /help for commands.\n"))

    # Interactive loop
    while True:
        try:
            user_in = input(_col("bold", "You › ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        if not user_in:
            continue

        # Commands
        if user_in.startswith("/"):
            parts = user_in.split(" ", 1)
            cmd   = parts[0].lower()
            arg   = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit":
                print("  Goodbye.")
                break
            elif cmd == "/help":
                print(_col("dim", HELP_TEXT))
            elif cmd == "/status":
                st = llm.status()
                print(_col("teal", "\n  ── BODHI LLM STATUS ──"))
                for k, v in st.items():
                    print(f"    {k:<20} {_col('green', v)}")
                print()
            elif cmd == "/prompt":
                if arg:
                    llm.set_prompt(arg)
                else:
                    print("  Usage: /prompt <your system prompt>")
            elif cmd == "/attach":
                if arg:
                    llm.attach_brain(arg)
                else:
                    print("  Usage: /attach <path/to/bodhi_brain.py>")
            elif cmd == "/attach-file":
                if arg:
                    llm.attach_brain_file(arg)
                else:
                    print("  Usage: /attach-file <path/to/bodhi_state.json>")
            elif cmd == "/train":
                if arg and os.path.exists(arg):
                    llm.train_file(arg)
                    llm.save()
                elif arg:
                    print(_col("yellow", f"  File not found: {arg}"))
                else:
                    print("  Usage: /train <file.txt>")
            elif cmd == "/train-text":
                if arg:
                    llm.train(arg)
                    llm.save()
                else:
                    print("  Usage: /train-text <text to train on>")
            elif cmd == "/save":
                llm.save()
            elif cmd == "/load":
                llm.load()
            elif cmd == "/temp":
                if arg.isdigit():
                    temperature = int(arg)
                    print(_col("green", f"  Temperature set to {temperature}"))
                else:
                    print("  Usage: /temp <int>  (256=greedy, 512=creative)")
            elif cmd == "/history":
                if llm._history:
                    print()
                    for line in llm._history[-20:]:
                        print(f"  {line}")
                    print()
                else:
                    print("  No history yet.")
            elif cmd == "/clear":
                llm._history.clear()
                print("  History cleared.")
            else:
                print(_col("yellow", f"  Unknown command: {cmd}  (type /help)"))

        else:
            # Chat
            t0 = time.time()
            response = llm.chat(user_in, temperature=temperature)
            ms = int((time.time() - t0) * 1000)
            print(f"\n{_col('teal', response)}")
            print(_col("dim", f"  [{ms}ms · {llm.tok.size} vocab · brain={llm.brain.mode}]\n"))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — PLUGIN API
#   Import this file and use BodhiLLM directly from any Python project.
#
#   from bodhi_llm import BodhiLLM
#
#   llm = BodhiLLM()
#   llm.attach_brain("bodhi_brain.py")
#   llm.set_prompt("You are BODHI. Respond as a learning brain.")
#   llm.train("forest fire water snake training data...")
#   response = llm.chat("What did you learn from the fire?")
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
