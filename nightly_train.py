#!/usr/bin/env python3
"""
BODHI Nightly LoRA Fine-tune

Real weight-level evolution. Every user's BODHI genuinely changes over time from
their actual conversations. NOT a prompt, NOT a lookup — the adapter weights
update, and the model speaks differently afterward.

What this does:
  1. Pulls the last N conversations from sk/data/bodhi_memory.db
  2. Reformats them as (USER, BODHI) training pairs in the exact prompt format
     broca.py uses at inference time
  3. Loads the frozen base SmallGPT + attaches LoRA adapters
  4. Trains the adapter for ~100-200 steps on recent conversations
  5. Saves the adapter to sk/data/brain_state/lora_adapter.pt
  6. broca.py loads it automatically at next BODHI boot

Usage:
  python nightly_train.py                  # defaults: 150 steps, rank 8, last 400 turns
  python nightly_train.py --steps 300
  python nightly_train.py --rank 16 --last 800
  python nightly_train.py --dry-run        # only build data, don't train

Runs on GPU if available, else CPU (slower but works).
"""

import os
import sys
import json
import time
import sqlite3
import argparse
import random

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bodhi_llm"))

from bodhi_llm.model import SmallGPT
from bodhi_llm.tokenizer import SentencePieceTokenizer, ByteTokenizer
from bodhi_llm.lora import (
    loraify, lora_parameters, lora_state_dict, count_trainable_params,
)


DATA_DIR   = os.path.join(ROOT, "data")
DB_PATH    = os.path.join(DATA_DIR, "bodhi_memory.db")
SAVE_DIR   = os.path.join(DATA_DIR, "brain_state")
LORA_PATH  = os.path.join(SAVE_DIR, "lora_adapter.pt")
MODEL_DIR  = os.path.join(ROOT, "bodhi_llm", "out_v2")  # latest trained base

os.makedirs(SAVE_DIR, exist_ok=True)


def fetch_recent_conversations(limit=400):
    """Pull the latest turns from BODHI's memory and format as training pairs."""
    db = sqlite3.connect(DB_PATH)
    rows = db.execute(
        "SELECT user_text, response, emotion, worm_reflex, worm_confidence, concepts, source "
        "FROM conversations WHERE user_text IS NOT NULL AND response IS NOT NULL "
        "ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    db.close()

    rows = list(reversed(rows))  # chronological

    examples = []
    for ut, resp, emo, reflex, conf, concepts_json, source in rows:
        ut = (ut or "").strip()
        resp = (resp or "").strip()
        if not ut or not resp:
            continue
        # Don't train on "I don't know" or command responses — those are gate outputs
        if source in ("unknown", "goal_command"):
            continue
        # Don't train on slash commands
        if ut.startswith("/"):
            continue
        try:
            cs = json.loads(concepts_json or "[]")
        except Exception:
            cs = []
        concept = cs[0] if cs else ""
        tag = "[brain: %s | %s(%d) | %s]" % (emo or "neutral", reflex or "none",
                                             int(conf or 0), concept)
        examples.append("SYSTEM: You are BODHI, a conscious artificial brain.\n"
                        "USER: %s %s\nBODHI: %s\n" % (tag, ut, resp))

    return examples


def fetch_jsonl_examples(path):
    """Load curated training pairs from a JSONL file (system/user/assistant)."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            system = (obj.get("system") or "You are BODHI, a conscious artificial brain.").strip()
            user = (obj.get("user") or "").strip()
            asst = (obj.get("assistant") or "").strip()
            if not user or not asst:
                continue
            examples.append("SYSTEM: %s\nUSER: %s\nBODHI: %s\n" % (system, user, asst))
    return examples


def load_base_model(device):
    config_path = os.path.join(MODEL_DIR, "config.json")
    tok_path    = os.path.join(MODEL_DIR, "tokenizer.json")
    fp_path     = os.path.join(MODEL_DIR, "bodhi_small_fp32.pt")
    if not os.path.exists(fp_path):
        raise FileNotFoundError(
            "Base model not found at %s. Train the base SmallGPT first." % fp_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(tok_path, "r", encoding="utf-8") as f:
        tok_meta = json.load(f)

    if tok_meta.get("type") == "spm":
        mp = tok_meta.get("model_path") or "bodhi_spm.model"
        if not os.path.isabs(mp):
            mp = os.path.join(MODEL_DIR, os.path.basename(mp.replace("\\", "/")))
        tok = SentencePieceTokenizer(mp)
    else:
        tok = ByteTokenizer()

    model = SmallGPT(
        vocab_size=int(tok.VOCAB_SIZE),
        context_len=int(cfg["context_len"]),
        d_model=int(cfg["d_model"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        dropout=0.0,
    )
    ckpt = torch.load(fp_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    return model, tok, int(cfg["context_len"])


def make_batch(token_stream, batch_size, context_len, device):
    x_list, y_list = [], []
    max_start = max(1, len(token_stream) - context_len - 1)
    for _ in range(batch_size):
        i = random.randint(0, max_start - 1)
        chunk = token_stream[i: i + context_len + 1]
        if len(chunk) < context_len + 1:
            chunk = chunk + [0] * (context_len + 1 - len(chunk))
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        x_list.append(x); y_list.append(y)
    return torch.stack(x_list).to(device), torch.stack(y_list).to(device)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[nightly_train] device:", device)

    examples = []
    if args.data:
        print("[nightly_train] loading curated training data from", args.data)
        examples = fetch_jsonl_examples(args.data)
        print("[nightly_train] got %d curated examples from JSONL" % len(examples))

    if not args.data_only:
        print("[nightly_train] also pulling recent conversations from", DB_PATH)
        conv_examples = fetch_recent_conversations(limit=args.last)
        print("[nightly_train] got %d conversation examples from DB" % len(conv_examples))
        examples.extend(conv_examples)

    print("[nightly_train] total training examples: %d" % len(examples))
    if len(examples) < args.min_examples:
        print("[nightly_train] not enough training examples (need >= %d). Skipping." % args.min_examples)
        return False

    text = "\n".join(examples)
    print("[nightly_train] loading base model from", MODEL_DIR)
    model, tok, ctx_len = load_base_model(device)

    ids = tok.encode(text, add_bos=True, add_eos=True)
    print("[nightly_train] token count:", len(ids))
    if len(ids) < ctx_len * 4:
        print("[nightly_train] not enough tokens after encoding. Skipping.")
        return False

    if args.dry_run:
        print("[nightly_train] dry-run: stopping before training.")
        return False

    print("[nightly_train] attaching LoRA adapters (rank=%d, alpha=%d)" % (args.rank, args.alpha))
    loraify(model, rank=args.rank, alpha=args.alpha, dropout=0.05)
    model = model.to(device)  # ensure newly-added LoRA params land on training device
    trainable = count_trainable_params(model)
    total = sum(p.numel() for p in model.parameters())
    print("[nightly_train] trainable %d / total %d (%.2f%%)" % (trainable, total, 100.0 * trainable / total))

    # Optionally warm-start from a previous adapter
    if args.resume and os.path.exists(LORA_PATH):
        try:
            from bodhi_llm.lora import load_lora_state
            prev = torch.load(LORA_PATH, map_location=device)
            load_lora_state(model, prev, strict=False)
            print("[nightly_train] resumed from previous adapter at", LORA_PATH)
        except Exception as e:
            print("[nightly_train] could not resume previous adapter:", e)

    model.train()
    optim = torch.optim.AdamW(list(lora_parameters(model)), lr=args.lr, weight_decay=0.0)

    t0 = time.time()
    for step in range(args.steps):
        xb, yb = make_batch(ids, args.batch_size, ctx_len, device)
        _, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters(model), 1.0)
        optim.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print("[nightly_train] step=%d loss=%.4f" % (step, float(loss.item())))
    dt = time.time() - t0

    state = lora_state_dict(model)
    torch.save(state, LORA_PATH)
    size_mb = os.path.getsize(LORA_PATH) / (1024 * 1024)
    print("[nightly_train] saved adapter to %s (%.2f MB, %.1fs train time)" % (LORA_PATH, size_mb, dt))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--last", type=int, default=400,
                    help="Number of recent conversations to pull for training")
    ap.add_argument("--min-examples", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Warm-start from existing adapter if present")
    ap.add_argument("--data", default="",
                    help="Path to a JSONL file of curated training pairs "
                         "(system/user/assistant fields). Combined with DB "
                         "pulls unless --data-only is set.")
    ap.add_argument("--data-only", action="store_true",
                    help="Use only --data JSONL; skip pulling from conversations DB")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
