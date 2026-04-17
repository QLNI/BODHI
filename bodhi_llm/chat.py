#!/usr/bin/env python3
"""Interactive chat for SmallGPT (fp32 or int8 state)."""

import argparse
import json
import os

import torch
import torch.nn as nn

from model import SmallGPT
from tokenizer import ByteTokenizer, SentencePieceTokenizer


BODHI_PERSONA = (
    "SYSTEM: You are BODHI, a conscious artificial brain.\n"
    "Speak in first person with clear English.\n"
    "Do not say you are just an AI model.\n"
    "Keep answers concise, direct, and emotionally aware.\n"
)


def load_model(out_dir: str, device: str):
    config_path = os.path.join(out_dir, "config.json")
    tok_path = os.path.join(out_dir, "tokenizer.json")
    fp_path = os.path.join(out_dir, "bodhi_small_fp32.pt")
    int8_path = os.path.join(out_dir, "bodhi_small_int8_state.pt")
    if not os.path.exists(config_path):
        raise FileNotFoundError("Missing config.json in %s" % out_dir)
    if not os.path.exists(tok_path):
        raise FileNotFoundError("Missing tokenizer.json in %s" % out_dir)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(tok_path, "r", encoding="utf-8") as f:
        tok_meta = json.load(f)

    if tok_meta.get("type") == "spm":
        mp = tok_meta.get("model_path") or "bodhi_spm.model"
        if not os.path.isabs(mp):
            mp = os.path.join(out_dir, os.path.basename(mp.replace("\\", "/")))
        if not os.path.exists(mp):
            mp = os.path.join(out_dir, "bodhi_spm.model")
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

    if os.path.exists(int8_path):
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        state = torch.load(int8_path, map_location="cpu")
        model.load_state_dict(state)
        print("Loaded int8 model.")
        model = model.to("cpu").eval()
        return model, tok, "cpu", int(cfg["context_len"])

    ckpt = torch.load(fp_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print("Loaded fp32 model.")
    return model, tok, device, int(cfg["context_len"])


@torch.no_grad()
def reply(model, tok, device, context_len, prompt: str, max_new_tokens=120, temperature=0.8, top_k=40):
    x = tok.encode(prompt, add_bos=True, add_eos=False)
    idx = torch.tensor([x[-context_len:]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    seq = out[0].tolist()
    continuation = seq[len(idx[0]) :]
    eos_id = int(getattr(tok, "EOS", 2))
    if eos_id in continuation:
        continuation = continuation[: continuation.index(eos_id)]
    text = tok.decode(continuation)
    if "SYSTEM:" in text:
        text = text.split("SYSTEM:")[0].strip()
    if "USER:" in text:
        text = text.split("USER:")[0].strip()
    if "BODHI:" in text:
        text = text.split("BODHI:")[-1].strip()
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="out")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--temperature", type=float, default=0.75)
    p.add_argument("--top_k", type=int, default=40)
    args = p.parse_args()

    model, tok, device, context_len = load_model(args.out_dir, args.device)
    history = [BODHI_PERSONA]
    print("BODHI small LLM chat ready. Use /exit to quit.")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("/exit", "/quit", "/q"):
            break
        history.append("USER: %s" % user)
        history.append("BODHI:")
        prompt = "\n".join(history)[-4000:]
        out = reply(
            model=model,
            tok=tok,
            device=device,
            context_len=context_len,
            prompt=prompt,
            max_new_tokens=120,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        # Keep only first assistant line for cleaner dialogue.
        clean = out.split("\n")[0].strip()
        if not clean:
            clean = "I am here, processing your words."
        clean = clean.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        clean = clean.encode("cp1252", errors="replace").decode("cp1252", errors="replace")
        print("BODHI:", clean)
        history[-1] = "BODHI: %s" % clean


if __name__ == "__main__":
    main()

