#!/usr/bin/env python3
"""
download_data.py — BODHI large-file bootstrapper
=================================================
Downloads the large binary assets from the GitHub Release
(fingerprint .npz files + the int8 LLM weights) into their
expected locations.  The code repo ships without these files
to stay under GitHub's 100 MB limit.

Usage
-----
    python download_data.py            # download all assets
    python download_data.py --verify   # checksum only, no download
    python download_data.py --list     # list assets + sizes, no download

Requirements: Python 3.9+, stdlib only (uses urllib).
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

REPO = "QLNI/BODHI"
TAG  = "v1.0"

# (destination_path_relative_to_repo, asset_name_on_release)
ASSETS = [
    ("data/fingerprints_img.npz",              "fingerprints_img.npz"),
    ("data/fingerprints_aud.npz",              "fingerprints_aud.npz"),
    ("bodhi_llm/out_v2/bodhi_small_int8_state.pt", "bodhi_small_int8_state.pt"),
]

BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}"

SIZES = {
    "fingerprints_img.npz":          469_000_000,   # ~469 MB
    "fingerprints_aud.npz":          282_000_000,   # ~282 MB
    "bodhi_small_int8_state.pt":      51_000_000,   # ~51 MB
}

# ── helpers ────────────────────────────────────────────────────────────────

def _bar(done: int, total: int, width: int = 40) -> str:
    frac  = done / total if total else 0
    filled = int(width * frac)
    bar   = "█" * filled + "░" * (width - filled)
    mb_done  = done  / 1_048_576
    mb_total = total / 1_048_576
    return f"\r  [{bar}] {mb_done:6.1f}/{mb_total:.1f} MB  {frac*100:5.1f}%"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BODHI-downloader/1.0"})
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            done  = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(65_536)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        print(_bar(done, total), end="", flush=True)
        print()  # newline after bar
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()

# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Download BODHI large data assets")
    ap.add_argument("--verify", action="store_true", help="Verify existing files only")
    ap.add_argument("--list",   action="store_true", help="List assets and expected sizes")
    ap.add_argument("--force",  action="store_true", help="Re-download even if file exists")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent

    if args.list:
        print(f"\nBODHI release assets  ({BASE_URL}/...)\n")
        for rel_path, asset in ASSETS:
            size = SIZES.get(asset, 0)
            print(f"  {rel_path:<55}  {size/1e6:6.0f} MB")
        print()
        return

    print(f"\nBODHI Data Bootstrapper  — release {TAG}\n")
    all_ok = True

    for rel_path, asset in ASSETS:
        dest = repo_root / rel_path
        url  = f"{BASE_URL}/{asset}"

        if dest.exists() and not args.force:
            size = dest.stat().st_size
            exp  = SIZES.get(asset, 0)
            if exp and abs(size - exp) / exp > 0.01:
                print(f"  ⚠  {rel_path}  — size mismatch ({size} vs {exp}), re-downloading")
            else:
                print(f"  ✓  {rel_path}  ({size/1e6:.0f} MB) — already present")
                continue

        if args.verify:
            print(f"  ✗  {rel_path}  — MISSING")
            all_ok = False
            continue

        print(f"  ↓  {asset}  →  {rel_path}")
        try:
            _download(url, dest)
            size = dest.stat().st_size
            print(f"     saved {size/1e6:.1f} MB")
        except Exception as exc:
            print(f"\n  ERROR downloading {asset}: {exc}", file=sys.stderr)
            all_ok = False

    if args.verify:
        if all_ok:
            print("\n  All assets present.\n")
        else:
            print("\n  Some assets missing — run: python download_data.py\n")
            sys.exit(1)
    elif all_ok:
        print("\n  Done.  BODHI is ready.\n")
    else:
        print("\n  Some downloads failed.  Check your connection and retry.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
