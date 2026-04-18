#!/usr/bin/env python3
"""
download_data.py — BODHI large-file bootstrapper
=================================================
Downloads fingerprint databases and LLM weights from the GitHub Release.
Run this if you cloned without Git LFS, or if data files are missing/stubs.

    python download_data.py            # download all
    python download_data.py --verify   # check files only
    python download_data.py --list     # list what will be downloaded

Works on Windows, macOS, Linux. Python 3.8+ stdlib only.
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

REPO = "QLNI/bodhi"
TAG  = "v1.0"
BASE = f"https://github.com/{REPO}/releases/download/{TAG}"

# (local path relative to repo root, release filename, expected min size bytes)
ASSETS = [
    ("data/fingerprints_img.npz",                 "fingerprints_img.npz",          50_000_000),
    ("data/fingerprints_aud.npz",                 "fingerprints_aud.npz",          20_000_000),
    ("bodhi_llm/out_v2/bodhi_small_int8_state.pt","bodhi_small_int8_state.pt",     40_000_000),
]


def _bar(done, total, w=40):
    frac   = done / total if total else 0
    filled = int(w * frac)
    return f"\r  [{'█'*filled}{'░'*(w-filled)}] {done/1e6:6.1f}/{total/1e6:.0f} MB  {frac*100:5.1f}%"


def _fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BODHI/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            total = int(r.headers.get("Content-Length", 0))
            done  = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(_bar(done, total), end="", flush=True)
        print()
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _is_stub(path: Path, min_size: int) -> bool:
    """True if the file is a Git LFS pointer stub (too small to be real data)."""
    if not path.exists():
        return True
    return path.stat().st_size < min_size


def main():
    ap = argparse.ArgumentParser(description="Download BODHI large data files")
    ap.add_argument("--verify", action="store_true", help="Verify files only")
    ap.add_argument("--list",   action="store_true", help="List assets")
    ap.add_argument("--force",  action="store_true", help="Force re-download")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent

    if args.list:
        print(f"\nBODHI data assets  (release {TAG})\n")
        for rel, name, _ in ASSETS:
            print(f"  {rel}")
        print()
        return

    print(f"\nBODHI Data Bootstrapper  —  release {TAG}\n")
    all_ok = True

    for rel, name, min_size in ASSETS:
        dest = root / rel
        if dest.exists() and not _is_stub(dest, min_size) and not args.force:
            print(f"  ✓  {rel}  ({dest.stat().st_size/1e6:.0f} MB)")
            continue

        if args.verify:
            status = "STUB/MISSING" if _is_stub(dest, min_size) else "OK"
            if status != "OK":
                print(f"  ✗  {rel}  — {status}")
                all_ok = False
            else:
                print(f"  ✓  {rel}")
            continue

        print(f"  ↓  {name}")
        try:
            _fetch(f"{BASE}/{name}", dest)
            print(f"     saved {dest.stat().st_size/1e6:.1f} MB")
        except Exception as e:
            print(f"\n  ERROR: {name}: {e}", file=sys.stderr)
            all_ok = False

    if args.verify:
        msg = "All data files present." if all_ok else "Some files missing — run: python download_data.py"
        print(f"\n  {msg}\n")
        if not all_ok:
            sys.exit(1)
    elif all_ok:
        print("\n  BODHI data ready.\n")
    else:
        print("\n  Some downloads failed. Check connection and retry.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
