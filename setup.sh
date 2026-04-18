#!/usr/bin/env bash
# ============================================================
#  BODHI Setup — macOS / Linux
#  Run: bash setup.sh
# ============================================================
set -e

echo ""
echo " ██████╗  ██████╗ ██████╗ ██╗  ██╗██╗"
echo " ██╔══██╗██╔═══██╗██╔══██╗██║  ██║██║"
echo " ██████╔╝██║   ██║██║  ██║███████║██║"
echo " ██╔══██╗██║   ██║██║  ██║██╔══██║██║"
echo " ██████╔╝╚██████╔╝██████╔╝██║  ██║██║"
echo " ╚═════╝  ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝"
echo " BODHI Setup for macOS / Linux"
echo ""

# Check Python 3
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 not found."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  Install with: brew install python3"
        echo "  Or download from: https://python.org/downloads"
    else
        echo "  Install with: sudo apt install python3 python3-pip   (Ubuntu/Debian)"
        echo "             or: sudo dnf install python3              (Fedora)"
    fi
    exit 1
fi

PY=$(command -v python3)
echo "[1/3] Using Python: $($PY --version)"

# Install packages
echo "[2/3] Installing Python packages..."
$PY -m pip install --quiet numpy pillow cryptography torch sentencepiece
echo "      Packages installed."

# Check fingerprint data (LFS stub = < 1 MB)
echo "[3/3] Checking data files..."
FP_IMG="data/fingerprints_img.npz"
if [ -f "$FP_IMG" ]; then
    SIZE=$(wc -c < "$FP_IMG")
    if [ "$SIZE" -lt 1000000 ]; then
        echo "  Fingerprint data not downloaded (LFS stub). Fetching now..."
        $PY download_data.py
    else
        echo "  Data files OK ($((SIZE/1000000)) MB)."
    fi
else
    echo "  Fingerprint data missing. Fetching..."
    $PY download_data.py
fi

echo ""
echo "  Setup complete!"
echo ""
echo "  To start BODHI:"
echo "    python3 bodhi.py"
echo ""
