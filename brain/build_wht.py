"""
BODHI WHT Codec — build script.

This script compiles the WHT codec to a platform-native binary (.so on Linux/Mac,
.pyd on Windows) so that the algorithm is not readable from the distributed
BODHI package.

WHY THIS EXISTS
---------------
The Walsh–Hadamard Transform codec (IP Australia 2026901656 image / 2026901657
audio) is the prize of BODHI's Phase-4 Final Match. For Phase 1–3 we ship the
codec only in compiled form so users can USE BODHI but cannot READ the algorithm.

USAGE
-----
From the repo root::

    python -m bodhi.brain.build_wht

Afterwards the ``_wht_core*.so`` (Linux/Mac) or ``_wht_core*.pyd`` (Windows)
appears alongside this file. ``sensor_wht.py`` then imports from it
transparently — nothing else in BODHI needs to change.

NOTE: The ``_wht_core.pyx`` source file is considered privileged. It is not
included in public distribution builds; the compiled binary alone ships.
"""
from __future__ import annotations

import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PYX  = os.path.join(HERE, "_wht_core.pyx")


def build():
    if not os.path.exists(PYX):
        print(f"[build_wht] ERROR: {PYX} not found.", file=sys.stderr)
        print("            The pyx source is required to rebuild the codec.",
              file=sys.stderr)
        print("            If you received only the compiled binary, you do not",
              file=sys.stderr)
        print("            need to rebuild — the .so/.pyd is already in place.",
              file=sys.stderr)
        sys.exit(1)

    # Lazy imports so this file is importable even without Cython available
    from setuptools import setup, Extension
    from Cython.Build import cythonize
    import numpy as np

    # Neutral module name so the .so works whether BODHI is imported as
    # ``bodhi.brain.sensor_wht`` or ``brain.sensor_wht``.
    ext = Extension(
        name="_wht_core",
        sources=[PYX],
        include_dirs=[np.get_include()],
    )

    # In-place build so the .so/.pyd appears next to this file.
    old_argv = sys.argv
    sys.argv = [old_argv[0], "build_ext", "--inplace",
                "--build-lib", HERE, "--build-temp", os.path.join(HERE, "_build")]

    try:
        setup(
            name="bodhi_wht_core",
            ext_modules=cythonize(
                [ext],
                language_level=3,
                compiler_directives={
                    "boundscheck": False,
                    "wraparound":  False,
                    "cdivision":   True,
                    # Strip docstrings / comments from the compiled binary
                    "embedsignature": False,
                },
            ),
        )
    finally:
        sys.argv = old_argv

    print("[build_wht] OK — compiled _wht_core binary is in", HERE)


if __name__ == "__main__":
    build()
