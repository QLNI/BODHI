#!/usr/bin/env python3
"""
Compress all WHT fingerprints to 20% (5x compression).
Keep only the largest coefficients by magnitude, zero the rest.
Store as single combined files: one for images, one for audio.

Input: bodhi_final/data/fingerprints/ (10K images + 632 audio = 652 MB)
Output: main/data/fingerprints_img.npz + fingerprints_aud.npz + index.json
"""
import numpy as np
import os, sys, glob, json, time

BODHI_FP = "C:/Users/USER/Desktop/bodhi_final/data/fingerprints"
OUT_DIR = "C:/Users/USER/Desktop/main/data"
KEEP_RATIO = 0.20  # 20% keep = 5x compression

os.makedirs(OUT_DIR, exist_ok=True)

def compress(fp, keep_ratio):
    n = len(fp)
    n_keep = max(1, int(n * keep_ratio))
    abs_fp = np.abs(fp)
    threshold = np.partition(abs_fp, -n_keep)[-n_keep]
    compressed = fp.copy()
    compressed[abs_fp < threshold] = 0
    return compressed


# === IMAGE FINGERPRINTS ===
print("=== Compressing image fingerprints (20%% keep) ===")
t0 = time.time()

img_files = sorted(glob.glob(os.path.join(BODHI_FP, "*_img.npz")))
print("Found: %d image fingerprints" % len(img_files))

img_names = []
img_data = []
for i, f in enumerate(img_files):
    name = os.path.basename(f).replace("_img.npz", "")
    fp = np.load(f)["fingerprint"]
    compressed = compress(fp, KEEP_RATIO)
    img_names.append(name)
    img_data.append(compressed.astype(np.int16))  # int16 saves space, range fits
    if (i + 1) % 2000 == 0:
        print("  %d/%d..." % (i + 1, len(img_files)))

# Stack into single array
img_array = np.stack(img_data)
fp_size = img_array.shape[1]
print("Shape: %s" % str(img_array.shape))

# Save
img_path = os.path.join(OUT_DIR, "fingerprints_img.npz")
np.savez_compressed(img_path, data=img_array)
img_file_size = os.path.getsize(img_path)
print("Saved: %s (%.1f MB)" % (img_path, img_file_size / 1024 / 1024))

orig_img_size = sum(os.path.getsize(f) for f in img_files)
print("Original: %.1f MB -> Compressed: %.1f MB (%.1fx)" % (
    orig_img_size / 1024 / 1024, img_file_size / 1024 / 1024,
    orig_img_size / max(1, img_file_size)))

elapsed = time.time() - t0
print("Time: %.1fs" % elapsed)
print()


# === AUDIO FINGERPRINTS ===
print("=== Compressing audio fingerprints (20%% keep) ===")
t1 = time.time()

aud_files = sorted(glob.glob(os.path.join(BODHI_FP, "*_aud.npz")))
print("Found: %d audio fingerprints" % len(aud_files))

aud_names = []
aud_offsets = [0]  # audio fps have different lengths, store offsets
aud_all = []
for f in aud_files:
    name = os.path.basename(f).replace("_aud.npz", "")
    fp = np.load(f)["fingerprint"]
    compressed = compress(fp, KEEP_RATIO)
    aud_names.append(name)
    aud_all.append(compressed.astype(np.int32))  # audio needs int32 range
    aud_offsets.append(aud_offsets[-1] + len(compressed))

# Concatenate all audio into one flat array
aud_concat = np.concatenate(aud_all)
aud_offsets = np.array(aud_offsets, dtype=np.int64)
print("Total audio coefficients: %d" % len(aud_concat))

# Save
aud_path = os.path.join(OUT_DIR, "fingerprints_aud.npz")
np.savez_compressed(aud_path, data=aud_concat, offsets=aud_offsets)
aud_file_size = os.path.getsize(aud_path)
print("Saved: %s (%.1f MB)" % (aud_path, aud_file_size / 1024 / 1024))

orig_aud_size = sum(os.path.getsize(f) for f in aud_files)
print("Original: %.1f MB -> Compressed: %.1f MB (%.1fx)" % (
    orig_aud_size / 1024 / 1024, aud_file_size / 1024 / 1024,
    orig_aud_size / max(1, aud_file_size)))

elapsed2 = time.time() - t1
print("Time: %.1fs" % elapsed2)
print()


# === INDEX ===
index = {
    "img_names": img_names,
    "img_name_to_idx": {n: i for i, n in enumerate(img_names)},
    "img_fp_size": fp_size,
    "img_count": len(img_names),
    "aud_names": aud_names,
    "aud_name_to_idx": {n: i for i, n in enumerate(aud_names)},
    "aud_count": len(aud_names),
    "keep_ratio": KEEP_RATIO,
}
index_path = os.path.join(OUT_DIR, "fingerprint_index.json")
with open(index_path, "w") as f:
    json.dump(index, f)
print("Index: %s (%d images, %d audio)" % (index_path, len(img_names), len(aud_names)))


# === SUMMARY ===
total_orig = orig_img_size + orig_aud_size
total_comp = img_file_size + aud_file_size + os.path.getsize(index_path)
print()
print("=" * 50)
print("  COMPRESSION SUMMARY")
print("=" * 50)
print("  Images: %.1f MB -> %.1f MB" % (orig_img_size/1024/1024, img_file_size/1024/1024))
print("  Audio:  %.1f MB -> %.1f MB" % (orig_aud_size/1024/1024, aud_file_size/1024/1024))
print("  Total:  %.1f MB -> %.1f MB (%.1fx compression)" % (
    total_orig/1024/1024, total_comp/1024/1024, total_orig/max(1, total_comp)))
print("  Saved:  %.1f MB" % ((total_orig - total_comp)/1024/1024))
