#!/usr/bin/env bash
# Example 03 — Fine-tune BODHI's language model on your own conversations.
#
# Run this after ~30-50 conversations have accumulated in data/bodhi_memory.db.
# Produces a ~2.5 MB LoRA adapter under data/brain_state/lora_adapter.pt.
# Next time BODHI boots, its language model speaks from weights that have
# drifted toward your use patterns.

set -e

# Default — small nightly run, resumes from any existing adapter
python nightly_train.py --steps 150 --resume

# Alternative — full fresh run on the curated 516-example seed corpus
# python nightly_train.py --data data/training_seed.jsonl --data-only --steps 600

# Alternative — bigger run, rank 16 adapter
# python nightly_train.py --rank 16 --alpha 32 --steps 500 --resume

echo
echo "Done. Launch BODHI to use the evolved adapter:"
echo "  python bodhi.py"
