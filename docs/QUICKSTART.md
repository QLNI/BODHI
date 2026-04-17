# BODHI Quickstart

The 3-minute install.

```bash
git clone <repo-url> bodhi
cd bodhi
pip install -r requirements.txt
python bodhi.py
```

You'll see:

```
Initialising BODHI brain...
BODHI ready. 10000 concepts, 91613 aliases, 9159 engrams.

You:
```

Try these:

```
You: What is fire?
You: Tell me about the ocean
You: I see a snake
You: Who are you?
You: What is xyzzy?        # unknown — BODHI says so honestly
You: /goal add learn more about storms
You: /status
You: /sleep
You: /quit
```

## Teach BODHI from your own images

With any image file (jpg, png, etc.):

```
You: this is a cat photos/my_cat.jpg
You: Tell me about cat
```

Or use the explicit form:

```
You: /teach cat photos/my_cat.jpg
```

## Let BODHI evolve its language model

After ~50 conversations:

```bash
python nightly_train.py --steps 150
```

This trains a LoRA adapter (~2.5 MB) on your conversation history. Next
time BODHI boots:

```
Broca's area loaded from out_v2 + LoRA adapter (evolved)
```

Now the language model has weights that have genuinely drifted from the
baseline toward your use patterns.

## Check it all still works

```bash
python eval_harness.py
```

Should print `SUMMARY: 12 passed, 0 failed, 0 skipped (of 12 total)`.

## Clean slate

```bash
python clear_memory.py
```

Wipes BODHI's conversations, emotional memory, Hebbian graph, and LoRA
adapter. The codec, fingerprints, and trained LLM base stay — those are
shipped, not learned.
