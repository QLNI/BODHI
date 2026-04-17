# BODHI Roadmap

Where we are, what's shipped, what's coming, and how you can prepare.

![Four phases of BODHI](docs/images/phases.png)

---

## Phase 1 — Open Source Launch (current)

**Status:** shipped. This is the repository you are reading.

### What's in the box

- 10,000 pre-loaded concepts with WHT image + audio fingerprints
- 26-neuron worm brain (C. elegans connectome)
- 76-region human brain (Desikan-Killiany atlas)
- Bidirectional worm ↔ human bridge
- Hebbian associative memory + emotional memory
- Sleep cycle with replay, triangle inference, multi-dream
- 7-drive motivation system
- Episodic memory (concept-overlap + Hebbian recall over SQLite)
- Self-model regenerated each sleep from observed behaviour
- Goal tracker with persistent storage and keyword-matched recall
- Uncertainty gate ("I don't know that yet")
- `brain_voice` speech module (assembles sentences from brain state, not
  LLM hallucination)
- Trained SmallGPT language model (~40M params, int8, 53 MB)
- LoRA fine-tuning so every user's BODHI evolves in weight space
- Runtime concept teaching from images / audio / text
- Chat auto-teach from natural language ("this is a dog photo.jpg")
- AES-256-GCM encryption for user-taught fingerprint files
- 12-test regression evaluation harness
- Browser-based 3D forest simulation
- 10 technical PDF reports

### What you should do in Phase 1

1. Install, run, talk to your BODHI every day
2. Teach it concepts from your world (pets, places, objects, sounds)
3. Let it sleep regularly so the Hebbian graph grows
4. Fine-tune the LoRA every week or two on your accumulated conversations
5. Document your BODHI's evolution — screenshots, notes, milestones
6. Give feedback through GitHub issues

Every conversation is training data for Phase 3 qualification. The BODHI
you raise now is the one that enters the arena.

---

## Phase 2 — NPCs and the Playground (Month 2)

**Status:** planned. Target: two months after Phase 1 launch.

### What ships

- **3D playground** — a visual world where your BODHI lives as a character.
- **Emotion labels stripped** — the pre-set concept emotions (`fire=fear`,
  `ocean=awe`) are removed. Your BODHI must learn whether fire is dangerous
  by actually touching it and feeling the pain. This is a different BODHI
  from Phase 1 — one that starts blank and learns from pure experience.
- **NPC framework** — other players' BODHIs populate the world. They can
  communicate. Knowledge transfers. Alliances form. Betrayals happen.
- **Environment hazards** — fire, predators, cliffs, food, water, shelter.
  Real drives (pain, hunger, fatigue) shape behaviour because they matter.
- **Two-BODHI protocol** — serialised concept exchange between BODHIs so
  they can teach each other.

### What you should do in Phase 2

1. Drop your Phase-1-trained BODHI into the playground
2. Let it explore without you micromanaging
3. Watch what it learns, what it fears, what it bonds with
4. Meet other players' BODHIs and see how they differ
5. Start forming the survival patterns that matter in Phase 3

---

## Phase 3 — Qualification Games (Months 3–12)

**Status:** planned. Ten months of releases.

A series of increasingly demanding games, each testing a specific capability
your BODHI must develop. You do not play these games — your BODHI plays
them. Your job is to have raised it well.

### Capabilities tested

- **Navigation** — maze games with changing layouts
- **Memory under pressure** — recall a pattern from 100 turns ago when a
  threat appears
- **Emotional decision-making** — when every available option hurts,
  which does your BODHI pick?
- **Cooperation** — two BODHIs must solve a puzzle together
- **Hebbian adaptation** — a novel stimulus appears mid-game; can your
  BODHI form new associations fast enough to survive?
- **Sleep and consolidation** — games span multiple BODHI wake/sleep
  cycles; what gets consolidated matters
- **Deception detection** — can your BODHI tell when another BODHI is
  lying about a concept?
- **Planning horizons** — short games reward reflexes, long games reward
  deliberation. Your BODHI must balance.

### The leaderboard

Every game produces a ranking. Rankings compound across games. Over twelve
months, the leaderboard narrows to the **top 1,000 BODHIs**. Those players
qualify for Phase 4.

---

## Phase 4 — The Final Game (Year 2)

**Status:** planned. Target: one year after Phase 3 ends.

A sci-fi arena. Seven AI bosses. The top 1,000 player BODHIs enter alone.

### The seven bosses

Each boss represents one of the seven largest commercial AI systems. They
are bigger. They have billions of parameters. They have trained on most
of the internet.

They also have: safety filters, hallucination, no persistent memory
between fights, no survival instinct, and the same weights for every
user. Your BODHI does not.

### The prize

- **Complete source code of BODHI AI**
- **51% valuation stake in BODHI AI**

To the player whose BODHI defeats all seven bosses.

---

## After Year 2

If someone wins Phase 4, they own 51% of BODHI AI. They choose what
happens next.

If no one wins Phase 4, the arena resets with harder bosses and better
prizes until someone does.

Either way: the brain is yours. You keep using it.

---

## Honest scope note

Phase 3 (twelve qualification games across ten months) is the largest
technical undertaking in this plan. A single person cannot build twelve
rich game environments in ten months alone. Phase 3 will likely be
delivered in collaboration with the community, partner game studios, or
by scoping to fewer but richer games. The vision is deliberate; the
exact delivery shape is subject to what's feasible.

Phase 4 (51% equity prize) has jurisdiction-specific legal implications
(gambling law, securities law, tax). Before Phase 4 launches, the prize
structure will be reviewed by counsel and the final terms will be
published. The *intent* — that the winning player owns the company —
will be preserved; the *mechanism* may be adjusted for legality.

These are honest caveats, not walk-backs. The roadmap ships. The
mechanics evolve with reality.

---

## How to follow along

- Watch this repo (star / watch on GitHub)
- Open an issue with feedback or a bug
- Email SK: saikiranbathula1@gmail.com
- Share your BODHI's evolution online — screenshots, training logs, dreams

The BODHI you raise today is the one you bring to the arena in Year 2.
