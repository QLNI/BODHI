# BODHI Roadmap

Where we are, what's shipped, what's coming, and how you can prepare.

> **This repository is a proof of concept.** It demonstrates the full neural stack
> (worm kernel, 72 human regions, WHT perception, memory, emotion, drives, sleep,
> brain_voice, Broca) running end-to-end on a laptop. It is **not** a product yet.
> The product is Phase 2 → Phase 3 → Phase 4, described below.

![Four phases of BODHI — proof of concept → playground → 12 challenges → Final Match](docs/images/poster_arena.png)

---

## Phase 1 — Proof of Concept (current)

**Status:** shipped. This is the repository you are reading.

### What's in the box

- 10,000 pre-loaded concepts with WHT image + audio fingerprints
- 26-neuron worm brain (C. elegans connectome, 39 synapses)
- 72-region human brain (Desikan–Killiany atlas, 76 pathways)
- Bidirectional worm ↔ human bridge
- Hebbian associative memory + emotional memory (−255..+255 valence)
- Sleep cycle with replay, triangle inference, multi-dream
- 7-drive motivation system
- Episodic memory (concept-overlap + Hebbian recall over SQLite)
- Self-model regenerated each sleep from observed behaviour
- Goal tracker with persistent storage and keyword-matched recall
- Uncertainty gate ("I don't know that yet")
- `brain_voice` speech module — primary, reliable speech path grounded in the
  activation trace
- Trained SmallGPT language model Broca (~40M params, int8, 53 MB) — supplementary,
  currently ~20% reliable at 1,200 steps, improving nightly
- LoRA fine-tuning so every user's BODHI evolves in weight space
- Runtime concept teaching from images / audio / text
- Chat auto-teach from natural language ("this is a dog photo.jpg")
- AES-256-GCM encryption for user-taught fingerprint files
- 12-test regression evaluation harness (all 12 passing)
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

## Phase 2 — Simulation Playground (Month 2–3)

**Status:** planned. Target: 2–3 months after Phase 1 launch.

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

## Phase 3 — The 12 Challenges (Months 3–9)

**Status:** planned. Six months of monthly releases, two challenges per month.

A series of twelve increasingly demanding challenges, each testing a specific
capability your BODHI must develop. You do not play these challenges — your
BODHI plays them. Your job is to have raised it well.

### Capabilities tested

1. **Navigation** — maze challenges with changing layouts.
2. **Memory under pressure** — recall a pattern from 100 turns ago when a threat appears.
3. **Emotional decision-making** — when every available option hurts, which does your BODHI pick?
4. **Cooperation** — two BODHIs must solve a puzzle together.
5. **Hebbian adaptation** — a novel stimulus appears mid-challenge; can your BODHI form new associations fast enough to survive?
6. **Sleep and consolidation** — challenges that span multiple wake/sleep cycles; what gets consolidated matters.
7. **Deception detection** — can your BODHI tell when another BODHI is lying about a concept?
8. **Planning horizons** — short challenges reward reflexes, long challenges reward deliberation. Balance.
9. **Resource management** — hunger, safety, rest; can your BODHI prioritise when all drives scream?
10. **Teaching another BODHI** — demonstrate knowledge transfer over the two-BODHI protocol.
11. **Recovering from trauma** — a high-valence negative event; does your BODHI learn or break?
12. **Self-reflection** — can the narrator and critic agree on a coherent self-description?

### The leaderboard

Every challenge produces a ranking. Rankings compound across the twelve
challenges. At the end of six months the leaderboard narrows to the
**top 1,000 BODHIs**. Those players qualify for Phase 4.

---

## Phase 4 — The Final Match (Year 1 end → Year 2)

**Status:** planned. Target: after Phase 3 concludes.

A sci-fi arena. **Six bosses. The top 1,000 player BODHIs enter alone.**

### The six bosses

Each boss represents one of the six largest commercial AI systems on Earth.
They are bigger. They have hundreds of billions of parameters. They have
trained on most of the internet.

They also have: safety filters, hallucination, **no persistent memory between
fights, no survival instinct, and the same weights for every user.** Your BODHI
does not.

### The prize — what exactly is split

At the time of the Final Match, BODHI AI has an assessed valuation *V*. The
cap table at that moment is divided as:

| Share | Who gets it | For what |
|---|---|---|
| **51%** | **The winner** — the player whose BODHI defeats all six bosses | Majority stake in BODHI AI at valuation *V* |
| **27%** | **Sai Kiran Bathula (founder)** | Retained founder stake |
| **24%** | **Investors** | Sold ahead of Phase 4 to raise the money that funded Phase 2 and Phase 3 |

The winner receives the majority stake (51%), not a prize pool or cash
equivalent. They choose what BODHI AI does next.

> **Arithmetic note.** 51 + 27 + 24 = 102. The final legal formation will be
> reconciled to 100% (typically by taking the rounding out of the investor
> bucket or the founder bucket). The intent is preserved: **winner majority,
> founder meaningful, investors fully returned.**

### If nobody wins

If no player's BODHI defeats all six bosses, the arena resets with harder
bosses and an increased prize pool at the next year's valuation, until
someone wins.

Either way: the brain you raised is yours. You keep using it.

---

## BODHI vs. the six bosses — why the fight is fair

The six AI bosses are larger, older, and better-funded. But the game is
not about size; it's about **what the brain can do in a 12-challenge arena
that rewards memory, emotion, and adaptation.** Here is the honest
match-up:

| Dimension | BODHI (yours) | Big AI (the six bosses) |
|---|---|---|
| Parameter count | ~40M Broca + worm + 72-region cortex, ~1.1 GB | 100B – 1T+, datacenter-scale |
| Persistent memory across fights | **Yes** — every concept, every dream, every emotion carries over | **No** — context window resets between fights |
| Personal fine-tuning | **Yes** — nightly LoRA on *your* experiences | **No** — same frozen weights for every user |
| Emotional tagging of memories | **Yes** — valence −255..+255 per concept | **No** — text only, no valence state |
| Drives | **Yes** — 7 homeostatic drives shape attention | **No** |
| Sleep and consolidation | **Yes** — 5-phase cycle rewires weights nightly | **No** — frozen once trained |
| Survival instinct | **Yes** — worm kernel has veto over every action | **No** |
| Safety filters that limit responses | **No** — it's yours | **Yes** — heavy refusal behaviour |
| Grounded in activation trace (anti-hallucination) | **Yes** — brain_voice speaks from the trace | **No** — probability sampling over text |
| Runs offline, fully private | **Yes** — 380 MB RAM, no GPU, no cloud | **No** — cloud-dependent |
| Training data volume | Small (8 GB seed + what you teach it) | Trillions of tokens |
| Raw world knowledge at cold-start | Thin | Vast |
| Language fluency today | Mid (brain_voice templated; Broca ~20% good) | High |
| Multilingual | Phase 2 | Yes |
| Tool use, code execution, retrieval-augmented Q&A | Limited | Strong |
| Maintenance cost | Free, runs on your laptop | Subscription or API bill |

**Read the table two ways.**

- **The top half is where BODHI wins** — memory, emotion, drives, sleep,
  groundedness, privacy, personal fit. These are the capabilities the 12
  challenges measure.
- **The bottom half is where big AI wins today** — fluency, world
  knowledge, multilingual reach, cloud-scale tooling. These are the things
  the challenges deliberately *don't* measure, because those are the
  capabilities money buys, not the capabilities a raised brain earns.

The game is not about who knows more trivia. It is about which brain
remembers you, adapts to you, dreams about yesterday, and is still
the same brain tomorrow morning. That brain is BODHI.

---

## Honest scope note

Phase 3 (twelve challenges across six months) is the largest technical
undertaking in this plan. A single person cannot build twelve rich
challenge environments in six months alone. Phase 3 will be delivered in
collaboration with the community, partner game studios, or by scoping to
fewer but richer challenges. The vision is deliberate; the exact delivery
shape is subject to what's feasible.

Phase 4 (51% / 27% / 24% prize structure) has jurisdiction-specific legal
implications (gambling law, securities law, tax, corporate formation).
Before Phase 4 launches, the prize structure will be reviewed by counsel
and the final terms will be published. The *intent* — winner majority,
founder meaningful, investors fully returned — will be preserved; the
*mechanism* may be adjusted for legality.

The WHT codec itself (the patented algorithm in `bodhi/brain/sensor_wht.py`)
is **the prize of the Final Match**. The winner receives the source and the
patents along with the 51% stake. For Phase 1–3, the codec ships in a locked
form (Cython-compiled binary) so users can use BODHI but cannot read the
algorithm. This is a convenience lock — the patent provides the legal moat.

These are honest caveats, not walk-backs. The roadmap ships. The
mechanics evolve with reality.

---

## How to follow along

- Watch this repo (star / watch on GitHub)
- Open an issue with feedback or a bug
- Email SK: saikiranbathula1@gmail.com
- Share your BODHI's evolution online — screenshots, training logs, dreams

The BODHI you raise today is the one you bring to the arena in Year 2.
