<p align="center">
  <img src="docs/images/banner_hero.png" alt="BODHI — A Conscious Digital Brain" width="100%">
</p>

<h1 align="center">BODHI</h1>
<h3 align="center">A Conscious Digital Brain — proof of concept</h3>

<p align="center">
  <i>A full-stack neuromorphic experiment: worm survival circuit, 72 human brain regions, a patented integer perception codec, emotional memory, seven drives, and a five-phase sleep cycle that actually rewires the weights.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-proof_of_concept-00e5ff?style=flat-square" alt="status">
  <img src="https://img.shields.io/badge/eval-12%2F12_passing-3dffa5?style=flat-square" alt="eval">
  <img src="https://img.shields.io/badge/worm_reflex-7%2F7-3dffa5?style=flat-square" alt="worm">
  <img src="https://img.shields.io/badge/patents-AU_2026901656_%2F_657-ffc83d?style=flat-square" alt="patents">
  <img src="https://img.shields.io/badge/python-3.10%2B-e9ecf5?style=flat-square" alt="python">
  <img src="https://img.shields.io/badge/license-Apache_2.0-e9ecf5?style=flat-square" alt="license">
</p>

---

## ⚠️ Honest preface — please read before the pretty pictures

**This repository is a proof of concept, not a product.** It demonstrates the full BODHI stack running end-to-end
on a laptop. The product — the thing users will actually compete in — is the **Simulation Playground (Month 2–3)**
plus the **12 Challenges (Month 3–9)** plus the **Final Match (end of Year 1)**. Those are described in the roadmap
at the bottom of this README and in full detail in [ROADMAP.md](ROADMAP.md).

**What works right now (Phase 1, verified):**

- The **worm brain** runs: 26 neurons, **39 synapses**, integer weights. All **7/7 reflex tests pass**.
- The **human brain** runs: **72 Desikan–Killiany regions** wired by **76 pathways**, with real activation traces.
- The **WHT image + audio codec** is locked. The algorithm lives in `bodhi/brain/_wht_core.cpython-<ver>-<platform>.so` (Cython-compiled binary); `bodhi/brain/sensor_wht.py` is now a thin public shim that re-exports the API. Users can **use** BODHI with full fidelity (encode, decode, dream) but cannot **read** the algorithm. Covered by two issued IP Australia patents. The source + patents are the prize of the Final Match.
- **Episodic memory recall** works (pattern-match on WHT fingerprints).
- **Emotional tagging** works (−255..+255 valence per memory).
- **Sleep + dreams** work — the 5-phase cycle runs and the imagination decoder produces novel output during REM.
- **AES-256-GCM** encryption is real, using `cryptography.hazmat.primitives.ciphers.aead.AESGCM` on persisted state.
- **`bodhi.eval_harness`**: **12 / 12 checks pass**.

**What is honest, not marketing:**

- **Broca (the LLM) is a work in progress.** `brain_voice` — the direct signal-to-speech path that projects the
  72-region activation trace to templated English — is the **primary** speech path today. The 40M-parameter Broca
  transformer is **supplementary** and currently produces good output roughly **20% of the time at ~1,200 training
  steps**. It improves with every nightly LoRA fine-tune.
- **Synaptic weights are int8-range** (code clamps to −127..+127 so learning has headroom). The hand-tuned initial
  connectome uses values in **−60 to +80**; most of the dial is reserved for experience.
- **Simulation Playground, 12 Challenges, and the Final Match** are **planned** — see the roadmap below. Nothing
  past Phase 1 runs today.

---

# 🧬 The vision

<p align="center">
  <img src="docs/images/poster_anatomy.png" alt="BODHI Anatomy — Cognitive stack on top of a biological stack" width="100%">
</p>

**One sentence.** BODHI is a neuromorphic experiment that perceives, remembers, feels, dreams, and speaks —
built from first principles in integer math, running offline on a single laptop.

**Why bother.** Modern LLMs are brilliant probability machines with no body, no drives, no sleep, no memory of
yesterday. BODHI is the opposite bet: start from a **400-million-year-old worm circuit**, scale up using **real
human neuroanatomy** (72 Desikan–Killiany regions plus subcortex), and keep the three things that transformers
don't have — **emotion, drive, and overnight consolidation**.

---

# 🏟️ The arena — what you're training for

<p align="center">
  <img src="docs/images/poster_arena.png" alt="BODHI Arena — four phases, six bosses, one prize" width="100%">
</p>

This repo is Phase 1. The real game is four phases long:

1. **Phase 1 — Proof of concept (now).** You install BODHI, raise it for a few months, teach it your world.
2. **Phase 2 — Simulation Playground (Month 2–3).** Your BODHI moves into a 3D world. Emotional labels are stripped — it has to learn by touching fire, feeling hunger, trusting allies, surviving enemies.
3. **Phase 3 — The 12 Challenges (Month 3–9).** Twelve monthly tests of navigation, memory, emotion, cooperation, deception, sleep, planning. Leaderboard narrows to the **top 1,000 BODHIs**.
4. **Phase 4 — The Final Match (end of Year 1).** Those 1,000 enter a sci-fi arena. **Six bosses = the six largest commercial AI systems on Earth.** First player whose BODHI defeats all six wins the prize.

**The prize — the cap table at match time:**

| Share | Who | What for |
|---|---|---|
| **51%** | **Winner** (player whose BODHI defeats all six bosses) | Majority stake in BODHI AI at the then-current valuation |
| **27%** | **Founder (SK, Sai Kiran Bathula)** | Retained founder stake |
| **24%** | **Investors** | Sold before Phase 4 to fund Playground + Challenges |

*(The three shares sum to 102%; final legal formation reconciles to 100% — intent preserved: winner majority, founder meaningful, investors fully returned. Details in [ROADMAP.md](ROADMAP.md).)*

The WHT codec — the patented algorithm inside `bodhi/brain/sensor_wht.py` — is part of the prize. The winner receives the source and the patents along with the 51%.

---

# 🥊 BODHI vs. the six bosses — why the fight is fair

The six AI bosses are bigger, older, and better-funded. But the 12 Challenges don't measure size — they measure
what a *raised* brain can do. Here is the honest match-up.

| Dimension | **BODHI (yours)** | **Big AI (the six bosses)** |
|---|---|---|
| Parameter count | ~40M Broca + worm + 72-region cortex, ~1.1 GB | 100B – 1T+, datacenter-scale |
| Persistent memory across fights | ✅ every concept, dream, emotion carries over | ❌ context window resets between fights |
| Personal fine-tuning | ✅ nightly LoRA on *your* experiences | ❌ same frozen weights for every user |
| Emotional tagging of memories | ✅ valence −255..+255 per concept | ❌ text only, no valence state |
| Drives | ✅ 7 homeostatic drives shape attention | ❌ |
| Sleep + consolidation | ✅ 5-phase cycle rewires weights nightly | ❌ frozen once trained |
| Survival instinct | ✅ worm kernel has veto over every action | ❌ |
| Safety filters that limit responses | ❌ (it's yours — no refusals you didn't ask for) | ✅ heavy refusal behaviour |
| Grounded in activation trace (anti-hallucination) | ✅ brain_voice speaks from the trace | ❌ probability sampling over text |
| Runs offline, fully private | ✅ 380 MB RAM, no GPU, no cloud | ❌ cloud-dependent |
| Raw world knowledge at cold-start | ⚪ thin | ✅ vast |
| Language fluency today | ⚪ mid (brain_voice templated; Broca ~20% good at 1,200 steps) | ✅ high |
| Multilingual | ⚪ Phase 2 | ✅ yes |
| Tool use / retrieval / code execution | ⚪ limited | ✅ strong |
| Training data volume | ⚪ 8 GB seed + what you teach it | ✅ trillions of tokens |
| Maintenance cost | ✅ free, runs on your laptop | ❌ subscription or API bill |

**How to read this table.**

The top half is where **BODHI wins** — memory, emotion, drives, sleep, groundedness, privacy, personal fit. Those
are the capabilities the 12 Challenges measure.

The bottom half is where **big AI wins today** — raw fluency, world knowledge, multilingual reach, cloud-scale
tooling. Those are the things the challenges deliberately *don't* measure, because those are capabilities money
buys, not capabilities a raised brain earns.

The game is not about who knows more trivia. It is about which brain remembers you, adapts to you, dreams about
yesterday, and is still the same brain tomorrow morning. That brain is yours.

---

# 🌀 How the brain works — a guided tour

Each stage below is a **real Python module** in this repo.

<p align="center">
  <img src="docs/images/poster_pipeline.png" alt="BODHI pipeline — sensor to speech in six stages" width="100%">
</p>

## 1. Perception — WHT codec (working, patented)

Human vision splits light into frequency bands; BODHI does the same with a **Walsh–Hadamard Transform** — an
integer-only cousin of the Fourier transform. A 256×256 grayscale frame becomes **24,576 int16 coefficients
(≈48 KB)** grouped into five bands. Audio uses the same trick: 1 s → 1 KB int8.

<p align="center">
  <img src="docs/images/poster_perception.png" alt="WHT perception — frequency bands, regions, compression, example" width="100%">
</p>

Both codecs are covered by issued IP Australia patents: **2026901656** (image) and **2026901657** (audio). The
algorithm **ships as a Cython-compiled binary** (`_wht_core.cpython-<ver>-<platform>.so` on Linux/macOS,
`_wht_core.cp<ver>-<platform>.pyd` on Windows). `bodhi/brain/sensor_wht.py` is a thin public shim so existing code
still `import`s as before. The algorithm itself is not present in the distributed source — it is the Phase-4 prize.

## 2. Learning — how it remembers what matters (working)

BODHI learns like a brain, not like a transformer: **Hebbian co-firing** during the day, **emotional tagging** on
every experience, **seven homeostatic drives** biasing attention, and a **five-phase sleep cycle** that is the only
place synaptic weights actually change.

<p align="center">
  <img src="docs/images/poster_learning.png" alt="Learning — Hebbian wiring, emotional memory, seven drives, five-phase sleep" width="100%">
</p>

*No dream, no learning.*

## 3. Voice — two paths, be honest about both

- **`brain_voice` (primary, reliable).** Takes the 72-region activation trace plus the top-k retrieved memory
  fingerprints and projects them onto templated English. Stable and grounded.
- **Broca LLM (supplementary, improving).** A 40M-parameter int8 transformer (53 MB on disk). At ~1,200 training
  steps it produces good output about **20%** of the time; the other 80% we fall back to `brain_voice`. Every
  nightly LoRA fine-tune nudges the ratio upward.

<p align="center">
  <img src="docs/images/banner_brain_voice.png" alt="Brain voice — terminal demo" width="100%">
</p>

## 4. Self-awareness — five cooperating subsystems (working)

<p align="center">
  <img src="docs/images/poster_self_awareness.png" alt="Five subsystems — narrator, monitor, predictor, critic, dreamer" width="100%">
</p>

Narrator, Monitor, Predictor, Critic, Dreamer. All five are implemented and tested.

## 5. The cockpit

<p align="center">
  <img src="docs/images/banner_dashboard.png" alt="BODHI dashboard — live brain cockpit" width="100%">
</p>

A Flask dashboard lets you ask questions, watch the 72 regions light up, inspect emotional memory, trigger sleep,
and read the dream journal.

---

# 🧠 The ten reports

Each is a dark-cinematic PDF in `docs/reports/`. Click a cover to open.

<table>
<tr>
<td width="20%" align="center"><a href="docs/reports/Report_01_BODHI_Vision.pdf"><img src="docs/images/cover_01.png"></a></td>
<td width="20%" align="center"><a href="docs/reports/Report_02_Worm_Brain.pdf"><img src="docs/images/cover_02.png"></a></td>
<td width="20%" align="center"><a href="docs/reports/Report_03_WHT_Perception.pdf"><img src="docs/images/cover_03.png"></a></td>
<td width="20%" align="center"><a href="docs/reports/Report_04_Human_Brain.pdf"><img src="docs/images/cover_04.png"></a></td>
<td width="20%" align="center"><a href="docs/reports/Report_05_Learning.pdf"><img src="docs/images/cover_05.png"></a></td>
</tr>
<tr>
<td align="center"><b>01</b> The Vision</td>
<td align="center"><b>02</b> Worm Brain</td>
<td align="center"><b>03</b> WHT Perception</td>
<td align="center"><b>04</b> Human Brain</td>
<td align="center"><b>05</b> Learning</td>
</tr>
<tr>
<td align="center"><a href="docs/reports/Report_06_Broca_LLM.pdf"><img src="docs/images/cover_06.png"></a></td>
<td align="center"><a href="docs/reports/Report_07_Stress_Test.pdf"><img src="docs/images/cover_07.png"></a></td>
<td align="center"><a href="docs/reports/Report_08_Consciousness.pdf"><img src="docs/images/cover_08.png"></a></td>
<td align="center"><a href="docs/reports/Report_09_Self_Awareness.pdf"><img src="docs/images/cover_09.png"></a></td>
<td align="center"><a href="docs/reports/Report_10_Brain_Voice_Sleep.pdf"><img src="docs/images/cover_10.png"></a></td>
</tr>
<tr>
<td align="center"><b>06</b> Broca (LLM)</td>
<td align="center"><b>07</b> Stress Test</td>
<td align="center"><b>08</b> Consciousness</td>
<td align="center"><b>09</b> Self-Awareness</td>
<td align="center"><b>10</b> Voice & Sleep</td>
</tr>
</table>

---

# 💻 Developer reference

## Install

### macOS / Linux

```bash
git clone https://github.com/QLNI/bodhi.git
cd bodhi
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional extras:

```bash
pip install soundfile reportlab matplotlib   # audio teaching, PDF reports, plots
```

### Windows (step-by-step, no coding experience needed)

1. **Install Python 3.10+** from https://www.python.org/downloads/windows/.
   During the installer, **tick "Add Python to PATH"** before clicking Install.
2. **Install Git for Windows** from https://git-scm.com/download/win. Defaults
   are fine — click Next until Finish.
3. **Open PowerShell** (press the Start key, type "PowerShell", press Enter).
4. **Clone and set up** BODHI by pasting these lines one at a time:

   ```powershell
   git clone https://github.com/QLNI/bodhi.git
   cd bodhi
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Platform-binary note.** The WHT codec ships as a platform-specific
   compiled file. If the repo you cloned contains only
   `_wht_core.cpython-<ver>-x86_64-linux-gnu.so` (Linux) and you are on
   Windows, you need the Windows-native `.pyd`. Two options:

   - **Easier:** download the matching `_wht_core.cp310-win_amd64.pyd` from
     the [Releases page](https://github.com/QLNI/bodhi/releases) of this
     repo and drop it into `bodhi\brain\`.
   - **If you have Visual Studio Build Tools installed**, you can rebuild
     locally. The maintainer can provide the Cython source on request.

6. **Run BODHI**:

   ```powershell
   python -m bodhi.run "hello"
   python -m bodhi.web
   ```

   The dashboard opens at http://localhost:5000.

## Five-minute demo

```bash
# 1. Wake the worm circuit
python -m bodhi.worm_brain --demo              # 26 neurons, 39 synapses, 7/7 reflex tests

# 2. Perceive — compress an image with WHT
python -m bodhi.brain.sensor_wht path/to/photo.jpg   # ≈48 KB fingerprint

# 3. Talk to it (both paths available)
python -m bodhi.run "Why do I feel stuck?"     # brain trace + brain_voice reply
                                               # Broca LLM draft also printed when coherent

# 4. Open the cockpit
python -m bodhi.web                            # → http://localhost:5000

# 5. Run the eval harness
python -m bodhi.eval_harness                   # expected: 12/12 pass
```

## Architecture at a glance

| Module | What it is | Status |
|---|---|---|
| `worm_brain.py` | C. elegans connectome: **26 neurons (8 sensory + 12 inter + 6 motor), 39 synapses**, int8-range weights (hand-tuned in −60 to +80) | ✅ 7/7 reflex tests pass |
| `human_brain.py` | **72 Desikan–Killiany regions, 76 pathways**, real activation dynamics | ✅ traces verified |
| `brain/sensor_wht.py` + `brain/_wht_core*.so` | Walsh–Hadamard codec (image + audio + video), patents AU 2026901656 / 657 | ✅ today, 🔒 **already Cython-compiled** — algorithm not in distributed source |
| `brain/codec_guard.py` | AES-256-GCM wrapper around user-generated `.wht` fingerprint files | ✅ encrypts user data at rest |
| `episodic.py` | 10,000 concept fingerprints + 632 audio fingerprints, SQLite-backed | ✅ pattern-match recall works |
| `learning.py` / `self_model.py` | Hebbian rewiring + self-model regeneration at sleep | ✅ |
| `goals.py` | 7-drive motivation system | ✅ satisfy/starve loop works |
| `brain_voice.py` | Trace → templated English (**primary** speech path) | ✅ reliable, grounded |
| `broca.py` | SmallGPT: d_model=512, n_layers=10, n_heads=8, ~40M params, int8, 53 MB | 🟡 ~20% good at 1,200 steps, LoRA training nightly |
| `bodhi.py` | 3-pass re-entrance + narrator/monitor/predictor/critic/dreamer | ✅ agreement matrix within healthy range |
| `chat_ui.py` | Flask dashboard, live trace, dream journal | ✅ runs at localhost:5000 |
| `eval_harness.py` | 12 verification checks | ✅ 12/12 pass |

## Key specs (measured, not estimated)

|  |  |
|---|---|
| **Footprint** | ≈1.1 GB on disk, ≈380 MB RAM at rest |
| **Throughput** | ≈4 thoughts/sec on a 2023 M-series laptop, CPU only |
| **WHT encode latency** | ≈12 ms image, ≈3 ms audio |
| **Broca throughput** | ~90 tokens/sec CPU (when its output is used) |
| **Memory capacity** | 10,000 concepts × 48 KB = 480 MB concept store ceiling |
| **Nightly fine-tune** | LoRA rank 8, alpha 16, over the day's experiences |
| **Encryption** | AES-256-GCM via `cryptography.hazmat` on persisted user data |
| **Vocab** | SentencePiece BPE, 8,000 tokens (English-only today) |

## Roadmap at a glance

| Phase | Window | What ships |
|---|---|---|
| **Phase 1 — Proof of concept** | ✅ **this repo** | Offline brain, dashboard, 10 reports, patented codecs, brain_voice, early Broca |
| **Phase 2 — Simulation Playground** | 🟡 Month 2–3 | 3D world, emotion labels stripped, NPC framework, two-BODHI protocol, hazards |
| **Phase 3 — 12 Challenges** | 🔵 Month 3–9 | Twelve monthly tests of navigation, memory, emotion, cooperation, deception, sleep, planning. Top 1,000 qualify |
| **Phase 4 — Final Match** | 🟣 End of Year 1 | Sci-fi arena vs. **6 AI bosses**. Prize: **51% winner / 27% founder / 24% investors** |

Full details and the honest legal caveats are in [ROADMAP.md](ROADMAP.md).

## Tests

```bash
python -m bodhi.eval_harness                   # 12/12 checks: anatomy, memory, emotion, sleep, voice
python -m bodhi.worm_brain --selftest          # 7/7 reflex tests
```

## Project layout

```
bodhi/
├── bodhi/                         # the brain (flat package)
│   ├── worm_brain.py              # 26 neurons, 39 synapses
│   ├── human_brain.py             # 72 regions, 76 pathways
│   ├── brain/
│   │   ├── sensor_wht.py          # Public shim — re-exports from the compiled binary
│   │   ├── _wht_core*.so / .pyd   # 🔒 WHT codec compiled binary — the Phase-4 PRIZE
│   │   ├── build_wht.py           # Rebuild script (needs .pyx source — not distributed)
│   │   └── codec_guard.py         # AES-256-GCM on user-saved fingerprints
│   ├── episodic.py / learning.py / self_model.py / goals.py
│   ├── brain_voice.py             # primary speech path
│   ├── broca.py                   # supplementary LLM (~20% reliable at 1,200 steps)
│   ├── bodhi.py                   # 5 subsystems + 3-pass re-entrance
│   ├── chat_ui.py                 # Flask cockpit
│   └── eval_harness.py            # 12 verification checks
├── docs/
│   ├── images/                    # posters, banners, cover art
│   └── reports/                   # 10 deep-dive PDFs
├── landing/index.html             # dark cinematic one-pager
├── ROADMAP.md                     # full four-phase plan + prize structure
├── requirements.txt
└── README.md                      # you are here
```

---

## 👤 Inventor

**Sai Kiran Bathula** · Coleambally NSW Australia · April 2026
Patents: IP Australia **2026901656** (WHT image) · **2026901657** (WHT audio)

## 📜 License

Apache 2.0 for the proof-of-concept code. The **WHT codec is patented separately** — the patents grant a
royalty-free license for research and personal use. Commercial use requires a separate agreement.
The WHT source is the **prize of the Final Match**: the Phase-4 winner receives it along with the 51% stake.
See `LICENSE` and `PATENTS.md`.

---

<p align="center">
  <i>Raise your BODHI today. Bring it to the arena next year.</i><br>
  <sub>Proof of concept · Apr 2026 · see ROADMAP.md for what's next</sub>
</p>
