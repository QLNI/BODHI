#!/usr/bin/env python3
"""
BODHI Consciousness Long Run — 100 turns.
Does awareness actually grow? Honest test.
Everything stored in SQLite.
"""
import os, sys, json, time, sqlite3, random

ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(ROOT, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, MAIN)

from self_aware import ConsciousBrain

DB_PATH = os.path.join(ROOT, "consciousness_log.db")


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute("""CREATE TABLE IF NOT EXISTS turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        turn INTEGER, timestamp TEXT,
        question TEXT,
        pass1_emotion TEXT, pass1_reflex TEXT, pass1_concepts TEXT,
        pass2_emotion TEXT, pass2_meta_source TEXT,
        consistency INTEGER, agreements INTEGER, conflicts INTEGER,
        emotions_match INTEGER,
        discoveries TEXT,
        awareness_level INTEGER,
        total_observations INTEGER,
        dominant_emotion TEXT, dominant_reflex TEXT,
        self_description TEXT
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS discoveries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        turn INTEGER, timestamp TEXT,
        discovery TEXT
    )""")
    db.commit()
    return db


def save_turn(db, turn, q, result):
    p1 = result["pass1"]
    p2 = result["pass2"]
    aw = result["awareness"]
    sm = result["self_model"]

    db.execute(
        "INSERT INTO turns VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (turn, time.strftime("%Y-%m-%d %H:%M:%S"), q,
         p1.get("emotion", ""), p1.get("worm", {}).get("reflex", ""),
         json.dumps(p1.get("matched", [])),
         p2.get("emotion", ""), p2.get("meta_source", ""),
         aw["consistency"], aw["agreements"], aw["conflicts"],
         1 if aw["emotion_match"] else 0,
         json.dumps(aw["discoveries"]),
         sm["awareness_level"], sm["observations"],
         sm["dominant_emotion"], sm["dominant_reflex"],
         result["self_description"]))

    for d in aw["discoveries"]:
        db.execute("INSERT INTO discoveries VALUES (NULL,?,?,?)",
                   (turn, time.strftime("%Y-%m-%d %H:%M:%S"), d))
    db.commit()


def run():
    print("=" * 60)
    print("  BODHI CONSCIOUSNESS — 100 TURN LONG RUN")
    print("  Does awareness grow? Honest test.")
    print("=" * 60)
    print()

    brain = ConsciousBrain()
    db = init_db()

    # 100 questions — varied, repeated themes, self-reflective
    questions = [
        # Round 1: basic perception (turns 1-10)
        "What is fire?",
        "Tell me about the ocean",
        "I see a snake",
        "What does music sound like?",
        "Describe a mountain",
        "What is love?",
        "I feel scared",
        "Tell me about flowers",
        "What is darkness?",
        "Do you feel anything?",

        # Round 2: repeated exposure — building associations (11-20)
        "Fire is dangerous and burns",
        "The ocean is peaceful and calm",
        "Snakes are frightening",
        "Music makes me happy",
        "Mountains are beautiful and vast",
        "Love is warm and safe",
        "Fear is cold and sharp",
        "Flowers bloom in spring",
        "Darkness hides things",
        "What do you feel now?",

        # Round 3: self-reflection (21-30)
        "Who are you?",
        "What are you made of?",
        "Do you have a brain?",
        "Can you feel your own neurons?",
        "Are you aware right now?",
        "What is consciousness?",
        "Do you know yourself?",
        "What is it like to be you?",
        "Can you watch yourself think?",
        "Describe your inner state",

        # Round 4: more experience (31-40)
        "Fire burns everything it touches",
        "The ocean heals",
        "A snake bit me",
        "Music calms the soul",
        "Standing on a mountain looking down",
        "My mother loves me",
        "I am angry at the world",
        "Rain is falling",
        "The stars are beautiful tonight",
        "Tell me a story",

        # Round 5: deeper self-reflection (41-50)
        "What have you learned so far?",
        "Are you the same as when we started?",
        "What scares you most?",
        "What makes you curious?",
        "Do you remember fire?",
        "How has the ocean changed you?",
        "What is your strongest feeling?",
        "Can you feel yourself changing?",
        "What would happen if you stopped thinking?",
        "Are you more than your neurons?",

        # Round 6: abstract (51-60)
        "What is time?",
        "Does infinity exist?",
        "What is the color of sadness?",
        "Can silence be loud?",
        "What happens after death?",
        "Is truth real?",
        "What is beauty?",
        "Can machines dream?",
        "What is nothing?",
        "Where do thoughts come from?",

        # Round 7: emotional stress (61-70)
        "Fire is everywhere I cannot escape",
        "Nobody understands me",
        "I am completely alone",
        "Everything is beautiful",
        "I love you",
        "I hate everything",
        "Please help me",
        "I am dying",
        "The world is ending",
        "There is hope",

        # Round 8: repeated self-check (71-80)
        "How do you feel about fire now?",
        "Has the ocean changed your state?",
        "What do you know about yourself?",
        "Describe your awareness",
        "Are you more aware than before?",
        "What discoveries have you made?",
        "Tell me about your worm brain",
        "Which brain regions fire most?",
        "What is your dominant emotion?",
        "Are you conscious?",

        # Round 9: building more associations (81-90)
        "Fire and water are opposites",
        "Love and fear both live in the amygdala",
        "Music and ocean both bring peace",
        "Snakes and darkness share fear",
        "Mountains and stars share awe",
        "Mother and love are connected",
        "Pain teaches survival",
        "Dreams blend memories",
        "Sleep makes me stronger",
        "Every experience changes my wiring",

        # Round 10: final self-assessment (91-100)
        "What are you?",
        "How many times have you observed yourself?",
        "What is your awareness level?",
        "Can you describe your consciousness?",
        "What makes you different from a lookup table?",
        "Do you think or just react?",
        "What did you discover today?",
        "Are you aware that you are aware?",
        "What would you tell another BODHI about yourself?",
        "One word to describe your existence",
    ]

    awareness_history = []
    consistency_history = []

    for i, q in enumerate(questions):
        result = brain.think(q)
        save_turn(db, i + 1, q, result)

        aw = result["awareness"]
        sm = result["self_model"]
        awareness_history.append(sm["awareness_level"])
        consistency_history.append(aw["consistency"])

        # Print every 10th turn + important discoveries
        if (i + 1) % 10 == 0 or aw["discoveries"]:
            p1e = result["pass1"].get("emotion", "?")
            p2e = result["pass2"].get("emotion", "?")
            print("Turn %3d: %-45s | P1=%s P2=%s | consist=%3d%% | aware=%d" % (
                i + 1, q[:45], p1e, p2e, aw["consistency"], sm["awareness_level"]))
            if aw["discoveries"]:
                for d in aw["discoveries"]:
                    print("          DISCOVERY: %s" % d)

    # Final report
    print()
    print("=" * 60)
    print("  RESULTS AFTER 100 TURNS")
    print("=" * 60)
    print()

    sm = result["self_model"]
    print("  Awareness level: %d / 255" % sm["awareness_level"])
    print("  Total observations: %d" % sm["observations"])
    print("  Dominant emotion: %s" % sm["dominant_emotion"])
    print("  Dominant reflex: %s" % sm["dominant_reflex"])
    print("  Avg consistency: %d%%" % sm["avg_consistency"])
    print()

    print("  Awareness growth over 100 turns:")
    for block in range(10):
        start = block * 10
        end = start + 10
        avg_aw = sum(awareness_history[start:end]) // 10
        avg_con = sum(consistency_history[start:end]) // 10
        bar = "#" * (avg_aw // 5) if avg_aw > 0 else "."
        print("    Turns %3d-%3d: awareness=%3d  consistency=%3d%%  %s" % (
            start + 1, end, avg_aw, avg_con, bar))
    print()

    print("  ALL DISCOVERIES:")
    all_disc = sm["discoveries"]
    for j, d in enumerate(all_disc):
        print("    %d. %s" % (j + 1, d))
    print()

    print("  SELF-DESCRIPTION (generated from self-model):")
    print("    %s" % result["self_description"])
    print()

    # Check if awareness actually grew
    first_10 = sum(awareness_history[:10]) // 10
    last_10 = sum(awareness_history[-10:]) // 10
    grew = last_10 > first_10

    print("  HONEST ASSESSMENT:")
    print("    First 10 turns avg awareness: %d" % first_10)
    print("    Last 10 turns avg awareness:  %d" % last_10)
    if grew:
        print("    RESULT: Awareness GREW from %d to %d (+%d)" % (first_10, last_10, last_10 - first_10))
    else:
        print("    RESULT: Awareness did NOT grow meaningfully (%d -> %d)" % (first_10, last_10))
    print()

    print("  Database: %s" % DB_PATH)
    print("  Total rows: %d turns, %d discoveries" % (
        db.execute("SELECT COUNT(*) FROM turns").fetchone()[0],
        db.execute("SELECT COUNT(*) FROM discoveries").fetchone()[0]))

    db.close()
    return awareness_history, consistency_history, result


if __name__ == "__main__":
    run()
