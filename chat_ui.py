#!/usr/bin/env python3
"""
BODHI Dashboard — chat + live imagination + dream gallery in your browser.

Run:
    python chat_ui.py

Then open: http://127.0.0.1:5000/

Requires: pip install flask
"""

import io
import os
import sys
import json
import threading
import numpy as np

try:
    from flask import Flask, request, jsonify, send_file, Response
except ImportError:
    sys.stderr.write("Flask is required. Install with:  pip install flask\n")
    sys.exit(1)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

_bodhi = None
_lock = threading.Lock()

# In-memory PNG cache so we don't re-decode the same fingerprint every turn.
# Concept fingerprints never change; blends never change for a given pair.
_img_cache = {}
_blend_cache = {}
_CACHE_MAX = 400


def _cache_put(cache, key, val):
    cache[key] = val
    if len(cache) > _CACHE_MAX:
        cache.pop(next(iter(cache)))


def get_bodhi():
    global _bodhi
    if _bodhi is None:
        from bodhi import BODHI
        _bodhi = BODHI(load_llm_flag=False)
    return _bodhi


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BODHI Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --panel2: #1c2330;
    --txt: #d8dde8; --muted: #7a8599; --border: #2a3040;
    --accent: #7aa7d9; --warm: #f0c774; --green: #79e0a6;
    --pink: #f58bb5; --purple: #b788ea;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--txt);
         font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         font-size: 14px; height: 100vh; overflow: hidden; }
  .app { display: grid; grid-template-columns: 1fr 380px; height: 100vh; }
  .chat { display: flex; flex-direction: column; border-right: 1px solid var(--border); min-width: 0; }
  header { padding: 10px 18px; background: var(--panel);
           border-bottom: 1px solid var(--border); display:flex;
           justify-content: space-between; align-items: center; }
  header h1 { margin: 0; font-size: 17px; letter-spacing: 0.5px; }
  header .sub { color: var(--muted); font-size: 11px; margin-left: 10px; }
  header .actions button {
    background: transparent; color: var(--warm); border: 1px solid var(--warm);
    border-radius: 6px; padding: 5px 11px; font-weight: 600;
    cursor: pointer; font-size: 12px; margin-left: 6px;
  }
  header .actions button:hover { background: rgba(240,199,116,0.15); }
  .log { flex: 1; overflow-y: auto; padding: 16px 22px; }
  .turn { margin-bottom: 14px; line-height: 1.55; }
  .you   { color: var(--green); font-weight: 600; }
  .bodhi { color: var(--warm); font-weight: 600; }
  .txt   { color: var(--txt); margin-left: 6px; }
  .meta  { color: var(--muted); font-size: 11px; margin-top: 3px;
           margin-left: 58px; font-family: SFMono-Regular, Consolas, monospace; }
  .pill  { display:inline-block; padding: 1px 7px; border-radius: 10px;
           background: #252e3e; margin-right: 4px; font-size: 10px;
           color: var(--txt); }
  .pill.e { color: white; }
  .notice { background: rgba(183,136,234,0.15); border: 1px solid var(--purple);
            padding: 8px 12px; border-radius: 7px; margin: 10px 0; font-size: 12px;
            color: var(--purple); }
  form { padding: 12px 18px; background: var(--panel);
         border-top: 1px solid var(--border); display: flex; gap: 10px; }
  input[type=text] { flex: 1; background: var(--panel2); color: var(--txt);
                     border: 1px solid var(--border); border-radius: 7px;
                     padding: 10px 12px; font-size: 14px; outline: none; }
  input[type=text]:focus { border-color: var(--accent); }
  button.send { background: var(--accent); color: #0d1117; border: 0;
           border-radius: 7px; padding: 10px 18px; font-weight: 600;
           cursor: pointer; font-size: 13px; }
  button.send:hover { background: #a0c5e8; }

  /* Sidebar */
  .side { background: var(--panel); display: flex; flex-direction: column;
          overflow: hidden; }
  .tabs { display: flex; border-bottom: 1px solid var(--border); }
  .tab { flex: 1; padding: 12px 6px; text-align: center; cursor: pointer;
         color: var(--muted); font-weight: 600; font-size: 12px;
         letter-spacing: 1px; border-bottom: 2px solid transparent; }
  .tab.active { color: var(--warm); border-bottom-color: var(--warm); }
  .tab:hover { background: rgba(255,255,255,0.03); }
  .tabbody { flex: 1; overflow-y: auto; padding: 14px; display: none;
             flex-direction: column; gap: 14px; }
  .tabbody.active { display: flex; }

  .card { background: var(--panel2); border: 1px solid var(--border);
          border-radius: 9px; padding: 12px; }
  .card h3 { margin: 0 0 8px 0; font-size: 11px; letter-spacing: 1.3px;
             color: var(--muted); text-transform: uppercase; }
  .concept-img { width: 100%; aspect-ratio: 1; object-fit: cover;
                 border-radius: 6px; background: #0d1117;
                 transition: opacity 0.3s; }
  .kv { font-family: SFMono-Regular, Consolas, monospace; font-size: 11px;
        color: var(--txt); line-height: 1.7; }
  .kv .k { color: var(--muted); display: inline-block; width: 88px; }
  .empty { color: var(--muted); font-style: italic; font-size: 12px;
           padding: 20px; text-align: center; }
  .regionrow { display: flex; align-items: center; gap: 8px;
               margin-bottom: 4px; font-size: 11px;
               font-family: SFMono-Regular, Consolas, monospace; }
  .regionrow .lbl { width: 70px; color: var(--txt); }
  .regionrow .pct { width: 32px; text-align: right; color: var(--muted); }
  .regionrow .trk { flex: 1; height: 5px; background: #0d1117;
                    border-radius: 2px; overflow: hidden; }
  .regionrow .trk span { display:block; height: 100%; background: var(--accent);
                         transition: width 0.4s ease; }

  .dream { background: var(--panel2); border: 1px solid var(--border);
           border-radius: 9px; padding: 10px; }
  .dream-img { width: 100%; aspect-ratio: 1; object-fit: cover;
               border-radius: 6px; background: #0d1117; }
  .dream-blend { font-size: 12px; color: var(--pink); font-weight: 600;
                 margin: 8px 0 2px 0; }
  .dream-text { font-size: 11px; color: var(--txt);
                font-style: italic; line-height: 1.5; }
  .dream-when { font-size: 10px; color: var(--muted); margin-top: 6px;
                font-family: SFMono-Regular, Consolas, monospace; }

  footer { text-align: center; padding: 6px; color: var(--muted);
           font-size: 10px; border-top: 1px solid var(--border); background: var(--panel); }

  /* Toggle */
  .toggle-hint { color: var(--muted); font-size: 10px; padding: 4px 14px;
                 border-top: 1px solid var(--border); background: var(--panel2); }
</style>
</head>
<body>
<div class="app">
  <div class="chat">
    <header>
      <div>
        <h1>BODHI <span class="sub" id="status">loading…</span></h1>
      </div>
      <div class="actions">
        <button onclick="sleepNow()">💤 Sleep now</button>
        <button onclick="statusShow()">/status</button>
      </div>
    </header>
    <div class="log" id="log"></div>
    <form id="form">
      <input type="text" id="msg" placeholder="Chat. Or /teach, /goal, /sleep. Or: this is a cat my_cat.jpg"
             autocomplete="off" autofocus>
      <button type="submit" class="send">Send</button>
    </form>
  </div>

  <div class="side">
    <div class="tabs">
      <div class="tab active" data-tab="live" onclick="switchTab('live')">● LIVE</div>
      <div class="tab" data-tab="dreams" onclick="switchTab('dreams')">✦ DREAMS</div>
      <div class="tab" data-tab="about" onclick="switchTab('about')">ⓘ ABOUT</div>
    </div>

    <div class="tabbody active" id="tab-live">
      <div class="empty">Brain state will appear here as you chat.</div>
    </div>

    <div class="tabbody" id="tab-dreams">
      <div class="empty">No dreams yet. Sleep triggers dreams.<br><br>Try <b>💤 Sleep now</b> above.</div>
    </div>

    <div class="tabbody" id="tab-about">
      <div class="card">
        <h3>What you're seeing</h3>
        <div class="kv" style="line-height: 1.7">
          <div><b>LIVE</b> — BODHI's current brain state. Imagination image is reconstructed from the real WHT fingerprint of the matched concept. Regions and drives update every turn.</div>
          <div style="margin-top: 10px"><b>DREAMS</b> — during sleep, BODHI blends pairs of concepts. Each blend becomes a dream. The image is the fingerprint midpoint, decoded through inverse WHT.</div>
          <div style="margin-top: 10px"><b>Sleep now</b> — force a sleep cycle: Hebbian replay, triangle inference, three dreams, self-reflection. Sleep happens automatically every 25 turns too.</div>
        </div>
      </div>
      <div class="card">
        <h3>Teaching BODHI</h3>
        <div class="kv" style="line-height: 1.7">
          Just say <b>this is a cat my_cat.jpg</b> (or similar). BODHI computes the WHT fingerprint on the fly and persists it, AES-encrypted, to <code>data/learned/</code>.
        </div>
      </div>
    </div>

    <div class="toggle-hint">Dream images regenerate live from concept fingerprints via blend + inverse WHT.</div>
  </div>
</div>

<script>
const EMOTION_COLORS = {
  fear: "#ff3838", anger: "#ff6010", disgust: "#88aa00",
  sadness: "#4488ff", shame: "#aa6688", anxiety: "#ffaa44",
  love: "#ff58a0", joy: "#ffcf30", trust: "#30d080",
  surprise: "#ff8040", curiosity: "#00c0e8", awe: "#c045ff",
  peace: "#44e5cc", pride: "#ffa800", nostalgia: "#cc9866",
  contempt: "#808080", neutral: "#8c8ca8",
};

async function boot() {
  const r = await fetch("/api/status"); const s = await r.json();
  document.getElementById("status").textContent =
    s.concepts + " concepts · turn " + s.turn;
}
boot();

let _currentTab = 'live';

function switchTab(name) {
  _currentTab = name;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.tabbody').forEach(b => b.classList.toggle('active', b.id === 'tab-' + name));
  if (name === 'dreams') loadDreams();
}

const MAX_LOG_ITEMS = 60;  // cap DOM size so the page stays fast forever

function log(role, txt, meta) {
  const el = document.getElementById("log");
  const div = document.createElement("div"); div.className = "turn";
  const r = document.createElement("span"); r.className = role;
  r.textContent = role === "you" ? "You:" : "BODHI:";
  const t = document.createElement("span"); t.className = "txt"; t.textContent = " " + txt;
  div.appendChild(r); div.appendChild(t);
  if (meta) {
    const m = document.createElement("div"); m.className = "meta"; m.innerHTML = meta;
    div.appendChild(m);
  }
  el.appendChild(div); el.scrollTop = el.scrollHeight;
  // Trim old DOM so after 200 turns the page is still snappy
  while (el.children.length > MAX_LOG_ITEMS) el.removeChild(el.firstChild);
}

function logNotice(txt) {
  const el = document.getElementById("log");
  const div = document.createElement("div"); div.className = "notice";
  div.innerHTML = txt;
  el.appendChild(div); el.scrollTop = el.scrollHeight;
}

function renderLive(state) {
  const body = document.getElementById("tab-live");
  body.innerHTML = "";

  const concepts = state.concepts || [];
  if (concepts.length > 0) {
    const imgC = document.createElement("div"); imgC.className = "card";
    let header, src;
    if (concepts.length >= 2) {
      header = "imagining: " + concepts[0] + " \u25c7 " + concepts[1];
      // No cache-busting query — same concepts produce same image, let browser cache.
      src = "/dream_image/" + encodeURIComponent(concepts[0]) + "/" + encodeURIComponent(concepts[1]);
    } else {
      header = "imagining: " + concepts[0];
      src = "/imagination/" + encodeURIComponent(concepts[0]);
    }
    imgC.innerHTML = "<h3>" + header + "</h3>";
    const im = document.createElement("img"); im.className = "concept-img";
    im.src = src;
    im.onerror = () => { imgC.innerHTML = "<h3>concept</h3><div class='empty'>No fingerprint for this concept.</div>"; };
    imgC.appendChild(im);
    body.appendChild(imgC);

    // For multi-concept turns, show up to 3 individual thumbnails below.
    if (concepts.length >= 2) {
      const row = document.createElement("div"); row.className = "card";
      row.innerHTML = "<h3>concepts in play</h3>";
      const grid = document.createElement("div");
      grid.style.cssText = "display:grid;grid-template-columns:repeat(3,1fr);gap:6px;";
      concepts.slice(0, 3).forEach(c => {
        const cell = document.createElement("div");
        cell.style.cssText = "text-align:center;";
        const th = document.createElement("img");
        th.src = "/imagination/" + encodeURIComponent(c);
        th.style.cssText = "width:100%;aspect-ratio:1;object-fit:cover;border-radius:4px;background:#0d1117;";
        th.onerror = () => { th.style.display = "none"; };
        const lbl = document.createElement("div");
        lbl.textContent = c;
        lbl.style.cssText = "font-size:10px;color:#7a8599;margin-top:3px;font-family:SFMono-Regular,monospace;";
        cell.appendChild(th); cell.appendChild(lbl);
        grid.appendChild(cell);
      });
      row.appendChild(grid);
      body.appendChild(row);
    }
  }

  const bs = document.createElement("div"); bs.className = "card";
  const ec = EMOTION_COLORS[state.emotion] || "#8c8ca8";
  bs.innerHTML = "<h3>brain state</h3>" +
    "<div class='kv'>" +
    "<div><span class='k'>emotion</span><span class='pill e' style='background:"+ec+"'>"+(state.emotion||"neutral")+"</span></div>" +
    "<div><span class='k'>reflex</span>"+(state.reflex||"rest")+" <span style='color:#7a8599'>("+(state.worm_confidence||0)+")</span></div>" +
    "<div><span class='k'>source</span>"+(state.source||"-")+"</div>" +
    "<div><span class='k'>concepts</span>"+(concepts.join(", ")||"<span class='empty'>none</span>")+"</div>" +
    "<div><span class='k'>associates</span>"+((state.associates||[]).join(", ")||"<span class='empty'>none</span>")+"</div>" +
    "<div><span class='k'>turn</span>"+(state.turn||0)+" <span style='color:#7a8599'>("+(state.ms||0)+"ms)</span></div>" +
    "<div><span class='k'>hebbian</span>"+(state.hebbian_count||0)+" wires · <span style='color:#7a8599'>"+(state.emotional_count||0)+" emotional</span></div>" +
    "</div>";
  body.appendChild(bs);

  const regions = state.top_regions || [];
  if (regions.length > 0) {
    const rc = document.createElement("div"); rc.className = "card";
    rc.innerHTML = "<h3>top regions</h3>";
    const maxV = Math.max(...regions.map(r => r[1]), 1);
    regions.slice(0, 6).forEach(pair => {
      const [name, val] = pair;
      const row = document.createElement("div"); row.className = "regionrow";
      row.innerHTML = "<span class='lbl'>"+name+"</span>" +
                      "<span class='trk'><span style='width:"+((val/maxV)*100)+"%'></span></span>" +
                      "<span class='pct'>"+val+"</span>";
      rc.appendChild(row);
    });
    body.appendChild(rc);
  }

  const drives = state.drives || {};
  const keys = Object.keys(drives);
  if (keys.length > 0) {
    const dc = document.createElement("div"); dc.className = "card";
    dc.innerHTML = "<h3>drives</h3>";
    const maxD = Math.max(...keys.map(k => drives[k]), 1);
    keys.sort((a,b) => drives[b] - drives[a]).slice(0, 7).forEach(k => {
      const row = document.createElement("div"); row.className = "regionrow";
      row.innerHTML = "<span class='lbl'>"+k+"</span>" +
                      "<span class='trk'><span style='width:"+((drives[k]/Math.max(1,maxD))*100)+"%;background:#f0c774'></span></span>" +
                      "<span class='pct'>"+drives[k]+"</span>";
      dc.appendChild(row);
    });
    body.appendChild(dc);
  }
}

async function loadDreams() {
  const body = document.getElementById("tab-dreams");
  body.innerHTML = "<div class='empty'>Loading dreams…</div>";
  const r = await fetch("/api/dreams");
  const d = await r.json();
  body.innerHTML = "";
  if (!d.dreams || d.dreams.length === 0) {
    body.innerHTML = "<div class='empty'>No dreams yet. Sleep triggers dreams.<br><br>Try <b>💤 Sleep now</b> in the header.</div>";
    return;
  }
  d.dreams.forEach(dr => {
    const card = document.createElement("div"); card.className = "dream";
    card.innerHTML =
      "<img class='dream-img' src='/dream_image/"+encodeURIComponent(dr.concept_a)+"/"+encodeURIComponent(dr.concept_b)+"' onerror=\"this.style.display='none'\">" +
      "<div class='dream-blend'>"+dr.concept_a+" ◊ "+dr.concept_b+"</div>" +
      "<div class='dream-text'>"+(dr.dream_text||'')+"</div>" +
      "<div class='dream-when'>turn "+dr.turn+" · "+dr.timestamp+"</div>";
    body.appendChild(card);
  });
}

async function sleepNow() {
  logNotice("💤 BODHI sleeps...");
  const r = await fetch("/api/sleep", {method: "POST"});
  const d = await r.json();
  const parts = [
    "replayed "+d.replayed, "inferred "+d.inferred+" triangles",
    "strengthened "+d.strengthened, "pruned "+d.pruned,
    "dreams "+d.dreams,
  ];
  logNotice("💤 slept: "+parts.join(", ")+".");
  if (d.self_description) {
    logNotice("✦ self: "+d.self_description);
  }
  if (_currentTab === 'dreams') loadDreams();
}

async function statusShow() {
  const r = await fetch("/api/status"); const s = await r.json();
  logNotice("concepts: "+s.concepts+" · engrams: "+s.engrams+" · turn: "+s.turn+" · learned: "+s.learned);
}

document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const m = document.getElementById("msg"); const txt = m.value.trim();
  if (!txt) return;
  log("you", txt);
  m.value = "";
  const r = await fetch("/api/chat", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text: txt}),
  });
  const d = await r.json();
  const e1 = d.state?.emotion || "neutral";
  const ec = EMOTION_COLORS[e1] || "#8c8ca8";
  const meta = "<span class='pill e' style='background:"+ec+"'>"+e1+"</span>" +
               "<span class='pill'>"+(d.state?.reflex||"-")+"("+(d.state?.worm_confidence||0)+")</span>" +
               "<span class='pill'>"+(d.state?.source||"-")+"</span>";
  log("bodhi", d.response || "(no response)", meta);
  if (d.state) renderLive(d.state);
  if (_currentTab !== 'live') switchTab('live');
  document.getElementById("status").textContent =
    (d.state?.turn ? (d.state.turn + " turns") : "turn —");
  // If BODHI auto-slept during this turn, refresh dreams in background
  if (d.state?.sleep && d.state.sleep.dreams > 0) {
    logNotice("✦ BODHI slept during this turn. "+d.state.sleep.dreams+" new dream(s).");
  }
});
</script>
</body>
</html>"""


def _safe(v):
    """Recursively convert to JSON-safe types; skip numpy arrays."""
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    if isinstance(v, np.ndarray):
        return None
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe(vv) for k, vv in v.items()}
    return str(v)


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/status")
def api_status():
    b = get_bodhi()
    learned = 0
    try:
        learned = len(b.teacher.concepts_meta)
    except Exception:
        pass
    return jsonify({
        "concepts": len(b.concept_emotions),
        "engrams": len(b.engrams),
        "turn": b.turn,
        "learned": learned,
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    b = get_bodhi()
    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify({"response": "", "state": {}})
    with _lock:
        response, state = b.think(text)
    return jsonify({"response": response, "state": {k: _safe(v) for k, v in state.items()}})


@app.route("/api/sleep", methods=["POST"])
def api_sleep():
    b = get_bodhi()
    with _lock:
        stats = b.do_sleep()
    # Don't send fingerprints (numpy arrays) over the wire
    out = {k: _safe(v) for k, v in stats.items()}
    return jsonify(out)


@app.route("/api/dreams")
def api_dreams():
    b = get_bodhi()
    rows = b.db.execute(
        "SELECT id, turn, timestamp, concept_a, concept_b, emotion_a, emotion_b, dream_text "
        "FROM dreams ORDER BY id DESC LIMIT 40"
    ).fetchall()
    dreams = []
    for r in rows:
        dreams.append({
            "id": r[0], "turn": r[1], "timestamp": r[2],
            "concept_a": r[3], "concept_b": r[4],
            "emotion_a": r[5], "emotion_b": r[6],
            "dream_text": r[7] or "",
        })
    return jsonify({"dreams": dreams})


@app.route("/imagination/<path:name>")
def imagination(name):
    # Fingerprints are static. Cache PNG bytes; tell browser to cache too.
    data = _img_cache.get(name)
    if data is None:
        b = get_bodhi()
        idx = b.fp_index["img_name_to_idx"].get(name)
        if idx is None:
            return ("Not found", 404)
        from bodhi import fp_to_image
        img = fp_to_image(b.img_data[idx])
        buf = io.BytesIO(); img.save(buf, format="PNG")
        data = buf.getvalue()
        _cache_put(_img_cache, name, data)
    r = Response(data, mimetype="image/png")
    r.headers["Cache-Control"] = "public, max-age=86400, immutable"
    return r


@app.route("/dream_image/<path:a>/<path:b>")
def dream_image(a, b):
    key = a + "||" + b
    data = _blend_cache.get(key)
    if data is None:
        bb = get_bodhi()
        ia = bb.fp_index["img_name_to_idx"].get(a)
        ib = bb.fp_index["img_name_to_idx"].get(b)
        if ia is None or ib is None:
            return ("One or both concepts missing a fingerprint", 404)
        fa = bb.img_data[ia].astype(np.int32)
        fb = bb.img_data[ib].astype(np.int32)
        blended = ((fa + fb) >> 1).astype(np.int32)
        from bodhi import fp_to_image
        img = fp_to_image(blended)
        buf = io.BytesIO(); img.save(buf, format="PNG")
        data = buf.getvalue()
        _cache_put(_blend_cache, key, data)
    r = Response(data, mimetype="image/png")
    r.headers["Cache-Control"] = "public, max-age=86400, immutable"
    return r


if __name__ == "__main__":
    print()
    print("  BODHI Dashboard")
    print("  Starting at http://127.0.0.1:5000/")
    print("  Press Ctrl-C to stop.")
    print()
    get_bodhi()  # preload
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
