import os
import uuid
import json
import random
import sqlite3
import requests
from dotenv import load_dotenv
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from functools import wraps

load_dotenv()

app = Flask(__name__)
app.secret_key = "prompt2pixel-secret-key-2026"

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
HF_API_KEY = os.getenv("HF_API_KEY", "hf_xxx")  # Replace with your HuggingFace API key or set as env variable
HF_API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"

LLM_API_KEY = os.getenv("LLM_API_KEY", "smb_xxx")  # Replace with your LLM API key or set as env variable
LLM_API_URL = "https://api.sambanova.ai/v1/chat/completions"
LLM_MODEL   = "Meta-Llama-3.1-8B-Instruct"

SAVE_DIR = os.path.join("static", "images")
DB_FILE  = "app.db"

os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# NSFW block list
# ─────────────────────────────────────────────
BLOCKED_WORDS = ["nude", "naked", "nsfw", "violence", "gore", "porn", "explicit"]

def is_nsfw(text: str) -> bool:
    return any(word in text.lower() for word in BLOCKED_WORDS)

# ─────────────────────────────────────────────
# Style modifiers
# ─────────────────────────────────────────────
STYLE_MODIFIERS = {
    "realistic": "photorealistic, 8k, sharp focus, professional photography",
    "anime":     "anime style, vibrant colors, cel shading, Studio Ghibli inspired",
    "sketch":    "pencil sketch, hand-drawn, black and white, fine line art",
    "fantasy":   "fantasy art, magical, ethereal, concept art, epic lighting",
    "cinematic": "cinematic, dramatic lighting, movie still, anamorphic lens",
}

# ─────────────────────────────────────────────
# Prompt bank
# ─────────────────────────────────────────────
PROMPT_BANK = [
    "A futuristic city at night with neon reflections on wet streets",
    "An astronaut floating above Earth during golden hour",
    "A lone samurai standing in a misty bamboo forest at dawn",
    "An ancient dragon resting on a mountain peak at sunset",
    "A cozy coffee shop interior on a rainy evening",
    "A cyberpunk street market with holographic signs and crowds",
    "A magical library with glowing books floating in the air",
    "A deep sea creature emerging from the ocean depths",
    "A snow-covered Japanese village under moonlight",
    "An enchanted forest with giant glowing mushrooms",
    "A steampunk airship docking at a cloud city",
    "A desert oasis with a mysterious ancient temple",
    "A little girl in a red coat walking through a dark forest",
    "A Viking longship sailing through a stormy sea at dusk",
    "A futuristic robot playing chess in a dimly lit room",
    "A phoenix rising from golden flames against a dark sky",
    "A peaceful countryside meadow with wildflowers and butterflies",
    "An underground cave system filled with glowing crystals",
    "A bustling alien marketplace on a distant planet",
    "A medieval castle perched on a cliff at sunrise",
]

# ═════════════════════════════════════════════
# DATABASE SETUP
# ═════════════════════════════════════════════
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT    UNIQUE NOT NULL,
            password TEXT    NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER NOT NULL,
            prompt    TEXT    NOT NULL,
            style     TEXT    DEFAULT 'realistic',
            image_url TEXT    NOT NULL,
            timestamp TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER NOT NULL,
            message   TEXT    NOT NULL,
            reply     TEXT    NOT NULL,
            timestamp TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Seed default users
    for username, password in [("admin", "admin123"), ("user1", "pass1234")]:
        c.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)",
                  (username, password))

    conn.commit()
    conn.close()

with app.app_context():
    init_db()

# ─────────────────────────────────────────────
# History helpers (DB-backed, per-user)
# ─────────────────────────────────────────────
def load_history(user_id=None):
    conn = get_db()
    if user_id:
        rows = conn.execute(
            "SELECT * FROM history WHERE user_id=? ORDER BY id DESC LIMIT 50",
            (user_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM history ORDER BY id DESC LIMIT 50"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_to_history(prompt, style, image_path, user_id):
    ts   = datetime.now().strftime("%d %b %Y, %H:%M")
    conn = get_db()
    conn.execute(
        "INSERT INTO history (user_id, prompt, style, image_url, timestamp) VALUES (?,?,?,?,?)",
        (user_id, prompt, style, image_path, ts)
    )
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────
# Chat helpers
# ─────────────────────────────────────────────
def load_chat_history(user_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM chat_history WHERE user_id=? ORDER BY id ASC LIMIT 100",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_chat(user_id, message, reply):
    ts   = datetime.now().strftime("%d %b %Y, %H:%M")
    conn = get_db()
    conn.execute(
        "INSERT INTO chat_history (user_id, message, reply, timestamp) VALUES (?,?,?,?)",
        (user_id, message, reply, ts)
    )
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────
# Login decorator
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("intro"))
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────
# Image generation (unchanged logic)
# ─────────────────────────────────────────────
def generate_image(prompt, style, negative_prompt,
                   width=512, height=512, guidance_scale=7.0, num_inference_steps=30):
    style_tag   = STYLE_MODIFIERS.get(style, "")
    full_prompt = f"{prompt}, {style_tag}" if style_tag else prompt

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type":  "application/json"
    }
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "width":               width,
            "height":              height,
            "guidance_scale":      guidance_scale,
            "num_inference_steps": num_inference_steps,
        },
        "options": {"wait_for_model": True},
    }
    if negative_prompt:
        payload["parameters"]["negative_prompt"] = negative_prompt

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=90)
    if response.status_code != 200:
        raise RuntimeError(f"HuggingFace error: {response.text}")

    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(response.content)
    return f"/static/images/{filename}"

# ─────────────────────────────────────────────
# AI Prompt Improvement (unchanged)
# ─────────────────────────────────────────────
def improve_prompt_with_llm(user_prompt):
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type":  "application/json"
    }
    data = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert AI prompt engineer. Expand prompts into ultra-detailed, cinematic, 8k, high-resolution, photorealistic prompts with lighting, textures, shadows."
            },
            {"role": "user", "content": f"Improve this prompt: {user_prompt}"}
        ]
    }
    response = requests.post(LLM_API_URL, headers=headers, json=data)
    if response.status_code != 200:
        return user_prompt
    try:
        return response.json()["choices"][0]["message"]["content"].strip()
    except:
        return user_prompt

# ═════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════

@app.route("/intro")
def intro():
    if session.get("user"):
        return redirect(url_for("index"))
    return render_template("intro.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        ).fetchone()
        conn.close()
        if user:
            session["user"]    = username
            session["user_id"] = user["id"]
            return redirect(url_for("index"))
        else:
            error = "Invalid username or password."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
@login_required
def index():
    uid     = session.get("user_id")
    history = load_history(uid)
    return render_template("index.html", history=history, user=session.get("user"))

@app.route("/generate", methods=["POST"])
@login_required
def generate():
    prompt          = request.form.get("prompt", "").strip()
    style           = request.form.get("style", "realistic").strip()
    negative_prompt = request.form.get("negative_prompt", "").strip()

    try:
        wh = int(request.form.get("img_size", "512"))
        wh = wh if wh in (512, 768, 1024) else 512
    except:
        wh = 512

    try:
        cfg = float(request.form.get("cfg_scale", "7"))
        cfg = max(1.0, min(10.0, cfg))
    except:
        cfg = 7.0

    try:
        steps = int(request.form.get("steps", "30"))
        steps = max(10, min(50, steps))
    except:
        steps = 30

    if not prompt:
        return jsonify({"error": "Enter a prompt"}), 400
    if is_nsfw(prompt):
        return jsonify({"error": "Restricted content"}), 400

    try:
        image_url = generate_image(prompt, style, negative_prompt,
                                   width=wh, height=wh,
                                   guidance_scale=cfg, num_inference_steps=steps)
        save_to_history(prompt, style, image_url, session["user_id"])
        return jsonify({"image_url": image_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/improve-prompt", methods=["POST"])
@login_required
def improve():
    data   = request.get_json()
    prompt = (data or {}).get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt"}), 400
    return jsonify({"improved_prompt": improve_prompt_with_llm(prompt)})

@app.route("/history")
@login_required
def history():
    return jsonify(load_history(session.get("user_id")))

@app.route("/gallery")
@login_required
def gallery():
    uid     = session.get("user_id")
    history = load_history(uid)
    return render_template("gallery.html", history=history, user=session.get("user"))

@app.route("/variations", methods=["POST"])
@login_required
def variations():
    prompt          = request.form.get("prompt", "").strip()
    style           = request.form.get("style", "realistic").strip()
    negative_prompt = request.form.get("negative_prompt", "").strip()
    uid             = session["user_id"]
    results         = []
    for i in range(3):
        try:
            img = generate_image(prompt, style, negative_prompt)
            save_to_history(f"{prompt} (v{i+1})", style, img, uid)
            results.append(img)
        except:
            pass
    return jsonify({"image_urls": results})

@app.route("/delete-image", methods=["POST"])
@login_required
def delete_image():
    data      = request.get_json()
    image_url = (data or {}).get("image_url", "").strip()
    if not image_url:
        return jsonify({"error": "No image_url provided"}), 400

    relative_path = image_url.lstrip("/")
    file_path     = os.path.join(os.getcwd(), relative_path)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        return jsonify({"error": f"Could not delete file: {e}"}), 500

    conn = get_db()
    conn.execute(
        "DELETE FROM history WHERE image_url=? AND user_id=?",
        (image_url, session["user_id"])
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route("/suggest-prompts", methods=["POST"])
@login_required
def suggest_prompts():
    data    = request.get_json() or {}
    partial = data.get("partial", "").strip().lower()
    if partial:
        matched   = [p for p in PROMPT_BANK if partial in p.lower()]
        unmatched = [p for p in PROMPT_BANK if partial not in p.lower()]
        pool      = matched + unmatched
    else:
        pool = PROMPT_BANK
    suggestions = random.sample(pool, min(5, len(pool)))
    return jsonify({"suggestions": suggestions})

# ─────────────────────────────────────────────
# CHAT routes
# ─────────────────────────────────────────────
@app.route("/chat-history")
@login_required
def get_chat_history():
    uid   = session.get("user_id")
    chats = load_chat_history(uid)
    return jsonify(chats)

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data    = request.get_json() or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "No message"}), 400

    uid = session.get("user_id")
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Pixel, a friendly AI assistant for Prompt2Pixel, "
                    "an AI image generation studio. Help users craft great prompts, "
                    "explain styles, and answer questions about the app. "
                    "Keep replies short and friendly (2-3 sentences max)."
                )
            },
            {"role": "user", "content": message}
        ]
    }
    try:
        resp  = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=30)
        reply = resp.json()["choices"][0]["message"]["content"].strip()
    except:
        reply = "Sorry, I'm having trouble right now. Please try again!"

    save_chat(uid, message, reply)
    return jsonify({"reply": reply})

# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    uid  = session.get("user_id")
    conn = get_db()

    total = conn.execute(
        "SELECT COUNT(*) as cnt FROM history WHERE user_id=?", (uid,)
    ).fetchone()["cnt"]

    today_str = datetime.now().strftime("%d %b %Y")
    today = conn.execute(
        "SELECT COUNT(*) as cnt FROM history WHERE user_id=? AND timestamp LIKE ?",
        (uid, f"{today_str}%")
    ).fetchone()["cnt"]

    daily_limit = 20
    remaining   = max(0, daily_limit - today)

    style_row = conn.execute(
        "SELECT style, COUNT(*) as cnt FROM history WHERE user_id=? GROUP BY style ORDER BY cnt DESC LIMIT 1",
        (uid,)
    ).fetchone()
    fav_style = style_row["style"].capitalize() if style_row else "—"

    recent = conn.execute(
        "SELECT * FROM history WHERE user_id=? ORDER BY id DESC LIMIT 12",
        (uid,)
    ).fetchall()

    conn.close()
    return render_template("dashboard.html",
        user      = session.get("user"),
        total     = total,
        today     = today,
        remaining = remaining,
        fav_style = fav_style,
        recent    = [dict(r) for r in recent],
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)