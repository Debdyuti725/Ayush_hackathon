# app.py — AyurBalance (EfficientNet-B3 + Gemini) with flashcards + centered modal popup (no gr.Dialog)
# FULL DROP-IN FILE
# CHANGES IN THIS VERSION:
# ✅ Nutrition card bubbles show ONLY words: Calories, Protein, Carbs, Fat, Fiber, Sodium (no numbers)
# ✅ All 4 flashcards forced to equal size (largest size applied to all)
# ✅ Image upload stays upload-only (no webcam / clipboard)
# ✅ Everything else kept the same

import os
import re
import html
import torch
import torch.nn.functional as F
from PIL import Image

import gradio as gr
import torchvision.transforms as T
import timm

from google import genai
from google.genai import types

# =========================
# 0) CONFIG
# =========================
IMG_SIZE = 320
TOP_K = 2

CLASS_NAMES = [
    'aloo gobi', 'aloo methi', 'aloo mutter', 'aloo paratha', 'amritsari kulcha', 'anda curry', 'balushahi',
    'banana chips', 'besan laddu', 'bhindi masala', 'biryani', 'boondi laddu', 'chaas', 'chana masala', 'chapati',
    'chicken pizza', 'chicken wings', 'chikki', 'chivda', 'chole bhature', 'dabeli', 'dal khichdi', 'dhokla',
    'falooda', 'fish curry', 'gajar ka halwa', 'garlic bread', 'garlic naan', 'ghevar', 'grilled sandwich',
    'gujhia', 'gulab jamun', 'hara bhara kabab', 'idiyappam', 'idli', 'jalebi', 'kaju katli', 'khakhra', 'kheer',
    'kulfi', 'margherita pizza', 'masala dosa', 'masala papad', 'medu vada', 'misal pav', 'modak',
    'moong dal halwa', 'murukku', 'mysore pak', 'navratan korma', 'neer dosa', 'onion pakoda', 'palak paneer',
    'paneer masala', 'paneer pizza', 'pani puri', 'paniyaram', 'papdi chaat', 'patrode', 'pav bhaji',
    'pepperoni pizza', 'phirni', 'poha', 'pongal', 'puri bhaji', 'rajma chawal', 'rasgulla', 'rava dosa',
    'sabudana khichdi', 'sabudana vada', 'samosa', 'seekh kebab', 'set dosa', 'sev puri', 'solkadhi',
    'steamed momo', 'thali', 'thukpa', 'uttapam', 'vada pav'
]

CKPT_PATH = os.environ.get("FOOD_CKPT", "efficientnet_b3_best.pth")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-flash-lite-latest")
GEMINI_USE_SEARCH = os.environ.get("GEMINI_USE_SEARCH", "0") == "1"

# =========================
# 1) PREPROCESS
# =========================
infer_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

def pil_to_tensor(img: Image.Image):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return infer_tfms(img).unsqueeze(0)

# =========================
# 2) LOAD MODEL
# =========================
def infer_num_classes(state):
    for k in ("classifier.weight", "head.fc.weight", "fc.weight"):
        if k in state and isinstance(state[k], torch.Tensor) and state[k].ndim == 2:
            return int(state[k].shape[0])
    for _, t in state.items():
        if isinstance(t, torch.Tensor) and t.ndim == 2 and 2 <= t.shape[0] <= 10000:
            return int(t.shape[0])
    raise ValueError("Cannot infer num_classes from checkpoint.")

def load_model():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CKPT_PATH}\n"
            f"Put it beside app.py or set FOOD_CKPT=/path/to/file.pth"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(CKPT_PATH, map_location="cpu")
    state = state["state_dict"] if (isinstance(state, dict) and "state_dict" in state) else state

    num_classes = infer_num_classes(state)
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    labels = CLASS_NAMES if len(CLASS_NAMES) == num_classes else [f"class_{i}" for i in range(num_classes)]

    gemini_status = "READY" if GEMINI_API_KEY else "NOT SET (missing GEMINI_API_KEY)"
    status = (
        f"✅ EfficientNet-B3 loaded\n"
        f"✅ Classes: {num_classes}\n"
        f"✅ Device: {device.type.upper()}\n"
        f"✅ Gemini: {gemini_status}\n"
        f"   - model: {GEMINI_MODEL}\n"
        f"   - google_search tool: {'ON' if GEMINI_USE_SEARCH else 'OFF'}"
    )
    return model, labels, status

MODEL, LABELS, LOAD_STATUS = load_model()

# =========================
# 3) GEMINI + POSTPROCESSING
# =========================
def get_gemini_client():
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)

def gemini_ping():
    client = get_gemini_client()
    if client is None:
        return "❌ GEMINI_API_KEY not set"
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Reply with exactly: OK",
            config=types.GenerateContentConfig(temperature=0)
        )
        return f"✅ Gemini connected: {(resp.text or '').strip()}"
    except Exception as e:
        return f"❌ Gemini ping failed: {str(e)}"

def _strip_intro(text: str) -> str:
    if not text:
        return text
    lines = [ln.rstrip() for ln in text.splitlines()]
    if lines and re.match(r"^\s*here\s+is\s+.*", lines[0], flags=re.I):
        lines = lines[1:]
        while lines and lines[0].strip() == "":
            lines = lines[1:]
    return "\n".join(lines).strip()

def _remove_avoid_section(text: str) -> str:
    if not text:
        return text
    avoid_patterns = [
        r"^\s*3\)\s*\*\*who\s+should\s+avoid.*\*\*.*$",
        r"^\s*\*\*who\s+should\s+avoid.*\*\*.*$",
        r"^\s*who\s+should\s+avoid.*$",
        r"^\s*caution.*$",
    ]
    lines = text.splitlines()
    out, skipping = [], False
    for ln in lines:
        if any(re.match(p, ln.strip(), flags=re.I) for p in avoid_patterns):
            skipping = True
            continue
        if skipping and re.match(r"^\s*(3|4)\)\s*", ln.strip()):
            skipping = False
        if not skipping:
            out.append(ln)
    return "\n".join(out).strip()

def _split_on_dashes(s: str):
    parts = [p.strip() for p in re.split(r"\s+[–—-]\s+", s) if p.strip()]
    return parts if len(parts) > 1 else [s]

def _force_section_format(md: str) -> str:
    if not md:
        return md

    md = md.replace("\r\n", "\n")
    lines = md.splitlines()
    out = []

    def push_section(tag):
        if out and out[-1] != "":
            out.append("")
        out.append(tag)

    for ln in lines:
        s = ln.strip()
        if not s:
            continue

        m = re.match(r"^\s*(\d)\)\s*(.*)$", s)
        if m and m.group(1) in ("1", "2", "3", "4"):
            n = m.group(1)
            if n == "1":
                push_section("AYURVEDIC PROFILE")
            elif n == "2":
                push_section("BEST TIME TO EAT")
            elif n == "3":
                push_section("NUTRITION ESTIMATE PER TYPICAL SERVING")
            elif n == "4":
                push_section("HEALTHIER ALTERNATIVES / TWEAKS")
            continue

        if re.match(r"^(ayurvedic|best time|nutrition|healthier)", s, flags=re.I):
            u = s.upper()
            if "AYURVEDIC" in u:
                push_section("AYURVEDIC PROFILE")
            elif "BEST TIME" in u:
                push_section("BEST TIME TO EAT")
            elif "NUTRITION" in u:
                push_section("NUTRITION ESTIMATE PER TYPICAL SERVING")
            else:
                push_section("HEALTHIER ALTERNATIVES / TWEAKS")
            continue

        if s.startswith(("-", "•", "*")):
            s = re.sub(r"^[-•*]\s*", "", s).strip()

        chunks = [c.strip() for c in re.split(r"\s+\*\s+", s) if c.strip()]
        if len(chunks) > 1:
            for c in chunks:
                for p in _split_on_dashes(c):
                    out.append(f"- {p}")
            continue

        for p in _split_on_dashes(s):
            out.append(f"- {p}")

    # Ensure nutrition one-per-line if returned in a single dash-separated sentence
    final = []
    in_nut = False
    nutrient_keys = ("calories", "protein", "carbs", "carbo", "fat", "fiber", "fibre", "sodium")

    for ln in out:
        if ln == "NUTRITION ESTIMATE PER TYPICAL SERVING":
            in_nut = True
            final.append(ln)
            continue
        if ln in ("AYURVEDIC PROFILE", "BEST TIME TO EAT", "HEALTHIER ALTERNATIVES / TWEAKS"):
            in_nut = False
            final.append(ln)
            continue

        if in_nut and ln.startswith("- "):
            body = ln[2:].strip()
            parts = _split_on_dashes(body)
            if len(parts) > 1 and any(k in body.lower() for k in nutrient_keys):
                for p in parts:
                    final.append(f"- {p}")
            else:
                final.append(ln)
        else:
            final.append(ln)

    return "\n".join(final).strip()

# ---------- keyword + number styling ----------
DOSHA_KEYWORDS = [
    "vata", "pitta", "kapha", "dosha", "agni", "ama",
    "rasa", "guna", "virya", "vipaka",
    "ushna", "sheeta", "snigdha", "ruksha", "guru", "laghu",
    "katu", "madhura", "amla", "tikta", "kashaya", "lavana",
]

NUM_TOKEN_RE = re.compile(
    r"(?i)\b\d+(?:\.\d+)?\s?(?:kcal|cals|cal|kj|g|mg|ml|l|%)\b|\b\d+(?:\.\d+)?\b"
)

def _strip_md_emphasis(text: str) -> str:
    if text is None:
        return ""
    text = re.sub(r"(?<!\*)\*{1,2}([^*]+?)\*{1,2}(?!\*)", r"\1", text)
    text = re.sub(r"(?<!_)_{1,2}([^_]+?)_{1,2}(?!_)", r"\1", text)
    return text

def _wrap_keywords_and_numbers_as_html(md: str, food_name: str) -> str:
    if not md:
        return md

    food_words = [w.strip().lower() for w in re.split(r"\s+", food_name) if w.strip()]
    kws = sorted(set(DOSHA_KEYWORDS + food_words), key=len, reverse=True)

    def style_text(text: str) -> str:
        text = _strip_md_emphasis(text)
        t = html.escape(text)

        for kw in kws:
            if not kw:
                continue
            t = re.sub(
                rf"(?i)\b{re.escape(kw)}\b",
                lambda m: f"<span class='kw'>{m.group(0)}</span>",
                t
            )

        t = NUM_TOKEN_RE.sub(lambda m: f"<span class='num'>{m.group(0)}</span>", t)

        # join "1-3 g" into one highlighted span if split
        t = re.sub(
            r"<span class='num'>(\d+(?:\.\d+)?(?:\s?(?:kcal|cals|cal|kj|g|mg|ml|l|%)?)?)</span>\s*([–—-])\s*<span class='num'>(\d+(?:\.\d+)?(?:\s?(?:kcal|cals|cal|kj|g|mg|ml|l|%)?)?)</span>",
            r"<span class='num'>\1\2\3</span>",
            t,
            flags=re.I
        )
        return t

    out_lines = []
    for ln in md.splitlines():
        if ln.strip().startswith("- "):
            body = ln.strip()[2:].strip()
            out_lines.append(f"- {style_text(body)}")
        else:
            out_lines.append(ln)
    return "\n".join(out_lines).strip()

# =========================
# 3.1) PARSE SECTIONS (for cards + popup)
# =========================
SECTION_KEYS = [
    "AYURVEDIC PROFILE",
    "BEST TIME TO EAT",
    "NUTRITION ESTIMATE PER TYPICAL SERVING",
    "HEALTHIER ALTERNATIVES / TWEAKS",
]

def _parse_sections_from_forced(forced_text: str):
    out = {k: [] for k in SECTION_KEYS}
    cur = None
    for raw in (forced_text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s in out:
            cur = s
            continue
        if cur and s.startswith("- "):
            out[cur].append(s[2:].strip())
        elif cur:
            out[cur].append(s)
    return out

def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()

# =========================
# 3.2) CHIP BUILDERS
# =========================
STOPWORDS = set([
    "the","a","an","and","or","to","of","in","on","for","with","is","are","was","were",
    "be","been","being","can","could","may","might","should","would","will","also",
    "often","usually","generally","typically","recommended","avoid","caution",
    "helps","help","helpful","benefit","good","best","time","eat","consumed"
])

AYUR_KEYMAP = [
    ("vata", "Vata"),
    ("pitta", "Pitta"),
    ("kapha", "Kapha"),
    ("cooling", "Cooling"),
    ("warming", "Warming"),
    ("heating", "Warming"),
    ("light", "Light"),
    ("heavy", "Heavy"),
    ("dry", "Dry"),
    ("oily", "Oily"),
    ("snigdha", "Snigdha"),
    ("ruksha", "Ruksha"),
    ("guru", "Guru"),
    ("laghu", "Laghu"),
    ("ushna", "Ushna"),
    ("sheeta", "Sheeta"),
    ("digestive", "Digestive"),
    ("agni", "Agni"),
    ("ama", "Ama"),
    ("hydrating", "Hydrating"),
    ("liquid", "Liquid"),
    ("spicy", "Spicy"),
    ("sweet", "Sweet"),
    ("sour", "Sour"),
    ("salty", "Salty"),
    ("bitter", "Bitter"),
]

MEAL_WORDS = [
    ("breakfast", "Breakfast"),
    ("mid-morning", "Mid morning"),
    ("morning", "Morning"),
    ("lunch", "Lunch"),
    ("afternoon snack", "Afternoon snack"),
    ("snack", "Snack"),
    ("afternoon", "Afternoon"),
    ("evening", "Evening"),
    ("dinner", "Dinner"),
    ("night", "Night"),
    ("noon", "Lunch"),
    ("midday", "Lunch"),
]

def _uniq_keep_order(items, k=6):
    out, seen = [], set()
    for x in items:
        if not x:
            continue
        key = x.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(x)
        if len(out) >= k:
            break
    return out

def _best_time_range_chip(text):
    txt = text.lower()
    m = re.search(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:[–—-]|to)\s*"
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
        txt
    )
    if not m:
        return None

    h1, mm1, ap1, h2, mm2, ap2 = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)

    def fmt(h, mm, ap):
        t = f"{int(h)}" + (f":{mm}" if mm else "")
        ap = (ap or "").lower()
        ap = ap.upper() if ap in ("am", "pm") else ""
        return t, ap

    t1, A1 = fmt(h1, mm1, ap1)
    t2, A2 = fmt(h2, mm2, ap2)

    if A1 and not A2:
        A2 = A1
    if A2 and not A1:
        A1 = A2

    if A1 and A2:
        return f"{t1}{A1}–{t2}{A2}"
    if A1:
        return f"{t1}–{t2}{A1}"
    return f"{t1}–{t2}"

def _best_time_chips(bullets_plain, max_chips=4):
    blob = " ".join(bullets_plain)
    chips = []

    rng = _best_time_range_chip(blob)
    if rng:
        chips.append(rng)

    low = blob.lower()
    for key, label in MEAL_WORDS:
        if re.search(rf"\b{re.escape(key)}\b", low):
            chips.append(label)

    if not chips and bullets_plain:
        words = [w for w in re.findall(r"[A-Za-z]+", bullets_plain[0]) if w.lower() not in STOPWORDS]
        if words:
            chips.append(words[0].title())

    return _uniq_keep_order(chips, k=max_chips)

def _nutrition_chips_words_only():
    # ✅ EXACTLY what you asked: only the nutrient names as bubbles
    return ["Calories", "Protein", "Carbs", "Fat", "Fiber", "Sodium"]

def _tweaks_chips_strict(bullets_plain, max_chips=4):
    txt = " ".join(bullets_plain).lower()

    whitelist = [
        ("air fryer", "Air fryer"),
        ("air-fry", "Air fryer"),
        ("bake", "Bake"),
        ("steam", "Steam"),
        ("grill", "Grill"),
        ("roast", "Roast"),
        ("whole wheat", "Whole wheat"),
        ("multigrain", "Multigrain"),
        ("brown rice", "Brown rice"),
        ("millet", "Millet"),
        ("jowar", "Jowar"),
        ("bajra", "Bajra"),
        ("ragi", "Ragi"),
        ("less oil", "Less oil"),
        ("reduce oil", "Less oil"),
        ("low oil", "Less oil"),
        ("salad", "Salad"),
        ("curd", "Curd"),
        ("yogurt", "Yogurt"),
        ("buttermilk", "Buttermilk"),
        ("add veggies", "Add veggies"),
        ("vegetables", "Add veggies"),
        ("portion control", "Portion control"),
    ]

    found = []
    for pat, label in whitelist:
        if re.search(rf"\b{re.escape(pat)}\b", txt):
            found.append(label)

    if not found and bullets_plain:
        words = [w for w in re.findall(r"[A-Za-z]+", bullets_plain[0]) if w.lower() not in STOPWORDS]
        if words:
            found.append(words[0].title())
            if len(words) > 1:
                found.append(f"{words[1].title()}")

    return _uniq_keep_order(found, k=max_chips)

def _ayurvedic_chips(bullets_plain, max_chips=4):
    txt = " ".join(bullets_plain).lower()
    chips = []

    for pat, label in AYUR_KEYMAP:
        if re.search(rf"\b{re.escape(pat)}\b", txt):
            chips.append(label)

    if len(chips) < max_chips:
        for b in bullets_plain:
            words = [w for w in re.findall(r"[A-Za-z]+", b) if w.lower() not in STOPWORDS]
            if not words:
                continue
            for w in words[:6]:
                W = w.title()
                if W.lower() not in (c.lower() for c in chips):
                    chips.append(W)
                if len(chips) >= max_chips:
                    break
            if len(chips) >= max_chips:
                break

    return _uniq_keep_order(chips, k=max_chips) or ["—"]

def _chips_for_section(section_key, bullets_html):
    bullets_plain = [_strip_tags(x) for x in (bullets_html or [])]

    if section_key == "BEST TIME TO EAT":
        out = _best_time_chips(bullets_plain, max_chips=4)
        return out or ["—"]

    if section_key == "NUTRITION ESTIMATE PER TYPICAL SERVING":
        return _nutrition_chips_words_only()  # ✅ words only

    if section_key == "HEALTHIER ALTERNATIVES / TWEAKS":
        out = _tweaks_chips_strict(bullets_plain, max_chips=4)
        return out or ["—"]

    return _ayurvedic_chips(bullets_plain, max_chips=4)

# =========================
# 3.3) Build Flash Card HTML + Popup HTML
# =========================
def _make_chip_row(chips):
    chips = [c for c in (chips or []) if c]
    if not chips:
        return ""
    # ✅ allow up to 6 chips now (nutrition card needs 6)
    chips_html = "".join([f"<span class='chip'>{html.escape(str(c))}</span>" for c in chips[:6]])
    return f"<div class='chiprow'>{chips_html}</div>"

def build_flashcard_html(title, emoji, chips):
    return f"""
    <div class="flashcard">
      <div class="flashhead">
        <div class="flashicon">{emoji}</div>
        <div class="flashtitle">{html.escape(title)}</div>
      </div>
      {_make_chip_row(chips)}
      <div class="tap">Tap for details</div>
    </div>
    """

def build_popup_html(title, emoji, chips, bullets_html):
    li = []
    for i, b in enumerate(bullets_html or []):
        n = i + 1
        li.append(f"""
          <div class="poprow">
            <div class="popnum">{n}</div>
            <div class="poptext">{b}</div>
          </div>
        """)
    body = "".join(li) if li else "<div class='popempty'>(No details)</div>"

    return f"""
    <div class="popcard">
      <div class="pophero">
        <div class="popemoji">{emoji}</div>
        <div class="popback" title="Back">←</div>
      </div>
      <div class="popcontent">
        <div class="poptitle">{html.escape(title)}</div>
        {_make_chip_row(chips)}
        <div class="popsection">
          {body}
        </div>
      </div>
    </div>
    """

# =========================
# 3.4) Gemini call -> returns forced_text + styled_html
# =========================
def gemini_food_info_text(food_name: str, top2_pairs):
    client = get_gemini_client()
    if client is None:
        forced = (
            "AYURVEDIC PROFILE\n"
            "- LLM unavailable (missing GEMINI_API_KEY)\n\n"
            "BEST TIME TO EAT\n"
            "- Set GEMINI_API_KEY and restart\n\n"
            "NUTRITION ESTIMATE PER TYPICAL SERVING\n"
            "- Calories: —\n- Protein: —\n- Carbs: —\n- Fat: —\n- Fiber: —\n- Sodium: —\n\n"
            "HEALTHIER ALTERNATIVES / TWEAKS\n"
            "- —"
        )
        styled = _wrap_keywords_and_numbers_as_html(forced, food_name)
        return forced, styled

    top2_str = ", ".join([f"{n} ({c})" for n, c in top2_pairs])

    prompt = f"""
Food detected: {food_name}
Other close predictions: {top2_str}

Return EXACTLY 4 sections with titles on separate lines:
AYURVEDIC PROFILE
BEST TIME TO EAT
NUTRITION ESTIMATE PER TYPICAL SERVING
HEALTHIER ALTERNATIVES / TWEAKS

Under each title use bullet points only.
Nutrition: Calories, Protein, Carbs, Fat, Fiber, Sodium each on its own bullet.
Do NOT include any intro sentence.
Do NOT include "Who should avoid / caution".
No JSON.
No markdown emphasis like **word** or *word*.
""".strip()

    tools = [types.Tool(google_search=types.GoogleSearch())] if GEMINI_USE_SEARCH else None

    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=750,
                tools=tools
            )
        )
        txt = (resp.text or "").strip() or "⚠️ Gemini returned empty text."
        txt = _strip_intro(txt)
        txt = _remove_avoid_section(txt)
        forced = _force_section_format(txt)
        forced = _strip_md_emphasis(forced)

        styled = _wrap_keywords_and_numbers_as_html(forced, food_name)
        return forced, styled
    except Exception as e:
        forced = (
            "AYURVEDIC PROFILE\n"
            f"- Gemini call failed: {str(e)}\n\n"
            "BEST TIME TO EAT\n- —\n\n"
            "NUTRITION ESTIMATE PER TYPICAL SERVING\n"
            "- Calories: —\n- Protein: —\n- Carbs: —\n- Fat: —\n- Fiber: —\n- Sodium: —\n\n"
            "HEALTHIER ALTERNATIVES / TWEAKS\n- —"
        )
        styled = _wrap_keywords_and_numbers_as_html(forced, food_name)
        return forced, styled

# =========================
# 4) PREDICT -> builds flashcards
# =========================
def predict(image):
    if image is None:
        c = build_flashcard_html("Ayurvedic Profile", "🌿", ["—"])
        return "❌ Please upload an image.", [], c, c, c, c, "", "", ""

    device = next(MODEL.parameters()).device
    x = pil_to_tensor(image).to(device)

    with torch.no_grad():
        probs = F.softmax(MODEL(x), dim=1)[0]

    vals, idxs = torch.topk(probs, k=TOP_K)

    rows = []
    top2_pairs = []
    for p, i in zip(vals.tolist(), idxs.tolist()):
        conf = f"{p * 100:.2f}%"
        name = LABELS[i]
        rows.append([name, conf])
        top2_pairs.append((name, conf))

    food_name = rows[0][0]
    forced_text, styled_text = gemini_food_info_text(food_name, top2_pairs)

    sec_styled = _parse_sections_from_forced(styled_text)
    mapping = [
        ("AYURVEDIC PROFILE", "Ayurvedic Profile", "🌿"),
        ("BEST TIME TO EAT", "Best Time to Eat", "🕒"),
        ("NUTRITION ESTIMATE PER TYPICAL SERVING", "Nutrition Estimate", "🍽️"),
        ("HEALTHIER ALTERNATIVES / TWEAKS", "Healthier Tweaks", "✨"),
    ]

    cards = []
    for key, title, emoji in mapping:
        bullets_html = sec_styled.get(key, [])
        chips = _chips_for_section(key, bullets_html)
        cards.append(build_flashcard_html(title, emoji, chips))

    return "✅ Analysis complete", rows, cards[0], cards[1], cards[2], cards[3], food_name, forced_text, styled_text

# =========================
# 5) POPUP OPEN/CLOSE (Overlay modal)
# =========================
def open_popup(section_key, title, emoji, food_name, forced_text, styled_text):
    sec_styled = _parse_sections_from_forced(styled_text)
    bullets_html = sec_styled.get(section_key, [])
    chips = _chips_for_section(section_key, bullets_html)
    pop_html = build_popup_html(title, emoji, chips, bullets_html)
    return pop_html, gr.update(visible=True)

def close_popup():
    return gr.update(visible=False)

# =========================
# 6) UI
# =========================
CSS = """
.gradio-container, .gradio-container *{
  font-family: ui-serif, Georgia, "Times New Roman", Times, serif !important;
  box-shadow: none !important;
}
.gradio-container{ background: #f6f3ee !important; }

#topbar{
  display:flex; align-items:center; justify-content:space-between;
  padding: 14px 18px;
  border-bottom: 1px solid rgba(20,20,20,0.08);
  background: rgba(246,243,238,0.92);
  position: sticky; top: 0; z-index: 5;
  backdrop-filter: blur(6px);
}
#brand{ display:flex; align-items:center; gap: 10px; }
#brand-icon{
  width: 32px; height: 32px;
  display:flex; align-items:center; justify-content:center;
  border-radius: 14px;
  background: #d7eadf;
  border: 1px solid rgba(20,20,20,0.10);
  color: #14532d;
  font-size: 18px;
}
#brand-title{ font-size: 30px; margin: 0; color: #1f2937; font-weight: 900; }
#reassess{
  font-size: 14px;
  color: rgba(31,41,55,0.80);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important;
}

#wrap{ max-width: 980px; margin: 20px auto 44px auto; padding: 0 18px; }
.center-col{ display:flex; flex-direction:column; align-items:center; text-align:center; }

.card-dark{
  width: 100%;
  background: linear-gradient(135deg, #121826, #1f2937);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 26px;
  padding: 22px;
  color: #f9fafb;
}
.big-title{ font-size: 58px; line-height: 1.03; margin: 10px 0 2px 0; font-weight: 900; }
.kicker{
  font-size: 12px; letter-spacing: 2px; text-transform: uppercase; opacity: 0.70;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important;
}

button.primary, .primary button, .gr-button-primary{
  background: #d97706 !important;
  border: 1px solid rgba(0,0,0,0.10) !important;
  color: #fff !important;
  border-radius: 14px !important;
  padding: 12px 18px !important;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important;
}

#img_card .wrap, #img_card{ border-radius: 20px !important; }
#pred_card .wrap{ border-radius: 16px !important; }

.header-pill{
  width: 100%;
  background: #c7e3d5;
  border: 1px solid rgba(20,20,20,0.12);
  border-radius: 18px;
  padding: 14px 16px;
  display:flex;
  align-items:center;
  gap: 10px;
  justify-content:flex-start;
}
.header-pill .title{
  font-size: 22px;
  color: #0f3d2e;
  font-weight: 900;
}
.header-pill .sub{
  margin-left:auto;
  font-size: 13px;
  color: rgba(15,61,46,0.70);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important;
}

#status_msg, #status_msg *, #status_msg p{
  color: #0f172a !important;
  font-weight: 900;
}

.kw, .num{
  color: #d97706 !important;
  font-weight: 1000 !important;
}

table, thead, tbody, tr, td, th{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New" !important;
}

/* =========================
   FLASH CARDS
   ✅ Equal sizing: fixed min-height + flex layout
   ========================= */
.flashbtn{
  width: 100%;
  padding: 0 !important;
  border: none !important;
  background: transparent !important;
  text-align: left !important;
}
.flashbtn:hover{ background: transparent !important; }

.flashcard{
  background: #ffffff;
  border: 1px solid rgba(20,20,20,0.12);
  border-radius: 22px;
  padding: 18px 18px 14px 18px;
  box-shadow: 0 10px 24px rgba(0,0,0,0.05) !important;

  /* ✅ key change: make all cards same "bigger" height */
  min-height: 230px;

  /* ✅ keep spacing consistent inside */
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.flashhead{
  display:flex;
  align-items:center;
  gap: 12px;
  margin-bottom: 10px;
}

.flashicon{
  width: 44px;
  height: 44px;
  border-radius: 16px;
  background: rgba(217,119,6,0.14);
  border: 1px solid rgba(217,119,6,0.22);
  display:flex;
  align-items:center;
  justify-content:center;
  font-size: 22px;
}

.flashtitle{
  font-size: 34px;
  font-weight: 900;
  color: #111827;
  line-height: 1;
}

.chiprow{
  display:flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 6px 0 10px 0;
}

.chip{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  padding: 7px 14px;
  border-radius: 999px;
  background: rgba(217,119,6,0.12);
  border: 1px solid rgba(217,119,6,0.28);
  color: #c2410c;
  font-weight: 900;
  font-size: 18px;
}

.tap{
  margin-top: 10px;
  text-align:center;
  color: rgba(17,24,39,0.55);
  font-weight: 900;
  font-size: 18px;
}

/* =========================
   MODAL OVERLAY (CENTERED)
   ========================= */
#popup_overlay{
  position: fixed !important;
  inset: 0 !important;
  z-index: 9999 !important;
}
#popup_overlay .wrap{
  width: 100%;
  height: 100%;
  display:flex;
  align-items:center;
  justify-content:center;
}

.popbackdrop{
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.35);
}

.popwrap{
  position: relative;
  z-index: 10000;
  width: min(880px, 92vw);
  max-height: 86vh;
  overflow: auto;
  border-radius: 26px;
  background: #fbf7f1;
  border: 1px solid rgba(255,255,255,0.20);
  box-shadow: 0 30px 80px rgba(0,0,0,0.25) !important;
  margin: 0 auto;
}

.popcard{
  border-radius: 26px;
  overflow: hidden;
}

.pophero{
  height: 210px;
  background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.18), rgba(255,255,255,0.00)),
              linear-gradient(135deg, #e07000, #d97706);
  display:flex;
  align-items:center;
  justify-content:center;
  position: relative;
}

.popemoji{
  width: 86px;
  height: 86px;
  border-radius: 28px;
  background: rgba(255,255,255,0.22);
  border: 1px solid rgba(255,255,255,0.35);
  display:flex;
  align-items:center;
  justify-content:center;
  font-size: 40px;
  box-shadow: 0 10px 26px rgba(0,0,0,0.18) !important;
}

.popback{
  position: absolute;
  top: 18px;
  right: 18px;
  width: 44px;
  height: 44px;
  border-radius: 999px;
  background: rgba(255,255,255,0.25);
  border: 1px solid rgba(255,255,255,0.35);
  display:flex;
  align-items:center;
  justify-content:center;
  font-size: 22px;
  color: #fff;
}

.popcontent{
  padding: 18px 22px 16px 22px;
}

.poptitle{
  font-size: 44px;
  font-weight: 900;
  color: #111827;
  margin-bottom: 8px;
  line-height: 1.05;
}

.popsection{ margin-top: 14px; }

.poprow{
  display:flex;
  gap: 12px;
  align-items:flex-start;
  padding: 10px 0;
  border-top: 1px solid rgba(17,24,39,0.10);
}
.poprow:first-child{ border-top: none; }

.popnum{
  width: 34px;
  height: 34px;
  border-radius: 999px;
  background: rgba(217,119,6,0.14);
  border: 1px solid rgba(217,119,6,0.28);
  color: #c2410c;
  font-weight: 900;
  display:flex;
  align-items:center;
  justify-content:center;
  flex: 0 0 auto;
  margin-top: 2px;
  font-size: 16px;
}

.poptext{
  font-size: 18px;
  color: rgba(17,24,39,0.90);
  line-height: 1.6;
}

.popcontrols{
  padding: 14px 18px 18px 18px;
  border-top: 1px solid rgba(17,24,39,0.10);
}

/* ===== Invisible full-card clickable overlay ON TOP of the HTML card ===== */
#card1wrap, #card2wrap, #card3wrap, #card4wrap{ position: relative !important; }
.flashhtml{ position: relative !important; z-index: 1 !important; }
#card1wrap .flashbtn, #card2wrap .flashbtn, #card3wrap .flashbtn, #card4wrap .flashbtn{
  position: absolute !important;
  inset: 0 !important;
  z-index: 5 !important;
  opacity: 0 !important;
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  cursor: pointer !important;
}
#card1wrap .flashbtn button, #card2wrap .flashbtn button, #card3wrap .flashbtn button, #card4wrap .flashbtn button{
  width: 100% !important;
  height: 100% !important;
  opacity: 0 !important;
  padding: 0 !important;
}
"""

THEME = gr.themes.Soft(
    primary_hue="orange",
    neutral_hue="stone",
    radius_size=gr.themes.sizes.radius_lg
)

with gr.Blocks(theme=THEME, css=CSS, title="AyurFood Vision") as demo:
    gr.HTML("""
    <div id="topbar">
      <div id="brand">
        <div id="brand-icon">🍃</div>
        <div><h1 id="brand-title">AyurBalance</h1></div>
      </div>
      <div id="reassess">⟵ Re-Assess</div>
    </div>
    """)

    with gr.Column(elem_id="wrap"):
        with gr.Accordion("Debug / Model + Gemini Status", open=False):
            gr.Markdown(f"```text\n{LOAD_STATUS}\n```")
            ping_btn = gr.Button("Test Gemini Connection", variant="secondary")
            ping_out = gr.Markdown("")
            ping_btn.click(fn=gemini_ping, inputs=[], outputs=[ping_out])

        gr.HTML("""
        <div class="card-dark">
          <div class="center-col">
            <div class="kicker">AYURFOOD VISION</div>
            <div class="big-title">Analyze a<br/>meal photo</div>
          </div>
        </div>
        """)

        with gr.Column(elem_classes=["center-col"]):
            with gr.Group(elem_id="img_card"):
                inp = gr.Image(
                    type="pil",
                    label="Upload Food Image",
                    height=360,
                    sources=["upload"],  # ✅ upload-only
                )
                btn = gr.Button("Analyze Food", variant="primary")

            msg = gr.Markdown("", elem_id="status_msg")

            with gr.Group(elem_id="pred_card"):
                gr.HTML('<div class="header-pill"><div class="title">🍽️ PREDICTIONS</div><div class="sub">Top-2</div></div>')
                out = gr.Dataframe(headers=["Food", "Confidence"], datatype=["str", "str"], interactive=False)

            gr.HTML('<div style="height:12px;"></div>')
            gr.HTML('<div class="header-pill"><div class="title">🌿 INSIGHTS</div><div class="sub">Tap cards</div></div>')

            food_state = gr.State("")
            forced_state = gr.State("")
            styled_state = gr.State("")

            with gr.Row():
                with gr.Column(elem_id="card1wrap"):
                    c1_html = gr.HTML(elem_classes=["flashhtml"])
                    c1 = gr.Button("", elem_classes=["flashbtn"], elem_id="card1btn")
                with gr.Column(elem_id="card2wrap"):
                    c2_html = gr.HTML(elem_classes=["flashhtml"])
                    c2 = gr.Button("", elem_classes=["flashbtn"], elem_id="card2btn")

            with gr.Row():
                with gr.Column(elem_id="card3wrap"):
                    c3_html = gr.HTML(elem_classes=["flashhtml"])
                    c3 = gr.Button("", elem_classes=["flashbtn"], elem_id="card3btn")
                with gr.Column(elem_id="card4wrap"):
                    c4_html = gr.HTML(elem_classes=["flashhtml"])
                    c4 = gr.Button("", elem_classes=["flashbtn"], elem_id="card4btn")

    popup = gr.Column(visible=False, elem_id="popup_overlay")
    with popup:
        gr.HTML("<div class='popbackdrop'></div>")
        with gr.Column(elem_classes=["popwrap"]):
            popup_html = gr.HTML()
            with gr.Row(elem_classes=["popcontrols"]):
                close_btn = gr.Button("Close", variant="secondary")

    btn.click(
        predict,
        inputs=[inp],
        outputs=[msg, out, c1_html, c2_html, c3_html, c4_html, food_state, forced_state, styled_state],
        api_name=False
    )

    c1.click(
        fn=lambda food, forced, styled: open_popup("AYURVEDIC PROFILE", "Ayurvedic Profile", "🌿", food, forced, styled),
        inputs=[food_state, forced_state, styled_state],
        outputs=[popup_html, popup]
    )
    c2.click(
        fn=lambda food, forced, styled: open_popup("BEST TIME TO EAT", "Best Time to Eat", "🕒", food, forced, styled),
        inputs=[food_state, forced_state, styled_state],
        outputs=[popup_html, popup]
    )
    c3.click(
        fn=lambda food, forced, styled: open_popup("NUTRITION ESTIMATE PER TYPICAL SERVING", "Nutrition Estimate", "🍽️", food, forced, styled),
        inputs=[food_state, forced_state, styled_state],
        outputs=[popup_html, popup]
    )
    c4.click(
        fn=lambda food, forced, styled: open_popup("HEALTHIER ALTERNATIVES / TWEAKS", "Healthier Tweaks", "✨", food, forced, styled),
        inputs=[food_state, forced_state, styled_state],
        outputs=[popup_html, popup]
    )

    close_btn.click(fn=close_popup, inputs=[], outputs=[popup])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_api=False)
