# app.py

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
    """
    Split on hyphen/en-dash/em-dash separators with spaces around them:
    "A - B – C — D" => ["A", "B", "C", "D"]
    """
    parts = [p.strip() for p in re.split(r"\s+[–—-]\s+", s) if p.strip()]
    return parts if len(parts) > 1 else [s]

def _force_section_format(md: str) -> str:
    """
    Output becomes:
    SECTION TITLE
    - bullet
    - bullet
    ...
    Ensures that inline dash-separated statements become separate bullet lines.
    """
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

        # Strip bullet markers if present
        if s.startswith(("-", "•", "*")):
            s = re.sub(r"^[-•*]\s*", "", s).strip()

        # Split " * " chunks into bullets (Gemini sometimes uses asterisks as separators)
        chunks = [c.strip() for c in re.split(r"\s+\*\s+", s) if c.strip()]
        if len(chunks) > 1:
            for c in chunks:
                for p in _split_on_dashes(c):
                    out.append(f"- {p}")
            continue

        # Split dash-separated statements into multiple bullet items
        for p in _split_on_dashes(s):
            out.append(f"- {p}")

    # Make sure nutrition becomes one-per-line nutrients even if Gemini returns all in one bullet
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

            # Split on dash separators inside nutrition line too:
            # "Calories: ... - Protein: ... - Carbs: ..." => separate bullets
            parts = _split_on_dashes(body)
            if len(parts) > 1 and any(k in body.lower() for k in nutrient_keys):
                for p in parts:
                    final.append(f"- {p}")
            else:
                final.append(ln)
        else:
            final.append(ln)

    return "\n".join(final).strip()

# ---------- keyword + number styling (NO ** markdown) ----------
DOSHA_KEYWORDS = [
    "vata", "pitta", "kapha", "dosha", "agni", "ama",
    "rasa", "guna", "virya", "vipaka",
    "ushna", "sheeta", "snigdha", "ruksha", "guru", "laghu",
    "katu", "madhura", "amla", "tikta", "kashaya", "lavana",
]

# Match numeric values INCLUDING attached units like 3g, 6.5g, 28g, 250kcal, 300mg, 99.8%
NUM_TOKEN_RE = re.compile(
    r"(?i)\b\d+(?:\.\d+)?\s?(?:kcal|cals|cal|kj|g|mg|ml|l|%)\b|\b\d+(?:\.\d+)?\b"
)

def _wrap_keywords_and_numbers_as_html(md: str, food_name: str) -> str:
    """
    - Split already done earlier.
    - Now:
      • keywords => orange + bold
      • numbers (including 3g / 28g / 6.5g / 250 kcal / 99.8%) => bold (via <span class="num">)
    """
    if not md:
        return md

    food_words = [w.strip().lower() for w in re.split(r"\s+", food_name) if w.strip()]
    kws = sorted(set(DOSHA_KEYWORDS + food_words), key=len, reverse=True)

    def style_text(text: str) -> str:
        """
        Converts a plain-text bullet line into safe HTML and applies:
        - removes Gemini markdown emphasis like *agni*, **Kapha**, _word_
        - wraps Ayurvedic keywords + food words with <span class='kw'>...</span>
        - wraps numbers/units with <span class='num'>...</span>
        - merges numeric ranges into a single <span class='num'>A-B</span>
        """
        if text is None:
            text = ""

        # ✅ 1) Strip Gemini markdown emphasis BEFORE escaping
        # Handles: *word*, **word**, _word_, __word__
        text = re.sub(r"(?<!\*)\*{1,2}([^*]+?)\*{1,2}(?!\*)", r"\1", text)
        text = re.sub(r"(?<!_)_{1,2}([^_]+?)_{1,2}(?!_)", r"\1", text)

        # ✅ 2) Escape to safe HTML
        t = html.escape(text)

        # ✅ 3) Highlight keywords (whole-word match)
        for kw in kws:
            if not kw:
                continue
            t = re.sub(
                rf"(?i)\b{re.escape(kw)}\b",
                lambda m: f"<span class='kw'>{m.group(0)}</span>",
                t
            )

        # ✅ 4) Highlight numbers + unit tokens (e.g., 28g, 141 kcal, 99.8%)
        t = NUM_TOKEN_RE.sub(lambda m: f"<span class='num'>{m.group(0)}</span>", t)

        # ✅ 5) Merge ranges like: 250 - 350, 250–350, 250 — 350 into one span
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

def _decorate_section_titles(md: str) -> str:
    mapping = {
        "AYURVEDIC PROFILE": ("🌿", "AYURVEDIC PROFILE"),
        "BEST TIME TO EAT": ("🕒", "BEST TIME TO EAT"),
        "NUTRITION ESTIMATE PER TYPICAL SERVING": ("🍽️", "NUTRITION ESTIMATE"),
        "HEALTHIER ALTERNATIVES / TWEAKS": ("✨", "HEALTHIER TWEAKS"),
    }
    out_lines = []
    for ln in md.splitlines():
        s = ln.strip()
        if s in mapping:
            emoji, title = mapping[s]
            out_lines.append(
                "<div class='section-head'>"
                f"<span class='section-emoji'>{emoji}</span>"
                f"<span class='section-title'>{title}</span>"
                "</div>"
            )
        else:
            out_lines.append(ln)
    return "\n".join(out_lines).strip()

def _ensure_section_separators(md: str) -> str:
    parts = md.splitlines()
    out = []
    first = True
    for ln in parts:
        if "class='section-head'" in ln:
            if not first:
                out.append("<hr/>")
            first = False
        out.append(ln)
    return "\n".join(out).strip()

def gemini_food_info_text(food_name: str, top2_pairs):
    client = get_gemini_client()
    if client is None:
        return (
            "⚠️ <b>LLM unavailable</b> (missing <code>GEMINI_API_KEY</code>).<br/><br/>"
            "Set it in terminal:<br/>"
            "<code>export GEMINI_API_KEY=\"YOUR_KEY\"</code> and restart <code>python app.py</code>."
        )

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
        txt = _force_section_format(txt)                     # ✅ dash splits -> separate bullet lines
        txt = _wrap_keywords_and_numbers_as_html(txt, food_name)  # ✅ 3g / 28g etc bold
        txt = _decorate_section_titles(txt)
        txt = _ensure_section_separators(txt)
        return txt
    except Exception as e:
        return f"⚠️ <b>Gemini call failed</b>: <code>{html.escape(str(e))}</code>"

# =========================
# 4) PREDICT
# =========================
def predict(image):
    if image is None:
        return "❌ Please upload an image.", [], "—"

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

    llm_text = gemini_food_info_text(rows[0][0], top2_pairs)
    return "✅ Analysis complete", rows, llm_text

# =========================
# 5) UI
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

#llm_out{
  width: 100%;
  background: #d4ecdd !important;
  border: 1px solid rgba(20,20,20,0.12) !important;
  border-radius: 18px !important;
  padding: 16px 16px 10px 16px !important;
  text-align: left;
}
#llm_out, #llm_out *{
  color: #0b1f14 !important;
  opacity: 1 !important;
  font-size: 18px;
  line-height: 1.65;
}

.section-head{
  display:flex;
  align-items:center;
  gap: 10px;
  padding: 6px 0 8px 0;
}
.section-emoji{ font-size: 20px; width: 28px; display:flex; align-items:center; justify-content:center; }
.section-title{
  font-size: 20px;
  font-weight: 1000;
  color: #0f3d2e !important;
  text-decoration: underline;
  text-decoration-thickness: 2px;
  text-underline-offset: 4px;
}

.kw, .num{
  color: #d97706 !important;
  font-weight: 1000 !important;
}

#llm_out hr{
  border: none;
  border-top: 1px solid rgba(20,20,20,0.12);
  margin: 10px 0;
}

#llm_out ul{ margin: 4px 0 12px 0; padding-left: 28px; }
#llm_out li{ margin: 7px 0; }

table, thead, tbody, tr, td, th{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New" !important;
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
                inp = gr.Image(type="pil", label="Upload Food Image", height=360)
                btn = gr.Button("Analyze Food", variant="primary")

            msg = gr.Markdown("", elem_id="status_msg")

            with gr.Group(elem_id="pred_card"):
                gr.HTML('<div class="header-pill"><div class="title">🍽️ PREDICTIONS</div><div class="sub">Top-2</div></div>')
                out = gr.Dataframe(
                    headers=["Food", "Confidence"],
                    datatype=["str", "str"],
                    interactive=False
                )

            gr.HTML('<div style="height:12px;"></div>')
            gr.HTML('<div class="header-pill"><div class="title">🌿 INSIGHTS</div><div class="sub">Gemini</div></div>')
            llm_out = gr.Markdown(elem_id="llm_out")

    btn.click(predict, inputs=inp, outputs=[msg, out, llm_out], api_name=False)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_api=False)
