import streamlit as st
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torchvision.transforms as T
import matplotlib.pyplot as plt
import io
import time
import os
import gdown

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IDD Semantic Segmentation",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0a0e17;
    --surface:   #111827;
    --surface2:  #1a2235;
    --accent:    #00d4ff;
    --accent2:   #7c3aed;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --success:   #10b981;
    --border:    rgba(255,255,255,0.07);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a0533 50%, #0a1628 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(0,212,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(124,58,237,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 0.8rem;
    background: linear-gradient(90deg, #ffffff 0%, #00d4ff 60%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 620px;
    line-height: 1.7;
    margin: 0;
}

/* ── Stat pills ── */
.stats-row { display: flex; gap: 1rem; margin-top: 2rem; flex-wrap: wrap; }
.stat-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.7rem 1.3rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.stat-pill .val {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--accent);
}
.stat-pill .lbl { font-size: 0.78rem; color: var(--muted); }

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}

/* ── Upload zone ── */
.upload-zone {
    background: linear-gradient(135deg, rgba(0,212,255,0.04), rgba(124,58,237,0.04));
    border: 2px dashed rgba(0,212,255,0.25);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    transition: all 0.3s;
}
.upload-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.upload-text { color: var(--muted); font-size: 0.9rem; }

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 24px rgba(0,212,255,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,212,255,0.4) !important;
}

/* ── Legend badge ── */
.legend-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.legend-item {
    display: flex; align-items: center; gap: 0.5rem;
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
    padding: 0.4rem 0.75rem;
    font-size: 0.82rem;
}
.legend-dot {
    width: 10px; height: 10px;
    border-radius: 3px; flex-shrink: 0;
}

/* ── Metric cards ── */
.metric-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--accent);
}
.metric-lbl { font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(124,58,237,0.2)) !important;
    color: var(--accent) !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, #00d4ff, #7c3aed) !important; border-radius: 99px !important; }

/* ── Images ── */
img { border-radius: 12px !important; }

/* ── Selectbox / slider ── */
.stSelectbox > div > div, .stSlider { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
NUM_CLASSES = 10

CLASS_INFO = {
    0:  {"name": "Road",          "color": (128, 64,128)},
    1:  {"name": "Sidewalk",      "color": ( 70,130,180)},
    2:  {"name": "Pedestrian",    "color": (220, 20, 60)},
    3:  {"name": "Two-Wheeler",   "color": (  0,  0,142)},
    4:  {"name": "Large Vehicle", "color": (  0, 60,100)},
    5:  {"name": "Animal",        "color": (119, 11, 32)},
    6:  {"name": "Traffic Sign",  "color": (220,220,  0)},
    7:  {"name": "Building",      "color": ( 70, 70, 70)},
    8:  {"name": "Sky",           "color": (135,206,235)},
    9:  {"name": "Background",    "color": (107,142, 35)},
}

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

val_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

# ─────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────
GDRIVE_FILE_ID = "1KP1X2h4fF9akU0Hsk3gZxYWWSzkLzDAH"
MODEL_PATH     = "model_v2_epoch_10.pth"

@st.cache_resource(show_spinner=False)
def load_model(_=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Download weights from Google Drive if not cached ──
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model weights (first run only)…"):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device

# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────
def predict(image_np, model, device):
    pil_img = Image.fromarray(image_np)
    tensor  = val_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    return pred

def mask_to_color(pred_mask):
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, info in CLASS_INFO.items():
        color_mask[pred_mask == cls_id] = info["color"]
    return color_mask

def overlay(image_np, color_mask, alpha=0.55):
    h, w = color_mask.shape[:2]
    img_resized = np.array(Image.fromarray(image_np).resize((w, h)))
    return ((1 - alpha) * img_resized + alpha * color_mask).astype(np.uint8)

def get_class_distribution(pred_mask):
    total = pred_mask.size
    dist = {}
    for cls_id, info in CLASS_INFO.items():
        pct = (pred_mask == cls_id).sum() / total * 100
        if pct > 0.1:
            dist[info["name"]] = round(pct, 1)
    return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#111827", edgecolor="none", dpi=150)
    buf.seek(0)
    return Image.open(buf)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 1rem;'>
        <div style='font-size:2.5rem; margin-bottom:0.4rem;'>🛣️</div>
        <div style='font-family:Syne,sans-serif; font-size:1.1rem; font-weight:800;
                    background:linear-gradient(90deg,#00d4ff,#7c3aed);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            IDD SegNet
        </div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:0.3rem;'>
            DeepLabV3+ · ResNet-50
        </div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.07); margin: 0.5rem 0 1.5rem;'>
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Settings**")
    overlay_alpha = st.slider("Overlay Opacity", 0.2, 0.9, 0.55, 0.05)
    show_distribution = st.checkbox("Show Class Distribution Chart", True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.07); margin:1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25);
                border-radius:10px; padding:0.8rem 1rem; font-size:0.82rem;'>
        <span style='color:#10b981; font-weight:700;'>✅ Model Auto-Loaded</span><br>
        <span style='color:#64748b;'>Weights download automatically on first run</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.07); margin:1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.78rem; color:#64748b; line-height:1.7;'>
        <b style='color:#94a3b8;'>Architecture</b><br>DeepLabV3+ with ASPP<br><br>
        <b style='color:#94a3b8;'>Backbone</b><br>ResNet-50 (SSL pretrained)<br><br>
        <b style='color:#94a3b8;'>Input Size</b><br>512 × 512 px<br><br>
        <b style='color:#94a3b8;'>Classes</b><br>10 semantic categories<br><br>
        <b style='color:#94a3b8;'>Dataset</b><br>India Driving Dataset (IDD)
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">🔬 Computer Vision · Semantic Segmentation</div>
    <h1>Road Scene Understanding</h1>
    <p>
        Upload any driving scene image and get pixel-level semantic segmentation
        using a <strong>DeepLabV3+</strong> model trained on the India Driving Dataset.
        Identifies 10 scene categories in real time.
    </p>
    <div class="stats-row">
        <div class="stat-pill"><span class="val">10</span><span class="lbl">Semantic Classes</span></div>
        <div class="stat-pill"><span class="val">512²</span><span class="lbl">Input Resolution</span></div>
        <div class="stat-pill"><span class="val">R-50</span><span class="lbl">Backbone</span></div>
        <div class="stat-pill"><span class="val">IDD</span><span class="lbl">Training Data</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  COLOUR LEGEND
# ─────────────────────────────────────────────
legend_html = '<div class="card"><div class="card-title">🎨 Class Colour Legend</div><div class="legend-grid">'
for cls_id, info in CLASS_INFO.items():
    r, g, b = info["color"]
    legend_html += f"""
    <div class="legend-item">
        <div class="legend-dot" style="background:rgb({r},{g},{b});"></div>
        <span>{info['name']}</span>
    </div>"""
legend_html += "</div></div>"
st.markdown(legend_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1.6], gap="large")

with col_upload:
    st.markdown('<div class="card-title">📤 Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop your driving scene image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        img_rgb = np.array(pil_img)
        st.image(img_rgb, caption="Input Image", use_container_width=True)

        h, w = img_rgb.shape[:2]
        st.markdown(f"""
        <div style='display:flex; gap:0.6rem; margin-top:0.8rem; flex-wrap:wrap;'>
            <div class="metric-box" style='flex:1'>
                <div class="metric-val">{w}</div>
                <div class="metric-lbl">Width (px)</div>
            </div>
            <div class="metric-box" style='flex:1'>
                <div class="metric-val">{h}</div>
                <div class="metric-lbl">Height (px)</div>
            </div>
            <div class="metric-box" style='flex:1'>
                <div class="metric-val">{round(uploaded.size/1024,1)}</div>
                <div class="metric-lbl">Size (KB)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.5rem;'>", unsafe_allow_html=True)
        run_btn = st.button("🚀  Run Segmentation", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-icon">🖼️</div>
            <div style="font-family:Syne,sans-serif; font-size:1rem; font-weight:600;
                        color:#e2e8f0; margin-bottom:0.4rem;">
                Drop image here
            </div>
            <div class="upload-text">JPG · JPEG · PNG &nbsp;|&nbsp; Any resolution</div>
        </div>
        """, unsafe_allow_html=True)
        run_btn = False

# ─── Results column ───
with col_result:
    st.markdown('<div class="card-title">🔍 Segmentation Results</div>', unsafe_allow_html=True)

    if uploaded and run_btn:
        with st.spinner(""):
            progress = st.progress(0, text="Loading model weights…")
            model, device = load_model()
            progress.progress(35, text="Running inference…")
            t0 = time.time()
            pred_mask  = predict(img_rgb, model, device)
            infer_ms   = (time.time() - t0) * 1000
            progress.progress(75, text="Rendering colour map…")
            color_mask = mask_to_color(pred_mask)
            ov         = overlay(img_rgb, color_mask, overlay_alpha)
            progress.progress(100, text="Done ✓")
            time.sleep(0.3)
            progress.empty()

        # ── Tabs ──
        tab1, tab2, tab3 = st.tabs(["🎨 Overlay", "🗺️ Mask", "📊 Analysis"])

        with tab1:
            st.image(ov, caption="Segmentation Overlay", use_container_width=True)

        with tab2:
            st.image(color_mask, caption="Semantic Mask", use_container_width=True)

        with tab3:
            dist = get_class_distribution(pred_mask)

            st.markdown(f"""
            <div style='display:flex; gap:0.8rem; margin-bottom:1.2rem; flex-wrap:wrap;'>
                <div class="metric-box" style='flex:1'>
                    <div class="metric-val">{infer_ms:.0f}</div>
                    <div class="metric-lbl">Inference (ms)</div>
                </div>
                <div class="metric-box" style='flex:1'>
                    <div class="metric-val">{len(dist)}</div>
                    <div class="metric-lbl">Classes Detected</div>
                </div>
                <div class="metric-box" style='flex:1'>
                    <div class="metric-val">{"GPU" if str(device)=="cuda" else "CPU"}</div>
                    <div class="metric-lbl">Device</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if show_distribution:
                names  = list(dist.keys())
                values = list(dist.values())
                colors_hex = []
                for n in names:
                    for c in CLASS_INFO.values():
                        if c["name"] == n:
                            r,g,b = c["color"]
                            colors_hex.append(f"#{r:02x}{g:02x}{b:02x}")
                            break

                fig, ax = plt.subplots(figsize=(6, max(3, len(names)*0.5)))
                fig.patch.set_facecolor("#111827")
                ax.set_facecolor("#1a2235")
                bars = ax.barh(names, values, color=colors_hex, edgecolor="none", height=0.55)
                for bar, val in zip(bars, values):
                    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                            f"{val}%", va="center", fontsize=8, color="#94a3b8", fontweight="600")
                ax.set_xlabel("Coverage (%)", color="#64748b", fontsize=9)
                ax.tick_params(colors="#94a3b8", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#1a2235")
                ax.invert_yaxis()
                ax.xaxis.label.set_color("#64748b")
                plt.tight_layout()
                st.image(fig_to_pil(fig), use_container_width=True)
                plt.close(fig)

            if dist:
                top_class = list(dist.keys())[0]
                top_pct   = list(dist.values())[0]
                st.markdown(f"""
                <div style='background:rgba(0,212,255,0.07); border:1px solid rgba(0,212,255,0.2);
                            border-radius:12px; padding:1rem 1.2rem; margin-top:0.5rem;'>
                    <span style='font-size:0.75rem; color:#64748b; text-transform:uppercase;
                                 letter-spacing:0.1em;'>Dominant Class</span><br>
                    <span style='font-family:Syne,sans-serif; font-size:1.4rem; font-weight:800;
                                 color:#00d4ff;'>{top_class}</span>
                    <span style='color:#64748b; font-size:0.9rem;'> · {top_pct}% of scene</span>
                </div>
                """, unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div style='height:380px; display:flex; flex-direction:column;
                    align-items:center; justify-content:center; gap:1rem;
                    background:rgba(255,255,255,0.02); border:1px dashed rgba(255,255,255,0.07);
                    border-radius:16px;'>
            <div style='font-size:3rem;'>🛣️</div>
            <div style='font-family:Syne,sans-serif; font-size:1rem; font-weight:600;
                        color:#334155;'>Awaiting image upload…</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-top:3rem; border-top:1px solid rgba(255,255,255,0.06);
            padding-top:1.5rem; text-align:center;
            font-size:0.78rem; color:#334155; line-height:2;'>
    <strong style='color:#64748b;'>IDD Semantic Segmentation</strong> ·
    DeepLabV3+ · ResNet-50 · 10 Classes ·
    India Driving Dataset<br>
    Built with <span style='color:#00d4ff;'>Streamlit</span> &
    <span style='color:#7c3aed;'>PyTorch</span>
</div>
""", unsafe_allow_html=True)
