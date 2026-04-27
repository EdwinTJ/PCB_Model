import sys
import io
import json
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="PCB Defect Detector", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Roboto+Mono:wght@400;700&display=swap');

    /* ── Base ── */
    .stApp { background-color: #0d1117; color: #e6edf3; font-family: 'Inter', sans-serif; }
    .block-container {
    padding: 2rem 4rem !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
    }
    h1, h2, h3 { color: #e6edf3 !important; font-family: 'Inter', sans-serif !important; }
    hr { border-color: #21262d; }

    /* ── Upload box ── */
    [data-testid="stFileUploader"] > div:first-child {
        border: 2px dashed #30363d !important;
        border-radius: 10px !important;
        background: #161b22 !important;
    }

    /* ── Select box ── */
    [data-testid="stSelectbox"] > div > div {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        color: #e6edf3 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #238636 !important;
        color: #fff !important;
        border: 1px solid #2ea043 !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 8px 20px !important;
        transition: background 0.15s;
    }
    .stButton > button:hover { background: #2ea043 !important; }

    /* ── Metric cards ── */
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 14px 16px; margin: 5px 0;
    }
    .metric-card h3 { margin: 0; font-size: 13px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; font-family: 'Roboto Mono', monospace; }
    .metric-card p  { margin: 4px 0 0; font-size: 26px; font-weight: 800; color: #58a6ff; font-family: 'Roboto Mono', monospace; }

    /* ── Detection rows ── */
    .det-row {
        display: flex; align-items: center; gap: 8px;
        padding: 7px 10px; border-radius: 6px; margin: 3px 0;
        background: #0d1117; border: 1px solid #21262d; font-size: 15px;
        font-family: 'Roboto Mono', monospace;
    }
    .det-badge {
        padding: 2px 7px; border-radius: 4px; font-weight: 700;
        font-size: 13px; min-width: 48px; text-align: center; flex-shrink: 0;
    }
    .score-bar-bg { flex: 1; height: 5px; background: #21262d; border-radius: 3px; overflow: hidden; min-width: 0; }
    .score-bar-fill { height: 100%; border-radius: 3px; }

    /* ── Legend ── */
    .legend-grid {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px;
        margin: 12px 0 20px;
    }
    .legend-item {
        display: flex; align-items: center; gap: 7px;
        background: #161b22; border: 1px solid #21262d;
        border-radius: 6px; padding: 6px 10px; font-size: 14px; color: #8b949e;
        font-family: 'Roboto Mono', monospace;
    }
    .legend-badge {
        padding: 1px 6px; border-radius: 3px; font-weight: 700;
        font-size: 10px; color: #000; flex-shrink: 0;
    }

    /* ── Section header ── */
    .section-label {
        font-size: 13px; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: #8b949e; margin: 18px 0 8px;
        font-family: 'Roboto Mono', monospace;
    }

    /* ── Info box ── */
    .info-box {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 16px 20px; margin: 12px 0;
        font-size: 15px; color: #8b949e; line-height: 1.6;
    }
    .info-box strong { color: #e6edf3; }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_ABBR = {
    0: "SH", 1: "SP", 2: "SC", 3: "OP", 4: "MB",
    5: "HB", 6: "CS", 7: "CFO", 8: "BMFO",
}
CLASS_FULL = {
    "SH":   "Short Circuit",
    "SP":   "Spur (Copper Spike)",
    "SC":   "Spurious Copper",
    "OP":   "Open Circuit",
    "MB":   "Mouse Bite",
    "HB":   "Hole Breakout",
    "CS":   "Conductor Scratch",
    "CFO":  "Conductor Foreign Object",
    "BMFO": "Base Material Foreign Object",
}
COLORS = [
    "#FF4B4B", "#FF9900", "#FFD700", "#00CC66", "#00BFFF",
    "#CC44FF", "#FF69B4", "#00CED1", "#FFA07A",
]

EXAMPLE_BASE = "https://huggingface.co/spaces/mcthebest/PCB_defect_detection/resolve/main/test_image"
N_EXAMPLES = 12


# ── Model ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from engine.core import YAMLConfig

    ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT))

    local_ckpt = ROOT / "outputs" / "rtv4_hgnetv2_x_pcb" / "last.pth"
    if local_ckpt.exists():
        print(f"Using local checkpoint: {local_ckpt}")
        ckpt_path = local_ckpt
    else:
        print(f"Local checkpoint not found, downloading from HF")
        ckpt_path = Path(hf_hub_download(
            repo_id="mcthebest/PCB_RTDETR",
            repo_type="model",
            filename="last.pth",
        ))
    cfg_path = ROOT / "configs" / "rtv4" / "rtv4_hgnetv2_x_pcb.yml"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class DeployModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    cfg = YAMLConfig(str(cfg_path), resume=str(ckpt_path))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["num_classes"] = 9

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("ema", {}).get("module") or ckpt.get("model")
    if state is None:
        raise RuntimeError("Model weights not found in checkpoint.")
    cfg.model.load_state_dict(state)

    return DeployModel(cfg).to(device).eval(), device


def run_inference(model, device, image: Image.Image, threshold: float):
    import torch
    import torchvision.transforms as T

    w, h = image.size
    tensor = T.Compose([T.Resize((640, 640)), T.ToTensor()])(image)
    tensor = tensor.unsqueeze(0).to(device)
    orig_size = torch.tensor([[w, h]], dtype=torch.float32, device=device)

    with torch.no_grad():
        labels, boxes, scores = model(tensor, orig_size)

    l, b, s = labels[0].cpu().numpy(), boxes[0].cpu().numpy(), scores[0].cpu().numpy()
    keep = s > threshold
    return l[keep], b[keep], s[keep]


def draw_results(image: Image.Image, labels, boxes, scores) -> Image.Image:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.imshow(image)
    ax.axis("off")

    for label, box, score in zip(labels, boxes, scores):
        cid = int(label)
        color = COLORS[cid % len(COLORS)]
        x1, y1, x2, y2 = box
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        ))
        ax.text(
            x1, y1 - 4, f"{CLASS_ABBR.get(cid, cid)} {score:.2f}",
            color="white", fontsize=8, fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.85, pad=1.5, edgecolor="none"),
        )

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# PCB Defect Detector")
st.markdown(
    '<div class="info-box">'
    '<strong>Model:</strong> RT-DETRv4 · HGNetV2-X &nbsp;|&nbsp; '
    'Upload a PCB image or select an example to detect manufacturing defects.'
    '</div>',
    unsafe_allow_html=True,
)

# ── Class legend (inline, no sidebar) ────────────────────────────────────────

st.markdown('<div class="section-label">Defect Classes</div>', unsafe_allow_html=True)
legend_html = '<div class="legend-grid">'
for cid, abbr in CLASS_ABBR.items():
    color = COLORS[cid % len(COLORS)]
    legend_html += (
        f'<div class="legend-item">'
        f'<span class="legend-badge" style="background:{color}">{abbr}</span>'
        f'{CLASS_FULL[abbr]}'
        f'</div>'
    )
legend_html += '</div>'
st.markdown(legend_html, unsafe_allow_html=True)

# ── Confidence threshold (inline) ────────────────────────────────────────────

st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
threshold = st.slider(
    "Confidence threshold",
    0.05, 0.95, 0.30, 0.05,
    help="Detections below this score are hidden",
)

# ── Image source ──────────────────────────────────────────────────────────────

st.markdown('<div class="section-label">Image Source</div>', unsafe_allow_html=True)
source = st.radio("Source", ["Upload your own", "Use an example"], horizontal=True, label_visibility="collapsed")

pil_image = None
image_name = None

if source == "Upload your own":
    uploaded = st.file_uploader(
        "Drop a PCB image here",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        pil_image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        image_name = uploaded.name

else:
    example_options = {f"Example {i}": i for i in range(1, N_EXAMPLES + 1)}
    chosen_label = st.selectbox(
        "Select example image",
        list(example_options.keys()),
        label_visibility="collapsed",
    )
    chosen_idx = example_options[chosen_label]

    if st.button("▶ Run on this example"):
        st.session_state["confirmed_example"] = chosen_idx
        # Clear any cached result from a previous example
        st.session_state.pop("result_img", None)
        st.session_state.pop("result_labels", None)
        st.session_state.pop("result_scores", None)
        st.session_state.pop("original_img", None)

    if "confirmed_example" in st.session_state:
        idx = st.session_state["confirmed_example"]
        if "original_img" not in st.session_state or st.session_state.get("loaded_idx") != idx:
            with st.spinner(f"Fetching example {idx}.jpg — hang tight…"):
                try:
                    resp = requests.get(f"{EXAMPLE_BASE}/{idx}.jpg", timeout=15)
                    resp.raise_for_status()
                    pil_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    image_name = f"{idx}.jpg"
                    st.session_state["original_img"] = pil_image
                    st.session_state["loaded_idx"] = idx
                except Exception as e:
                    st.error(f"Could not fetch image: {e}")
        else:
            pil_image = st.session_state["original_img"]
            image_name = f"{st.session_state['loaded_idx']}.jpg"

if pil_image is None:
    st.stop()

# ── Inference ─────────────────────────────────────────────────────────────────

try:
    model, device = load_model()
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# Run inference (or use cached result for this image)
cache_key = image_name + str(threshold)
if st.session_state.get("_last_cache_key") != cache_key:
    with st.spinner("Running inference… this may take a moment ⏳"):
        labels, boxes, scores = run_inference(model, device, pil_image, threshold)
        result_img = draw_results(pil_image, labels, boxes, scores)
    st.session_state["_last_cache_key"] = cache_key
    st.session_state["result_labels"] = labels
    st.session_state["result_boxes"] = boxes
    st.session_state["result_scores"] = scores
    st.session_state["result_img"] = result_img
else:
    labels = st.session_state["result_labels"]
    boxes  = st.session_state["result_boxes"]
    scores = st.session_state["result_scores"]
    result_img = st.session_state["result_img"]

# ── Results layout ────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)

col_img, col_det = st.columns([3, 1], gap="large")

with col_img:
    # Show annotated result by default; toggle to see original
    show_original = st.toggle("See original image", value=False)
    if show_original:
        st.image(pil_image, use_container_width=True, caption=f"Original — {image_name}")
    else:
        st.image(result_img, use_container_width=True, caption=f"Detected — {image_name}")

with col_det:
    for label, value in [("Detections", str(len(labels))), ("Threshold", f"{threshold:.2f}")]:
        st.markdown(
            f'<div class="metric-card"><h3>{label}</h3><p>{value}</p></div>',
            unsafe_allow_html=True,
        )
    if len(scores):
        st.markdown(
            f'<div class="metric-card"><h3>Top Score</h3><p>{scores.max():.2f}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="section-label">Detections</div>', unsafe_allow_html=True)

    if len(labels) == 0:
        st.markdown('<p style="color:#8b949e;font-size:13px">Nothing above threshold.</p>', unsafe_allow_html=True)
    else:
        for i in scores.argsort()[::-1]:
            cid   = int(labels[i])
            score = float(scores[i])
            abbr  = CLASS_ABBR.get(cid, str(cid))
            color = COLORS[cid % len(COLORS)]
            pct   = int(score * 100)
            st.markdown(
                f'<div class="det-row">'
                f'<span class="det-badge" style="background:{color};color:#000">{abbr}</span>'
                f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%;background:{color}"></div></div>'
                f'<span style="color:#e6edf3;font-weight:600;min-width:36px;text-align:right">{score:.2f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ── Download ──────────────────────────────────────────────────────────────────

st.markdown("---")
stem = Path(image_name).stem

# PNG buffer
png_buf = io.BytesIO()
result_img.save(png_buf, format="PNG")

# JSON — boxes in original image pixel space [x1, y1, x2, y2]
sorted_indices = scores.argsort()[::-1]
detections_payload = {
    "image": image_name,
    "threshold": threshold,
    "image_size": {"width": pil_image.width, "height": pil_image.height},
    "detections": [
        {
            "rank": int(rank + 1),
            "label_id": int(labels[i]),
            "label": CLASS_ABBR.get(int(labels[i]), str(int(labels[i]))),
            "label_full": CLASS_FULL.get(CLASS_ABBR.get(int(labels[i]), ""), ""),
            "score": round(float(scores[i]), 4),
            "box_xyxy": [round(float(v), 2) for v in boxes[i]],
        }
        for rank, i in enumerate(sorted_indices)
    ],
}
json_buf = json.dumps(detections_payload, indent=2)

dl_col1, dl_col2 = st.columns(2)
with dl_col1:
    st.download_button(
        "⬇ Download annotated image (.png)",
        data=png_buf.getvalue(),
        file_name=f"{stem}_detected.png",
        mime="image/png",
        use_container_width=True,
    )
with dl_col2:
    st.download_button(
        "⬇ Download detections (.json)",
        data=json_buf,
        file_name=f"{stem}_detections.json",
        mime="application/json",
        use_container_width=True,
    )