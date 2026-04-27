"""
FastAPI sidecar for PCB defect detection.

Run:
    uvicorn api:app --host 127.0.0.1 --port 8000

POST /detect
    multipart form:
        image: file (jpg/png/...)
        threshold: float (optional, default 0.30)
    returns JSON:
        {
          "width": int, "height": int,
          "detections": [
            {"label": "SH", "label_id": 0, "label_full": "Short Circuit",
             "score": 0.91, "box": [x1, y1, x2, y2]},
            ...
          ]
        }
"""

import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig  # noqa: E402

CKPT_REPO = "mcthebest/PCB_RTDETR"
CKPT_FILE = "last.pth"
CFG_REL = "configs/rtv4/rtv4_hgnetv2_x_pcb.yml"
LOCAL_CKPT = "outputs/rtv4_hgnetv2_x_pcb/last.pth"

CLASS_ABBR = {
    0: "SH", 1: "SP", 2: "SC", 3: "OP", 4: "MB",
    5: "HB", 6: "CS", 7: "CFO", 8: "BMFO",
}
CLASS_FULL = {
    "SH": "Short Circuit",
    "SP": "Spur (Copper Spike)",
    "SC": "Spurious Copper",
    "OP": "Open Circuit",
    "MB": "Mouse Bite",
    "HB": "Hole Breakout",
    "CS": "Conductor Scratch",
    "CFO": "Conductor Foreign Object",
    "BMFO": "Base Material Foreign Object",
}

_TRANSFORM = T.Compose([T.Resize((640, 640)), T.ToTensor()])


class DeployModel(nn.Module):
    def __init__(self, cfg: YAMLConfig) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
        return self.postprocessor(self.model(images), orig_target_sizes)


state: dict = {"model": None, "device": "cpu"}


def load_model() -> tuple[nn.Module, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_ckpt = ROOT / LOCAL_CKPT
    if local_ckpt.exists():
        print(f"Using local checkpoint: {local_ckpt}")
        ckpt_path = str(local_ckpt)
    else:
        print(f"Local checkpoint not found at {local_ckpt}, downloading from {CKPT_REPO}")
        ckpt_path = hf_hub_download(repo_id=CKPT_REPO, repo_type="model", filename=CKPT_FILE)
    cfg_path = ROOT / CFG_REL

    cfg = YAMLConfig(str(cfg_path), resume=str(ckpt_path))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["num_classes"] = 9

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    weights = ckpt.get("ema", {}).get("module") or ckpt.get("model")
    if weights is None:
        raise RuntimeError("Model weights not found in checkpoint.")
    cfg.model.load_state_dict(weights)

    model = DeployModel(cfg).to(device).eval()
    return model, device


@asynccontextmanager
async def lifespan(_: FastAPI):
    print("Loading PCB_RTDETR model... (first run downloads weights from HF)")
    state["model"], state["device"] = load_model()
    print(f"Model ready on {state['device']}")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"ok": state["model"] is not None, "device": state["device"]}


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    threshold: float = Form(0.30),
) -> dict:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    raw = await image.read()
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    w, h = pil.size
    device = state["device"]
    tensor = _TRANSFORM(pil).unsqueeze(0).to(device)
    orig_size = torch.tensor([[w, h]], dtype=torch.float32, device=device)

    with torch.no_grad():
        labels, boxes, scores = state["model"](tensor, orig_size)

    labels = labels[0].cpu().tolist()
    boxes = boxes[0].cpu().tolist()
    scores = scores[0].cpu().tolist()

    detections = []
    for label, box, score in zip(labels, boxes, scores):
        if score < threshold:
            continue
        cid = int(label)
        abbr = CLASS_ABBR.get(cid, str(cid))
        x1, y1, x2, y2 = box
        detections.append({
            "label": abbr,
            "label_id": cid,
            "label_full": CLASS_FULL.get(abbr, ""),
            "score": float(score),
            "box": [float(x1), float(y1), float(x2), float(y2)],
        })

    detections.sort(key=lambda d: d["score"], reverse=True)
    return {"width": w, "height": h, "detections": detections}
