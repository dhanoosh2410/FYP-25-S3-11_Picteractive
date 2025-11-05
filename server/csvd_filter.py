from __future__ import annotations
from io import BytesIO
from typing import Literal, Tuple, Union

import numpy as np
from PIL import Image
from colorspacious import cspace_convert

# ---- FastAPI bits live in the same file ----
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse

# Public router you will import in main.py
router = APIRouter()

# ---------- Core CVD logic ----------
CVDChoice = Literal["protan", "deutan", "tritan", "none"]
ModeChoice = Literal["simulate", "daltonize"]

def _normalize_type(cvd: str) -> str:
    cvd = (cvd or "none").lower()
    if cvd.startswith("prot"):  return "protanomaly"
    if cvd.startswith("deut"):  return "deuteranomaly"
    if cvd.startswith("trit"):  return "tritanomaly"
    return "none"

def _to_numpy_rgb(img: Image.Image) -> Tuple[np.ndarray, np.ndarray | None]:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        return arr[..., :3], arr[..., 3:4]
    return arr, None

def _to_image(rgb: np.ndarray, a: np.ndarray | None) -> Image.Image:
    rgb = np.clip(rgb, 0.0, 1.0)
    if a is not None:
        rgba = np.concatenate([rgb, np.clip(a, 0.0, 1.0)], axis=-1)
        return Image.fromarray((rgba * 255 + 0.5).astype(np.uint8), "RGBA")
    return Image.fromarray((rgb * 255 + 0.5).astype(np.uint8), "RGB")

def simulate(rgb: np.ndarray, cvd_type: CVDChoice, severity: float) -> np.ndarray:
    if severity <= 0 or cvd_type == "none":
        return rgb.copy()
    space = {
        "name": "sRGB1+CVD",
        "cvd_type": _normalize_type(cvd_type),
        "severity": int(round(100 * float(severity))),  # 0..100
    }
    return cspace_convert(rgb, space, "sRGB1")

def daltonize(
    rgb: np.ndarray,
    cvd_type: CVDChoice,
    severity: float,
    amount: float = 1.0
) -> np.ndarray:
    if severity <= 0 or cvd_type == "none":
        return rgb.copy()
    sim = simulate(rgb, cvd_type, severity)
    o = cspace_convert(rgb, "sRGB1", "CAM02-UCS")
    s = cspace_convert(sim, "sRGB1", "CAM02-UCS")
    delta = o - s
    delta[..., 0] = 0.0  # keep lightness stable
    return cspace_convert(o + amount * delta, "CAM02-UCS", "sRGB1")

def apply(
    image: Union[str, BytesIO, Image.Image],
    *,
    mode: ModeChoice,
    cvd_type: CVDChoice,
    severity: float,
    amount: float = 1.0
) -> Image.Image:
    pil = Image.open(image) if isinstance(image, (str, BytesIO)) else image
    rgb, a = _to_numpy_rgb(pil)
    out = simulate(rgb, cvd_type, severity) if mode == "simulate" else daltonize(rgb, cvd_type, severity, amount)
    return _to_image(out, a)

# ---------- FastAPI route (same file) ----------
@router.post("/cvd/apply")
async def cvd_apply_api(
    image: UploadFile = File(...),
    mode: str = Form("simulate"),          # "simulate" | "daltonize"
    cvd_type: str = Form("deutan"),        # "protan" | "deutan" | "tritan" | "none"
    severity: float = Form(1.0),           # 0..1
    amount: float = Form(1.0),             # daltonize strength
):
    raw = await image.read()
    out_img = apply(
        BytesIO(raw),
        mode="daltonize" if mode == "daltonize" else "simulate",
        cvd_type=cvd_type if cvd_type in ("protan", "deutan", "tritan", "none") else "deutan",
        severity=max(0.0, min(1.0, severity)),
        amount=amount,
    )
    buf = BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
