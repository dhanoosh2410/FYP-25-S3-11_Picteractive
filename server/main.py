from __future__ import annotations

import io
import json
import time
import uuid
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .auth_DB import app as auth_subapp
from .story_gen import StoryGenerator, BLIPCaptioner
from .quiz_gen import QuizGenerator
from .showtellpyTorch import ShowTellCaptioner
import re


# ---------- Optional libs ----------
import torch
from PIL import Image

# Try to import open_clip for simple sketch recognition; degrade gracefully if missing
try:
    import open_clip  # pip install open_clip_torch
    _HAS_OPEN_CLIP = True
except Exception:
    open_clip = None
    _HAS_OPEN_CLIP = False

# Try to import Ultralytics YOLO (optional)
try:
    from ultralytics import YOLO  # pip install ultralytics
    _HAS_ULTRALYTICS = True
except Exception:
    YOLO = None
    _HAS_ULTRALYTICS = False

# Try to import transformers pipeline for DETR fallback
try:
    from transformers import pipeline as hf_pipeline
    _HAS_TRANSFORMERS_PIPE = True
except Exception:
    hf_pipeline = None
    _HAS_TRANSFORMERS_PIPE = False

# ---------- App ----------
app = FastAPI(title="Picteractive API")

# CORS (credentials + configurable origins)
# CORS (credentials + configurable origins)
import os as _os
_origins_env = _os.getenv("ALLOWED_ORIGINS", "")
if _origins_env.strip():
    _origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
else:
    _origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount auth routes
app.include_router(auth_subapp.router)

# ---------- Storage paths ----------
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
IMG_DIR = DATA / "scenes"
ITEMS_JSON = DATA / "items.json"
IMG_DIR.mkdir(parents=True, exist_ok=True)
if not ITEMS_JSON.exists():
    ITEMS_JSON.write_text("[]", encoding="utf-8")


def _load_items():
    try:
        return json.loads(ITEMS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_items(items):
    ITEMS_JSON.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- Health / Status ----------
@app.get("/api/health")
async def health():
    eng = story_engine
    cap = getattr(eng, "captioner", None)
    sk = get_quickdraw_labeler()
    return {
        "ok": bool(getattr(eng, "ready", False)) and bool(getattr(cap, "ready", False)),
        "captioner": bool(getattr(cap, "ready", False)),
        "storygen": bool(getattr(eng, "ready", False)),
        "sketch_index": bool(sk is not None),
        "mode": getattr(eng, "_mode", None),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "error": getattr(cap, "err", None) or getattr(eng, "err", None),
    }



@app.get("/api/story_status")
def story_status():
    eng = story_engine
    return {
        "ready": bool(getattr(eng, "ready", False)),
        "mode": getattr(eng, "_mode", None),
        "model": getattr(eng, "model_name", ""),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "err": getattr(eng, "err", None),
    }


@app.post("/api/story_test")
def story_test():
    # Smoke test payload (no images here, just a static check that endpoint is reachable)
    t = "A LITTLE ADVENTURE"
    p = [
        "Something begins in the first picture.",
        "Something changes in the second picture.",
        "A friendly ending appears in the third picture.",
    ]
    return {"title": t, "panels": p, "story": "\n".join(p)}



# ---------- CLIP recognizer (optional) ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = None
CLIP_PREPROC = None
CLIP_TOKENIZER = None
TEXT_EMB = None

LABELS = [
    # People & Characters
    "boy","girl","child","man","woman","teacher","farmer","chef","doctor",
    "nurse","policeman","firefighter","artist","student","singer","dancer",

    # Animals
    "dog","cat","bird","rabbit","fish","horse","cow","sheep","goat","pig",
    "elephant","tiger","lion","monkey","chicken","duck","frog","bear",
    "deer","snake","giraffe","zebra","kangaroo","panda","crocodile",
    "mouse","butterfly","bee","ant","spider","snail","worm","crab","octopus",
    "whale","dolphin","penguin","peacock","parrot","eagle","owl",
    "heron","egret","seagull","sparrow","pigeon","crow","flamingo",
    # Additional common farm/zoo animals for accuracy
    "alpaca","llama",
    "goldfish","salmon","shark","ray",

    # Nature & Environment
    "tree","flower","grass","leaf","forest","mountain","hill","river","lake",
    "sea","beach","sun","moon","cloud","rain","snow","wind","rainbow","sand",
    "rock","soil","waterfall","island","volcano","field","garden","park",

    # Places & Structures
    "house","school","hospital","market","temple","church","bridge","building",
    "farm","road","street","village","city","playground","garden","office",
    "restaurant","shop","station","airport","bus stop","harbor","tower","castle",

    # Vehicles & Transport
    "car","bus","truck","bicycle","motorcycle","train","boat","ship","submarine",
    "airplane","helicopter","scooter","van","tractor","ambulance","police car",
    "fire truck","rocket","balloon","skateboard",

    # Objects & Tools
    "chair","table","bed","sofa","cup","plate","bottle","spoon","fork","knife",
    "book","pen","pencil","bag","clock","lamp","mirror","phone","computer",
    "television","remote","camera","guitar","piano","drum","microphone",
    "umbrella","hat","shoe","shirt","dress","ball","kite","toy","doll",
    "brush","broom","bucket","mop","key","door","window","fan","light","towel",
    # Containers / props for food scenes
    "basket","wicker basket","bowl","plate",

    # Fruits
    "apple","banana","orange","mango","grape","strawberry","pineapple",
    "watermelon","lemon","lime","pear","peach","cherry","tomato","coconut",
    "durian","jackfruit","papaya","guava","lychee","longan","rambutan","melon",
    "blueberry","blackberry","plum","pomegranate","kiwi","fig","date","dragonfruit",
    "avocado","passionfruit","cranberry","pear","sapodilla","tangerine","olive",

    # Vegetables
    "broccoli","cucumber","carrot","onion","garlic","potato","sweet potato",
    "eggplant","brinjal","pumpkin","cabbage","lettuce","spinach","bell pepper",
    "red pepper","green pepper","chili","pepper","cauliflower","zucchini",
    "radish","beetroot","yam","ginger","okra","bitter gourd","celery",
    "asparagus","corn","peas","bean","capsicum","spring onion","leek",
    "tomato","turnip","mushroom","brussels sprout","artichoke","coriander",
    "parsley","mint","basil","curry leaf","mustard leaf","drumstick","lotus root",

    # Food & Drinks
    "pizza","cookie","cake","bread","burger","sandwich","rice","noodles","pasta",
    "ice cream","chocolate","milk","coffee","tea","juice","water","egg","cheese",
    "soup","salad","hotdog","fries","doughnut","pancake","biscuit",

    # School & Learning
    "blackboard","whiteboard","book","ruler","eraser","bag","notebook",
    "calculator","scissors","glue","marker","compass","telescope","microscope",

    # Technology & Electronics
    "laptop","mouse","keyboard","printer","speaker","headphones","microphone",
    "projector","camera","drone","robot","tablet","smartwatch","controller",
    "charger","battery","cable",

    # Sports & Recreation
    "football","soccer ball","basketball","cricket bat","tennis racket",
    "badminton racket","baseball bat","golf club","skateboard","surfboard",
    "swimming pool","goalpost","whistle","trophy","medal","helmet",

    # Miscellaneous
    "flag","map","sign","poster","newspaper","gift","balloon","toy car",
    "shopping cart","money","wallet","coin","credit card","ticket","passport",
    "calendar","booklet","clipboard","notepad","suitcase","rope","ladder",
    "plant pot","vase","painting","photo frame","clock tower","bench",
    "dustbin","mailbox","fence","gate","pathway","stairs","handbag","watch"
]

# --- Quick, Draw! combined label maps + sketch labeler (self-contained) ---

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# Canonicalization for label variants (merge synonyms / variants)
_LABEL_CANON = {
    "stick man": "stick figure",
    "smiley face": "face",
    "surprised face": "face_surprised",
    "angry face": "face_angry",
    "sad face": "face_sad",
    "night": "night_sky",
    "day": "day_sky",
    "sea": "ocean",
    "t-shirt": "tshirt",
}
def _canon(label: str) -> str:
    l = (label or "").strip().lower()
    return _LABEL_CANON.get(l, l)

# Optional HOG features. If scikit-image is not present, we fall back automatically.
try:
    from skimage.feature import hog as _hog
    _HAS_HOG = True
except Exception:
    _hog = None
    _HAS_HOG = False

_QD_IMG_SIZE = 64
_ROOT = Path(__file__).resolve().parents[0]        # .../server
_MODEL_DIR = _ROOT / "models"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)
_QD_INDEX_PATH = _MODEL_DIR / "quickdraw_index.npz"

def _qd_feat_from_pil(pil: Image.Image) -> np.ndarray:
    g = pil.convert("L").resize((_QD_IMG_SIZE, _QD_IMG_SIZE))
    arr = np.asarray(g, dtype=np.float32)
    if _HAS_HOG:
        f = _hog(arr, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
        return f.astype(np.float32)
    arr = (255.0 - arr) / 255.0
    return arr.flatten().astype(np.float32)

def _cosine_topk(x: np.ndarray, X: np.ndarray, k: int = 25):
    x = x / (np.linalg.norm(x) + 1e-8)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sims = Xn @ x
    idx = np.argpartition(sims, -k)[-k:]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

class _QDIndexSingleton:
    inst = None

class QuickDrawSketchLabeler:
    """Memory-mapped cosine kNN over precomputed features from Quick, Draw! sketches."""
    def __init__(self, index_path: Path = _QD_INDEX_PATH):
        self.index_path = Path(index_path)
        self.ready = False
        self.labels = []
        self.X = None
        self.y = None
        self._try_load()

    def _try_load(self):
        if not self.index_path.exists():
            self.ready = False
            return
        z = np.load(self.index_path, allow_pickle=True)
        self.X = z["X"].astype(np.float32)
        self.y = z["y"].astype(object)
        self.labels = sorted(list(set(self.y.tolist())))
        self.ready = True

    def predict_topk(self, pil: Image.Image, k: int = 5):
        if not self.ready or self.X is None or self.y is None:
            return []
        f = _qd_feat_from_pil(pil)
        idx, sims = _cosine_topk(f, self.X, k=50)
        votes = {}
        for i, s in zip(idx, sims):
            lb = _canon(str(self.y[i]))
            sc = float(max(0.0, s))
            if lb not in votes or sc > votes[lb]:
                votes[lb] = sc
        return sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[:k]

def get_quickdraw_labeler():
    if _QDIndexSingleton.inst is None:
        _QDIndexSingleton.inst = QuickDrawSketchLabeler()
    return _QDIndexSingleton.inst if _QDIndexSingleton.inst.ready else None

def enrich_scene_with_sketch_labels(
    pil: Image.Image,
    scene: dict,
    min_conf: float = 0.30,
    *,
    topk: int = 5,
    merge_mode: str = "union",   # "union" or "intersect"
) -> dict:
    """
    Merge Quick, Draw! labels into scene['objects'].
    - topk: how many sketch labels to consider
    - min_conf: cosine threshold to keep a label
    - merge_mode:
        "union"     -> add QuickDraw labels to existing objects
        "intersect" -> keep only objects that also appear in QuickDraw
    Safe no-op if index not loaded.
    """
    sk = get_quickdraw_labeler()
    if not sk:
        return scene

    preds = sk.predict_topk(pil, k=topk)  # [(label, score)]
    qd = [lb for lb, sc in preds if sc >= min_conf]
    base = scene.get("objects") or []

    if merge_mode == "intersect":
        merged = [o for o in base if o in qd]
    else:  # union
        merged = list(dict.fromkeys(qd + base))

    scene["objects"] = merged

    # mood from face_* classes (face_angry/face_sad/face_surprised)
    mood = next((x.split("_", 1)[1] for x in merged if x.startswith("face_") and "_" in x), None)
    if mood:
        scene["mood"] = mood
        scene["sentiment"] = {"angry": "tense", "sad": "gentle", "surprised": "curious"}.get(
            mood, scene.get("sentiment", "neutral")
        )
    return scene

# --- End Quick, Draw! block ---



def _init_clip() -> bool:
    """Load CLIP and precompute averaged text embeddings (kid-sketch templates)."""
    global CLIP_MODEL, CLIP_PREPROC, CLIP_TOKENIZER, TEXT_EMB
    if not _HAS_OPEN_CLIP:
        return False
    if CLIP_MODEL is not None:
        return True

    try:
        model, preproc, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        tok = open_clip.get_tokenizer("ViT-B-32")
        model = model.to(DEVICE).eval()

        templates = [
            "a child's line drawing of a {}",
            "a doodle of a {}",
            "a cartoon {}",
            "a simple sketch of a {}",
            "a cute {}",
        ]
        prompts, owners = [], []
        for i, lab in enumerate(LABELS):
            variants = [t.format(lab) for t in templates]
            prompts.extend(variants)
            owners.extend([i] * len(variants))

        with torch.no_grad():
            txt_tokens = tok(prompts).to(DEVICE)
            emb = model.encode_text(txt_tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        dim = emb.shape[-1]
        avg = torch.zeros((len(LABELS), dim), device=emb.device)
        cnt = torch.zeros((len(LABELS), 1), device=emb.device)
        for r, lab_idx in enumerate(owners):
            avg[lab_idx] += emb[r]
            cnt[lab_idx] += 1
        avg = avg / torch.clamp(cnt, min=1.0)
        avg = avg / avg.norm(dim=-1, keepdim=True)

        CLIP_MODEL, CLIP_PREPROC, CLIP_TOKENIZER, TEXT_EMB = model, preproc, tok, avg
        return True
    except Exception:
        CLIP_MODEL = CLIP_PREPROC = CLIP_TOKENIZER = TEXT_EMB = None
        return False


def _preprocess_sketch(pil: Image.Image) -> Image.Image:
    # If it's a colorful photo (not a line drawing), keep it as-is.
    import numpy as np
    arr = np.asarray(pil.convert("RGB"))
    # crude photo-ness: std of pixel values across channels
    photoish = arr.std() > 25  # tweak threshold if needed
    if photoish:
        return pil

    # else: use the sketch pipeline (great for drawings)
    from PIL import ImageOps, ImageFilter
    g = pil.convert("L")
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.MedianFilter(3))
    g = g.filter(ImageFilter.MaxFilter(3))
    return g.convert("RGB")



def clip_recognize(pil: Image.Image, topk: int = 3):
    if not _init_clip():
        raise RuntimeError("clip_unavailable")
    with torch.no_grad():
        sk = _preprocess_sketch(pil)
        img = CLIP_PREPROC(sk).unsqueeze(0).to(DEVICE)
        img_emb = CLIP_MODEL.encode_image(img)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sims = (img_emb @ TEXT_EMB.T).squeeze(0)
        vals, idx = sims.topk(topk)
        probs = torch.softmax(vals, dim=0).tolist()
        idx = idx.tolist()
        return [{"label": LABELS[i], "score": float(probs[j])} for j, i in enumerate(idx)]


# ---------- Object detector (optional) ----------
YOLO_DETECTOR = None
DETR_PIPELINE = None

def _init_yolo() -> bool:
    global YOLO_DETECTOR
    if not _HAS_ULTRALYTICS:
        return False
    if YOLO_DETECTOR is not None:
        return True
    try:
        # Small, fast model; downloads weights on first use
        YOLO_DETECTOR = YOLO("yolov8n.pt")
        return True
    except Exception:
        YOLO_DETECTOR = None
        return False

def _init_detr() -> bool:
    global DETR_PIPELINE
    if not _HAS_TRANSFORMERS_PIPE:
        return False
    if DETR_PIPELINE is not None:
        return True
    try:
        device = 0 if torch.cuda.is_available() else -1
        DETR_PIPELINE = hf_pipeline("object-detection", model="facebook/detr-resnet-50", device=device)
        return True
    except Exception:
        DETR_PIPELINE = None
        return False

def _canon_det_label(s: str) -> str:
    # Minimal canonicalization for detector labels
    lab = (s or "").strip().lower()
    # Normalize a few common COCO phrases
    repl = {
        "motorbike": "motorcycle",
        "aeroplane": "airplane",
        "tv": "television",
        "tvmonitor": "television",
        "sports ball": "ball",
    }
    return repl.get(lab, lab)

def detect_objects(pil: Image.Image, min_conf: float = 0.28, topk: int = 30):
    """Return list of {label, score} from YOLOv8n or DETR fallback.
    Gracefully returns [] if both unavailable/fail.
    """
    # 1) YOLOv8n (Ultralytics)
    try:
        if _init_yolo():
            res = YOLO_DETECTOR.predict(source=pil, conf=min_conf, max_det=topk, verbose=False)
            if res and len(res) > 0:
                r = res[0]
                names = getattr(r, "names", {}) or {}
                boxes = getattr(r, "boxes", None)
                out = []
                if boxes is not None:
                    try:
                        # ultralytics Results API
                        agg_score = {}
                        agg_ct = {}
                        for i in range(len(boxes)):
                            cls_id = int(boxes.cls[i].item())
                            conf = float(boxes.conf[i].item())
                            label = _canon_det_label(names.get(cls_id, str(cls_id)))
                            if conf >= min_conf:
                                prev = agg_score.get(label, 0.0)
                                if conf > prev:
                                    agg_score[label] = conf
                                agg_ct[label] = agg_ct.get(label, 0) + 1
                        out = [{"label": l, "score": float(agg_score[l]), "count": int(agg_ct.get(l, 1))} for l in agg_score.keys()]
                    except Exception:
                        pass
                # Rank and cap
                out.sort(key=lambda d: -float(d.get("score", 0.0)))
                return out[:topk]
    except Exception:
        pass

    # 2) DETR fallback (Transformers pipeline)
    try:
        if _init_detr():
            preds = DETR_PIPELINE(pil)
            agg_score = {}
            agg_ct = {}
            for p in preds:
                try:
                    label = _canon_det_label(p.get("label", ""))
                    score = float(p.get("score", 0.0))
                    if score >= min_conf and label:
                        prev = agg_score.get(label, 0.0)
                        if score > prev:
                            agg_score[label] = score
                        agg_ct[label] = agg_ct.get(label, 0) + 1
                except Exception:
                    continue
            out = [{"label": l, "score": float(agg_score[l]), "count": int(agg_ct.get(l, 1))} for l in agg_score.keys()]
            out.sort(key=lambda d: -float(d.get("score", 0.0)))
            return out[:topk]
    except Exception:
        pass

    return []


@app.post("/api/recognize")
async def api_recognize(image: UploadFile = File(...)):
    """Always returns 200; best-effort top-k labels for guidance.
    On CLIP failure, fallback to BLIP noun overlap from a small kid-friendly set.
    """
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")

        # Try CLIP
        try:
            top = clip_recognize(pil, topk=3)
            if isinstance(top, list) and top:
                return {"top": top}
        except Exception:
            pass

        # Fallback: BLIP caption -> first noun from small set
        small = [
            "dog","cat","bird","apple","ball","kite","tree","flower","house",
            "boy","girl","child","car","bus","bicycle","train","boat","airplane",
            "sun","cloud","rain","snow","book","cake","cookie","pizza","guitar",
            "chair","table","bed","worm"
        ]
        try:
            eng = story_engine
            cap_low = (eng.captioner.caption(pil) or "").lower()
            cand = [lab for lab in small if lab in cap_low]
            guess = cand[0] if cand else "child"
            return {"top": [{"label": guess, "score": 0.12}]}
        except Exception:
            # last resort neutral guess
            return {"top": [{"label": "child", "score": 0.10}]}
    except Exception as e:
        return {"top": [], "error": f"{type(e).__name__}: {e}"}


# Description engine (Flickr8k retrieval -> paragraph)
try:
    retrieval_captioner = ShowTellCaptioner(model_root=BASE)
except Exception as e:
    retrieval_captioner = None
    print("ShowTellCaptioner init failed:", e)

# ---------- Lightweight caption (for UI hints) ----------
@app.post("/api/caption")
async def caption(
    image: UploadFile = File(...),
    region: Optional[str] = Form(None),
    mode: Optional[str] = Form(None)
):
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        return JSONResponse(content={"error": f"invalid_image: {e}"}, status_code=400)

    # Parse region if provided
    region_box = None
    if region:
        try:
            region_box = json.loads(region)
        except Exception:
            region_box = None

    # Fast path: keep old behaviour when mode != "detailed"
    if (mode or "").lower() != "detailed":
        try:
            eng = story_engine
            text = eng.captioner.caption(pil) or ""
            return {"caption": text.strip()}
        except Exception:
            return {"caption": ""}

    # ---- DETAILED MODE ----
    # 1) Base BLIP (short, fluent)
    base = ""
    try:
        base = (story_engine.captioner.caption(pil, region=region_box) or "").strip()
    except Exception:
        base = ""

    # 2) Retrieval paragraph from neighbours (richer detail)
    para = ""
    sentences = []
    try:
        if retrieval_captioner and getattr(retrieval_captioner, "ready", False):
            desc = retrieval_captioner.describe(pil, region=region_box, mode="paragraph", n_candidates=5)
            if isinstance(desc, dict):
                para = (desc.get("paragraph") or "").strip()
                sentences = list(desc.get("sentences") or [])
            else:
                para = (desc or "").strip()
    except Exception:
        pass

    # 3) Object detector (YOLOv8n -> DETR fallback) to list concrete nouns
    det_raw = []
    try:
        det_raw = detect_objects(pil, min_conf=0.28, topk=30)
    except Exception:
        det_raw = []
    # Prepare objects list for response (filled later after merging with CLIP)
    objects = []

    # 4) CLIP label hints (fine-grained nouns + scores kept for thresholds)
    clip_raw = []
    clip_labels = []
    try:
        clip_raw = clip_recognize(pil, topk=64)  # returns [{"label": "...", "score": ...}]
        clip_labels = [t.get("label", "") for t in (clip_raw or []) if t.get("label")]
    except Exception:
        clip_raw = []
        clip_labels = []

    # Canonicalize labels (plural/synonyms -> base) and merge CLIP + detector
    def _canon_label_token(s: str) -> str:
        s = (s or "").strip().lower()
        syn = {
            "grapes": "grape",
            "green grapes": "grape",
            "purple grapes": "grape",
            "bunch of grapes": "grape",
            "apples": "apple",
            "oranges": "orange",
            "bananas": "banana",
            "peaches": "peach",
            "lemons": "lemon",
            "limes": "lime",
            "tomatoes": "tomato",
            "potatoes": "potato",
            "pears": "pear",
            "carrots": "carrot",
            "cherries": "cherry",
        }
        return syn.get(s, s)

    labels = []
    seen_labs = set()
    for arr in (clip_labels, [d["label"] for d in det_raw]):
        for lab in arr:
            ll = _canon_label_token(lab)
            if ll and ll not in seen_labs:
                seen_labs.add(ll)
                labels.append(ll)

    # Map label->score using max across sources
    score_of = {}
    for t in (clip_raw or []):
        k = _canon_label_token(t.get("label") or "")
        if not k:
            continue
        score_of[k] = max(float(t.get("score", 0.0)), score_of.get(k, 0.0))
    for t in (det_raw or []):
        k = _canon_label_token(t.get("label") or "")
        if not k:
            continue
        score_of[k] = max(float(t.get("score", 0.0)), score_of.get(k, 0.0))
    # Detector counts per canonical label (0 if absent)
    det_counts = { _canon_label_token(t.get("label") or ""): int(t.get("count", 0) or 0) for t in (det_raw or []) if t.get("label") }


    # 5) Upgrade generic nouns -> specific ones based on labels (now includes detector hints)
    FRUITS = {
        "apple","banana","orange","mango","grape","grapes","strawberry","pineapple",
        "watermelon","lemon","lime","pear","peach","cherry","tomato","coconut",
        "durian","jackfruit","papaya","guava","lychee","rambutan","melon",
        "pomegranate","kiwi","dragonfruit","avocado","plum","tangerine"
    }
    VEGGIES = {
        "broccoli","cucumber","carrot","onion","garlic","potato","sweet potato",
        "eggplant","brinjal","pumpkin","cabbage","lettuce","spinach",
        "bell pepper","red pepper","green pepper","chili","pepper",
        "cauliflower","zucchini","radish","beetroot","yam","ginger","okra",
        "bitter gourd","celery","asparagus","corn","peas","bean","capsicum",
        "mushroom","leek","turnip","mint","basil","coriander","parsley"
    }
    VEHICLES = {"car","bus","bicycle","train","boat","airplane","truck","motorcycle","van","ship"}
    FURN = {"chair","table","bed","sofa","lamp"}
    BIRDS = {"bird","eagle","owl","penguin","parrot","peacock","duck","chicken","heron","egret","seagull","sparrow","pigeon","crow","flamingo"}
    FISH  = {"fish","whale","dolphin","goldfish","salmon","shark","ray"}
    ANIMALS = {"dog","cat","rabbit","horse","cow","sheep","goat","pig","elephant","tiger","lion","monkey","bear","deer","snake","giraffe","zebra","kangaroo","panda","crocodile","mouse","frog","alpaca","llama"}
    FRUIT_FAMILY_SPIKY = {"durian","jackfruit","rambutan","soursop"}

    label_set = set(x.lower() for x in labels)

    def _fmt_list(words, cap: int = 4) -> str:
        """Join a small ranked list ('a, b and c'). Deduplicate and rank by CLIP score."""
        uniq = []
        seen = set()
        for w in words:
            lw = w.lower()
            if lw not in seen:
                seen.add(lw)
                uniq.append(w)
        # rank by score (desc)
        uniq.sort(key=lambda w: -score_of.get(w.lower(), 0.0))
        uniq = uniq[:cap]
        if not uniq:
            return ""
        if len(uniq) == 1:
            return uniq[0]
        return ", ".join(uniq[:-1]) + " and " + uniq[-1]
    
    detected_set = set((d.get("label") or "").lower() for d in (det_raw or []))

    def _top_hits(pool: set[str], k: int = 3, min_score: float = 0.22, prefer_detected: bool = False) -> list[str]:
        """Return top-k labels from 'pool' with optional preference for detector-backed items.
        If prefer_detected is True, detector-backed labels get a small boost; CLIP-only labels
        need a slightly higher score to pass the threshold to reduce color-name confusions
        (e.g., apples vs oranges).
        """
        cands = []
        for w in labels:
            lw = w.lower()
            if lw not in pool:
                continue
            s = float(score_of.get(lw, 0.0))
            if prefer_detected:
                if lw in detected_set:
                    eff = s + 0.06
                else:
                    eff = s - 0.04  # downweight CLIP-only
                # boost for grape/grapes which detectors rarely cover
                if lw in {"grape"}:
                    eff += 0.07
            else:
                eff = s
            if eff >= min_score:
                cands.append((w, eff))
        cands.sort(key=lambda t: -t[1])
        return [w for (w, _eff) in cands[:k]]

    def _suppress_confusables(words: list[str]) -> list[str]:
        """If multiple spiky fruits appear, keep only the single best-scoring one."""
        picked = []
        max_spiky = None
        max_spiky_score = -1.0
        for w in words:
            lw = w.lower()
            s = score_of.get(lw, 0.0)
            if lw in FRUIT_FAMILY_SPIKY:
                if s > max_spiky_score:
                    max_spiky, max_spiky_score = lw, s
            else:
                picked.append(w)
        if max_spiky is not None:
            picked.append(max_spiky)
        picked.sort(key=lambda w: -score_of.get(w.lower(), 0.0))
        return picked

    def _destutter(text: str) -> str:
        """Collapse repeated tokens like 'jack jack jack' or 'fanny fanny fanny'."""
        t = re.sub(r"\b(\w+)(?:\s+\1\b){1,}", r"\1", text, flags=re.I)
        t = re.sub(r"\s{2,}", " ", t).strip()
        return t
    
    def upgrade(text: str) -> str:
        if not text:
            return text
        t = text
        
        # --- fruits ---
        if re.search(r"\bfruits\b", t, flags=re.I) or re.search(r"\bfruit\b", t, flags=re.I):
            fruit_hits = _top_hits(FRUITS, k=4, min_score=0.24)
            fruit_hits = _suppress_confusables(fruit_hits)
            if fruit_hits:
                if re.search(r"\bfruits\b", t, flags=re.I):
                    t = re.sub(r"\b[Ff]ruits\b", _fmt_list(fruit_hits, cap=3), t)
                else:
                    t = re.sub(r"\b[Ff]ruit\b", fruit_hits[0], t)
                    
        # --- vegetables ---
        if re.search(r"\bvegetables\b", t, flags=re.I) or re.search(r"\bvegetable\b", t, flags=re.I):
            veg_hits = _top_hits(VEGGIES, k=5, min_score=0.24)
            if veg_hits:
                if re.search(r"\bvegetables\b", t, flags=re.I):
                    t = re.sub(r"\b[Vv]egetables\b", _fmt_list(veg_hits, cap=4), t)
                else:
                    t = re.sub(r"\b[Vv]egetable\b", veg_hits[0], t)

        # --- animals vs birds vs fish ---
        # Prefer a *specific* bucket when BLIP says "animal(s)"
        if re.search(r"\banimals?\b", t, flags=re.I):
            bird_hits = _top_hits(BIRDS, k=3, min_score=0.23)
            fish_hits = _top_hits(FISH,  k=2, min_score=0.23)
            mammal_hits = _top_hits(ANIMALS, k=3, min_score=0.23)
            repl = ""
            # choose the bucket with the highest top score
            top_cand = []
            for arr in (bird_hits, fish_hits, mammal_hits):
                if arr:
                    top_cand.append((score_of.get(arr[0].lower(), 0.0), arr))
            if top_cand:
                _, best_arr = max(top_cand, key=lambda t:t[0])
                repl = _fmt_list(best_arr, cap=3)
            if repl:
                t = re.sub(r"\b[Aa]nimals?\b", repl, t)

        # --- vehicles ---
        if re.search(r"\bvehicles?\b", t, flags=re.I):
            v_hits = _top_hits(VEHICLES, k=3, min_score=0.23)
            if v_hits:
                if re.search(r"\bvehicles\b", t, flags=re.I):
                    t = re.sub(r"\b[Vv]ehicles\b", _fmt_list(v_hits, cap=3), t)
                else:
                    t = re.sub(r"\b[Vv]ehicle\b", v_hits[0], t)

        # --- camelids vs sheep disambiguation (prefer alpaca/llama when confident) ---
        if re.search(r"\bsheep\b", t, flags=re.I):
            s_sheep = float(score_of.get("sheep", 0.0))
            best_cand = None
            best_score = 0.0
            for cand in ("alpaca", "llama"):
                sc = float(score_of.get(cand, 0.0))
                if sc > best_score:
                    best_score, best_cand = sc, cand
            # flip only if camelid is reasonably confident and beats sheep by a margin
            if best_cand and best_score >= max(0.23, s_sheep + 0.05):
                singular = best_cand
                plural = best_cand + "s"
                # grammar-sensitive article for singular replacement
                art = "an" if singular[0] in "aeiou" else "a"
                # 1) specific articles
                t = re.sub(r"\b([Aa]n|[Aa])\s+[Ss]heep\b", f"{art} {singular}", t)
                t = re.sub(r"\b([Oo]ne)\s+[Ss]heep\b", f"one {singular}", t)
                # 2) generic sheep tokens -> plural camelid
                t = re.sub(r"\b[Ss]heep\b", plural, t)

        # --- furniture ---
        if re.search(r"\bfurniture\b", t, flags=re.I):
            f_hits = _top_hits(FURN, k=3, min_score=0.25)
            if f_hits:
                t = re.sub(r"\b[Ff]urniture\b", _fmt_list(f_hits, cap=3), t)

        # --- catch-all for very generic words (only if confident labels exist) ---
        generic_pat = r"\b(things|items|objects|stuff|produce|food|groceries|ingredients|toys|tools|electronics)\b"
        if re.search(generic_pat, t, flags=re.I):
            top_any = [w for w in labels if score_of.get(w.lower(), 0.0) >= 0.26]
            if top_any:
                t = re.sub(generic_pat, _fmt_list(top_any, cap=5), t)

        # final cleanup: no stutters, tidy spaces
        return _destutter(t)

    # 6) Build objects[] with category-aware selection to improve accuracy
    #    - Prioritize clear fruits/vegetables and birds vs. other animals
    #    - Then add a few high-confidence detector-only items for breadth
    fruit_hits = _top_hits(FRUITS, k=4, min_score=0.24, prefer_detected=True)
    fruit_hits = _suppress_confusables(fruit_hits)
    veg_hits = _top_hits(VEGGIES, k=4, min_score=0.24, prefer_detected=True)
    bird_hits = _top_hits(BIRDS, k=4, min_score=0.23, prefer_detected=False)
    # animals excluding birds
    animals_core = ANIMALS - BIRDS
    mammal_hits = _top_hits(animals_core, k=4, min_score=0.23, prefer_detected=False)

    # other strong detector labels not in our category pools
    cat_pool = FRUITS | VEGGIES | BIRDS | ANIMALS | FISH | VEHICLES | FURN
    det_only = []
    for d in (det_raw or []):
        lab = (d.get("label") or "").lower()
        if not lab or lab in cat_pool:
            continue
        if score_of.get(lab, 0.0) >= 0.30:
            det_only.append((lab, score_of[lab]))
    det_only.sort(key=lambda t: -t[1])
    other_hits = [l for l, _ in det_only[:5]]

    # stricter filtering for produce to avoid spurious second fruit/veg types (e.g., oranges with apples)
    def _filter_produce(seq: list[str]) -> list[str]:
        if not seq:
            return []
        out = []
        for i, w in enumerate(seq):
            lw = w.lower()
            sc = float(score_of.get(lw, 0.0))
            ct = int(det_counts.get(lw, 0))
            # first produce: allow if detector saw it OR CLIP is strong
            if i == 0:
                if ct >= 1 or sc >= 0.32:
                    out.append(w)
            else:
                # additional produce: need stronger evidence to keep
                # grapes are often not a COCO class; allow slightly lower CLIP score
                if lw == "grape":
                    if ct >= 1 or sc >= 0.30:
                        out.append(w)
                elif ct >= 1 or sc >= 0.36:
                    out.append(w)
        return out

    fruit_hits = _filter_produce(fruit_hits)
    veg_hits = _filter_produce(veg_hits)

    # Note: generic filtering only; no special-case differentiation between similar fruits

    # stitch final objects list (dedup, preserve order)
    merged_seq = fruit_hits + veg_hits + bird_hits + mammal_hits + other_hits
    seen = set()
    objects = []
    for w in merged_seq:
        lw = (w or "").lower()
        if not lw or lw in seen:
            continue
        seen.add(lw)
        objects.append(w)

    # 7) If we clearly see multiple fruits/veggies or multiple animals/birds,
    #    compose a precise list-style caption to avoid generic mistakes.
    def _pluralize(w: str) -> str:
        w = (w or "").strip()
        if not w:
            return w
        lower = w.lower()
        IRREG = {
            "sheep": "sheep", "deer": "deer", "fish": "fish", "goose": "geese",
            "mouse": "mice", "ox": "oxen", "cactus": "cacti", "fungus": "fungi",
            "tomato": "tomatoes", "potato": "potatoes", "cherry": "cherries",
            "strawberry": "strawberries", "peach": "peaches", "leaf": "leaves",
            "wolf": "wolves", "knife": "knives", "loaf": "loaves", "zucchini": "zucchini",
            "okra": "okra", "corn": "corn", "rice": "rice", "spinach": "spinach",
            "lettuce": "lettuce", "broccoli": "broccoli", "cauliflower": "cauliflower"
        }
        if lower in IRREG:
            return IRREG[lower]
        # keep multiword as-is (e.g., "green pepper" -> "green peppers")
        if " " in lower:
            head, *rest = lower.split()
            tail = " ".join(rest)
            return (head + "s" + (" " + tail if tail else ""))
        if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
            return lower[:-1] + "ies"
        if lower.endswith(("sh","ch","x","z","o")):
            return lower + "es"
        return lower + "s"

    def _compose_list_caption():
        # prioritize produce lists; otherwise animals/birds
        prod = fruit_hits + veg_hits
        fa = bird_hits + mammal_hits
        # rank by score, keep unique
        def rank_unique(seq):
            seen=set(); out=[]
            for w in seq:
                lw=w.lower();
                if lw in seen: continue
                seen.add(lw); out.append(w)
            out.sort(key=lambda w: -score_of.get(w.lower(), 0.0))
            return out
        prod = rank_unique(prod)
        fa = rank_unique(fa)

        container = ""
        ctx = set(x.lower() for x in labels)
        if "basket" in ctx: container = "in a basket"
        elif "bowl" in ctx: container = "in a bowl"
        elif "plate" in ctx: container = "on a plate"
        elif "table" in ctx: container = "on a table"
        elif "tree" in ctx: container = "on a tree branch"

        def to_sentence(item_list, container_phrase):
            phrase = _fmt_list(item_list, cap=5)
            if not phrase:
                return ""
            if container_phrase in ("in a basket", "in a bowl"):
                # Use a natural container-led sentence
                return f"A {container_phrase.split()[-1]} filled with {phrase}.".replace("  ", " ")
            if container_phrase:
                return f"{phrase.capitalize()} are {container_phrase}."
            return f"The picture shows {phrase}."

        if len(prod) >= 2:
            items = prod[:5]
            items_pl = [_pluralize(w) for w in items]
            return to_sentence(items_pl, container)
        if len(fa) >= 2:
            items = fa[:5]
            items_pl = [_pluralize(w) for w in items]
            return to_sentence(items_pl, container)
        return ""

    list_caption = _compose_list_caption()

    best_sentence = list_caption or upgrade(base)
    best_paragraph = upgrade(para or base)
    _cap = (best_sentence or best_paragraph or "").strip()
    if _cap and not re.search(r"[\.!?]$", _cap):
        _cap += "."
        # Universal prefix enforcement
    if _cap:
        if not _cap.lower().startswith("the image shows"):
            _cap = re.sub(r"^(in the picture[, ]*)", "The image shows ", _cap, flags=re.I)
            if not _cap.lower().startswith("the image shows"):
                _cap = "The image shows " + _cap

    return {
        "caption": _cap,
        "labels": labels,
        "objects": objects,
        "sentences": sentences,
        "paragraph": best_paragraph.strip(),
        "mode": "detailed"
    }


# ---------- Save / Recent / Serve image ----------
@app.post("/api/save")
async def save_item(caption: str = Form(...), image: UploadFile = File(...)):
    try:
        data = await image.read()
        img_id = f"{uuid.uuid4().hex}.jpg"
        img_path = IMG_DIR / img_id
        Image.open(io.BytesIO(data)).convert("RGB").save(img_path, "JPEG", quality=92)

        items = _load_items()
        obj = {
            "id": uuid.uuid4().hex,
            "imageUrl": f"/api/image/{img_id}",
            "caption": caption,
            "savedAt": int(time.time() * 1000),
        }
        items.append(obj)
        _save_items(items)
        return obj
    except Exception as e:
        return JSONResponse(content={"error": f"save_error: {e}"}, status_code=500)


@app.get("/api/recent")
async def recent():
    items = _load_items()
    return items[-1] if items else {}


@app.get("/api/image/{name}")
async def serve_image(name: str):
    path = IMG_DIR / name
    if not path.exists():
        return JSONResponse(content={"error": "not_found"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg")

# ---------- CVD (color-vision) filter ----------
from typing import Literal
from fastapi import UploadFile, File, Form
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
import io, uuid, tempfile

@app.post("/api/cvd/apply")
async def cvd_apply(
    image: UploadFile = File(...),
    mode: Literal["simulate","daltonize"] = Form("simulate"),
    cvd_type: str = Form("deuteranopia"),
    severity: float = Form(1.0),
    amount: float = Form(1.0),
):
    """
    Apply colour-vision simulation/daltonization using the proper CVD pipeline
    backed by colorspacious (see server/csvd_filter.py). Accepts broader cvd_type
    values (e.g., "protanopia"/"deuteranopia"/"tritanopia" as well as
    "protan"/"deutan"/"tritan"). Returns a PNG stream.
    """
    raw = await image.read()

    try:
        # Lazy import to avoid importing heavy deps unless needed
        from .csvd_filter import apply as cvd_apply_image
    except Exception:
        # Fallback to original image if pipeline unavailable
        buf = io.BytesIO(raw)
        return StreamingResponse(buf, media_type="image/png")

    t = (cvd_type or "").lower()
    if t.startswith("prot"):  t = "protan"
    elif t.startswith("deut"): t = "deutan"
    elif t.startswith("trit"): t = "tritan"
    else: t = "none"

    sev = float(max(0.0, min(1.0, float(severity))))
    mode_norm = "daltonize" if mode == "daltonize" else "simulate"

    out_img = cvd_apply_image(io.BytesIO(raw), mode=mode_norm, cvd_type=t, severity=sev, amount=float(amount))
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# ---------- Story Generation ----------
story_engine = StoryGenerator()


@app.post("/api/story")
async def story_api(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    image3: UploadFile = File(...),
    mood: str = Form("friendly"),
):
    """
    Story generation with Quick, Draw! enrichment:
      1) Read 3 uploads as PIL.
      2) Get CLIP label hints (best-effort).
      3) Build scenes via StoryGenerator.
      4) Enrich each scene with Quick, Draw! labels (from index).
      5) Generate title, 3 panels, moral; save the 3 inputs to server/data/scenes.
    Always returns exactly 3 panel lines plus title & moral.
    """
    try:
        # 1) Read images
        raw1, raw2, raw3 = await image1.read(), await image2.read(), await image3.read()
        p1 = Image.open(io.BytesIO(raw1)).convert("RGB")
        p2 = Image.open(io.BytesIO(raw2)).convert("RGB")
        p3 = Image.open(io.BytesIO(raw3)).convert("RGB")
        eng = story_engine

        # 2) CLIP label suggestions (best-effort; safe if CLIP unavailable)
        def labels_for(img):
            try:
                top = clip_recognize(img, topk=3)  # [{"label": "...", "score": ...}]
                return [t["label"] for t in (top or []) if "label" in t]
            except Exception:
                return []
        lab1, lab2, lab3 = labels_for(p1), labels_for(p2), labels_for(p3)

        # 3) Build scenes (+ deltas) using your existing engine
        scenes, deltas = eng.build_scenes([p1, p2, p3], [lab1, lab2, lab3])

        # 4) Quick, Draw! enrichment temporarily disabled per request
        #    Keep the pipeline simple and rely on BLIP + CLIP only.
        #    If you want to re-enable later, restore the block below.
        # enriched = []
        # for pil_img, scene in zip([p1, p2, p3], scenes):
        #     enriched.append(
        #         enrich_scene_with_sketch_labels(
        #             pil_img, dict(scene),
        #             min_conf=0.20,
        #             topk=3,
        #             merge_mode="union"
        #         )
        #     )
        enriched = scenes

        # 5) Generate story text from enriched scenes (keep your planner unchanged)
        title, panels, moral = eng.generate_from_scenes(enriched, deltas, mood=mood)
        story_text = "\n".join(panels)

        # Ensure exactly 3 panel sentences
        panels = [(panels[i] if i < len(panels) and panels[i] else "") for i in range(3)]

        # Original BLIP captions (useful for UI debugging)
        captions = [s.get("caption", "") for s in enriched]

        # 6) Persist the three input drawings under server/data/scenes
        names = []
        for idx, img in enumerate([p1, p2, p3], start=1):
            fname = f"scene_{int(time.time())}_{uuid.uuid4().hex[:8]}_{idx}.jpg"
            out_path = IMG_DIR / fname
            try:
                img.save(out_path, "JPEG", quality=92)
                names.append(fname)
            except Exception:
                names.append(None)
        image_urls = [f"/api/image/{n}" if n else "" for n in names]

        # 7) Response payload (backward compatible)
        return {
            "title": title,
            "story": story_text,
            "panels": panels,
            "moral": moral,
            "captions": captions,
            "scenes": enriched,      # enriched scenes (contains objects/mood)
            "deltas": deltas,        # as computed during build_scenes
            "labels": [lab1, lab2, lab3],
            "images": image_urls,
        }

    except Exception as e:
        # Graceful fallback so the UI never hangs on errors
        return {
            "error": f"{type(e).__name__}: {e}",
            "title": "A LITTLE ADVENTURE",
            "panels": [
                "We see a simple scene.",
                "Then something changes.",
                "Finally, it ends happily."
            ],
            "moral": "We learn and smile together."
        }




# ---------- (Optional) Simple quiz stub to keep route alive ----------
class QuizIn(BaseModel):
    caption: str
    count: Optional[int] = 3

quiz_engine = QuizGenerator()


from fastapi import Body

@app.post("/api/quiz")
def api_quiz(payload: dict = Body(...)):
    """
    Input: { "caption": str, "count": 3 }
    Output: { "questions": [{question, options[3], answer_index}] }
    """
    try:
        caption = (payload.get("caption") or "").strip()
        count = int(payload.get("count") or 3)
        # Always serve 3 questions (as requested), clamp any smaller value up to 3
        count = 3 if count < 3 else min(count, 3)

        # Optional: include detected objects (from CLIP/QuickDraw) into caption
        # Accepts payload fields: objects | labels | sketch_labels (list[str] or comma-separated str)
        obj_field = None
        for key in ("objects", "labels", "sketch_labels"):
            if key in payload and payload.get(key):
                obj_field = payload.get(key)
                break

        objects: list[str] = []
        if isinstance(obj_field, list):
            objects = [str(x).strip() for x in obj_field if str(x).strip()]
        elif isinstance(obj_field, str):
            parts = [p.strip() for p in obj_field.replace("/", ",").replace("|", ",").split(",")]
            objects = [p for p in parts if p]

        # Deduplicate and keep short
        seen = set()
        objects = [o for o in objects if not (o.lower() in seen or seen.add(o.lower()))]
        if objects:
            # Only append if not already present to avoid repetition
            if not re.search(r"\bobjects?\s*:\s*", caption, flags=re.I):
                tail = ", ".join(objects[:8])
                caption = (caption.rstrip(".") + f" Objects: {tail}.").strip()

        qs = quiz_engine.generate(caption, num_questions=count)
        return {"questions": qs}
    except Exception as e:
        # safe fallback with generic but valid payload
        fb = quiz_engine._dynamic_questions(payload.get("caption") or "", 3, quiz_engine._extract_facts(payload.get("caption") or ""))
        return {"questions": fb, "error": f"{type(e).__name__}: {e}"}


# ---------- Translate ----------
from typing import Literal
from pydantic import BaseModel
from fastapi.responses import JSONResponse

class TranslateIn(BaseModel):
    text: str
    lang: Literal["en", "zh", "ms", "ta"]  # include 'en' for quick revert

@app.post("/api/translate")
def api_translate(payload: TranslateIn):
    text = (payload.text or "").strip()
    if not text:
        return JSONResponse(content={"error": "empty_text"}, status_code=400)

    target_map = {"en": "en", "zh": "zh-CN", "ms": "ms", "ta": "ta"}
    target = target_map[payload.lang]

    # If user picked English, just return the original (avoid any network calls)
    if target == "en":
        return {"text": text, "lang": payload.lang}

    # Try multiple providers in order; fall back gracefully if a provider is unavailable.
    translated = None
    errors = []

    try:
        from deep_translator import GoogleTranslator as DTGoogle
        translated = (DTGoogle(source="auto", target=target).translate(text) or "").strip()
    except Exception as e:
        errors.append(f"google:{type(e).__name__}")

    if not translated:
        try:
            from deep_translator import MyMemoryTranslator
            translated = (MyMemoryTranslator(source="en", target=target).translate(text) or "").strip()
        except Exception as e:
            errors.append(f"mymemory:{type(e).__name__}")

    if not translated:
        # final fallback: return original text with a warning (so UI doesn’t look broken)
        return {"text": text, "lang": payload.lang, "warning": "translator_unavailable", "providers": errors}

    return {"text": translated, "lang": payload.lang}

# ---------- TTS ----------
class TTSIn(BaseModel):
    text: str
    voice: Optional[str] = None  # 'male' | 'female' | None
    rate: Optional[float] = None  # 0.5 .. 1.5


@app.post("/api/tts")
async def tts(payload: TTSIn):
    import pyttsx3
    text = (payload.text or "").strip()
    if not text:
        return JSONResponse(content={"error": "empty_text"}, status_code=400)

    tmp = Path(tempfile.gettempdir()) / f"tts_{uuid.uuid4().hex}.wav"
    try:
        engine = pyttsx3.init()
        try:
            if isinstance(payload.rate, (int, float)) and payload.rate > 0:
                base = engine.getProperty("rate") or 200
                rate_val = int(max(50, min(300, float(base) * float(payload.rate))))
                engine.setProperty("rate", rate_val)
        except Exception:
            pass

        try:
            vp = (payload.voice or "").strip().lower()
            voices = engine.getProperty("voices") or []
            chosen = None
            if vp in {"male", "female"} and voices:
                male_pat = re.compile(r"male|dan|fred|sam|david|george|barry|paul|mike|john", re.I)
                female_pat = re.compile(r"female|susan|sara|ava|samantha|victoria|zira|zoe|karen|tessa|anna|jess", re.I)
                for v in voices:
                    name = getattr(v, "name", "") or getattr(v, "id", "")
                    if vp == "male" and male_pat.search(name):
                        chosen = v; break
                    if vp == "female" and female_pat.search(name):
                        chosen = v; break
                if not chosen and voices:
                    chosen = voices[0]
                if chosen:
                    engine.setProperty("voice", getattr(chosen, "id", None) or getattr(chosen, "name", None))
        except Exception:
            pass

        engine.save_to_file(text, str(tmp))
        engine.runAndWait()
        return FileResponse(str(tmp), media_type="audio/wav", filename="speech.wav")
    except Exception as e:
        return JSONResponse(content={"error": f"tts_error: {e}"}, status_code=500)

# ---------- Dictionary ----------
try:
    import nltk
    from nltk.corpus import wordnet as wn
except Exception:
    wn = None  # graceful fallback if missing

@app.get("/api/dictionary")
def api_dictionary(word: str):
    """Return a short dictionary entry for the given word."""
    w = (word or "").strip().lower()
    if not w:
        return JSONResponse(content={"error": "empty_word"}, status_code=400)

    if wn is None:
        return {"definition": "", "synonyms": [], "examples": []}

    try:
        synsets = wn.synsets(w)
    except Exception:
        synsets = []

    definition = ""
    examples = []
    synonyms = set()

    for s in synsets:
        if not definition and s.definition():
            definition = s.definition()
        ex = s.examples()
        if ex:
            examples.extend(ex[:1])
        for l in s.lemmas():
            name = l.name().replace("_", " ")
            synonyms.add(name)

    synonyms.discard(w)
    return {
        "definition": definition,
        "synonyms": sorted(synonyms)[:8],
        "examples": examples[:3],
    }
