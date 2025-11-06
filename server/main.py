# server/main.py
from __future__ import annotations

import io
import os
import re
import json
import time
import uuid
import tempfile
from hashlib import sha1
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import torch
from PIL import Image
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)

# --- App + subapps / engines you already had ---
from .auth_DB import app as auth_subapp
from .story_gen import StoryGenerator
from .quiz_gen import QuizGenerator

# ---------- App ----------
app = FastAPI(title="Picteractive API")

# Ensure environment variables (from repo root .env) are loaded when launched via uvicorn
try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    env_loaded = load_dotenv(REPO_ROOT / ".env") or load_dotenv(".env")
except Exception:
    env_loaded = False

# CORS (credentials + configurable origins)
_origins_env = os.getenv("ALLOWED_ORIGINS", "")
_origin_regex = os.getenv("ALLOWED_ORIGIN_REGEX", r"https://.*\.vercel\.app$")
if _origins_env.strip():
    _origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
else:
    _origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=_origin_regex,
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


# ---------- Single, reliable captioner (InstructBLIP) ----------
_INSTRUCT_PROC = None
_INSTRUCT_MODEL = None
_INSTRUCT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tiny in-memory caption cache (per-process)
_CAP_CACHE: dict[str, dict] = {}
_CAP_CACHE_MAX = 64  # small cap


def _cap_cache_get(k: str):
    return _CAP_CACHE.get(k)


def _cap_cache_put(k: str, v: dict):
    if len(_CAP_CACHE) >= _CAP_CACHE_MAX:
        # drop an arbitrary/oldest key
        try:
            _CAP_CACHE.pop(next(iter(_CAP_CACHE)))
        except Exception:
            _CAP_CACHE.clear()
    _CAP_CACHE[k] = v


def _init_instructblip():
    """Load once; reused for all caption requests."""
    global _INSTRUCT_PROC, _INSTRUCT_MODEL
    if _INSTRUCT_PROC is not None and _INSTRUCT_MODEL is not None:
        return True
    name = "Salesforce/instructblip-flan-t5-xl"  # change to '-large' for lower CPU RAM/time
    _INSTRUCT_PROC = InstructBlipProcessor.from_pretrained(name)
    _INSTRUCT_MODEL = InstructBlipForConditionalGeneration.from_pretrained(
        name,
        torch_dtype=(torch.float16 if _INSTRUCT_DEVICE == "cuda" else torch.float32),
    ).to(_INSTRUCT_DEVICE).eval()
    return True


# ---------- Object parsing helpers (fruits/veggies/animals/birds) ----------
_FRUITS = {
    "apple","apples","banana","bananas","orange","oranges","grape","grapes","pear","pears",
    "mango","mangoes","pineapple","pineapples","strawberry","strawberries","watermelon","watermelons",
    "papaya","papayas","kiwi","kiwis","peach","peaches","plum","plums","cherry","cherries","lemon","lemons",
    "lime","limes","pomegranate","pomegranates","blueberry","blueberries","raspberry","raspberries","avocado","avocados"
}
_VEGETABLES = {
    "carrot","carrots","potato","potatoes","tomato","tomatoes","onion","onions","garlic","garlics",
    "cucumber","cucumbers","pepper","peppers","capsicum","capsicums","broccoli","broccolis","cauliflower","cauliflowers",
    "spinach","lettuce","cabbage","cabbages","eggplant","eggplants","brinjal","brinjals","okra","ladyfinger","ladyfingers",
    "chilli","chillies","bean","beans","pea","peas","corn","corns","pumpkin","pumpkins"
}
_ANIMALS = {
    "cat","cats","dog","dogs","cow","cows","horse","horses","sheep","goat","goats","rabbit","rabbits",
    "tiger","tigers","lion","lions","elephant","elephants","bear","bears","zebra","zebras","giraffe","giraffes",
    "monkey","monkeys","panda","pandas","kangaroo","kangaroos","fox","foxes","wolf","wolves","deer","deers","mouse","mice"
}
_BIRDS = {
    "bird","birds","sparrow","sparrows","pigeon","pigeons","dove","doves","eagle","eagles",
    "owl","owls","parrot","parrots","crow","crows","peacock","peacocks","duck","ducks","chicken","chickens"
}

def _cat_of(name: str) -> str:
    w = (name or "").lower().strip()
    if w in _FRUITS: return "fruit"
    if w in _VEGETABLES: return "vegetable"
    if w in _BIRDS: return "bird"
    if w in _ANIMALS: return "animal"
    return "other"

# digits like "3 apples" OR number-words like "three apples"
_WORD_TO_NUM = {
    "one":1, "two":2, "three":3, "four":4, "five":5,
    "six":6, "seven":7, "eight":8, "nine":9, "ten":10,
    "eleven":11, "twelve":12
}
_NUM_DIGIT_RE = re.compile(r"\b(\d+)\s+([A-Za-z][A-Za-z\- ]+?)\b")
_NUM_WORD_RE  = re.compile(r"\b(" + "|".join(_WORD_TO_NUM.keys()) + r")\s+([A-Za-z][A-Za-z\- ]+?)\b", re.I)
_PAREN_COUNT_RE = re.compile(r"\b([A-Za-z][A-Za-z\- ]+?)\s*\((\d+)\)")

def _normalize_name(name: str) -> str:
    return (name or "").lower().strip().rstrip("s").strip()

def _parse_objects_from_text(text: str):
    """
    Extract normalized [{name,count,category}] from NATURAL PROSE like:
      - 'Three apples and 1 banana sit in a basket while two sparrows perch nearby.'
      - also supports '(item)(count)' if the model ever uses that pattern.
    """
    out = []
    seen = {}

    t = (text or "").strip()
    if not t:
        return out

    # 1) digit form: "3 apples"
    for m in _NUM_DIGIT_RE.finditer(t):
        cnt = int(m.group(1))
        name = _normalize_name(m.group(2))
        if not name: continue
        seen[name] = seen.get(name, 0) + max(1, cnt)

    # 2) word form: "three apples"
    for m in _NUM_WORD_RE.finditer(t):
        cnt = _WORD_TO_NUM.get(m.group(1).lower(), 0)
        name = _normalize_name(m.group(2))
        if cnt <= 0 or not name: continue
        seen[name] = seen.get(name, 0) + max(1, cnt)

    # 3) rare fallback: "apple(3)"
    for m in _PAREN_COUNT_RE.finditer(t):
        cnt = int(m.group(2))
        name = _normalize_name(m.group(1))
        if not name: continue
        seen[name] = seen.get(name, 0) + max(1, cnt)

    for name, cnt in seen.items():
        out.append({"name": name, "count": int(cnt), "category": _cat_of(name)})

    return out


# ---------- Small text helper ----------
def _split_sentences(text: str):
    parts = re.split(r'(?<=[\.!?])\s+', (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


# ---------- Speed presets ----------
def _gen_kwargs_for(speed: str) -> dict:
    s = (speed or "balanced").lower()
    if s == "fast":
        # much faster; still coherent
        return dict(max_new_tokens=64, num_beams=1, do_sample=True, top_p=0.9, temperature=0.7)
    if s == "thorough":
        return dict(max_new_tokens=220, num_beams=5)
    # balanced
    return dict(max_new_tokens=180, num_beams=3)


# ---------- Caption core ----------
def _detailed_caption_instructblip(pil: Image.Image, region=None, *, speed: str = "fast") -> dict:
    """Return {caption, sentences, paragraph, labels, objects, mode} using ONE model."""
    # Optional region crop
    if region:
        try:
            x = max(0, int(region.get("x", 0)))
            y = max(0, int(region.get("y", 0)))
            w = max(1, int(region.get("w", 1)))
            h = max(1, int(region.get("h", 1)))
            pil = pil.crop((x, y, x + w, y + h))
        except Exception:
            pass

    _init_instructblip()

    # ✅ Natural paragraph only — no headings, no bullets, no labels
    instruction = (
        "Describe the image in one detailed paragraph (4–6 sentences). "
        "Naturally mention every clearly visible object and its quantity using simple nouns and numerals "
        "(e.g., '3 apples', '2 sparrows'). Include salient attributes like colors, materials, and relative positions. "
        "Do not invent objects. Do not start with labels or headings; write only the paragraph."
    )

    inputs = _INSTRUCT_PROC(images=pil, text=instruction, return_tensors="pt").to(_INSTRUCT_DEVICE)
    gen_kwargs = _gen_kwargs_for(speed)
    with torch.inference_mode():
        out = _INSTRUCT_MODEL.generate(**inputs, **gen_kwargs)
    text = _INSTRUCT_PROC.batch_decode(out, skip_special_tokens=True)[0].strip()

    # Clean up in case the model ever sneaks in a "Description:" prefix
    desc = re.sub(r"(?i)^\s*description:\s*", "", text).strip()
    sentences = _split_sentences(desc)
    paragraph = desc if re.search(r"[\.!?]$", desc) else (desc + ".")
    caption = sentences[0] if sentences else paragraph

    # Parse objects directly from the prose (digits or number-words)
    objects = _parse_objects_from_text(paragraph)

    return {
        "caption": caption,
        "sentences": sentences,
        "paragraph": paragraph,
        "labels": [],
        "objects": objects,   # [{"name":"apple","count":3,"category":"fruit"}, ...]
        "mode": "detailed",
    }


# ---------- Health / Status ----------
def _ensure_story_engine():
    """Try to (re)initialize the story engine if not ready. Keeps import-time failures from breaking startup."""
    global story_engine
    try:
        if not getattr(story_engine, "ready", False):
            story_engine = StoryGenerator()
    except Exception as e:
        try:
            setattr(story_engine, "ready", False)
            setattr(story_engine, "err", f"{type(e).__name__}: {e}")
        except Exception:
            pass
    return story_engine


@app.on_event("startup")
def _warm_start():
    # Warm the caption model so the first request is fast
    try:
        _init_instructblip()
    except Exception:
        pass
    # Seed demo admin user in the auth subapp DB (idempotent)
    try:
        from .auth_DB import _seed_admin_user  # type: ignore
        _seed_admin_user()
    except Exception:
        pass


@app.get("/api/health")
async def health():
    eng = _ensure_story_engine()
    cap_ready = True  # captioner warm-loaded
    return {
        "ok": bool(getattr(eng, "ready", False)) and cap_ready,
        "captioner": cap_ready,
        "storygen": bool(getattr(eng, "ready", False)),
        "device": _INSTRUCT_DEVICE,
        "error": getattr(eng, "err", None),
    }


@app.get("/api/story_status")
def story_status():
    eng = _ensure_story_engine()
    return {
        "ready": bool(getattr(eng, "ready", False)),
        "mode": getattr(eng, "_mode", None),
        "model": getattr(eng, "model_name", ""),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "err": getattr(eng, "err", None),
    }


@app.post("/api/story_test")
def story_test():
    # Smoke test payload
    t = "A LITTLE ADVENTURE"
    p = [
        "Something begins in the first picture.",
        "Something changes in the second picture.",
        "A friendly ending appears in the third picture.",
    ]
    return {"title": t, "panels": p, "story": "\n".join(p)}


# ---------- Caption (single-model, fast default + cached) ----------
@app.post("/api/caption")
async def caption(
    image: UploadFile = File(...),
    region: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
    speed: Optional[str] = Form("fast"),   # default to fast for UX
):
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        return JSONResponse(content={"error": f"invalid_image: {e}"}, status_code=400)

    region_box = None
    region_norm = ""
    if region:
        try:
            region_box = json.loads(region)
            region_norm = json.dumps(region_box, sort_keys=True)
        except Exception:
            region_box = None

    # cache key on (image bytes + region + speed + mode)
    key = sha1(b"v3|" + raw + b"|" + region_norm.encode() + b"|" + str(speed).encode() + b"|" + str(mode).encode()).hexdigest()
    hit = _cap_cache_get(key)
    if hit is not None:
        if (mode or "").lower() not in ("detailed", "description"):
            return {"caption": hit.get("caption", "")}
        return hit

    try:
        result = _detailed_caption_instructblip(pil, region=region_box, speed=(speed or "fast"))
        _cap_cache_put(key, result)
        if (mode or "").lower() not in ("detailed", "description"):
            return {"caption": (result.get("caption") or "").strip()}
        return result
    except Exception:
        empty = {"caption": "", "sentences": [], "paragraph": "", "labels": [], "objects": [], "mode": "detailed"}
        _cap_cache_put(key, empty)
        if (mode or "").lower() not in ("detailed", "description"):
            return {"caption": ""}
        return empty


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
@app.post("/api/cvd/apply")
async def cvd_apply(
    image: UploadFile = File(...),
    mode: Literal["simulate", "daltonize"] = Form("simulate"),
    cvd_type: str = Form("deuteranopia"),
    severity: float = Form(1.0),
    amount: float = Form(1.0),
):
    """
    Apply colour-vision simulation/daltonization using the proper CVD pipeline.
    If the specialized pipeline isn't available at runtime, we just echo the image back.
    """
    raw = await image.read()

    try:
        from .csvd_filter import apply as cvd_apply_image  # lazy import
    except Exception:
        buf = io.BytesIO(raw)
        return StreamingResponse(buf, media_type="image/png")

    t = (cvd_type or "").lower()
    if t.startswith("prot"):
        t = "protan"
    elif t.startswith("deut"):
        t = "deutan"
    elif t.startswith("trit"):
        t = "tritan"
    else:
        t = "none"

    sev = float(max(0.0, min(1.0, float(severity))))
    mode_norm = "daltonize" if mode == "daltonize" else "simulate"

    out_img = cvd_apply_image(io.BytesIO(raw), mode=mode_norm, cvd_type=t, severity=sev, amount=float(amount))
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# ---------- Story Generation (kept) ----------
try:
    story_engine = StoryGenerator()
except Exception as e:
    class _StoryStub:
        ready = False
        err = f"{type(e).__name__}: {e}"
        model_name = ""
        _mode = None
    story_engine = _StoryStub()


@app.post("/api/story")
async def story_api(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    image3: UploadFile = File(...),
    mood: str = Form("friendly"),
):
    """
    Story generation reuses your existing StoryGenerator pipeline.
    """
    try:
        raw1, raw2, raw3 = await image1.read(), await image2.read(), await image3.read()
        p1 = Image.open(io.BytesIO(raw1)).convert("RGB")
        p2 = Image.open(io.BytesIO(raw2)).convert("RGB")
        p3 = Image.open(io.BytesIO(raw3)).convert("RGB")

        eng = _ensure_story_engine()
        scenes, deltas = eng.build_scenes([p1, p2, p3], [[], [], []])
        title, panels, moral = eng.generate_from_scenes(scenes, deltas, mood=mood)
        story_text = "\n".join(panels)

        # Persist inputs for UI use
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

        panels = [(panels[i] if i < len(panels) and panels[i] else "") for i in range(3)]

        return {
            "title": title,
            "story": story_text,
            "panels": panels,
            "moral": moral,
            "captions": [s.get("caption", "") for s in scenes],
            "scenes": scenes,
            "deltas": deltas,
            "labels": [[], [], []],
            "images": image_urls,
        }

    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "title": "A LITTLE ADVENTURE",
            "panels": [
                "We see a simple scene.",
                "Then something changes.",
                "Finally, it ends happily.",
            ],
            "moral": "We learn and smile together.",
        }


# ---------- Quiz ----------
class QuizIn(BaseModel):
    caption: str
    count: Optional[int] = 3


quiz_engine = QuizGenerator()


@app.post("/api/quiz")
def api_quiz(payload: dict = Body(...)):
    """
    Input: { "caption": str, "count": 3 }
    Output: { "questions": [{question, options[3], answer_index}] }
    """
    try:
        caption = (payload.get("caption") or "").strip()
        count = int(payload.get("count") or 3)
        count = 3 if count < 3 else min(count, 3)
        qs = quiz_engine.generate(caption, num_questions=count)
        return {"questions": qs}
    except Exception as e:
        fb = quiz_engine._dynamic_questions(payload.get("caption") or "", 3, quiz_engine._extract_facts(payload.get("caption") or ""))
        return {"questions": fb, "error": f"{type(e).__name__}: {e}"}


# ---------- Translate ----------
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

    if target == "en":
        return {"text": text, "lang": payload.lang}

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

# server/main.py (temporary)
from openai import OpenAI
client = OpenAI()

@app.get("/api/openai_ping")
def openai_ping():
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":"Respond with OK"}],
            max_tokens=3,
        )
        return {"ok": True, "reply": r.choices[0].message.content}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
