# server/main.py
from __future__ import annotations

import io
import os
import re
import json
import time
import uuid
import base64
import tempfile
from hashlib import sha1
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from PIL import Image

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


# ---------- OpenAI client (caption provider) ----------
from openai import OpenAI
_openai_client = OpenAI()

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


# ---------- Caption core (OpenAI Vision) ----------
def _detailed_caption_openai(pil: Image.Image, region=None, *, speed: str = "fast") -> dict:
    """
    Returns a ONE-SENTENCE caption (plus derived fields).

    Style:
    - Always start with: "The image shows"
    - Natural, child-friendly wording
    - No need to mention exact counts of objects unless really important
    """
    # Optional crop
    if region:
        try:
            x = max(0, int(region.get("x", 0)))
            y = max(0, int(region.get("y", 0)))
            w = max(1, int(region.get("w", 1)))
            h = max(1, int(region.get("h", 1)))
            pil = pil.crop((x, y, x + w, y + h))
        except Exception:
            pass

    # Encode image → base64 data URL
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    # New style instruction
    instruction = (
        "You are writing a simple caption for children.\n"
        "Describe this image in EXACTLY ONE clear sentence (under 30 words).\n"
        "The sentence MUST start with the words: 'The image shows'.\n"
        "Briefly mention the main objects and setting.\n"
        "Avoid giving exact numbers or counts unless it is very important.\n"
        "Do not hedge or invent details. Output only the sentence."
    )

    max_tokens = 60
    temperature = 0.4

    try:
        r = _openai_client.chat.completions.create(
            model=os.getenv("CAPTION_OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = (r.choices[0].message.content or "").strip()
    except Exception as e:
        if (os.getenv("CAPTION_DEBUG", "").lower() in {"1", "true", "yes", "on"}):
            import sys, traceback
            print(f"[caption] OpenAI error: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
        text = ""

    # Normalize → ONE sentence
    text = re.sub(r"(?i)^\s*description:\s*", "", text).strip()
    parts = re.split(r'(?<=[\.!?])\s+', text) if text else []
    one = (parts[0] if parts else "").strip()

    # Ensure it starts with "The image shows"
    if one:
        if not one.lower().startswith("the image shows"):
            # avoid double capitalisation weirdness
            one = one.lstrip()
            if one and one[0].isupper():
                # make rest sentence start lower-case to read naturally
                one = "The image shows " + one[0].lower() + one[1:]
            else:
                one = "The image shows " + one
    else:
        one = "The image shows a scene that could not be described clearly."

    # Ensure terminal punctuation
    if one and one[-1] not in ".!?":
        one += "."

    # Derive objects (still works if the model occasionally uses numbers)
    objects = _parse_objects_from_text(one)

    return {
        "caption": one,
        "sentences": [one],
        "paragraph": one,
        "labels": [],
        "objects": objects,
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
    # Nothing to warm for OpenAI caption path
    try:
        from .auth_DB import _seed_admin_user  # type: ignore
        _seed_admin_user()
    except Exception:
        pass


@app.get("/api/health")
async def health():
    eng = _ensure_story_engine()
    cap_ready = True  # OpenAI path
    return {
        "ok": bool(getattr(eng, "ready", False)) and cap_ready,
        "captioner": cap_ready,
        "storygen": bool(getattr(eng, "ready", False)),
        "device": "api",   # previously cuda/cpu
        "error": getattr(eng, "err", None),
    }


@app.get("/api/story_status")
def story_status():
    eng = _ensure_story_engine()
    return {
        "ready": bool(getattr(eng, "ready", False)),
        "mode": getattr(eng, "_mode", None),
        "model": getattr(eng, "model_name", ""),
        "device": "api",
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


# ---------- Caption (OpenAI-backed + cached) ----------
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
    key = sha1(b"v4|" + raw + b"|" + region_norm.encode() + b"|" + str(speed).encode() + b"|" + str(mode).encode()).hexdigest()
    hit = _cap_cache_get(key)
    if hit is not None:
        if (mode or "").lower() not in ("detailed", "description"):
            return {"caption": hit.get("caption", "")}
        return hit

    try:
        result = _detailed_caption_openai(pil, region=region_box, speed=(speed or "fast"))
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
    rate: Optional[float] = None  # speaking speed hint (handled on frontend)


@app.post("/api/tts")
async def tts(payload: TTSIn):
    """
    Text‑to‑speech using gTTS (Google Text‑to‑Speech).
    This avoids OS‑level engines like pyttsx3 so it works on Render.
    """
    from gtts import gTTS

    text = (payload.text or "").strip()
    if not text:
        return JSONResponse(content={"error": "empty_text"}, status_code=400)

    try:
        # Map simple male/female choices to slightly different
        # English variants via TLD. This is not a true gender
        # switch but gives users an audible difference.
        # Map simple male/female choices to different TLDs (accent proxy)
        vp = (payload.voice or "").strip().lower()
        if vp == "male":
            tld = "co.uk"   # UK English (used for the “Male” option)
        else:
            # Treat anything else (including "female" and "app voice")
            # as the same female/default voice.
            tld = "us"      # US English – shared by App voice and Female


        # Always generate at normal speed; the frontend controls
        # playbackRate (0.5x / 1.0x / 1.5x).
        tts_obj = gTTS(text=text, lang="en", tld=tld, slow=False)
        buf = io.BytesIO()
        tts_obj.write_to_fp(buf)
        buf.seek(0)
        # Frontend treats this as a generic audio blob; mp3 is fine.
        return StreamingResponse(buf, media_type="audio/mpeg")
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

    definition = ""
    examples: list[str] = []
    synonyms: set[str] = set()

    # 1) Try free online API (dictionaryapi.dev) – works on Render, no auth.
    try:
        import httpx

        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{w}"
        r = httpx.get(url, timeout=5.0)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                entry = data[0] or {}
                meanings = entry.get("meanings") or []
                for m in meanings:
                    defs = m.get("definitions") or []
                    for d in defs:
                        if not definition and d.get("definition"):
                            definition = str(d["definition"]).strip()
                        ex = d.get("example")
                        if ex:
                            examples.append(str(ex).strip())
                        for syn in d.get("synonyms") or []:
                            name = str(syn).replace("_", " ").strip()
                            if name:
                                synonyms.add(name)
            if definition or examples or synonyms:
                synonyms.discard(w)
                return {
                    "definition": definition,
                    "synonyms": sorted(synonyms)[:8],
                    "examples": examples[:3],
                }
    except Exception:
        # Network errors or unexpected formats fall through to the WordNet fallback.
        pass

    # 2) Fallback to local WordNet if available (primarily for offline dev).
    if wn is not None:
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
        if definition or examples or synonyms:
            return {
                "definition": definition,
                "synonyms": sorted(synonyms)[:8],
                "examples": examples[:3],
            }

    # 3) Final safe fallback: empty entry so UI still renders.
    return {"definition": "", "synonyms": [], "examples": []}

# Simple OpenAI ping (kept)
@app.get("/api/openai_ping")
def openai_ping():
    try:
        r = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":"Respond with OK"}],
            max_tokens=3,
        )
        return {"ok": True, "reply": r.choices[0].message.content}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
