from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, JSON, or_, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from datetime import datetime, timedelta
from copy import deepcopy
import secrets
import re
import json
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import os

# ---------------------------------------------------------------------------
# Database (SQLite local by default; Postgres on Render via DATABASE_URL)
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH = Path("server/data/app.db")
DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _normalize_db_url(raw: str) -> str:
    """
    - Accepts sqlite / postgres URLs from env
    - Normalizes postgres:// -> postgresql:// for SQLAlchemy
    - Appends sslmode=require for hosted Postgres if missing
    """
    if not raw:
        return f"sqlite:///{DEFAULT_DB_PATH}"

    url = raw.strip()

    # Normalize scheme
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # If it's sqlite, keep as-is
    if url.startswith("sqlite:///"):
        return url

    # Ensure sslmode=require for hosted Postgres
    if url.startswith("postgresql://"):
        parsed = urlparse(url)
        q = dict(parse_qsl(parsed.query or ""))
        if "sslmode" not in q:
            q["sslmode"] = "require"
        url = urlunparse(parsed._replace(query=urlencode(q)))
    return url

DATABASE_URL = _normalize_db_url(os.getenv("DATABASE_URL", "").strip())

# Create engine with appropriate args
if DATABASE_URL.startswith("sqlite:///"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        future=True,
    )
else:
    # Postgres (Render) or any other server DB
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        future=True,
    )

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# ---------------------------------------------------------------------------
# User Model
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)   # plaintext for simplicity
    # Use callables for defaults to avoid shared mutable dicts
    settings = Column(JSON, default=dict)
    achievements = Column(JSON, default=dict)

# ---------------------------------------------------------------------------
# Persistent Sessions (DB-backed)
# ---------------------------------------------------------------------------
class SessionRow(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, index=True)  # session_id token
    user_id = Column(Integer, index=True, nullable=False)
    expiry = Column(DateTime, nullable=False)

# Create all tables
Base.metadata.create_all(bind=engine)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI()

# CORS: allow credentials and configurable origins for dev (include LAN IPs if needed)
_origins_env = os.getenv("ALLOWED_ORIGINS", "")
_origin_regex = os.getenv("ALLOWED_ORIGIN_REGEX", r"https://.*\.vercel\.app$")
if _origins_env.strip():
    _origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
else:
    _origins = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",  # vite preview
        "http://localhost:4173",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session TTL in minutes (persisted in DB)
SESSION_TTL_MIN = 60 * 24  # 24h

# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def create_session(user_id: int, db: Session):
    sid = secrets.token_urlsafe(24)
    row = SessionRow(
        id=sid,
        user_id=user_id,
        expiry=datetime.utcnow() + timedelta(minutes=SESSION_TTL_MIN),
    )
    db.add(row)
    db.commit()
    return sid

def current_user_from_cookie(request: Request, db: Session):
    sid = request.cookies.get("session_id")
    if not sid:
        return None
    row = db.query(SessionRow).filter(SessionRow.id == sid).first()
    if row is None:
        return None
    if row.expiry < datetime.utcnow():
        try:
            db.delete(row)
            db.commit()
        except Exception:
            db.rollback()
        return None
    return db.get(User, row.user_id)

def _cookie_params_for(request: Request) -> dict:
    """Decide cookie security flags based on request scheme and env.

    - In production (HTTPS / proxies set x-forwarded-proto=https) -> SameSite=None; Secure
    - In local HTTP dev -> SameSite=Lax; not Secure
    You can force secure cookies by setting COOKIE_SECURE=true in the env.
    """
    xf_proto = (request.headers.get("x-forwarded-proto") or request.url.scheme or "").lower()
    force_secure = (os.getenv("COOKIE_SECURE", "").strip().lower() in {"1", "true", "yes", "on"})
    secure = force_secure or xf_proto == "https"
    samesite = "none" if secure else "lax"
    return dict(
        httponly=True,
        samesite=samesite,
        secure=secure,
        path="/",
        max_age=SESSION_TTL_MIN * 60,
    )

def gmail_like(email: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9._%+-]+@gmail\.com", email))

# Reasonable defaults per account
DEFAULT_SETTINGS = {
    "daily_objective_time": "17:00",
    "enable_notifications": True,
    "streak_reminders": True,
    "storage_path": "",

    # Accessibility
    "dyslexia_font": "Off",          # Off | OpenDyslexic | Lexend
    "reading_guide": True,           # word-by-word highlight toggle
    "high_contrast": False,          # site-wide
    "bw_mode": False,                # site-wide black & white

    # TTS
    "tts_voice": "female",           # male | female
    "speaking_rate": 1.0,            # 0.5 .. 1.5
    "word_highlight_enable": True,
    "word_highlight_color": "#FFD54F",

    # Drawing
    "grid_guides": False,
}

DEFAULT_ACHIEVEMENTS = {
    "streak_days": 0,
    "last_active_at": None,            # ISO date string in UTC, e.g. "2025-10-14"
    "points": 0,
    "counts": {"captions": 0, "quizzes": 0, "stories": 0},
    "badges": [],
}

def _ensure_achievements_shape(ach: dict | str | None) -> dict:
    """Return a normalized achievements dict ensuring all expected keys exist."""
    base = deepcopy(DEFAULT_ACHIEVEMENTS)
    data: dict = {}
    if isinstance(ach, str):
        try:
            data = json.loads(ach) or {}
        except Exception:
            data = {}
    elif isinstance(ach, dict):
        data = deepcopy(ach)
    # Shallow merge
    out = {**base, **(data or {})}
    # Ensure nested counts exists with all keys
    counts = out.get("counts") or {}
    out["counts"] = {
        "captions": int(counts.get("captions", 0) or 0),
        "quizzes": int(counts.get("quizzes", 0) or 0),
        "stories": int(counts.get("stories", 0) or 0),
    }
    # Ensure badges is a list of dicts
    badges = out.get("badges")
    out["badges"] = [b for b in (badges if isinstance(badges, list) else []) if isinstance(b, (dict, str))]
    # Coerce streak
    out["streak_days"] = int(out.get("streak_days", 0) or 0)
    # last_active_at left as string/None
    return out

# --- Settings normalization ---
def _ensure_settings_shape(s: dict | str | None) -> dict:
    """Return a normalized settings dict with proper types and defaults."""
    base = deepcopy(DEFAULT_SETTINGS)
    data: dict = {}
    if isinstance(s, str):
        try:
            data = json.loads(s) or {}
        except Exception:
            data = {}
    elif isinstance(s, dict):
        data = deepcopy(s)

    out = {**base, **(data or {})}

    def _b(v):
        return bool(v) if isinstance(v, (bool, int)) else str(v).lower() in {"1", "true", "on", "yes"}

    # Booleans
    for k in [
        "enable_notifications",
        "streak_reminders",
        "reading_guide",
        "high_contrast",
        "bw_mode",
        "word_highlight_enable",
        "grid_guides",
    ]:
        out[k] = _b(out.get(k, base.get(k, False)))

    # Speaking rate
    try:
        rate = float(out.get("speaking_rate", base["speaking_rate"]))
    except Exception:
        rate = base["speaking_rate"]
    rate = max(0.5, min(1.5, rate))
    out["speaking_rate"] = rate

    # Voice canonicalization: allow 'male' | 'female' | 'App voice'
    v = str(out.get("tts_voice", base["tts_voice"]))
    lv = v.strip().lower()
    if lv in {"male", "female"}:
        out["tts_voice"] = lv
    else:
        out["tts_voice"] = "App voice"

    # Strings
    out["daily_objective_time"] = str(out.get("daily_objective_time", base["daily_objective_time"]))
    out["storage_path"] = str(out.get("storage_path", base["storage_path"]))
    out["word_highlight_color"] = str(out.get("word_highlight_color", base["word_highlight_color"]))

    return out

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# Seed a default admin account at startup (for demo/verification on Vercel)
@app.on_event("startup")
def _seed_admin_user():
    try:
        db = SessionLocal()
        # If either username or email already exists, skip seeding
        existing = db.query(User).filter(or_(User.username == "admin", User.email == "admin@example.com")).first()
        if existing is None:
            demo_settings = _ensure_settings_shape(DEFAULT_SETTINGS)
            demo_ach = _ensure_achievements_shape({})
            user = User(
                username="admin",
                email="admin@example.com",
                password="admin1",
                settings=demo_settings,
                achievements=demo_ach,
            )
            db.add(user)
            db.commit()
    except Exception:
        # Best-effort seed; do not crash app if seeding fails
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        try:
            db.close()
        except Exception:
            pass

@app.post("/api/auth/register")
def register_user(payload: dict, request: Request, response: Response, db: Session = Depends(get_db)):
    username = payload.get("username", "").strip()
    email = payload.get("email", "").strip().lower()
    password = payload.get("password", "").strip()
    settings = payload.get("settings") or {}
    achievements = payload.get("achievements") or {}

    if not username or not email or not password:
        raise HTTPException(status_code=400, detail="Missing required fields")

    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already exists")

    # seed with defaults and normalize types if provided
    base_settings = _ensure_settings_shape({**DEFAULT_SETTINGS, **(settings or {})})
    # Normalize achievements into the full expected shape
    base_ach = _ensure_achievements_shape(achievements)

    user = User(
        username=username,
        email=email,
        password=password,
        settings=base_settings,
        achievements=base_ach,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    sid = create_session(user.id, db)
    # Cross-site compatible cookie (Vercel -> Render): SameSite=None; Secure on HTTPS
    response.set_cookie(
        key="session_id",
        value=sid,
        **_cookie_params_for(request),
    )

    return {"message": "Registration successful", "user": {"id": user.id, "username": user.username}}

@app.post("/api/auth/login")
def login_user(payload: dict, request: Request, response: Response, db: Session = Depends(get_db)):
    identifier = payload.get("username_or_email", "").strip().lower()
    password = payload.get("password", "").strip()

    if not identifier or not password:
        raise HTTPException(status_code=400, detail="Missing credentials")

    user = db.query(User).filter(or_(User.username == identifier, User.email == identifier)).first()
    if not user or user.password != password:
        raise HTTPException(status_code=401, detail="Invalid username/email or password")

    sid = create_session(user.id, db)
    response.set_cookie(
        key="session_id",
        value=sid,
        **_cookie_params_for(request),
    )
    return {"message": "Login successful", "user": {"id": user.id, "username": user.username}}

@app.post("/api/auth/logout")
def logout_user(response: Response, request: Request, db: Session = Depends(get_db)):
    sid = request.cookies.get("session_id")
    if sid:
        row = db.query(SessionRow).filter(SessionRow.id == sid).first()
        if row:
            try:
                db.delete(row)
                db.commit()
            except Exception:
                db.rollback()
    response.delete_cookie("session_id", path="/")
    return {"message": "Logged out"}

@app.get("/api/auth/me")
def get_me(request: Request, db: Session = Depends(get_db)):
    user = current_user_from_cookie(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Normalize and persist any missing keys on-the-fly so the client always
    # receives a complete structure (prevents UI from showing zeros forever).
    fixed_ach = _ensure_achievements_shape(user.achievements)
    fixed_settings = _ensure_settings_shape(user.settings)
    if user.achievements != fixed_ach or user.settings != fixed_settings:
        user.achievements = fixed_ach
        user.settings = fixed_settings
        db.add(user)
        db.commit()
        db.refresh(user)

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "settings": user.settings or DEFAULT_SETTINGS,
        "achievements": fixed_ach,
    }

# ---- Achievements & Settings (existing) ----
@app.get("/api/me/progress")
def me_progress(request: Request, db: Session = Depends(get_db)):
    user = current_user_from_cookie(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    ach = _ensure_achievements_shape(user.achievements)
    streak = int(ach.get("streak_days", 0) or 0)
    return {"streak": max(0, streak)}

@app.get("/api/me/achievements")
def me_achievements(request: Request, db: Session = Depends(get_db)):
    user = current_user_from_cookie(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    ach = _ensure_achievements_shape(user.achievements)
    badges = ach.get("badges")
    return badges if isinstance(badges, list) else []

@app.patch("/api/user/settings")
def update_settings(payload: dict, request: Request, db: Session = Depends(get_db)):
    user = current_user_from_cookie(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    incoming = payload.get("settings")
    if incoming is None:
        incoming = payload.get("preferences")
    if incoming is None and isinstance(payload, dict):
        incoming = payload
    if not isinstance(incoming, dict):
        raise HTTPException(status_code=400, detail="Invalid settings payload")

    base = _ensure_settings_shape(user.settings)
    # Shallow update then re-normalize types/values
    base.update(incoming)
    user.settings = _ensure_settings_shape(base)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"ok": True, "settings": user.settings}

# ---- Profile actions (new) ----
@app.post("/api/account/change_email")
def change_email(payload: dict, request: Request, db: Session = Depends(get_db)):
    """
    Change the email address for the current user.

    The client currently sends { email: ... } while this endpoint
    originally expected { new_email: ... }, so we accept both keys
    for robustness.
    """
    user = current_user_from_cookie(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    new_email = (
        payload.get("new_email")
        or payload.get("email")
        or ""
    )
    new_email = new_email.strip().lower()
    if not gmail_like(new_email):
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid Gmail address (e.g., name@gmail.com)",
        )

    # If the email is unchanged, treat as a no-op success
    if new_email == user.email:
        return {"ok": True, "email": user.email}

    # Enforce uniqueness across users
    existing = db.query(User).filter(User.email == new_email).first()
    if existing and existing.id != user.id:
        raise HTTPException(status_code=400, detail="Email already exists")

    user.email = new_email
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"ok": True, "email": user.email}


# ---- Profile actions (new) ----
@app.post("/api/account/change_display_name")
def change_display_name(payload: dict, request: Request, db: Session = Depends(get_db)):
    """
    Update the user's display name (username).

    The front-end treats display name as the login username, so we
    enforce non-empty, reasonably short names and uniqueness.
    """
    user = current_user_from_cookie(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    new_name = (payload.get("display_name") or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="Display name cannot be empty")
    if len(new_name) < 3:
        raise HTTPException(status_code=400, detail="Display name must be at least 3 characters")
    if len(new_name) > 50:
        raise HTTPException(status_code=400, detail="Display name is too long")

    # Ensure no other user already has this username
    existing = db.query(User).filter(User.username == new_name).first()
    if existing and existing.id != user.id:
        raise HTTPException(status_code=400, detail="That display name is already taken")

    user.username = new_name
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"ok": True, "username": user.username}


@app.post("/api/account/change_password")
def change_password(payload: dict, request: Request, db: Session = Depends(get_db)):
    user = current_user_from_cookie(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    current_pw = (payload.get("current_password") or "").strip()
    new_pw = (payload.get("new_password") or "").strip()
    if not current_pw or not new_pw:
        raise HTTPException(status_code=400, detail="Missing password fields")
    if user.password != current_pw:
        raise HTTPException(status_code=403, detail="Incorrect current password")
    if len(new_pw) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    user.password = new_pw
    db.add(user); db.commit(); db.refresh(user)
    return {"ok": True}

@app.post("/api/account/clear_data")
def clear_data(payload: dict, request: Request, db: Session = Depends(get_db)):
    """
    Resets user preferences & achievements only. Account (username/email/password) remains.
    Requires correct password.
    """
    user = current_user_from_cookie(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    pw = (payload.get("password") or "").strip()
    if user.password != pw:
        raise HTTPException(status_code=403, detail="Incorrect password")

    user.settings = deepcopy(DEFAULT_SETTINGS)
    user.achievements = deepcopy(DEFAULT_ACHIEVEMENTS)
    db.add(user); db.commit(); db.refresh(user)
    return {"ok": True}

@app.post("/api/account/delete")
def delete_account(payload: dict, request: Request, response: Response, db: Session = Depends(get_db)):
    """
    Deletes the entire account. Requires correct password.
    """
    user = current_user_from_cookie(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    pw = (payload.get("password") or "").strip()
    if user.password != pw:
        raise HTTPException(status_code=403, detail="Incorrect password")

    db.delete(user); db.commit()

    # Remove the current session row if present
    sid = request.cookies.get("session_id")
    if sid:
        row = db.query(SessionRow).filter(SessionRow.id == sid).first()
        if row:
            try:
                db.delete(row)
                db.commit()
            except Exception:
                db.rollback()
    response.delete_cookie("session_id", path="/")

    return {"ok": True}

# ---- Achievements: client events ----
@app.post("/api/achievements/event")
def ach_event(payload: dict, request: Request, db: Session = Depends(get_db)):
    """
    Record an achievement event (caption / quiz / story) and update the user's daily streak.
    """
    user = current_user_from_cookie(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    kind = (payload.get("type") or "").strip().lower()
    if kind not in {"caption", "quiz", "story"}:
        raise HTTPException(status_code=400, detail="Invalid event type")

    # Load and normalize existing achievements
    ach = _ensure_achievements_shape(user.achievements)
    ach_counts = dict(ach.get("counts") or {"captions": 0, "quizzes": 0, "stories": 0})
    badges = list(ach.get("badges") or [])
    streak = int(ach.get("streak_days", 0) or 0)

    now_iso = datetime.utcnow().isoformat() + "Z"

    def has_badge(bid: str) -> bool:
        return any(isinstance(b, dict) and b.get("id") == bid for b in badges)

    # === EVENT COUNTS & FIRST-TIME BADGES ===
    if kind == "caption":
        ach_counts["captions"] += 1
        if not has_badge("first_caption"):
            badges.append({
                "id": "first_caption",
                "title": "First Caption",
                "description": "Generate your first caption",
                "unlocked_at": now_iso
            })
    elif kind == "quiz":
        ach_counts["quizzes"] += 1
        if not has_badge("quiz_whiz"):
            badges.append({
                "id": "quiz_whiz",
                "title": "Quiz Whiz",
                "description": "Complete your first quiz",
                "unlocked_at": now_iso
            })
    elif kind == "story":
        ach_counts["stories"] += 1
        if not has_badge("storyteller"):
            badges.append({
                "id": "storyteller",
                "title": "Storyteller",
                "description": "Create your first story",
                "unlocked_at": now_iso
            })

    # === DAILY STREAK TRACKING ===
    today = datetime.utcnow().date()
    last_raw = ach.get("last_active_at")
    last_date = None
    if isinstance(last_raw, str):
        try:
            last_date = datetime.fromisoformat(last_raw.replace("Z", "")).date()
        except Exception:
            pass

    if last_date == today:
        pass  # already counted today
    elif last_date == (today - timedelta(days=1)):
        streak += 1
    else:
        streak = 1

    ach["last_active_at"] = today.isoformat()

    # === STREAK BADGES ===
    def add_badge(bid, title, desc):
        if not has_badge(bid):
            badges.append({
                "id": bid,
                "title": title,
                "description": desc,
                "unlocked_at": now_iso
            })

    if streak >= 7:
        add_badge("streak_7", "7-Day Streak", "Use the app 7 days in a row")
    if streak >= 30:
        add_badge("streak_30", "30-Day Streak", "Use the app 30 days in a row")

    # === SAVE BACK ===
    ach["streak_days"] = streak
    ach["counts"] = ach_counts
    ach["badges"] = badges
    user.achievements = ach

    db.add(user)
    db.commit()
    db.refresh(user)

    return {"ok": True, "achievements": user.achievements}

# --- Stories: save / list / get ---
class StoryIn(BaseModel):
    title: str
    panels: List[str]              # exactly 3 lines of text
    images: List[str]              # exactly 3 image URLs as returned by /api/story
    story: Optional[str] = None    # optional full story text

def _get_user_stories(user: User) -> list:
    settings = user.settings or {}
    stories = settings.get("stories")
    if not isinstance(stories, list):
        stories = []
    return stories

@app.post("/api/stories/save")
def save_story(payload: StoryIn, request: Request, db: Session = Depends(get_db)):
    user = current_user_from_cookie(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    data = payload.dict()
    # Basic validation/trim
    title = (data["title"] or "").strip()[:140] or "Untitled Story"
    panels = [str(x or "").strip() for x in (data.get("panels") or [])][:3]
    images = [str(x or "").strip() for x in (data.get("images") or [])][:3]
    if len(panels) != 3 or len(images) != 3:
        raise HTTPException(status_code=400, detail="Need 3 panels and 3 images")

    stories = _get_user_stories(user)
    story_id = secrets.token_urlsafe(10)
    stories.append({
        "id": story_id,
        "title": title,
        "panels": panels,
        "images": images,
        "story": (data.get("story") or "\n".join(panels)).strip(),
        "saved_at": datetime.utcnow().isoformat() + "Z",
    })
    # persist back under settings
    s = user.settings or {}
    s["stories"] = stories
    user.settings = s
    db.add(user); db.commit(); db.refresh(user)
    return {"ok": True, "id": story_id}

@app.get("/api/stories/list")
def list_stories(request: Request, db: Session = Depends(get_db)):
    user = current_user_from_cookie(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    stories = _get_user_stories(user)
    # Most recent first (if saved_at present)
    try:
        stories = sorted(stories, key=lambda s: s.get("saved_at",""), reverse=True)
    except Exception:
        pass
    # return just a tiny index for the Draw page
    return [{"id": s.get("id"), "title": s.get("title","Untitled Story")} for s in stories]

@app.get("/api/stories/get")
def get_story(id: str, request: Request, db: Session = Depends(get_db)):
    user = current_user_from_cookie(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    for s in _get_user_stories(user):
        if s.get("id") == id:
            return {"ok": True, "story": s}
    raise HTTPException(status_code=404, detail="Story not found")

@app.post("/api/stories/delete")
def delete_story(payload: dict, request: Request, db: Session = Depends(get_db)):
    """Delete a saved story by id for the current user."""
    user = current_user_from_cookie(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    sid = (payload.get("id") or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="Missing story id")

    stories = _get_user_stories(user)
    new_list = [s for s in stories if s.get("id") != sid]
    if len(new_list) == len(stories):
        raise HTTPException(status_code=404, detail="Story not found")

    # persist
    s = user.settings or {}
    s["stories"] = new_list
    user.settings = s
    db.add(user); db.commit(); db.refresh(user)
    return {"ok": True}
