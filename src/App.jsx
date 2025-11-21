import React,{createContext,useContext,useEffect,useMemo,useRef,useState} from "react";
import {BrowserRouter,Routes,Route,Link,Navigate,useLocation,useNavigate} from "react-router-dom";
import { fabric } from "fabric";
import logo from "./assets/picteractive_logo.png";
import profileLogo from "./assets/profile_logo.png";
import imgIcon from "./assets/img-file.png";
import drawIcon from "./assets/draw_file.png";
import textIcon from "./assets/text_file.png";
import arrowIcon from "./assets/arrow.png";
import cropIcon from "./assets/crop.png";
import speakIcon from "./assets/speaker.png";
import trashIcon from "./assets/trash.png";
import uploadIcon from "./assets/upload.png";
import cameraIcon from "./assets/cam.png";
import sparklesIcon from "./assets/ai_stars.png";
import translateIcon from './assets/translate.png';
import backIcon from './assets/back.png';
import editIcon from './assets/edit.png';
import saveIcon from './assets/save.png';
import siteLogo from "./assets/picteractive_logo.png";
import accessIcon from "./assets/access.png";
import profileIcon from "./assets/profile_logo.png";
import scene1 from "./assets/scene1.jpg";
import scene2 from "./assets/scene2.jpg";
import scene3 from "./assets/scene3.jpg";

// Removed save button usage on Story page


// Resolve API base ensuring same hostname as the frontend for cookie same-site
const API = (() => {
  const env = (import.meta?.env?.VITE_API_BASE || "").trim();
  try {
    if (env) {
      const u = new URL(env);
      const h = window.location.hostname;
      // If dev uses 127.0.0.1 vs localhost, align to page host for same-site cookies
      if ((u.hostname === "127.0.0.1" && h === "localhost") || (u.hostname === "localhost" && h === "127.0.0.1")) {
        u.hostname = h;
        return u.toString().replace(/\/$/, "");
      }
      return env.replace(/\/$/, "");
    }
  } catch {}
  const proto = window.location.protocol === "https:" ? "https" : "http";
  const host = window.location.hostname || "localhost";
  return `${proto}://${host}:8000`;
})();
const cx = (...xs) => xs.filter(Boolean).join(" ");

function useToast(){
  const [msg, setMsg] = useState("");
  useEffect(()=>{ if(!msg) return; const id=setTimeout(()=>setMsg(""),1500); return ()=>clearTimeout(id); },[msg]);
  return { show:setMsg, Toast:()=> msg ? <div className="toast">{msg}</div> : null };
}

// --- Accessibility application helpers ---
// Accept both flat and nested settings shapes and coerce into a single flat map
function toFlatSettings(raw = {}){
  if (!raw || typeof raw !== 'object') return {};
  // If already flat, prefer those keys
  const flat = {
    dyslexia_font: raw.dyslexia_font,
    reading_guide: raw.reading_guide,
    high_contrast: raw.high_contrast,
    bw_mode: raw.bw_mode,
    tts_voice: raw.tts_voice,
    speaking_rate: raw.speaking_rate,
    word_highlight_enable: raw.word_highlight_enable,
    word_highlight_color: raw.word_highlight_color,
    grid_guides: raw.grid_guides,
  };
  // Also merge from any nested shape (e.g., settings.accessibility.*)
  const acc = raw.accessibility || {};
  const tts = acc.tts || {};
  const canvas = acc.canvas || {};
  // Only overwrite if value is defined
  if (acc.dyslexiaFont != null) flat.dyslexia_font = acc.dyslexiaFont;
  if (acc.readingGuide != null) flat.reading_guide = !!acc.readingGuide;
  if (acc.highContrast != null) flat.high_contrast = !!acc.highContrast;
  if (canvas.gridGuides != null) flat.grid_guides = !!canvas.gridGuides;
  if (tts.voice != null) flat.tts_voice = tts.voice;
  if (tts.rate != null) flat.speaking_rate = tts.rate;
  if (tts.wordColor != null) flat.word_highlight_color = tts.wordColor;
  // Keep sensible defaults if still missing
  // Coerce rate to number even if stored as a string
  let rateVal = flat.speaking_rate;
  if (typeof rateVal !== 'number') {
    const n = Number(rateVal);
    rateVal = Number.isFinite(n) && n > 0 ? n : undefined;
  }
  if (rateVal == null) rateVal = 1.0;
  return {
    dyslexia_font: flat.dyslexia_font ?? 'Off',
    reading_guide: flat.reading_guide ?? true,
    high_contrast: flat.high_contrast ?? false,
    bw_mode: flat.bw_mode ?? false,
    tts_voice: flat.tts_voice ?? undefined,
    speaking_rate: rateVal,
    word_highlight_enable: flat.word_highlight_enable ?? true,
    word_highlight_color: flat.word_highlight_color ?? '#FFD54F',
    grid_guides: flat.grid_guides ?? false,
  };
}

function applyAccessibility(settings = {}) {
  const s = toFlatSettings(settings);
  const root = document.documentElement;

  const {
    dyslexia_font,
    high_contrast,
    bw_mode,
    word_highlight_enable,
    word_highlight_color,
    tts_voice,
    speaking_rate,
    grid_guides,
  } = s;

  // Update DOM attributes / CSS vars
  root.setAttribute("data-font", String(dyslexia_font));
  root.setAttribute("data-contrast", high_contrast ? "1" : "0");
  root.setAttribute("data-bw", bw_mode ? "1" : "0");
  // Expose reading guide state separately (if UI wants to use it)
  // Keep data-reading-guide for legacy consumers; defaults to "1" if unset
  const rg = (s.reading_guide ?? true) ? "1" : "0";
  root.setAttribute("data-reading-guide", rg);
  // TTS word highlight follows the dedicated toggle
  root.setAttribute("data-tts-highlight", word_highlight_enable ? "1" : "0");
  root.style.setProperty("--word-hilite", word_highlight_color || "#FFD54F");

  // Merge to global without overwriting with undefined
  const prev = (window.__picteractive_settings || {});
  const merged = { ...prev };
  Object.entries(s).forEach(([k, v]) => { if (v !== undefined) merged[k] = v; });

  // Normalize TTS rate
  if (typeof merged.speaking_rate !== 'number' || isNaN(merged.speaking_rate)) {
    merged.speaking_rate = 1.0;
  }

  window.__picteractive_settings = merged;

  try {
    window.dispatchEvent(new CustomEvent('picteractive:settings-applied', {
      detail: { settings: window.__picteractive_settings }
    }));
  } catch {}
}


// ======= AUTH CONTEXT =======
const AuthCtx = createContext(null);
export const useAuth = () => useContext(AuthCtx);

function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(false);

  // Seed accessibility BEFORE we show children
  useEffect(() => {
    (async () => {
      try {
        const me = await fetch(`${API}/api/auth/me`, { credentials: "include" })
          .then(r => (r.ok ? r.json() : null));
        if (me?.settings) applyAccessibility(me.settings);
        setUser(me);
      } finally {
        setReady(true);
      }
    })();
  }, []);

  const value = useMemo(
    () => ({
      user,
      refetchMe: () =>
        fetch(`${API}/api/auth/me`, { credentials: "include" })
          .then(r => (r.ok ? r.json() : null))
          .then(setUser),
      logout: async () => {
        await fetch(`${API}/api/auth/logout`, { method: "POST", credentials: "include" });
        setUser(null);
      },
    }),
    [user]
  );

  if (!ready) return null; // or a tiny loader/spinner
  return <AuthCtx.Provider value={value}>{children}</AuthCtx.Provider>;
}


// ======= PROTECTED ROUTE =======
function ProtectedRoute({ children }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  return children;
}

// ======= NAVBAR (Auth variant) =======
function AuthNavBar() {
  const { user, logout } = useAuth();
  return (
    <header className="site-nav">
      <div className="site-nav-inner">
        <Link to="/" className="site-brand"><img src={logo} alt="Picteractive" /></Link>
        <nav className="site-menu">
          <Link className="site-menu-link" to="/draw">DRAW</Link>
          <Link className="site-menu-link" to="/story">STORY</Link>
          <Link className="site-menu-link" to="/quiz">QUIZ</Link>
          <Link className="site-menu-link" to="/instructions">INSTRUCTIONS</Link>
          <Link className="site-menu-link" to="/settings">SETTINGS</Link>
        </nav>
        <div className="site-nav-spacer" />
        <div className="profile">
          {user ? (
            <div className="profile-row">
              <img src={profileLogo} className="profile-avatar" alt="profile" />
              <span className="profile-name">{user.username}</span>
              <button className="auth-mini-btn" onClick={logout}>Logout</button>
            </div>
          ) : (
            <div className="profile-row">
              <Link className="auth-mini-btn" to="/login">Login</Link>
              <Link className="auth-mini-btn" to="/register">Sign up</Link>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

// ======= AUTH INPUT =======
function CardField({ label, type="text", value, onChange, name, placeholder, onBlur, error, autoComplete, minLength, maxLength, required }) {
  return (
    <label className="auth-label">
      <span>{label}</span>
      <input
        className="auth-input"
        name={name}
        type={type}
        value={value}
        placeholder={placeholder}
        onChange={(e)=>onChange(e.target.value)}
        onBlur={onBlur}
        autoComplete={autoComplete}
        minLength={minLength}
        maxLength={maxLength}
        required={required}
        aria-invalid={Boolean(error) || undefined}
        aria-describedby={error ? `${name}-error` : undefined}
      />
      {error ? <div id={`${name}-error`} className="auth-error" role="alert" style={{marginTop:6}}>{error}</div> : null}
    </label>
  );
}

// ======= LOGIN =======
function LoginPage() {
  const [username,setUsername] = useState("");
  const [password,setPassword] = useState("");
  const [err,setErr] = useState("");
  const [touched, setTouched] = useState({ user:false, pass:false });
  const navigate = useNavigate();
  const { user, refetchMe } = useAuth();

  useEffect(()=>{ if(user){ navigate('/whats-this', { replace:true }); } }, [user]);

  function isEmail(x){ return /.+@.+\..+/.test(x); }
  function validate(){
    const u = username.trim();
    const p = password;
    let ue = "";
    if(!u) ue = "Username or email is required";
    else if(u.includes('@') && !isEmail(u)) ue = "Please enter a valid email";
    else if(!u.includes('@') && u.length < 3) ue = "Username must be at least 3 characters";
    let pe = "";
    if(!p) pe = "Password is required"; else if(p.length < 6) pe = "Password must be at least 6 characters";
    return { ue, pe, ok: !(ue||pe) };
  }

async function handleSubmit(e){
  e.preventDefault();
  const v = validate();
  if(!v.ok) return;

  try{
    const res = await fetch(`${API}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",           // <- send/receive cookie
      body: JSON.stringify({
        username_or_email: username.trim(),
        password
      })
    });
    if(!res.ok) throw new Error("Invalid credentials");
    await refetchMe();                  // <- populate user from cookie session
    navigate("/whats-this", { replace: true });
  }catch(e){
    setErr(e.message || "Login failed");
  }
}
  const v = validate();

  return (
    <div className="min-h-screen" style={{ background:"var(--forest)" }}>
      <NavBar />
      <div className="auth-wrap">
        <form className="auth-card" onSubmit={handleSubmit}>
          <div className="auth-title">WELCOME BACK!</div>
          <CardField label="USERNAME:" name="username" value={username} onChange={setUsername} onBlur={()=>setTouched(s=>({...s,user:true}))} error={touched.user ? v.ue : ""} autoComplete="username" required minLength={3} />
          <CardField label="PASSWORD:" name="password" type="password" value={password} onChange={setPassword} onBlur={()=>setTouched(s=>({...s,pass:true}))} error={touched.pass ? v.pe : ""} autoComplete="current-password" required minLength={6} />
          <div className="auth-subtext">FORGOT PASSWORD?</div>
          {err && <div className="auth-error">{err}</div>}
          <button className="auth-cta" disabled={!v.ok}>LOGIN</button>
        </form>
      </div>
    </div>
  );
}

// ======= REGISTER =======
function RegisterPage() {
  const [username,setUsername] = useState("");
  const [email,setEmail] = useState("");
  const [password,setPassword] = useState("");
  const [confirm,setConfirm] = useState("");
  const [err,setErr] = useState("");
  const [touched, setTouched] = useState({ user:false, email:false, pass:false, confirm:false });
  const navigate = useNavigate();
  const { user, refetchMe } = useAuth();

  useEffect(()=>{ if(user){ navigate('/whats-this', { replace:true }); } }, [user]);

  const USER_RE = /^[A-Za-z0-9_.-]+$/;
  const isEmail = (x)=> /.+@.+\..+/.test(x);
  function validate(){
    const u = username.trim();
    const m = email.trim();
    const p = password;
    const c = confirm;
    let ue = ""; let me = ""; let pe = ""; let ce = "";
    if(!u) ue = "Username is required"; else if(u.length < 3) ue = "At least 3 characters"; else if(u.length>32) ue = "Max 32 characters"; else if(!USER_RE.test(u)) ue = "Letters, numbers, _ . - only";
    if(!m) me = "Email is required"; else if(!isEmail(m)) me = "Enter a valid email";
    if(!p) pe = "Password is required"; else if(p.length < 6) pe = "At least 6 characters"; else if(p.length>128) pe = "Max 128 characters";
    if(!c) ce = "Confirm your password"; else if(p !== c) ce = "Passwords do not match";
    return { ue, me, pe, ce, ok: !(ue||me||pe||ce) };
  }


async function handleSubmit(e){
  e.preventDefault();
  const v = validate();
  if(!v.ok) return;

  try{
    const res = await fetch(`${API}/api/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({
        username: username.trim(),
        email: email.trim().toLowerCase(),
        password,
        // optional defaults for your app:
        settings: {
          daily_objective_time: "17:00",
          streak_reminders: true,
          dyslexia_font: "Off",
          reading_guide: true,
          high_contrast: false,
          tts_voice: "App voice",
          speaking_rate: 1.0,
          word_highlight_color: "#FFD54F",
          grid_guides: false,
          storage_path: ""
        },
        achievements: { streak_days: 0, points: 0, badges: [] }
      })
    });
    if(!res.ok) throw new Error("Registration failed");
    await refetchMe();
    navigate("/whats-this", { replace: true });
  }catch(e){
    setErr(e.message || "Registration failed");
  }
}

  const v = validate();

  return (
    <div className="min-h-screen" style={{ background:"var(--forest)" }}>
      <NavBar />
      <div className="auth-wrap">
        <form className="auth-card" onSubmit={handleSubmit}>
          <div className="auth-title">REGISTER NOW!</div>
          <CardField label="USERNAME:" name="username" value={username} onChange={setUsername} onBlur={()=>setTouched(s=>({...s,user:true}))} error={touched.user ? v.ue : ""} autoComplete="username" required minLength={3} maxLength={32} />
          <CardField label="EMAIL:" name="email" type="email" value={email} onChange={setEmail} onBlur={()=>setTouched(s=>({...s,email:true}))} error={touched.email ? v.me : ""} autoComplete="email" required />
          <CardField label="PASSWORD:" name="password" type="password" value={password} onChange={setPassword} onBlur={()=>setTouched(s=>({...s,pass:true}))} error={touched.pass ? v.pe : ""} autoComplete="new-password" required minLength={6} maxLength={128} />
          <CardField label="CONFIRM PASSWORD:" name="confirm" type="password" value={confirm} onChange={setConfirm} onBlur={()=>setTouched(s=>({...s,confirm:true}))} error={touched.confirm ? v.ce : ""} autoComplete="new-password" required />
          {err && <div className="auth-error">{err}</div>}
          <button className="auth-cta" disabled={!v.ok}>REGISTER</button>
        </form>
      </div>
    </div>
  );
}

// ======= SIMPLE HOME (Legacy) =======
function HomeLegacy() {
  return (
    <div className="min-h-screen" style={{ background:"var(--forest)" }}>
      <NavBar />
      <section className="hero hero-home">
        <div className="hero-card hero-home-card">
          <div className="hero-icon-row">
            <img src={imgIcon} alt="Image" className="hero-icon" />
            <img src={drawIcon} alt="Draw" className="hero-icon" />
            <img src={arrowIcon} alt="Arrow" className="hero-arrow" />
            <img src={textIcon} alt="Caption" className="hero-icon" />
          </div>
        </div>
        <p className="hero-tagline">TURN IMAGES AND SKETCHES INTO CAPTIONS, AND SPIN YOUR DRAWINGS INTO STORIES.</p>
      </section>
    </div>
  );
}

/// ======= APP ROUTES (Fixed with real pages) =======
function AppLegacy(){
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          {/* Public */}
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

          {/* Protected: mount the actual pages, not <Home /> */}
          {/* If you have a WhatsThisPage component, use it here; otherwise keep Home */}
          <Route
            path="/whats-this"
            element={<ProtectedRoute><Home /></ProtectedRoute>}
          />
          <Route
            path="/draw"
            element={<ProtectedRoute><DrawPage /></ProtectedRoute>}
          />
          <Route
            path="/story"
            element={<ProtectedRoute><StoryPage /></ProtectedRoute>}
          />
          <Route
            path="/quiz"
            element={<ProtectedRoute><QuizPage /></ProtectedRoute>}
          />
          {/* If you have dedicated components for these, swap them in */}
          <Route
            path="/instructions"
            element={<ProtectedRoute><Home /></ProtectedRoute>}
          />
          <Route
            path="/settings"
            element={<ProtectedRoute><Home /></ProtectedRoute>}
          />

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}


async function apiStoriesList(){
  const r = await fetch(`${API}/api/stories/list`, { credentials:'include' });
  const d = await r.json().catch(()=>({}));
  if (!r.ok || d?.error) {
    const msg = d?.detail || d?.error || `list failed (${r.status})`;
    throw new Error(msg);
  }
  return d; // an array of {id,title}
}

async function apiStoriesSave(payload){
  const r = await fetch(`${API}/api/stories/save`, {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    credentials:'include',
    body: JSON.stringify(payload)
  });
  const d = await r.json().catch(()=>({}));
  if(!r.ok || d?.error){
    const msg = d?.detail || d?.error || `save failed (${r.status})`;
    throw new Error(msg);
  }
  return d; // { ok:true, id }
}

async function apiStoryGet(id){
  const r = await fetch(`${API}/api/stories/get?id=${encodeURIComponent(id)}`, { credentials:'include' });
  const d = await r.json().catch(()=>({}));
  if(!r.ok || d?.error){
    const msg = d?.detail || d?.error || `get failed (${r.status})`;
    throw new Error(msg);
  }
  return d.story; // { id,title,panels,images,saved_at }
}

async function apiStoryDelete(id){
  const r = await fetch(`${API}/api/stories/delete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ id })
  });
  const d = await r.json().catch(()=>({}));
  if (!r.ok || d?.error) {
    const msg = d?.detail || d?.error || `delete failed (${r.status})`;
    throw new Error(msg);
  }
  return d; // { ok: true }
}

async function apiStory(blobs){
  const fd = new FormData();
  fd.append('image1', blobs[0], 'scene1.png');
  fd.append('image2', blobs[1], 'scene2.png');
  fd.append('image3', blobs[2], 'scene3.png');
  const r = await fetch(`${API}/api/story`, { method:'POST', body: fd });
  const data = await r.json().catch(()=>({}));
  if(!r.ok || data?.error) throw new Error(data?.error || 'story failed');
  return data;
}


async function apiCaption(imageBlob, region){
  const fd = new FormData();
  fd.append('image', imageBlob, 'img.jpg');
  if (region) fd.append('region', JSON.stringify(region));
  fd.append('mode', 'detailed');
  const r = await fetch(`${API}/api/caption`, { method: 'POST', body: fd });
  const d = await r.json().catch(()=>({}));
  if (!r.ok || d?.error) throw new Error(d?.error || 'caption failed');
  return d.caption;           // still returns the string you set into state
 }

async function apiDescribe(imageBlob, region){
  const fd = new FormData();
  fd.append('image', imageBlob, 'img.jpg');
  if (region) fd.append('region', JSON.stringify(region));
  fd.append('mode', 'description');   // was used before; harmless to keep
  const r = await fetch(`${API}/api/caption`, { method: 'POST', body: fd });
  const d = await r.json().catch(()=>({}));
  if (!r.ok || d?.error) throw new Error(d?.error || 'description failed');
  return d; // { caption, sentences, paragraph } in your original flow
}

async function apiTranslate(text, lang /* 'zh' | 'ms' | 'ta' */){
  const r = await fetch(`${API}/api/translate`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ text, lang })
  });
  const d = await r.json().catch(()=>({}));
  if(!r.ok || d?.error) throw new Error(d?.error || 'translate failed');
  return d.text || '';
}


async function apiDict(word){ const r=await fetch(`${API}/api/dictionary?word=${encodeURIComponent(word)}`); const d=await r.json().catch(()=>({})); if(!r.ok) throw new Error('dict failed'); return d; }
// in App.jsx (where apiTTS lives)
// in App.jsx
async function apiTTS(text, opts = {}) {
  // Guard: nothing to read
  if (!text || !String(text).trim()) return null;

  const hasWindow = typeof window !== "undefined";
  const gs = hasWindow ? (window.__picteractive_settings || {}) : {};

  // ----- VOICE (App voice + Female use the SAME voice) -----
  // Priority: explicit opts.voice -> saved settings -> "App voice"
  const effectiveVoice =
    opts.voice != null && opts.voice !== ""
      ? opts.voice
      : (gs.tts_voice || "App voice");

  const rawVoice = effectiveVoice.toString().trim().toLowerCase();

  // Only exactly "male" is male. Everything else (including "female" and "app voice") is female.
  let normalizedVoice = "female";
  if (rawVoice === "male") {
    normalizedVoice = "male";
  }
  // (App voice, Female, default, etc. all share the same 'female' branch)

  // ----- RATE -----
  // Priority: explicit opts.rate -> saved speaking_rate -> 1.0
  const rateSource =
    opts.rate != null && Number(opts.rate) > 0
      ? Number(opts.rate)
      : (Number(gs.speaking_rate) > 0 ? Number(gs.speaking_rate) : 1.0);

  let rate = Number(rateSource);
  if (!isFinite(rate) || rate <= 0) rate = 1.0;
  // Clamp to a sensible Web Speech range
  if (rate < 0.25) rate = 0.25;
  if (rate > 3.0) rate = 3.0;

  // ----- WEB SPEECH PATH (primary) -----
  if (
    hasWindow &&
    "speechSynthesis" in window &&
    "SpeechSynthesisUtterance" in window
  ) {
    const synth = window.speechSynthesis;

    // Helper: wait for voices to be loaded
    function loadVoices() {
      return new Promise((resolve) => {
        let voices = synth.getVoices();
        if (voices && voices.length) {
          resolve(voices);
          return;
        }
        const interval = setInterval(() => {
          voices = synth.getVoices();
          if (voices && voices.length) {
            clearInterval(interval);
            resolve(voices);
          }
        }, 50);
        // Safety timeout
        setTimeout(() => {
          clearInterval(interval);
          resolve(voices || []);
        }, 1000);
      });
    }

    function pickVoice(voices, kind /* "male" | "female" */) {
      if (!Array.isArray(voices) || !voices.length) return null;

      const isEnglish = (v) =>
        v && typeof v.lang === "string" && /^en(-|_|$)/i.test(v.lang);

      const lowerMatches = (v, list) => {
        const name = (v.name || "").toLowerCase();
        return list.some((m) => name.includes(m));
      };

      if (kind === "male") {
        // Prefer an English male-ish voice if we can guess it
        const maleNames = ["male", "david", "christopher", "daniel", "fred"];
        const byName = voices.find(
          (v) => isEnglish(v) && lowerMatches(v, maleNames)
        );
        if (byName) return byName;

        // Fallback: English UK often sounds "different" enough for kids
        const enGb = voices.find(
          (v) => isEnglish(v) && /en-GB/i.test(v.lang || "")
        );
        if (enGb) return enGb;
      }

      // Female / App voice → same base voice
      const femaleNames = [
        "female",
        "zira",
        "susan",
        "samantha",
        "karen",
        "kathy",
        "victoria",
      ];
      const femaleByName = voices.find(
        (v) => isEnglish(v) && lowerMatches(v, femaleNames)
      );
      if (femaleByName) return femaleByName;

      const firstEn = voices.find(isEnglish);
      if (firstEn) return firstEn;

      return voices[0] || null;
    }

    const voices = await loadVoices();
    const chosen = pickVoice(voices, normalizedVoice);

    // Cancel any previous utterances so settings apply immediately
    synth.cancel();

    const utterance = new SpeechSynthesisUtterance(String(text));
    utterance.rate = rate;
    if (chosen) utterance.voice = chosen;

    synth.speak(utterance);

    // IMPORTANT:
    // We return null to signal to callers (Story page, What’s This page)
    // that Web Speech is already speaking and no <audio src> is needed.
    return null;
  }

  // ----- FALLBACK: call backend /api/tts (gTTS) if Web Speech is not available -----
  const serverVoice = normalizedVoice === "male" ? "male" : "female";

  const res = await fetch(`${API}/api/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice: serverVoice, rate }),
  });

  if (!res.ok) throw new Error("TTS failed");

  const blob = await res.blob();
  return URL.createObjectURL(blob); // used with <audio src="...">
}



async function apiQuiz(caption, count=3){ const r=await fetch(`${API}/api/quiz`,{ method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ caption, count })}); const d=await r.json().catch(()=>({})); if(!r.ok||d?.error) throw new Error(d?.error||'quiz failed'); return (Array.isArray(d.questions)? d.questions: []).map(q=>({ question:q.question, options:q.options?.slice(0,3)||[], answerIndex: (typeof q.answerIndex==='number'? q.answerIndex : (typeof q.answer_index==='number'? q.answer_index : 0)) })); }
async function apiAchEvent(type){ try{ await fetch(`${API}/api/achievements/event`, { method:'POST', credentials:'include', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ type })}); }catch{} }

function NavBar(){
  const loc = useLocation();
  const navigate = useNavigate();
  const path = loc.pathname;
  const [open, setOpen] = useState(false);
  const { user, logout } = useAuth();
  const go = (p)=>{ setOpen(false); navigate(p); };
  return (
    <header className="site-nav">
      <div className="site-nav-inner">
        <Link to="/" className="site-brand"><img src={logo} alt="Picteractive" /></Link>
        <nav className="site-menu">
          {user && (
            <>
              <Link to="/whats-this" className={cx('site-menu-link', path==='/whats-this'&&'active')}>WHAT'S THIS?</Link>
              <Link to="/draw" state={{ refresh: true }} className={cx('site-menu-link', path==='/draw'&&'active')}>DRAW</Link>
              <Link to="/instructions" className={cx('site-menu-link', path==='/instructions'&&'active')}>INSTRUCTIONS</Link>
              <Link to="/achievements" className={cx('site-menu-link', path==='/achievements'&&'active')}>ACHIEVEMENTS</Link>
              <Link to="/settings" className={cx('site-menu-link', path==='/settings'&&'active')}>SETTINGS</Link>
            </>
          )}
        </nav>
        <div className="site-nav-spacer" />
        <div className="profile">
          <button className="profile-btn" onClick={()=>setOpen(v=>!v)} aria-haspopup="menu" aria-expanded={open}><img src={profileLogo} alt="Profile"/></button>
          {open && (
            <div className="dropdown" role="menu">
              {!user && (
                <>
                  <button className="dropdown-item" onClick={()=>go('/login')}>Log in</button>
                  <button className="dropdown-item" onClick={()=>go('/register')}>Sign up</button>
                </>
              )}
              {user && (
                <>
                  <div className="dropdown-item" style={{pointerEvents:'none'}}>{user?.username || 'Account'}</div>
                  <button className="dropdown-item" onClick={()=>{ setOpen(false); logout(); navigate('/'); }}>Logout</button>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

// --- Quiz: Confetti overlay ---
function ConfettiOverlay({ show }){
  const colors = ['#fb4c00','#ffd166','#06b6d4','#8b5cf6','#22c55e','#f59e0b'];
  if(!show) return null;
  const pieces = Array.from({length:120}).map((_,i)=>{
    const left = Math.random()*100; // vw
    const delay = Math.random()*0.6; // s
    const dur = 2.2 + Math.random()*1.1; // s
    const bg = colors[i % colors.length];
    const rotate = Math.floor(Math.random()*360);
    const style = {
      left: left+"vw",
      top: "-10vh",
      background: bg,
      transform: `rotate(${rotate}deg)`,
      animationDuration: `${dur}s`,
      animationDelay: `${delay}s`,
    };
    return <div key={i} className="confetti-piece" style={style}/>;
  });
  return <div className="confetti-layer">{pieces}</div>;
}

function DictionaryModal({ word, data, loading, error, onClose }){
  if(!word) return null;
  const synonyms = Array.isArray(data?.synonyms) ? data.synonyms.filter(Boolean) : [];
  const displayWord = word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
  const examples = Array.isArray(data?.examples) ? data.examples.filter(Boolean).slice(0,3) : [];
  return (
    <div className="modal" role="dialog" aria-modal="true" aria-labelledby="wt-dict-word" onClick={(ev)=>{ if(ev.target===ev.currentTarget) onClose(); }}>
      <div className="wt-dict-card">
        <div className="wt-dict-header">
          <div id="wt-dict-word" className="wt-dict-word">{displayWord}</div>
          <button className="wt-dict-close" onClick={onClose} aria-label="Close dictionary">&times;</button>
        </div>
        {loading && (<div className="wt-dict-loading">Looking up...</div>)}
        {!loading && error && (<div className="wt-dict-error">{error}</div>)}
        {!loading && !error && data && (
          <div className="wt-dict-body">
            <div className="wt-dict-section">Meaning (WordNet):</div>
            <div className="wt-dict-definition">{data.definition || 'Definition not available.'}</div>
            {examples.length ? (
              <div className="wt-dict-examples">
                <div className="wt-dict-section">Examples:</div>
                <ul>
                  {examples.map((ex,i)=>(<li key={i}>{ex}</li>))}
                </ul>
              </div>
            ) : null}
            <div className="wt-dict-section">Synonyms (WordNet):</div>
            <div className="wt-dict-synonyms">
              {synonyms.length ? synonyms.map((syn,i)=>(<span key={i}>{syn}</span>)) : <span className="wt-dict-synonyms-empty">No close synonyms found.</span>}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// --- Quiz Page ---
function QuizPage(){
  const { state } = useLocation();
  const navigate = useNavigate();
  const { refetchMe } = useAuth();
  const caption = state?.caption || '';
  const imageUrl = state?.imageUrl || '';
  const [loading, setLoading] = React.useState(true);
  const [questions, setQuestions] = React.useState([]);
  const [idx, setIdx] = React.useState(0);
  const [picked, setPicked] = React.useState(null);
  const [checked, setChecked] = React.useState(false);
  const [corrects, setCorrects] = React.useState([]);
  const [allRight, setAllRight] = React.useState(false);
  const total = questions.length || 0;

  React.useEffect(()=>{
    if(!caption){ navigate('/whats-this'); return; }
    (async()=>{
      setCorrects([]); setAllRight(false); setIdx(0); setPicked(null); setChecked(false);
      try{ setLoading(true); const q = await apiQuiz(caption, 3); setQuestions(q); }
      catch{ setQuestions([]); }
      finally{ setLoading(false); }
    })();
  },[]);

  React.useEffect(()=>{
    // reset selection on question change
    setPicked(null); setChecked(false);
  },[idx]);

  React.useEffect(()=>{
    if(allRight){ (async()=>{ try{ await apiAchEvent('quiz'); await refetchMe(); }catch{} })(); }
  }, [allRight]);

  function confirm(){
    if(picked==null || checked) return;
    const isRight = (picked === (questions[idx]?.answerIndex ?? -1));
    setChecked(true);
    setCorrects(prev=>{ const next=[...prev]; next[idx]=isRight; if(next.length===questions.length && next.every(Boolean)) setAllRight(true); return next; });
  }
  function next(){ if(idx < total-1){ setIdx(i=>i+1);} }
  function handleNext(){
    if(picked==null) return;
    if(!checked){
      confirm();
      return;
    }
    if(idx < total-1){
      next();
    }
  }

  return (
    <div className="min-h-screen" style={{ background:'var(--forest)' }}>
      <NavBar />
      <div className="quiz-wrap">
        <button className="quiz-back" aria-label="Back" onClick={()=>navigate(-1)}><img src={backIcon} alt="Back" style={{ width: 28, height: 28, display: 'block' }}/></button>
        <div className="quiz-title">QUIZ TIME!</div>
        <div className="quiz-stage">
          {imageUrl? <img src={imageUrl} alt="quiz subject"/> : <div className="quiz-placeholder">No image</div>}
        </div>
        <div className="quiz-card">
          {loading && (<div className="quiz-loading">Generating questions...</div>)}
          {!loading && total>0 && (
            <div>
              <div className="quiz-question">{questions[idx]?.question || ''}</div>
              <div className="quiz-options">
                {(questions[idx]?.options||[]).map((opt,i)=>{
                  const isCorrect = checked && i===questions[idx].answerIndex;
                  const isWrong = checked && picked===i && !isCorrect;
                  const cls = cx('quiz-option', picked===i && 'selected', isCorrect && 'correct', isWrong && 'wrong');
                  return (
                    <button key={i} className={cls} onClick={()=>!checked && setPicked(i)} disabled={checked}>
                      <span className="quiz-letter">{'ABC'[i]}</span>
                      <span>{opt}</span>
                    </button>
                  );
                })}
              </div>
              <div className="quiz-foot">
                <div>{Math.min(idx+1,total)}/{total||3}</div>
                <button
                  className="quiz-next"
                  onClick={handleNext}
                  disabled={picked==null || (checked && idx>=total-1)}
                >
                  {checked ? (idx>=total-1 ? 'DONE' : 'NEXT →') : 'NEXT →'}
                </button>
              </div>
              {checked && (
                <div className={cx('quiz-feedback', (picked === questions[idx].answerIndex)? 'correct':'incorrect')}>
                  {(picked === questions[idx].answerIndex)? 'Correct!' : 'Not quite. Try the next one!'}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      <ConfettiOverlay show={allRight} />
    </div>
  );
}

function Home(){
  return (
    <div className="min-h-screen" style={{ background:'var(--forest)' }}>
      <NavBar />
      <section className="hero hero-home">
        <div className="hero-card hero-home-card">
          <div className="hero-icon-row">
            <img src={imgIcon} alt="Image" className="hero-icon" />
            <img src={drawIcon} alt="Draw" className="hero-icon" />
            <img src={arrowIcon} alt="Arrow" className="hero-arrow" />
            <img src={textIcon} alt="Caption" className="hero-icon" />
          </div>
        </div>
        <p className="hero-tagline">TURN IMAGES AND SKETCHES INTO CAPTIONS, AND SPIN YOUR DRAWINGS INTO STORIES.</p>
      </section>
    </div>
  );
}



function DrawPage(){
  const canvasRef = useRef(null);
  const fabricRef = useRef(null);
  const inited = useRef(false);
  const navigate = useNavigate();
  const [tool, setTool] = useState('brush');
  const toolRef = useRef('brush');
  const [color, setColor] = useState('#000000');
  const colorRef = useRef('#000000');
  const [size, setSize] = useState(12);
  const [frames, setFrames] = useState([]);
  const dragIndex = useRef(null);
  const undo = useRef([]); const redo = useRef([]); const restoring = useRef(false);
  const { show, Toast } = useToast();
  const gridLayerRef = useRef(null);
  const [saved, setSaved] = useState([]);
  const location = useLocation();
  

// --- Refresh Saved Stories and listen for updates ---
const refreshSaved = React.useCallback(async () => {
  let ok = true;
  try {
    const list = await apiStoriesList(); // includes credentials
    if (!ok) return;
    setSaved(Array.isArray(list) ? list : []);
  } catch (e) {
    if (!ok) return;
    const msg = String(e?.message || "").toLowerCase();
    if (msg.includes("unauthorized") || msg.includes("(401)")) {
      show("Please log in to see your saved stories.");
      navigate("/login", { replace: true });
    } else {
      show("Could not load saved stories.");
      setSaved([]);
    }
  }
  return () => { ok = false; };
}, [navigate, show]);

useEffect(() => {
  refreshSaved();
  if (location.state?.justSavedId || location.state?.refresh) {
    navigate(location.pathname, { replace: true, state: {} });
  }
}, [refreshSaved, location.state?.justSavedId, location.state?.refresh]);

// ✅ NEW listener: updates list immediately when StoryPage saves
useEffect(() => {
  const onUpdated = (ev) => {
    const { id, title } = ev.detail || {};
    if (id && title) {
      // Optimistic insert so the user sees it instantly
      setSaved((prev) => [{ id, title }, ...prev.filter((s) => s.id !== id)]);
    }
    refreshSaved(); // then sync with server
  };
  window.addEventListener("stories:updated", onUpdated);
  return () => window.removeEventListener("stories:updated", onUpdated);
}, [refreshSaved]);


  useEffect(()=>{
    if (inited.current) return;
    const el = canvasRef.current;
    if(!el) return;
    inited.current = true;
    const SIZE = 700;
    el.width = SIZE;
    el.height = SIZE;
    const canvas = new fabric.Canvas(el, {
      isDrawingMode: true,
      backgroundColor: '#ffffff',
      width: SIZE,
      height: SIZE,
      enableRetinaScaling: false,
      renderOnAddRemove: true,
    });
    canvas.setBackgroundColor('#ffffff', () => canvas.renderAll());
    canvas.selection = false;
    canvas.skipTargetFind = true;
    canvas.defaultCursor = 'crosshair';
    canvas.hoverCursor = 'crosshair';
    if(canvas.upperCanvasEl){
      canvas.upperCanvasEl.style.touchAction = 'none';
      canvas.upperCanvasEl.style.pointerEvents = 'auto';
      canvas.upperCanvasEl.style.cursor = 'crosshair';
    }
    if(canvas.lowerCanvasEl){
      canvas.lowerCanvasEl.style.touchAction = 'none';
      canvas.lowerCanvasEl.style.pointerEvents = 'auto';
      canvas.lowerCanvasEl.style.cursor = 'crosshair';
    }
    fabricRef.current = canvas;

    // Build grid layer on mount based on current settings
    try { drawGridLayer(); } catch {}


    const rec = ()=>push();
    canvas.on('path:created', rec);
    canvas.on('object:modified', rec);
    canvas.on('object:removed', rec);

    const onDown = (opt) => {
      if(toolRef.current !== 'fill') return;
      const p = canvas.getPointer(opt.e);
      floodFillAt(canvas, Math.floor(p.x), Math.floor(p.y), colorRef.current);
    };
    canvas.on('mouse:down', onDown);

    push();
    applyBrush();

    return ()=>{
      canvas.off('path:created',rec);
      canvas.off('object:modified',rec);
      canvas.off('object:removed',rec);
      canvas.off('mouse:down', onDown);
      canvas.dispose();
      fabricRef.current = null;
      inited.current = false;
    };
  },[]);

  useEffect(()=>{
    toolRef.current = tool;
    colorRef.current = color;
    applyBrush();
  }, [tool, color, size]);

  
  function drawGridLayer(){
    const c = fabricRef.current;
    if(!c) return;
    // remove existing
    if(gridLayerRef.current){
      c.remove(gridLayerRef.current);
      gridLayerRef.current = null;
    }
    // add if enabled
    const enabled = (window.__picteractive_settings?.grid_guides ?? false);
    if(!enabled) { c.renderAll(); return; }
    
    const size = c.getWidth();
    const step = 40; // px
    const lines = [];
    for(let x=step; x<size; x+=step){
      lines.push(new fabric.Line([x,0,x,size], { stroke: '#cbd5e1', selectable:false, evented:false, opacity:0.6 }));
    }
    for(let y=step; y<size; y+=step){
      lines.push(new fabric.Line([0,y,size,y], { stroke: '#cbd5e1', selectable:false, evented:false, opacity:0.6 }));
    }
    const grp = new fabric.Group(lines, { selectable:false, evented:false, excludeFromExport:true });
    // keep grid below drawings
    c.add(grp);
    grp.sendToBack();
    c.renderAll();
    gridLayerRef.current = grp;
  }

  useEffect(() => {
  const onApply = () => {
    try { drawGridLayer(); } catch {}
  };
  window.addEventListener('picteractive:settings-applied', onApply);
  return () => window.removeEventListener('picteractive:settings-applied', onApply);
}, []);


  function applyBrush(){
  const c = fabricRef.current;
  if (!c) return;

  // Helpful cursor feedback
  if (c.upperCanvasEl) {
    c.upperCanvasEl.style.cursor = (tool === 'fill') ? 'pointer' : 'crosshair';
  }

  // Fill: no free drawing; click handled by mouse:down -> floodFillAt
  if (tool === 'fill') {
    c.isDrawingMode = false;
    c.requestRenderAll();
    return;
  }

  // Eraser vs Pencil
  const b = new fabric.PencilBrush(c);
  b.width = Number(size) || 3;
  b.color = (tool === 'eraser') ? '#ffffff' : color;

  c.freeDrawingBrush = b;
  c.isDrawingMode = true;
  c.requestRenderAll();
}

  function push(){ const c=fabricRef.current; if(!c||restoring.current) return; redo.current=[]; undo.current.push(c.toJSON()); if(undo.current.length>30) undo.current.shift(); }
  function restore(s){ const c=fabricRef.current; if(!c) return; restoring.current=true; c.loadFromJSON(s, ()=>{ c.renderAll(); restoring.current=false; }); }
  function handleUndo(){ if(undo.current.length<=1) return; const cur=undo.current.pop(); if(cur) redo.current.push(cur); const prev=undo.current[undo.current.length-1]; if(prev) restore(prev); }
  function handleRedo(){ if(!redo.current.length) return; const s=redo.current.pop(); if(s){ undo.current.push(s); restore(s); } }
  function handleClear(){
  const c = fabricRef.current;
  if (!c) return;

  // 1) Remove all drawable objects
  c.getObjects().forEach(o => c.remove(o));

  // 2) Reset background image (persistent fills) and background color
  c.setBackgroundImage(null, () => {
    c.setBackgroundColor('#ffffff', () => {
      // 3) Rebuild grid (if guides are enabled)
      if (gridLayerRef.current) {
        gridLayerRef.current = null;
      }
      drawGridLayer();

      // 4) Re-render and reset history
      c.requestRenderAll();
      undo.current = [];
      redo.current = [];
      push(); // record the cleared state
    });
  });
}


  function hexToRgba(hex){
    let h = (hex || '').toString().trim();
    if(!h) return [0,0,0,255];
    if(h.startsWith('#')) h = h.slice(1);
    if(h.length===3){ h = h.split('').map(ch=>ch+ch).join(''); }
    if(h.length!==6){ return [0,0,0,255]; }
    const num = parseInt(h,16);
    return [(num >> 16) & 255, (num >> 8) & 255, num & 255, 255];
  }

  function floodFillAt(canvas, startX, startY, fillHex){
  if(!canvas) return;
  const lower = canvas.lowerCanvasEl;
  if(!lower) return;
  const ctx = lower.getContext('2d');
  if(!ctx) return;

  // Make sure we’re reading the most recent render
  canvas.renderAll();

  const width = canvas.getWidth();
  const height = canvas.getHeight();
  if(startX < 0 || startY < 0 || startX >= width || startY >= height) return;

  // Get a snapshot of the current pixels
  const img = ctx.getImageData(0, 0, width, height);
  const data = img.data;

  const fill = hexToRgba(fillHex);
  const idx0 = (startY * width + startX) * 4;
  const target = [data[idx0], data[idx0+1], data[idx0+2], data[idx0+3]];

  // If already same color, skip
  if(target[0]===fill[0] && target[1]===fill[1] && target[2]===fill[2] && target[3]===fill[3]) return;

  // Small tolerance helps fill slightly anti-aliased borders
  const tol = 8;
  const matches = (i) => {
    return Math.abs(data[i]   - target[0]) <= tol &&
           Math.abs(data[i+1] - target[1]) <= tol &&
           Math.abs(data[i+2] - target[2]) <= tol &&
           Math.abs(data[i+3] - target[3]) <= tol;
  };

  // Iterative stack-based flood fill
  const stack = [[startX, startY]];
  const visited = new Uint8Array(width * height);

  while(stack.length){
    const [x, y] = stack.pop();
    if(x<0 || y<0 || x>=width || y>=height) continue;
    const i = (y*width + x);
    if(visited[i]) continue;
    const di = i*4;
    if(!matches(di)) continue;

    // color the pixel
    data[di]   = fill[0];
    data[di+1] = fill[1];
    data[di+2] = fill[2];
    data[di+3] = fill[3];
    visited[i] = 1;

    // push neighbors
    stack.push([x+1,y],[x-1,y],[x,y+1],[x,y-1]);
  }

  // Draw the updated pixels into an offscreen canvas
  const tmp = document.createElement('canvas');
  tmp.width = width; tmp.height = height;
  const tctx = tmp.getContext('2d');
  tctx.putImageData(img, 0, 0);

  // Persist as backgroundImage so Fabric doesn’t overwrite it on the next render
  fabric.Image.fromURL(tmp.toDataURL(), (bgImg) => {
    // Once we start using a bitmap background, stop using backgroundColor
    canvas.setBackgroundColor(null, ()=>{});
    canvas.setBackgroundImage(bgImg, () => {
      canvas.requestRenderAll();
      // record in undo stack if you keep one
      if (typeof push === 'function') { try { push(); } catch {} }
    });
  }, { crossOrigin: 'anonymous' });
}


  // Turn an imported asset URL into a {id,url,blob} frame
async function urlToFrame(src){
  const blob = await fetch(src).then(r => r.blob());
  const url = URL.createObjectURL(blob);
  return { id: Date.now() + Math.random(), url, blob };
}

async function tryExample(){
  // Free any existing object URLs
  setFrames(prev => { prev.forEach(f => URL.revokeObjectURL(f.url)); return prev; });

  const [f1, f2, f3] = await Promise.all([
    urlToFrame(scene1),
    urlToFrame(scene2),
    urlToFrame(scene3),
  ]);

  setFrames([f1, f2, f3]);
  show("Loaded example scenes");
}


  async function capture(){ const c=fabricRef.current; if(!c) return null; c.discardActiveObject(); c.renderAll(); const url=c.toDataURL({format:'png'}); const blob=await (await fetch(url)).blob(); return blob; }
  // ---------- QUALITY GATE HELPERS ----------
function measureVarianceFromCanvas(){
  const c = fabricRef.current; if(!c) return 0;
  const off = document.createElement('canvas'); off.width = 64; off.height = 64;
  const ctx = off.getContext('2d'); if(!ctx) return 0;
  // ⬇️ use the real canvas element
  ctx.drawImage(c.lowerCanvasEl, 0, 0, 64, 64);
  const { data } = ctx.getImageData(0,0,64,64);
  let sum=0, sum2=0, n=64*64;
  for(let i=0;i<data.length;i+=4){ const g=(data[i]+data[i+1]+data[i+2])/3; sum+=g; sum2+=g*g; }
  const mean=sum/n; const varg=Math.max(0,(sum2/n)-mean*mean);
  return varg;
}
function captionOK(c){
  const t=(c||'').toLowerCase().trim();
  if(!t) return false;
  if(t.split(/\s+/).length<=3) return false;
  if(/\b(drawing|image|picture|photo|sketch|line|clip ?art|black and white)\b/.test(t)) return false;
  return true;
}
// small token-set similarity (0..1)
function sim(a,b){
  const A=new Set(String(a||'').toLowerCase().split(/\W+/).filter(Boolean));
  const B=new Set(String(b||'').toLowerCase().split(/\W+/).filter(Boolean));
  if(!A.size||!B.size) return 0;
  let inter=0; A.forEach(x=>{ if(B.has(x)) inter++; });
  return inter/Math.min(A.size,B.size);
}

async function addToStory() {
  if (frames.length >= 3) {
    show('All three frames are filled');
    return;
  }

  const b = await capture();
  if (!b) { show('Draw something first'); return; }

  // Gate 1: size + variance (keeps obvious blanks out)
  const variance = measureVarianceFromCanvas();
  if (b.size < 3000 || variance < 400) {
    show('I can’t see enough detail yet — add a clear shape or character.');
    return;
  }

  // ✅ No recognizer: just keep the image
  const u = URL.createObjectURL(b);
  setFrames(p => [...p, { id: Date.now(), url: u, blob: b }]);
}


  function removeFrame(id){ setFrames(p=>{ const n=p.filter(f=>f.id!==id); const r=p.find(f=>f.id===id); if(r) URL.revokeObjectURL(r.url); return n; }); }
  function dragStart(i,has){ if(!has) return; dragIndex.current=i; }
  function dragOver(e){ e.preventDefault(); }
  function drop(e,i){ e.preventDefault(); if(dragIndex.current===null||dragIndex.current===i) return; setFrames(p=>{ const n=[...p]; const [it]=n.splice(dragIndex.current,1); n.splice(i,0,it); return n; }); dragIndex.current=null; }
  async function createStory(){
  if(frames.length!==3){ show('Add three drawings'); return; }

  // Gate 3: diversity across panels
  try{
  const labels = await Promise.all(frames.map(async f => {
    if (f.label) return f.label;
    const r = await apiRecognize(f.blob);
    return r[0]?.label || "";
  }));
} catch {}
// then proceed to navigate('/story' ...)


  const blobs = frames.map(f=>f.blob);
  const urls  = frames.map(f=>f.url);
  navigate('/story',{ state:{ images: urls, blobs } });
}

// Light-grey look for any button that's not selected
const inactiveBtnStyle = { background: '#e5e7ebff', color: '#111' };
  
return (
  <div className="min-h-screen" style={{ background: 'var(--forest)' }}>
    <NavBar />
    <section className="draw-stage">
      <h1 className="draw-heading">DRAW SOMETHING</h1>

      <div className="draw-board">
        {/* Canvas */}
        <div className="draw-canvas-shell">
          <div className="draw-canvas-frame">
            <canvas ref={canvasRef} />
          </div>

          <div className="draw-toolbar-row">
            <button
              className={cx('draw-tool-btn', tool === 'brush' && 'active')}
              onClick={() => setTool('brush')}
              style={tool === 'brush' ? undefined : inactiveBtnStyle}
            >
              Brush
            </button>
            <button
              className={cx('draw-tool-btn', tool === 'eraser' && 'active')}
              onClick={() => setTool('eraser')}
              style={tool === 'eraser' ? undefined : inactiveBtnStyle}
            >
              Eraser
            </button>
            <button
              className={cx('draw-tool-btn', tool === 'fill' && 'active')}
              onClick={() => setTool('fill')}
              style={tool === 'fill' ? undefined : inactiveBtnStyle}
            >
              Fill
            </button>

            <label className="draw-color-picker">
              Color
              <input
                type="color"
                value={color}
                onChange={(e) => setColor(e.target.value)}
              />
            </label>

            <label className="draw-color-picker">
              Size
              <input
                type="range"
                min="2"
                max="48"
                step="1"
                value={size}
                onChange={(e) => setSize(Number(e.target.value))}
              />
            </label>

            <button className="draw-tool-btn" onClick={handleUndo} style={inactiveBtnStyle}>Undo</button>
            <button className="draw-tool-btn" onClick={handleRedo} style={inactiveBtnStyle}>Redo</button>
            <button className="draw-tool-btn" onClick={handleClear} style={inactiveBtnStyle}>Clear</button>
          </div>

          <button className="draw-add-btn" onClick={addToStory}>ADD TO STORY</button>
        </div>

        {/* RIGHT: Frames & actions */}
        <div className="draw-sidebar">
          <div className="draw-frames">
            {Array.from({ length: 3 }).map((_, i) => {
              const f = frames[i];
              return (
                <div
                  key={f?.id ?? `slot-${i}`}
                  className={cx('draw-frame-slot', f ? 'has-image' : '')}
                  draggable={Boolean(f)}
                  onDragStart={() => dragStart(i, Boolean(f))}
                  onDragOver={dragOver}
                  onDrop={(ev) => drop(ev, i)}
                >
                  {f ? (
                    <>
                      <img src={f.url} alt={`Storyboard ${i + 1}`} />
                      <button className="draw-frame-remove" onClick={() => removeFrame(f.id)}>×</button>
                    </>
                  ) : (
                    <span className="draw-frame-placeholder">Frame {i + 1}</span>
                  )}
                </div>
              );
            })}
          </div>

          <button className="draw-create-btn" onClick={tryExample}>TRY EXAMPLE</button>
          <button
            className="draw-create-btn"
            disabled={frames.length !== 3}
            onClick={createStory}
          >
            CREATE STORY
          </button>
        </div>
      </div>

      <Toast />
    </section>
  </div>
);
}
function StoryPage(){
  const { state } = useLocation();
  const navigate = useNavigate();
  const { refetchMe } = useAuth();
  const { show, Toast } = useToast();

  // If navigated from Draw: we get images+blobs; if opened from Saved list: we get preload
  const preload = state?.preload || null;
  const initialImages = preload?.images || state?.images || [];
  const blobs = state?.blobs || null;

  // Keep images in state so we can swap blob: URLs with server URLs from /api/story (durable)
  const [images, setImages] = React.useState(initialImages);
  const [title, setTitle] = React.useState(preload?.title || "");
  const [panels, setPanels] = React.useState(preload?.panels || ["", "", ""]);
  const [loading, setLoading] = React.useState(!preload);
  const [editing, setEditing] = React.useState(false);

  // Dictionary modal state
  const [dictWord, setDictWord] = React.useState("");
  const [dictData, setDictData] = React.useState(null);
  const [dictLoading, setDictLoading] = React.useState(false);
  const [dictErr, setDictErr] = React.useState("");
  const dictCache = React.useRef(new Map());


  // ----- TTS highlighting (same model as "What’s This?")
  const [storyFocusIndex, setStoryFocusIndex] = React.useState(-1);
  const ttsRef = React.useRef(null); // <audio> element

  // Split title + each panel into tokens (keep spaces to align timing)
  const segWords = React.useMemo(() => {
    const segs = [String(title || ""), ...panels.map(p => String(p || ""))];
    return segs.map(s => s.split(/(\s+)/));
  }, [title, panels]);

  // Cumulative offsets → global index for each token
  const segOffsets = React.useMemo(() => {
    let acc = 0;
    return segWords.map(ws => { const off = acc; acc += ws.length; return off; });
  }, [segWords]);

  const flatWords = React.useMemo(() => segWords.flat(), [segWords]);

  async function speakStory() {
  const fullText = [title, ...panels].filter(Boolean).join(" ").trim();
  if (!fullText) return;

  try {
    const gs =
      typeof window !== "undefined" ? window.__picteractive_settings || {} : {};

    // Normalise rate from settings
    const rate = (() => {
      const rv =
        gs && typeof gs.speaking_rate !== "undefined" ? gs.speaking_rate : 1.0;
      if (typeof rv === "number" && isFinite(rv) && rv > 0) return rv;
      const n = Number(rv);
      return isFinite(n) && n > 0 ? n : 1.0;
    })();

    const highlightOn =
      typeof document !== "undefined"
        ? document.documentElement.getAttribute("data-tts-highlight") !== "0"
        : true;

    const a = ttsRef.current;

    // Ask apiTTS for Web Speech or backend audio
    const src = await apiTTS(fullText, {
      voice: gs.tts_voice,
      rate: rate,
    });

    // --- WEB SPEECH PATH (src === null) ---
    if (!src) {
      if (!highlightOn) return;

      const nonSpace = flatWords.filter((w) => /\w/.test(w)).length || 1;
      const estDuration = nonSpace * 0.30;
      const baseStep = estDuration / nonSpace;
      const step = Math.max(0.15, Math.min(0.10, baseStep / rate));

      setStoryFocusIndex(-1);
      let i = 0;
      const timer = setInterval(() => {
        if (i >= flatWords.length) {
          clearInterval(timer);
          setStoryFocusIndex(-1);
          return;
        }
        if (/\w/.test(flatWords[i])) setStoryFocusIndex(i);
        i++;
      }, step * 1000);

      return;
    }

    // --- FALLBACK AUDIO ELEMENT PATH (when Web Speech not available) ---
    if (!a) return;

    a.src = src;
    a.playbackRate = rate;

    if (!highlightOn) {
      a.play().catch(() => {});
      return;
    }

    const clean = () => {
      a.removeEventListener("timeupdate", onTime);
      a.removeEventListener("ended", onEnd);
      a.removeEventListener("loadedmetadata", onMeta);
      a.removeEventListener("canplaythrough", onMeta);
    };

    const onEnd = () => {
      setStoryFocusIndex(-1);
      clean();
    };

    const onTime = () => {
      if (!a.duration || !isFinite(a.duration) || a.duration <= 0) return;
      const ratio = Math.min(1, Math.max(0, a.currentTime / a.duration));
      let i = Math.floor(ratio * flatWords.length);
      while (i < flatWords.length && !/\w/.test(flatWords[i])) i++;
      setStoryFocusIndex(i >= flatWords.length ? flatWords.length - 1 : i);
    };

    const onMeta = () => {
      a.removeEventListener("loadedmetadata", onMeta);
      a.removeEventListener("canplaythrough", onMeta);
      a.addEventListener("timeupdate", onTime);
      a.addEventListener("ended", onEnd, { once: true });
    };

    a.addEventListener("loadedmetadata", onMeta);
    a.addEventListener("canplaythrough", onMeta);
    a.addEventListener("timeupdate", onTime);
    a.addEventListener("ended", onEnd, { once: true });

    a.play().catch(() => {});
  } catch {
    show("TTS failed");
  }
}




  // React to settings changes (e.g., voice/rate) mid-session
  React.useEffect(() => {
    const onApply = (e) => {
      // No local state to sync right now; apiTTS reads globals.
      // This ensures the page picks up changes without reload.
    };
    window.addEventListener('picteractive:settings-applied', onApply);
    return () => window.removeEventListener('picteractive:settings-applied', onApply);
  }, []);

  React.useEffect(() => () => setStoryFocusIndex(-1), []);

  // Normalize relative server paths like "/files/..." → absolute URL
  function norm(u){
    if (!u) return "";
    if (u.startsWith("http") || u.startsWith("blob:") || u.startsWith("data:")) return u;
    return `${API}${u.startsWith("/") ? u : "/" + u}`;
  }

  // Build the story from blobs if we were sent here from Draw
  React.useEffect(() => {
    if (preload) return;                 // opening a saved story: nothing to generate
    if (!blobs || blobs.length !== 3) {  // safety
      navigate("/draw");
      return;
    }
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        // Keep current images visible—do NOT clear them while generating
        const data = await apiStory(blobs); // -> { title, panels, images? }
        if (!alive) return;

        setTitle(data.title || "STORY TIME!");
        const p = (Array.isArray(data.panels) && data.panels.length === 3)
          ? data.panels
          : (data.story
             ? String(data.story).split(/\n+/).slice(0,3)
             : ["A fun beginning.", "An exciting middle.", "A happy ending."]);
        setPanels(p);

        // Only swap to server URLs when valid
        if (Array.isArray(data.images) && data.images.length === 3) {
          setImages(data.images.map(norm));
        }

        try { await apiAchEvent("story"); await refetchMe(); } catch {}
      } catch (e) {
        show(e.message || "Story failed");
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, []);
  // Open dictionary helper

  function closeDictionary() {
  setDictWord("");
  setDictErr("");
  setDictData(null);
  setDictLoading(false);
}

  async function openDictionary(rawWord) {
  const cleaned = (rawWord || "").replace(/^[^A-Za-z0-9']+|[^A-Za-z0-9']+$/g, "");
  if (!cleaned) return;
  const key = cleaned.toLowerCase();
  setDictWord(cleaned);
  setDictErr("");

  // cache lookup
  const cached = dictCache.current.get(key);
  if (cached) {
    setDictData(cached);
    setDictLoading(false);
    return;
  }
  
  setDictData(null);
  setDictLoading(true);
  try {
    const data = await apiDict(key);
    dictCache.current.set(key, data);
    setDictData(data);
  } catch (e) {
    setDictErr(e?.message || "Unable to load definition.");
  } finally {
    setDictLoading(false);
  }}

// Save title + (edited) panel texts + the 3 panel images to a PNG
async function saveStoryAsImage(){
  // 1) Use CURRENT STATE first so edited text is captured
  const titleNow  = (title || "").trim() || "My Story";
  const panelsNow = Array.isArray(panels) ? panels.slice(0,3) : [];
  while (panelsNow.length < 3) panelsNow.push("");

  // 2) Load panel images from state (fallback to DOM if needed)
  const urls = Array.isArray(images) && images.length ? images.slice(0,3) : [];
  function loadImage(srcOrImg){
    return new Promise((resolve)=>{
      if (srcOrImg instanceof HTMLImageElement){
        if (srcOrImg.complete && srcOrImg.naturalWidth>0) return resolve(srcOrImg);
        srcOrImg.addEventListener('load', ()=>resolve(srcOrImg), {once:true});
        srcOrImg.addEventListener('error', ()=>resolve(null), {once:true});
        return;
      }
      const im = new Image();
      if (typeof srcOrImg === 'string' && /^https?:/i.test(srcOrImg)) im.crossOrigin = 'anonymous';
      im.src = typeof srcOrImg === 'string' ? srcOrImg : '';
      im.onload = ()=>resolve(im);
      im.onerror = ()=>resolve(null);
    });
  }

  // Try state images; if missing, look in the DOM (left thumbs)
  let panelImgs = await Promise.all(urls.map(loadImage));
  if (!panelImgs.filter(Boolean).length){
    const domImgs = Array.from(document.querySelectorAll(
      '.story-comic-panel img, .story-panel img, .story-thumb img, .panel img, .panel-thumb img'
    )).slice(0,3);
    panelImgs = await Promise.all(domImgs.map(loadImage));
  }
  while (panelImgs.length < 3) panelImgs.push(null);

  // Logo
  const logoImg = await loadImage(siteLogo);

    // --- Layout (export image only) ---
    // Aim for a roughly 1080-wide square-friendly image. Height is dynamic
    // but usually close to 1080px for 3 panels.
    const W = 1080, P = 40, cardR = 34, rowGap = 34, pillH = 90;
    const titleSize = 60, bodySize = 30;
    // Saved-image thumbnail (drawing panel) sizing: make it larger than on-screen,
    // with a thinner black frame.
    const thumbBox = 260, thumbPad = 24, thumbBorder = 6;
    const panelHeight = Math.max(thumbBox + thumbPad*2, 260);

  const measure = document.createElement('canvas').getContext('2d');
  const fontTitle = `900 ${titleSize}px "League Spartan", system-ui, Arial`;
  const fontBody  = `700 ${bodySize}px "League Spartan", system-ui, Arial`;
  measure.canvas.width = W;

  const MAX_W = Math.floor(W * 0.78);
  function wrap(text, font, maxW){
    measure.font = font;
    const words = String(text||'').split(/\s+/);
    const rows = [];
    let line = '';
    for (const w of words){
      const next = line ? line + ' ' + w : w;
      if (measure.measureText(next).width > maxW){
        if (line) rows.push(line);
        line = w;
      } else line = next;
    }
    if (line) rows.push(line);
    return rows;
  }

  const titleRows = wrap(titleNow, fontTitle, W - P*2 - 60);
  const panelRows = panelsNow.map(t => wrap(t, fontBody, W - P*2 - (thumbBox + thumbPad*2 + 40)));

  let H = P + pillH + 30;
  for (let i=0;i<3;i++){
    const textH = (panelRows[i]?.length || 1) * (bodySize + 10);
    H += Math.max(panelHeight, textH + thumbPad*2) + rowGap;
  }
  H += pillH + P;

  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  const c = canvas.getContext('2d');

  function roundRect(x,y,w,h,r){
    c.beginPath();
    c.moveTo(x+r,y);
    c.arcTo(x+w,y,x+w,y+h,r);
    c.arcTo(x+w,y+h,x,y+h,r);
    c.arcTo(x,y+h,x,y,r);
    c.arcTo(x,y,x+w,y,r);
    c.closePath();
  }

  // forest bg + orange card
  c.fillStyle = '#fb4c00'; c.fillRect(0,0,W,H);
  c.fillStyle = '#fb4c00'; roundRect(24,24,W-48,H-48,cardR); c.fill();

  // Title pill
  const innerX = P, innerW = W - P*2;
  let y = P;
  c.fillStyle = '#fff'; roundRect(innerX,y,innerW,pillH,40); c.fill();
  c.font = fontTitle; c.fillStyle = '#004aad'; c.textAlign = 'center'; c.textBaseline = 'middle';
  c.fillText(titleRows.join(' '), W/2, y + pillH/2);
  y += pillH + 30;

  // Panels
    for (let i=0;i<3;i++){
      c.fillStyle = '#fff'; roundRect(innerX,y,innerW,panelHeight,26); c.fill();
  
      // thumb frame
      const tX = innerX + 24;
      const tY = y + (panelHeight - (thumbBox + thumbBorder*2)) / 2;
      c.fillStyle = '#ffffff'; roundRect(tX, tY, thumbBox + thumbBorder*2, thumbBox + thumbBorder*2, 22); c.fill();
      c.lineWidth = thumbBorder; c.strokeStyle = '#000000';
      roundRect(tX + thumbBorder/2, tY + thumbBorder/2, thumbBox + thumbBorder, thumbBox + thumbBorder, 22); c.stroke();
  
      const im = panelImgs[i];
      if (im){
        const box = thumbBox - thumbBorder;
        const scale = Math.min(box / im.width, box / im.height);
        const dw = im.width * scale, dh = im.height * scale;
        const drawX = tX + thumbBorder + (box - dw)/2;
        const drawY = tY + thumbBorder + (box - dh)/2;
        c.save(); roundRect(tX + thumbBorder, tY + thumbBorder, box, box, 18); c.clip();
        c.drawImage(im, drawX, drawY, dw, dh); c.restore();
      }
  
      // text
      const textLeft = tX + thumbBorder*2 + thumbBox + 32;
      c.font = fontBody; c.fillStyle = '#000'; c.textAlign = 'left'; c.textBaseline = 'top';
      const rows = panelRows[i] || [''];
      let ty = y + (panelHeight - (rows.length * (bodySize + 10))) / 2;
      for (const r of rows){ c.fillText(r, textLeft, ty); ty += (bodySize + 10); }
  
      y += panelHeight + rowGap;
    }

  // Footer pill + logo + text
  c.fillStyle = '#fff'; roundRect(innerX,y,innerW,pillH,40); c.fill();
  if (logoImg){
    const pad = 26, s = pillH - pad*2, lx = innerX + pad, ly = y + pad;
    c.save(); c.beginPath(); c.arc(lx + s/2, ly + s/2, s/2, 0, Math.PI*2); c.closePath(); c.clip();
    c.drawImage(logoImg, lx, ly, s, s); c.restore();
  }
  c.font = `900 ${Math.floor(pillH*0.36)}px "League Spartan", system-ui, Arial`;
  c.fillStyle = '#000'; c.textAlign = 'left'; c.textBaseline = 'middle';
  c.fillText('Created with PICTERACTIVE.COM', innerX + pillH + 12, y + pillH/2);

  // Download
  const a = document.createElement('a');
  a.href = canvas.toDataURL('image/png');
  const safe = titleNow.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/(^-|-$)/g,'');
  a.download = (safe || 'story') + '.png';
  document.body.appendChild(a); a.click(); a.remove();
}


  return (
    <div className="min-h-screen" style={{ background:'var(--forest)' }}>
      <NavBar />
      <section className="story-stage">
        <button className="story-back" onClick={()=>navigate('/draw', { state: { refresh: Date.now() } })} aria-label="Back"><img src={backIcon} alt="Back"/></button>
        <h1 className="story-heading">STORY TIME!</h1>

        {/* hidden audio element used by speakStory() */}
        <audio ref={ttsRef} preload="auto" style={{ display: "none" }} />

        {/* Title (highlightable + dictionary-clickable) */}
        <div className="story-comic">
          <div className="story-comic-title">
            {editing ? (
              <input
                className="story-title-input"
                value={title}
                onChange={(e)=>setTitle(e.target.value)}
                placeholder="Story title"
              />
            ) : (
              <h2 className="story-title">
                {segWords[0].map((w, j) => {
                  const trimmed = w.replace(/^[^A-Za-z0-9']+|[^A-Za-z0-9']+$/g, "");
                  const clickable = Boolean(trimmed);
                  const idx = segOffsets[0] + j;
                  const onKey = (ev) => {
                    if (!clickable) return;
                    if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); openDictionary(trimmed); }
                  };
                  return (
                    <span
                      key={`title-${j}`}
                      className={cx("word", clickable && "clickable", idx === storyFocusIndex && "focus")}
                      onClick={() => clickable && openDictionary(trimmed)}
                      onKeyDown={onKey}
                      role={clickable ? "button" : undefined}
                      tabIndex={clickable ? 0 : undefined}
                    >
                      {w}
                    </span>
                  );
                })}
              </h2>
            )}
          </div>

          {/* Panels (each word as a span; highlight index = segOffsets[i+1] + j) */}
          <div className="story-comic-body">
            {[0,1,2].map(i => (
              <div key={i} className="story-comic-row">
                <div className="story-comic-image">
                  {images[i]
                    ? <img src={norm(images[i])} alt={`Panel ${i+1}`} />
                    : <div className="story-image-placeholder">Panel {i+1}</div>}
                </div>

                <div className="story-comic-text">
                  {loading ? (
                    "Generating..."
                  ) : editing ? (
                    <textarea
                      className="story-panel-input"
                      rows={3}
                      value={panels[i] || ""}
                      onChange={(e)=>{
                        const next = [...panels];
                        next[i] = e.target.value;
                        setPanels(next);
                      }}
                      placeholder={`Panel ${i+1} text`}
                    />
                  ) : (
                    segWords[i + 1].map((w, j) => {
                      const trimmed = w.replace(/^[^A-Za-z0-9']+|[^A-Za-z0-9']+$/g, "");
                      const clickable = Boolean(trimmed);
                      const idx = segOffsets[i + 1] + j;
                      const onKey = (ev) => {
                        if (!clickable) return;
                        if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); openDictionary(trimmed); }
                      };
                      return (
                        <span
                          key={`p${i}-w${j}`}
                          className={cx("word", clickable && "clickable", idx === storyFocusIndex && "focus")}
                          onClick={() => clickable && openDictionary(trimmed)}
                          onKeyDown={onKey}
                          role={clickable ? "button" : undefined}
                          tabIndex={clickable ? 0 : undefined}
                        >
                          {w}
                        </span>
                      );
                    })
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div className="story-actions">
          <button type="button" className="tts-button" onClick={speakStory}>
            <img src={speakIcon} alt="" aria-hidden="true" />
            <span>TEXT-TO-SPEECH</span>
          </button>
          <button
          type="button"
          className="btn-orange save-btn"
          onClick={saveStoryAsImage}
          title="Save story as image">
            <img src={saveIcon} alt="" aria-hidden="true" style={{width:18,height:18}} />
            SAVE</button>
          <button
            type="button"
            className="btn-orange edit-btn"
            onClick={()=>setEditing(e=>!e)}
            style={{ marginLeft: 12 }}
          >
            {!editing && <img src={editIcon} alt="" aria-hidden="true" style={{width:18,height:18}} />}
            {editing ? "DONE EDITING" : "EDIT TEXT"}
          </button>
        </div>

        {/* Dictionary modal */}
        <DictionaryModal
        word={dictWord}
        data={dictData}
        loading={dictLoading}
        error={dictErr}
        onClose={closeDictionary}
        />
        <Toast />
      </section>
    </div>
  );
}




function Placeholder(){ return (<div className="min-h-screen"><NavBar /><section className="hero" style={{minHeight:'calc(100vh - 96px)'}}><div className="hero-text">Coming soon</div></section></div>); }


function RegionSelector({ src, onConfirm, onCancel }){
  const imgRef = React.useRef(null);
  const [drag,setDrag]=React.useState(null); const [box,setBox]=React.useState(null);
  function start(ev){ const img=imgRef.current; if(!img) return; const r=img.getBoundingClientRect(); const p=('touches' in ev? ev.touches[0]:ev); setDrag({r,x0:p.clientX-r.left,y0:p.clientY-r.top}); }
  function move(ev){ if(!drag) return; const p=('touches' in ev? ev.touches[0]:ev); const {r,x0,y0}=drag; const x=Math.min(Math.max(p.clientX-r.left,0),r.width); const y=Math.min(Math.max(p.clientY-r.top,0),r.height); setBox({left:Math.min(x0,x),top:Math.min(y0,y),w:Math.abs(x-x0),h:Math.abs(y-y0)}); }
  function end(){ if(!drag||!box){ setDrag(null); setBox(null); return;} const img=imgRef.current; const {r}=drag; const sx=img.naturalWidth/r.width, sy=img.naturalHeight/r.height; onConfirm({x:Math.round(box.left*sx),y:Math.round(box.top*sy),w:Math.round(box.w*sx),h:Math.round(box.h*sy)}); setDrag(null); setBox(null); }
  React.useEffect(()=>{ if(!drag) return; const mm=e=>move(e), up=()=>end(); window.addEventListener('mousemove',mm); window.addEventListener('mouseup',up); window.addEventListener('touchmove',mm,{passive:false}); window.addEventListener('touchend',up); return ()=>{ window.removeEventListener('mousemove',mm); window.removeEventListener('mouseup',up); window.removeEventListener('touchmove',mm); window.removeEventListener('touchend',up); }; },[drag,box]);
  return (<div className="modal"><div className="modal-card"><div className="flex items-center justify-between mb-3"><b>Region Select</b><button className="btn btn-plain" onClick={onCancel}>Cancel</button></div><div className="relative" onMouseDown={start} onTouchStart={start}><img ref={imgRef} src={src} alt="select" draggable={false}/>{drag&&box?(<div className="crop-mask" style={{ left: drag.r.left+box.left, top: drag.r.top+box.top, width: box.w, height: box.h }}/>):null}</div></div></div>);
}


function WhatsThisV2(){
  
  const [imgUrl,setImgUrl]=React.useState("");
  const [imgBlob,setImgBlob]=React.useState(null);
  const [origBlob,setOrigBlob]=React.useState(null); // original file
  const [origUrl,setOrigUrl]=React.useState("");     // original object URL
  const [cvdBase,setCvdBase]=React.useState(null);   // unfiltered base used for CVD
  const [isCropped,setIsCropped]=React.useState(false);
  const [caption,setCaption]=React.useState("");
  const [regionOpen,setRegionOpen]=React.useState(false);
  const [loading,setLoading]=React.useState(false);
  const [focusIndex,setFocusIndex]=React.useState(-1);
  const dictCache=React.useRef(new Map());
  const [dictWord,setDictWord]=React.useState("");
  const [dictData,setDictData]=React.useState(null);
  const [dictLoading,setDictLoading]=React.useState(false);
  const [dictError,setDictError]=React.useState("");
  const [dictOpen,setDictOpen]=React.useState(false);
  
  const navigate=useNavigate(); const { show, Toast } = useToast(); const { refetchMe } = useAuth();
  // Colour-blind controls
  const [cbCond,setCbCond]=React.useState('deuteranomaly');
  const [cbSeverity,setCbSeverity]=React.useState(50); // 0..100
  const [cbMode,setCbMode]=React.useState('simulate'); // simulate | enhance
  const [cbSplit,setCbSplit]=React.useState(false);
  const [cbPreview,setCbPreview]=React.useState('');
  const fileRef   = React.useRef(null);
  const videoRef  = React.useRef(null);
  const canvasRef = React.useRef(null);
  const streamRef = React.useRef(null);
  const audioRef  = React.useRef(null);
  const [cameraOn, setCameraOn] = React.useState(false);


  React.useEffect(()=>()=>{ streamRef.current?.getTracks().forEach(t=>t.stop()); streamRef.current = null; },[]);
  function openPicker(){ fileRef.current?.click(); }
  function onFile(e){const f=e.target.files?.[0];if(!f)return;const o=URL.createObjectURL(f);setOrigBlob(f);setOrigUrl(o);setCvdBase(f);setIsCropped(false);setImgBlob(f);setImgUrl(o);stopCamera();doCaption();}
  
  

async function startCamera() {
  try {
    // reset caption + image state
    setCaption("");
    setImgUrl("");
    setOrigUrl("");
    setCbPreview("");
    setImgBlob(null);
    setCvdBase(null);
    setIsCropped(false);
    setFocusIndex(-1);

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    show("Camera is not supported in this browser.");
    setCameraOn(false);
    return;
  }

  // stop any previous stream
  if (streamRef.current) {
    streamRef.current.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }

  // request camera
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" }, // or { ideal: "environment" } to prefer back camera
    audio: false,
  });

  // store the stream, hook to <video>, then turn camera on
  streamRef.current = stream;

  if (videoRef.current) {
    videoRef.current.srcObject = stream;
    try { await videoRef.current.play(); } catch {}
  }

  setCameraOn(true);
} catch (err) {
  console.error("camera error", err);
  show(
    err && err.name === "NotAllowedError"
      ? "Camera permission was blocked."
      : "Failed to start camera."
  );
  setCameraOn(false);
  if (streamRef.current) {
    streamRef.current.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }
}
}


  function stopCamera() {
  const stream = streamRef.current;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }
  setCameraOn(false);
  if (videoRef.current) {
    videoRef.current.srcObject = null;
  }
}


  // SNAP CURRENT FRAME INTO HIDDEN CANVAS → BLOB
  async function captureFrame() {
  if (!videoRef.current || !canvasRef.current || !streamRef.current) {
    return;
  }

  const videoEl = videoRef.current;
  const canvasEl = canvasRef.current;
  const ctx = canvasEl.getContext("2d");

  canvasEl.width = videoEl.videoWidth || 640;
  canvasEl.height = videoEl.videoHeight || 480;

  ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);

  canvasEl.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    setImgBlob(blob);
    setImgUrl(url);
    setOrigUrl(url);
    setCvdBase(blob);
    setIsCropped(false);
    setCameraOn(false); // stop showing live video after snap
    stopCamera();
  }, "image/jpeg", 0.92);
}


  async function doCaption(region=null, blobOverride=null){ const blob=blobOverride||imgBlob; if(!blob){ show('Please upload or capture an image first'); return; } closeDictionary(); setLoading(true); setCaption(''); setFocusIndex(-1); try{ const text=await apiCaption(blob, region); const clean=String(text||'').trim(); const out = clean && clean.toLowerCase()!=='none'? clean : ''; setCaption(out); if(out) originalCaptionRef.current = out; if(out){ await apiAchEvent('caption'); await refetchMe().catch(()=>{}); } if(region) setIsCropped(true);} catch(e){ show(e.message||'Caption failed'); } finally{ setLoading(false);} }  
  async function speak() {
  const text = caption.trim();
  if (!text) return;

  try {
    const hasWindow = typeof window !== "undefined";
    const gs = hasWindow ? (window.__picteractive_settings || {}) : {};

    // Normalise rate from settings
    const rate = (() => {
      const rv =
        gs && typeof gs.speaking_rate !== "undefined" ? gs.speaking_rate : 1.0;
      if (typeof rv === "number" && isFinite(rv) && rv > 0) return rv;
      const n = Number(rv);
      return isFinite(n) && n > 0 ? n : 1.0;
    })();

    // Check if word highlight is enabled (Settings → Word highlight)
    const highlightOn =
      typeof document !== "undefined"
        ? document.documentElement.getAttribute("data-tts-highlight") !== "0"
        : true;

    // Ask apiTTS to either:
    //  - speak via Web Speech (returns null), OR
    //  - give us an audio src (fallback path)
    const src = await apiTTS(text, {
      voice: gs.tts_voice,
      rate: rate,
    });

    // --- WEB SPEECH PATH (src === null) ---
    // Web Speech is already speaking; we just drive a timer for highlights.
    if (!src) {
      if (!highlightOn) return;

      const nonSpace = words.filter((w) => /\w/.test(w)).length || 1;
      // Rough estimate: ~0.35s per word at rate 1.0
      const estDuration = nonSpace * 0.30;
      const baseStep = estDuration / nonSpace;
      const step = Math.max(0.15, Math.min(0.10, baseStep / rate)); // seconds per word

      setFocusIndex(-1);
      let i = 0;
      const timer = setInterval(() => {
        if (i >= words.length) {
          clearInterval(timer);
          setFocusIndex(-1);
          return;
        }
        if (/\w/.test(words[i])) setFocusIndex(i);
        i++;
      }, step * 1000);

      return;
    }

    // --- FALLBACK AUDIO ELEMENT PATH (old behaviour, when Web Speech not available) ---
    const a = audioRef.current;
    if (!a) return;

    a.src = src;
    a.playbackRate = rate;

    // If highlight is off, just play the audio
    if (!highlightOn) {
      a.play().catch(() => {});
      return;
    }

    setFocusIndex(-1);

    let timer;
    const onReady = () => {
      const nonSpace = words.filter((w) => /\w/.test(w)).length || 1;
      const duration =
        a.duration && isFinite(a.duration) && a.duration > 0
          ? a.duration
          : nonSpace * 0.35;
      const baseStep = duration / nonSpace;
      const step = Math.max(0.2, Math.min(0.8, baseStep / rate));

      let i = 0;
      timer = setInterval(() => {
        if (i >= words.length) {
          clearInterval(timer);
          setFocusIndex(-1);
          return;
        }
        if (/\w/.test(words[i])) setFocusIndex(i);
        i++;
      }, step * 1000);
    };

    const onEnd = () => {
      if (timer) clearInterval(timer);
      setFocusIndex(-1);
      a.removeEventListener("canplaythrough", onReady);
      a.removeEventListener("ended", onEnd);
    };

    a.addEventListener("canplaythrough", onReady, { once: true });
    a.addEventListener("ended", onEnd, { once: true });

    a.play().catch(() => {});
  } catch {
    show("TTS failed");
  }
}



  function clearAll(){ setImgUrl(""); setImgBlob(null); setOrigBlob(null); setCvdBase(null); setIsCropped(false); setCaption(""); stopCamera(); setFocusIndex(-1); closeDictionary(); }
  function revertCrop(){ if(origBlob){ setImgBlob(origBlob); setImgUrl(URL.createObjectURL(origBlob)); setCvdBase(origBlob); setIsCropped(false); } }

  const words = caption.split(/(\s+)/);
  const hasImage = Boolean(imgUrl || cameraOn);
  const captionReady = caption.trim().length>0;

  // React to settings changes (e.g., voice/rate) mid-session
  React.useEffect(() => {
    const onApply = (e) => {
      // Optional: update any local preview state from e.detail.settings
    };
    window.addEventListener('picteractive:settings-applied', onApply);
    return () => window.removeEventListener('picteractive:settings-applied', onApply);
  }, []);

  React.useEffect(() => {
  // When cameraOn becomes true and we have a stream + video element,
  // hook them together.
  if (cameraOn && videoRef.current && streamRef.current) {
    const videoEl = videoRef.current;
    videoEl.srcObject = streamRef.current;
    videoEl.play().catch(() => {});
  }
}, [cameraOn]);


  // ----- Colour-blindness preview/apply helpers -----
  // Map UI conditions to server cvd_type values expected by /api/cvd/apply
  // Server expects: "protanopia" | "deuteranopia" | "tritanopia"
  const CB_MAP = {
    'protanomaly': { type: 'protanopia' },
    'protanopia': { type: 'protanopia' },
    'deuteranomaly': { type: 'deuteranopia' },
    'deuteranopia': { type: 'deuteranopia' },
    'tritanomaly': { type: 'tritanopia' },
    'tritanopia': { type: 'tritanopia' },
    'achromatomaly': { type: 'mono' },
    'achromatopsia': { type: 'mono' },
    'blue-cone monochromacy': { type: 'mono' },
  };
  const severity01 = (v)=> Math.max(0, Math.min(1, (Number(v)||0)/100));
  async function toMonochromeBlob(blob){ const bm = await createImageBitmap(blob); const c=document.createElement('canvas'); c.width=bm.width; c.height=bm.height; const x=c.getContext('2d'); x.drawImage(bm,0,0); const im=x.getImageData(0,0,c.width,c.height); const d=im.data; for(let i=0;i<d.length;i+=4){ const g=0.2126*d[i]+0.7152*d[i+1]+0.0722*d[i+2]; d[i]=d[i+1]=d[i+2]=g; } x.putImageData(im,0,0); return await new Promise(res=>c.toBlob(b=>res(b),'image/png',0.92)); }
  async function makePreview(){ if(!cvdBase || !cbSplit){ setCbPreview(''); return; } const entry=CB_MAP[cbCond]||{type:'deuteranopia'}; if(entry.type==='mono'){ const b=await toMonochromeBlob(cvdBase); setCbPreview(URL.createObjectURL(b)); return; } const fd=new FormData(); fd.append('image', cvdBase,'img.png'); fd.append('mode', cbMode==='enhance'?'daltonize':'simulate'); fd.append('cvd_type', entry.type); fd.append('severity', String(severity01(cbSeverity))); fd.append('amount','1.0'); const r=await fetch(`${API}/api/cvd/apply`,{method:'POST', body: fd}); if(!r.ok){ setCbPreview(''); return;} const b=await r.blob(); setCbPreview(URL.createObjectURL(b)); }
  React.useEffect(()=>{ makePreview(); }, [cvdBase, cbCond, cbSeverity, cbMode, cbSplit]);
  async function applyCVD(){ if(!cvdBase) return; const entry=CB_MAP[cbCond]||{type:'deuteranopia'}; if(entry.type==='mono'){ const b=await toMonochromeBlob(cvdBase); setImgBlob(b); setImgUrl(URL.createObjectURL(b)); return; } const fd=new FormData(); fd.append('image', cvdBase,'img.png'); fd.append('mode', cbMode==='enhance'?'daltonize':'simulate'); fd.append('cvd_type', entry.type); fd.append('severity', String(severity01(cbSeverity))); fd.append('amount','1.0'); const r=await fetch(`${API}/api/cvd/apply`,{method:'POST', body: fd}); if(!r.ok) return; const b=await r.blob(); setImgBlob(b); setImgUrl(URL.createObjectURL(b)); }
  function handleReset(){setImgBlob(origBlob);setImgUrl(origUrl);setCbSplit(false);}


  function closeDictionary(){
    setDictOpen(false);
    setDictWord("");
    setDictData(null);
    setDictError("");
    setDictLoading(false);
  }

  // --- Add these hooks near the top of your component ---
const [transLang, setTransLang] = React.useState('zh');
const [translating, setTranslating] = React.useState(false);
const [dropdownOpen, setDropdownOpen] = React.useState(false);
const originalCaptionRef = React.useRef("");

// originalCaptionRef is updated when a new caption is generated in doCaption()


// --- Place this helper above your return() ---
async function doTranslate(langCode){
  // Revert to original English instantly
  if (langCode === 'en') {
    if (originalCaptionRef.current) {
      setCaption(originalCaptionRef.current);
      setFocusIndex(-1);
      show('Reverted to original English caption!');
    }
    setDropdownOpen(false);
    return;
  }

  try {
    setDropdownOpen(false);
    setTranslating(true);
    const base = originalCaptionRef.current || caption; // always prefer original English
    const t = await apiTranslate(base, langCode);
    setCaption(t);
    setFocusIndex(-1);
    show('Translated!');
  } catch (e) {
    show(e.message || 'Translate failed');
  } finally {
    setTranslating(false);
  }
}




  async function openDictionary(rawWord){
    const cleaned = rawWord.replace(/^[^A-Za-z0-9']+|[^A-Za-z0-9']+$/g,"");
    if(!cleaned) return;
    const key = cleaned.toLowerCase();
    setDictWord(cleaned);
    setDictOpen(true);
    setDictError("");
    const cached = dictCache.current.get(key);
    if(cached){
      setDictData(cached);
      setDictLoading(false);
      return;
    }
    setDictData(null);
    setDictLoading(true);
    try{
      const data = await apiDict(key);
      dictCache.current.set(key, data);
      setDictData(data);
    } catch (e){
      setDictError(e?.message || 'Unable to load definition.');
    } finally{
      setDictLoading(false);
    }
  }

    return (
    <div className="min-h-screen" style={{ background: 'var(--forest)' }}>
      <NavBar />
      <div className="wt-wrap">
        <div className="wt-title">IMAGE DESCRIPTION GENERATION</div>
        <div className="wt-row">
          <div className="wt-card">
            {!hasImage && (
              <div className="wt-actions">
                <button className="btn-orange" onClick={openPicker}>
                  <img src={uploadIcon} alt="Upload" width="22" height="22" /> UPLOAD IMAGE
                </button>
                <input
                  ref={fileRef}
                  type="file"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={onFile}
                />
                <button className="btn-orange" onClick={startCamera}>
                  <img src={cameraIcon} alt="Camera" width="22" height="22" /> TAKE A PHOTO
                </button>
              </div>
            )}

            {hasImage && (
              <div>
                <div className="wt-stage">
                  {/* Hidden canvas used only for capturing camera frames */}
                  <canvas
                    ref={canvasRef}
                    width={640}
                    height={480}
                    style={{ display: 'none' }}
                  />

                  {imgUrl ? (
                    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                      {!cbSplit ? (
                        <img
                          src={imgUrl}
                          alt="preview"
                          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                        />
                      ) : (
                        <>
                          {/* LEFT = original always */}
                          <img
                            src={origUrl || imgUrl}
                            alt="original"
                            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                          />
                          {/* RIGHT = filtered preview */}
                          <img
                            src={cbPreview || origUrl || imgUrl}
                            alt="cvd"
                            style={{
                              position: 'absolute',
                              inset: 0,
                              width: '100%',
                              height: '100%',
                              objectFit: 'contain',
                              clipPath: 'inset(0 0 0 50%)',
                            }}
                          />
                        </>
                      )}
                    </div>
                  ) : cameraOn ? (
                    <video ref={videoRef} autoPlay muted playsInline style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                  ) : null}
                </div>

                {cameraOn && !imgUrl && (
                  <div className="mt-3 grid place-items-center">
                    <button className="btn-orange" onClick={captureFrame}>
                      SNAP
                    </button>
                    <button className="btn btn-plain mt-2" onClick={stopCamera}>
                      Close Camera
                    </button>
                  </div>
                )}

                {isCropped && (
                  <div className="mt-2 text-center">
                    <button className="btn btn-plain" onClick={revertCrop}>
                      Revert to original
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Generate description button(s) */}
        {!hasImage && (
          <>
            <div className="wt-readout" style={{ marginTop: 12 }}>
              ADD IMAGE TO GENERATE DESCRIPTION
            </div>
            <div className="wt-generate" style={{ marginTop: 16 }}>
              <button
                className="btn-orange"
                onClick={() => doCaption()}
                disabled={!imgBlob || loading}
              >
                <img src={sparklesIcon} alt="Generate" width="22" height="22" />{' '}
                {loading ? 'GENERATING...' : 'GENERATE DESCRIPTION'}
              </button>
            </div>
          </>
        )}

        {hasImage && (
          <div className="wt-generate" style={{ marginTop: 16 }}>
            <button
              className="btn-orange"
              onClick={() => doCaption()}
              disabled={!imgBlob || loading}
            >
              <img src={sparklesIcon} alt="Generate" width="22" height="22" />{' '}
              {loading ? 'GENERATING...' : 'GENERATE DESCRIPTION'}
            </button>
          </div>
        )}

        {/* Caption output with clickable words (dictionary) */}
        {!!caption && (
          <div className="wt-readout">
            {words.map((w, i) => {
              const trimmed = w.replace(/^[^A-Za-z0-9']+|[^A-Za-z0-9']+$/g, '');
              const clickable = Boolean(trimmed);
              const handleClick = () => clickable && openDictionary(trimmed);
              const handleKey = (ev) => {
                if (!clickable) return;
                if (ev.key === 'Enter' || ev.key === ' ') {
                  ev.preventDefault();
                  openDictionary(trimmed);
                }
              };
              return (
                <span
                  key={i}
                  className={cx('word', clickable && 'clickable', i === focusIndex && 'focus')}
                  onClick={handleClick}
                  onKeyDown={handleKey}
                  role={clickable ? 'button' : undefined}
                  tabIndex={clickable ? 0 : undefined}
                >
                  {w}
                </span>
              );
            })}
          </div>
        )}

        {/* Tools row: Region select, translate, TTS, clear */}
        {captionReady && (
          <div className="wt-tools">
            <button className="wt-tool-btn" onClick={() => imgUrl && setRegionOpen(true)}>
              <img src={cropIcon} alt="Crop" width="22" height="22" /> REGION SELECT
            </button>

            {/* Translate controls with dropdown */}
            <div className="wt-translate" style={{ position: 'relative', display: 'inline-block' }}>
              <button
                className="wt-tool-btn"
                onClick={() => setDropdownOpen(!dropdownOpen)}
                onBlur={() => setTimeout(() => setDropdownOpen(false), 180)}
                disabled={!captionReady || translating}
                aria-haspopup="menu"
                aria-expanded={dropdownOpen}
              >
                <img src={translateIcon} alt="Translate" className="wt-icon" />
                <span>{translating ? 'TRANSLATING…' : 'TRANSLATE'}</span>
                {!translating && <span aria-hidden> ▾</span>}
              </button>

              {dropdownOpen && (
                <div className="wt-dropdown" role="menu">
                  <button className="wt-dropdown-item" onMouseDown={() => doTranslate('en')}>English</button>
                  <button className="wt-dropdown-item" onMouseDown={() => doTranslate('zh')}>Mandarin (中文)</button>
                  <button className="wt-dropdown-item" onMouseDown={() => doTranslate('ms')}>Malay (Bahasa Melayu)</button>
                  <button className="wt-dropdown-item" onMouseDown={() => doTranslate('ta')}>Tamil (தமிழ்)</button>
                </div>
               )}
            </div>
            <button className="wt-tool-btn" onClick={speak} disabled={!captionReady}>
              <img src={speakIcon} alt="Speak" width="22" height="22" /> TEXT-TO-SPEECH
            </button>

            <button className="wt-tool-btn" onClick={clearAll}>
              <img src={trashIcon} alt="Clear" width="22" height="22" /> CLEAR
            </button>
          </div>
        )}

        {/* Colour-blindness settings */}
        {captionReady && (
          <div
            style={{
              background: '#fff',
              borderRadius: 24,
              padding: 18,
              marginTop: 18,
              boxShadow: '0 2px 0 rgba(0,0,0,0.15)',
            }}
          >
            <div
              className="text-center text-slate-800"
              style={{
                fontWeight: 800,
                marginBottom: 10,
                textTransform: 'uppercase',
              }}
            >
              COLOUR BLINDNESS SETTINGS
            </div>
            <div style={{ display: 'grid', gap: 12 }}>
              <label style={{ display: 'grid', gap: 6 }}>
                <span className="text-sm font-semibold">Condition</span>
                <select value={cbCond} onChange={(e) => setCbCond(e.target.value)}>
                  <optgroup label="Red–Green deficiencies">
                    <option value="protanomaly">Protanomaly (red-weak)</option>
                    <option value="protanopia">Protanopia (red-blind)</option>
                    <option value="deuteranomaly">Deuteranomaly (green-weak)</option>
                    <option value="deuteranopia">Deuteranopia (green-blind)</option>
                  </optgroup>
                  <optgroup label="Blue–Yellow deficiencies">
                    <option value="tritanomaly">Tritanomaly (blue-weak)</option>
                    <option value="tritanopia">Tritanopia (blue-blind)</option>
                  </optgroup>
                  <optgroup label="Monochromacy">
                    <option value="achromatomaly">Achromatomaly (partial)</option>
                    <option value="achromatopsia">Achromatopsia (total)</option>
                    <option value="blue-cone monochromacy">Blue-cone monochromacy</option>
                  </optgroup>
                </select>
              </label>

              <label style={{ display: 'grid', gap: 6 }}>
                <span className="text-sm font-semibold">Severity (mild → full)</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={cbSeverity}
                  onChange={(e) => setCbSeverity(Number(e.target.value))}
                />
              </label>

              <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
                <span className="text-sm font-semibold">Mode</span>
                <label>
                  <input
                    type="radio"
                    name="cbmode"
                    checked={cbMode === 'simulate'}
                    onChange={() => setCbMode('simulate')}
                  />{' '}
                  Simulate
                </label>
                <label style={{ marginLeft: 10 }}>
                  <input
                    type="radio"
                    name="cbmode"
                    checked={cbMode === 'enhance'}
                    onChange={() => setCbMode('enhance')}
                  />{' '}
                  Enhance
                </label>
                <label style={{ marginLeft: 16 }}>
                  <input
                    type="checkbox"
                    checked={cbSplit}
                    onChange={(e) => setCbSplit(e.target.checked)}
                  />{' '}
                  Split view
                </label>
              </div>
            </div>

            <div
              className="mt-3"
              style={{ display: 'flex', gap: 12, justifyContent: 'center' }}
            >
              <button className="btn-orange" onClick={handleReset}>
                RESET
              </button>
              <button className="btn-orange" onClick={applyCVD}>
                APPLY CHANGES
              </button>
            </div>
          </div>
        )}

        {/* Quiz button */}
        {captionReady && (
          <div
            className="wt-test-row"
            style={{ marginTop: 24, marginBottom: 8 }}
          >
            <button
              className="wt-test-btn"
              onClick={() =>
                navigate('/quiz', { state: { caption, imageUrl: imgUrl } })
              }
            >
              <span className="wt-test-icon">?</span>TEST YOURSELF!
            </button>
          </div>
        )}

        {/* Audio + region selector + dictionary + toast */}
        <audio ref={audioRef} hidden />

        {regionOpen && imgUrl && (
          <RegionSelector
            src={imgUrl}
            onCancel={() => setRegionOpen(false)}
            onConfirm={async (box) => {
              setRegionOpen(false);
              try {
                const cropped = await new Promise((res) => {
                  const im = new Image();
                  im.onload = () => {
                    const c = document.createElement('canvas');
                    c.width = box.w;
                    c.height = box.h;
                    const x = c.getContext('2d');
                    x.drawImage(
                      im,
                      box.x,
                      box.y,
                      box.w,
                      box.h,
                      0,
                      0,
                      box.w,
                      box.h
                    );
                    c.toBlob((b) => res(b), 'image/jpeg', 0.92);
                  };
                  im.src = imgUrl;
                });
                if (cropped) {
                  setImgBlob(cropped);
                  setImgUrl(URL.createObjectURL(cropped));
                  setCvdBase(cropped);
                  setIsCropped(true);
                }
              } catch {}
              await doCaption(null);
            }}
          />
        )}

        <DictionaryModal
          word={dictOpen ? dictWord : ''}
          data={dictData}
          loading={dictLoading}
          error={dictError}
          onClose={closeDictionary}
        />

        <Toast />
      </div>
    </div>
  );
}

function AchievementsPage() {
  const { user } = useAuth();
  const [loading, setLoading] = React.useState(false);
  const [popup, setPopup] = React.useState(null); // { title, description }
  const progress = React.useMemo(() => ({ streak: Number(user?.achievements?.streak_days || 0) }), [user]);
  const badges = React.useMemo(() => (Array.isArray(user?.achievements?.badges) ? user.achievements.badges : []), [user]);

  // Catalogue of available awards (client-side)
  const AWARDS = [
    { id: 'first_caption', title: 'First Caption', description: 'Generate your first caption', target: 1, metric: 'captions', icon: '📝' },
    { id: 'quiz_whiz', title: 'Quiz Whiz', description: 'Complete your first quiz', target: 1, metric: 'quizzes', icon: '❓' },
    { id: 'storyteller', title: 'Storyteller', description: 'Create your first story', target: 1, metric: 'stories', icon: '📖' },
    { id: 'streak_7', title: '7-Day Streak', description: 'Use the app 7 days in a row', target: 7, metric: 'streak', icon: '🔥' },
    { id: 'streak_30', title: '30-Day Streak', description: 'Use the app 30 days in a row', target: 30, metric: 'streak', icon: '🏆' },
  ];
  React.useEffect(() => {
    // Detect newly unlocked awards and show a one-time popup (per browser)
    const unlockedIds = new Set((Array.isArray(badges) ? badges : []).map(b => b.id));
    const current = (id) => {
      const a = AWARDS.find(x => x.id === id);
      if (!a) return 0;
      if (a.metric === 'streak') return Number(progress?.streak || 0);
      return 0;
    };
    const seenRaw = localStorage.getItem('pic:ach:seen') || '{}';
    let seen = {};
    try { seen = JSON.parse(seenRaw) || {}; } catch { seen = {}; }
    for (const a of AWARDS) {
      const isUnlocked = unlockedIds.has(a.id) || current(a.id) >= a.target;
      if (isUnlocked && !seen[a.id]) {
        // mark and show popup once
        seen[a.id] = true;
        localStorage.setItem('pic:ach:seen', JSON.stringify(seen));
        setPopup({ title: `Achievement Unlocked: ${a.title}`, description: a.description });
        break;
      }
    }
  }, [progress, badges]);

  const streak = progress?.streak ?? 0;
  const today = new Date().toLocaleDateString();

  const unlockedIds = new Set((Array.isArray(badges) ? badges : []).map(b => b.id));
  const computed = AWARDS.map(aw => {
    const counts = (user?.achievements?.counts) || {};
    const cur = (() => {
      if (aw.metric === 'streak') return Number(progress?.streak || 0);
      if (aw.metric === 'captions') return Number(counts.captions || 0);
      if (aw.metric === 'quizzes') return Number(counts.quizzes || 0);
      if (aw.metric === 'stories') return Number(counts.stories || 0);
      return 0;
    })();
    const pct = Math.max(0, Math.min(100, Math.round((cur / aw.target) * 100)));
    const unlocked = unlockedIds.has(aw.id) || cur >= aw.target;
    return { ...aw, current: cur, pct, unlocked };
  });

  return (
    <div className="min-h-screen" style={{ background: 'var(--forest)' }}>
      <NavBar />
      <section className="ach-wrap">
        <h1 className="ach-title">ACHIEVEMENTS</h1>

        {/* Big rounded panel like your screenshot */}
        <div className="ach-card">
          {/* Streak bar */}
          <div className="ach-streak">
            <div className="ach-streak-left">
              <div className="ach-streak-emoji" aria-hidden>🔥</div>
              <div>
                <div className="ach-streak-label">Current Streak</div>
                <div className="ach-streak-value">{streak} day{streak === 1 ? '' : 's'}</div>
              </div>
            </div>
            <div className="ach-streak-right" role="img" aria-label="Mini calendar">
              {/* simple 7-slot indicator */}
              {Array.from({ length: 7 }).map((_, i) => {
                const filled = i < Math.min(streak, 7);
                return <span key={i} className={filled ? 'ach-dot filled' : 'ach-dot'} />;
              })}
            </div>
          </div>

          {/* Badges grid (show all awards with progress) */}
          <div className="ach-grid">
            {loading && (
              <>
                {Array.from({ length: 6 }).map((_, i) => (
                  <div key={i} className="ach-badge skeleton" />
                ))}
              </>
            )}

            {!loading && computed.map((b) => (
              <div key={b.id} className={`ach-badge ${b.unlocked ? 'unlocked' : 'locked'}`}>
                <div className="ach-badge-icon" aria-hidden>{b.unlocked ? '🏅' : '🔒'}</div>
                <div className="ach-badge-title">{b.title}</div>
                <div className="ach-badge-desc">{b.description}</div>
                {!b.unlocked && (<div className="ach-locked-label">Locked</div>)}
                <div className="ach-progress" aria-label="Progress">
                  <div className="ach-progress-bar" style={{ width: `${b.pct}%` }} />
                  <div className="ach-progress-text">{Math.min(b.current, b.target)} / {b.target}</div>
                </div>
              </div>
            ))}
          </div>
          {popup && (
            <div className="modal" role="dialog" aria-modal="true">
              <div className="modal-card">
                <div className="flex items-center justify-between mb-2">
                  <b>{popup.title}</b>
                  <button className="btn btn-plain" onClick={()=>setPopup(null)}>Close</button>
                </div>
                <div>{popup.description}</div>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

// --- Settings helpers ---
const DEFAULTS = {
  profile: { displayName: "", email: "", folderPath: "" },
  notifications: { allOn: true, dailyTime: "17:00", streakOn: true },
  accessibility: {
    dyslexiaFont: "Off",            // "Off" | "OpenDyslexic" | "Lexend"
    readingGuide: true,
    highContrast: false,
    tts: { voice: "App voice", rate: 1.0, wordColor: "#fde047" },
    canvas: { gridGuides: false }
  }
};

// Merge two plain objects shallowly/recursively for nested settings
function mergeDeep(base, patch){
  if(!patch || typeof patch !== 'object') return base;
  const out = Array.isArray(base) ? [...base] : { ...base };
  for(const k of Object.keys(patch)){
    const bv = out[k];
    const pv = patch[k];
    if(bv && typeof bv === 'object' && !Array.isArray(bv) && pv && typeof pv === 'object' && !Array.isArray(pv)){
      out[k] = mergeDeep(bv, pv);
    }else{
      out[k] = pv;
    }
  }
  return out;
}

// Convert any legacy flat server settings into our structured DEFAULTS shape
function normalizeSettings(raw, user){
  const base = JSON.parse(JSON.stringify(DEFAULTS));
  const flat = raw && typeof raw === 'object' ? raw : {};
  // Map legacy keys when present
  const mapped = {
    profile: {
      displayName: flat.display_name || flat.displayName || base.profile.displayName,
      email: user?.email || base.profile.email,
      folderPath: flat.storage_path || base.profile.folderPath,
    },
    notifications: {
      allOn: typeof flat.notifications_all_on === 'boolean' ? flat.notifications_all_on : base.notifications.allOn,
      dailyTime: flat.daily_objective_time || base.notifications.dailyTime,
      streakOn: typeof flat.streak_reminders === 'boolean' ? flat.streak_reminders : base.notifications.streakOn,
    },
    accessibility: {
      dyslexiaFont: flat.dyslexia_font || base.accessibility.dyslexiaFont,
      readingGuide: typeof flat.reading_guide === 'boolean' ? flat.reading_guide : base.accessibility.readingGuide,
      highContrast: typeof flat.high_contrast === 'boolean' ? flat.high_contrast : base.accessibility.highContrast,
      tts: {
        voice: flat.tts_voice || base.accessibility.tts.voice,
        rate: typeof flat.speaking_rate === 'number' ? flat.speaking_rate : base.accessibility.tts.rate,
        wordColor: flat.word_highlight_color || base.accessibility.tts.wordColor,
      },
      canvas: {
        gridGuides: typeof flat.grid_guides === 'boolean' ? flat.grid_guides : base.accessibility.canvas.gridGuides,
      }
    }
  };
  // If server has some of the new nested keys already, merge them in as well
  const nested = {};
  if (flat.profile || flat.notifications || flat.accessibility) {
    if (flat.profile) nested.profile = flat.profile;
    if (flat.notifications) nested.notifications = flat.notifications;
    if (flat.accessibility) nested.accessibility = flat.accessibility;
  }
  return mergeDeep(base, mergeDeep(mapped, nested));
}

function loadVoices() {
  const synth = window.speechSynthesis;
  if (!synth) return ["App voice"];
  const list = synth.getVoices();
  const names = list.map(v => v.name);
  return names.length ? [...names, "App voice"] : ["App voice"];
}

function useStickySave(initial, onSaved) {
  // Returns {value,setValue,dirty,save,resetAll,restore(sectionKey)}
  const [value, setValue] = React.useState(initial);
  const [dirty, setDirty] = React.useState(false);
  const seedingRef = React.useRef(false);
  // Keep local state in sync if the seed changes (e.g., after login)
  React.useEffect(() => {
    seedingRef.current = true;
    setValue(initial);
    setDirty(false);
  }, [JSON.stringify(initial)]);
  React.useEffect(() => {
    if (seedingRef.current) { seedingRef.current = false; return; }
    setDirty(true);
  }, [value]);
  const resetAll = () => { setValue(DEFAULTS); setDirty(true); };
  const restore = (key) => {
    setValue(v => ({ ...v, [key]: DEFAULTS[key] }));
    setDirty(true);
  };
  const save = async () => {
    // Only persist via API; DB is the single source of truth.
    const res = await fetch(`${API}/api/user/settings`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ settings: value })
    });
    if (!res.ok) throw new Error("save_failed");
    await res.json().catch(() => ({}));
    onSaved?.();
    setDirty(false);
  };
  return { value, setValue, dirty, save, resetAll, restore };
}

// --- Settings Page ---
function SettingsPage(){
  const { user, refetchMe } = useAuth();
  const [tab, setTab] = useState("profile"); // "profile" | "access"
  const [form, setForm] = useState(() => ({
    // profile
    display_name: user?.username || "",
    email: user?.email || "",

    // notifications
    daily_objective_time: user?.settings?.daily_objective_time ?? "17:00",
    streak_reminders: user?.settings?.streak_reminders ?? true,

    // accessibility - reading
    dyslexia_font: user?.settings?.dyslexia_font ?? "Off",
    high_contrast: user?.settings?.high_contrast ?? false,
    bw_mode: user?.settings?.bw_mode ?? false,

    // TTS
    tts_voice: (() => {
      const v = (user?.settings?.tts_voice ?? "App voice").toString();
      const low = v.toLowerCase();
      return (low === 'male' || low === 'female') ? low : 'App voice';
    })(),
    speaking_rate: String(user?.settings?.speaking_rate ?? 1.0),
    word_highlight_enable: user?.settings?.word_highlight_enable ?? true,
    word_highlight_color: user?.settings?.word_highlight_color ?? "#FFD54F",

    // Drawing
    grid_guides: user?.settings?.grid_guides ?? false,
  }));
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  function set(key, val){ setForm(v => ({ ...v, [key]: val })); }

  // live preview of accessibility items
    // live preview of accessibility items
  useEffect(() => {
    try {
      applyAccessibility({
        dyslexia_font: form.dyslexia_font,
        high_contrast: !!form.high_contrast,
        bw_mode: !!form.bw_mode,
        word_highlight_enable: !!form.word_highlight_enable,
        grid_guides: !!form.grid_guides,
        tts_voice: form.tts_voice,
        speaking_rate: Number(form.speaking_rate),
        word_highlight_color: form.word_highlight_color,
      });
    } catch {}
  }, [
    form.dyslexia_font,
    form.high_contrast,
    form.bw_mode,
    form.word_highlight_enable, // <-- added so toggle triggers applyAccessibility
    form.grid_guides,
    form.tts_voice,
    form.speaking_rate,
    form.word_highlight_color,
  ]);


  // profile actions (keep existing endpoints)
  async function changeDisplayName() {
  const newName = window.prompt("Enter new display name", form.display_name);
  if (newName == null || !newName.trim()) return;
  setBusy(true);
  try{
    const res = await fetch(`${API}/api/account/change_display_name`, {
      method:"POST", headers:{ "Content-Type":"application/json" }, credentials:"include",
      body: JSON.stringify({ display_name: newName.trim() })
    });
    if(!res.ok) throw new Error((await res.json()).detail || "Change display name failed");
    set("display_name", newName.trim());
    await refetchMe?.();
    setMsg("Display name updated");
  }catch(e){ setMsg(e.message); }
  finally{ setBusy(false); setTimeout(()=>setMsg(""),1200); }
}

async function changeEmail() {
  const newEmail = window.prompt("Enter new email", form.email);
  if (newEmail == null || !newEmail.trim()) return;
  setBusy(true);
  try{
    const res = await fetch(`${API}/api/account/change_email`, {
      method:"POST", headers:{ "Content-Type":"application/json" }, credentials:"include",
      body: JSON.stringify({ email: newEmail.trim() })
    });
    if(!res.ok) throw new Error((await res.json()).detail || "Change email failed");
    set("email", newEmail.trim());
    await refetchMe?.();
    setMsg("Email updated");
  }catch(e){ setMsg(e.message); }
  finally{ setBusy(false); setTimeout(()=>setMsg(""),1200); }
}

  async function changePassword(current_password, new_password){
    setBusy(true);
    try{
      const res = await fetch(`${API}/api/account/change_password`, {
        method:"POST", headers:{ "Content-Type":"application/json" }, credentials:"include",
        body: JSON.stringify({ current_password, new_password })
      });
      if(!res.ok) throw new Error((await res.json()).detail || "Password change failed");
      setMsg("Password changed");
    }catch(e){ setMsg(e.message); }
    finally{ setBusy(false); setTimeout(()=>setMsg(""),1200); }
  }

  async function clearData(password){
    setBusy(true);
    try{
      const res = await fetch(`${API}/api/account/clear_data`, {
        method:"POST", headers:{ "Content-Type":"application/json" }, credentials:"include",
        body: JSON.stringify({ password })
      });
      if(!res.ok) throw new Error((await res.json()).detail || "Clear failed");
      await refetchMe();
      applyAccessibility(DEFAULTS);
      setMsg("All app data cleared (kept your account).");
    }catch(e){ setMsg(e.message); }
    finally{ setBusy(false); setTimeout(()=>setMsg(""),1400); }
  }

  async function deleteAccount(password){
    if(!window.confirm("This will permanently delete your account. Proceed?")) return;
    setBusy(true);
    try{
      const res = await fetch(`${API}/api/account/delete`, {
        method:"POST", headers:{ "Content-Type":"application/json" }, credentials:"include",
        body: JSON.stringify({ password })
      });
      if(!res.ok) throw new Error((await res.json()).detail || "Delete failed");
      window.location.href = "/";
    }catch(e){ setMsg(e.message); }
    finally{ setBusy(false); setTimeout(()=>setMsg(""),1400); }
  }

  // persist (without the removed fields)
  async function persistSettings(){
  setBusy(true);
  try{
    // 1) Build normalized payload (booleans + numeric speaking_rate)
    const speaking_rate_num = Number(form.speaking_rate);
    const payloadSettings = {
      // notifications
      daily_objective_time: form.daily_objective_time,
      streak_reminders: !!form.streak_reminders,

      // accessibility
      dyslexia_font: form.dyslexia_font,
      high_contrast: !!form.high_contrast,
      bw_mode: !!form.bw_mode,

      // TTS
      tts_voice: form.tts_voice,
      speaking_rate: Number.isFinite(speaking_rate_num) ? speaking_rate_num : 1,
      word_highlight_enable: !!form.word_highlight_enable,
      word_highlight_color: form.word_highlight_color,

      // drawing
      grid_guides: !!form.grid_guides,
    };

    const res = await fetch(`${API}/api/user/settings`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ settings: payloadSettings })
    });
    if (!res.ok) throw new Error("server save failed");

    // 2) Use server-merged settings if returned; otherwise fall back to our payload
    const data = await res.json().catch(()=> ({}));
    const saved = (data && data.settings) ? data.settings : payloadSettings;

    // 3) Apply immediately (updates globals + CSS and notifies listeners)
    if (typeof applyAccessibility === "function") {
      applyAccessibility(saved);
    } else {
      // Fallback: fire a custom event if your pages listen for it
      window.__picteractive_settings = { ...(window.__picteractive_settings || {}), ...saved };
      window.dispatchEvent(new CustomEvent('picteractive:settings-applied', { detail: { settings: window.__picteractive_settings } }));
    }

    // 4) Refresh auth state so other pages re-read user.settings
    await refetchMe?.();

    setMsg("Settings saved");
  } catch (e){
    setMsg(e.message || "Save failed");
  } finally {
    setBusy(false);
    setTimeout(()=> setMsg(""), 1200);
  }
}


  function resetToDefaults(){
    // reset only to the keys we show on the page
    setForm(v => ({
      ...v,
      daily_objective_time: "17:00",
      streak_reminders: true,
      dyslexia_font: "Off",
      high_contrast: false,
      bw_mode: false,
      tts_voice: "App voice",
      speaking_rate: 1.0,
      word_highlight_enable: true,
      word_highlight_color: "#FFD54F",
      grid_guides: false,
    }));
  }

  return (
    <div className="min-h-screen" style={{ background:'var(--forest)' }}>
      <NavBar />
      <h1 className="settings-heading" style={{ textAlign: 'center', fontWeight: 900, fontSize: '32px', textTransform: 'uppercase', marginTop: '24px', marginBottom: '8px', letterSpacing: '.6px', color: 'var(--cream)' }}> SETTINGS </h1>
      <div className="settings-wrap">
        {/* Left orange sidebar with image icons */}
        <aside className="settings-side" aria-label="Settings sections">
          <button
            className={`side-pill ${tab === "profile" ? "active" : ""}`}
            onClick={() => setTab("profile")}
            title="Profile"
            aria-pressed={tab === "profile"}
          >
            <img src={profileIcon} alt="Profile" width="36" height="36" />
          </button>

          <button
            className={`side-pill ${tab === "access" ? "active" : ""}`}
            onClick={() => setTab("access")}
            title="Accessibility"
            aria-pressed={tab === "access"}
          >
            <img src={accessIcon} alt="Accessibility" width="36" height="36" />
          </button>
        </aside>

        {/* Right main panel */}
        <section className="settings-panel">
          <div className="settings-panel-inner">
            <div className="settings-title">
              {tab === "profile" ? "PROFILE ACCOUNT" : "ACCESSIBILITY"}
            </div>

            {tab === "profile" ? (
              <>
                <div className="settings-card">
                  {/* Profile (use same heading style as section headings) */}
                  <h3>Profile</h3>

                  {/* 1) Display Name, 2) Email */}
                  <div className="grid-2">
                    <label>Display Name
                      <div className="row gap">
                      <input type="text" value={form.display_name} readOnly />
                      <button className="btn btn-plain" onClick={changeDisplayName}>Change</button>
                      </div>
                    </label>
                    <label>Email
                      <div className="row gap">
                      <input type="email" value={form.email} readOnly />
                      <button className="btn btn-plain" onClick={changeEmail}>Change</button>
                      </div>
                    </label>
                  </div>

                  {/* 3) Change Password */}
                  <div className="row" style={{ marginTop: 14 }}>
                    <button
                      className="btn btn-plain"
                      onClick={() => {
                        const cur = prompt("Enter current password"); if(cur==null) return;
                        const nw  = prompt("Enter new password (min 6 chars)"); if(nw==null) return;
                        changePassword(cur, nw);
                      }}
                    >
                      Change password
                    </button>
                  </div>

                  {/* Notifications (only the two items you asked for) */}
                  <h3 style={{marginTop:18}}>Notifications</h3>
                  <div className="grid2">
                    <label>Daily Objective Time
                      <input type="time" value={form.daily_objective_time}
                      onChange={e=>set("daily_objective_time", e.target.value)} />
                      </label>
                      <label className="field inline">
                        <span>Streak reminders</span>
                        <input type="checkbox"
                        checked={!!form.streak_reminders}
                        onChange={e=>set("streak_reminders", e.target.checked)} />
                        </label>
                        </div>

                  {/* Danger zone */}
                  <div className="danger">
                    <div className="danger-title"></div>
                    <button
                      className="btn btn-red"
                      onClick={()=>{
                        const pw = prompt("Enter your password to CLEAR all app data (keeps account)");
                        if(pw==null) return; clearData(pw);
                      }}
                    >
                      Clear All Data
                    </button>
                    <button
                      className="btn btn-red"
                      onClick={()=>{
                        const pw = prompt("Enter your password to DELETE your account");
                        if(pw==null) return; deleteAccount(pw);
                      }}
                    >
                      Delete Account
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <>
                {/* ACCESSIBILITY (unchanged content – keep your existing controls here) */}
                <div className="settings-card">
                  <h3>Reading</h3>
                  <div className="grid-3">
                    <label>Dyslexia font
                      <select value={form.dyslexia_font}
                              onChange={e=>set("dyslexia_font", e.target.value)}>
                        <option>Off</option>
                        <option>OpenDyslexic</option>
                        <option>Lexend</option>
                      </select>
                    </label>
                    <label className="field inline">
                      <input
                        type="checkbox"
                        checked={!!form.high_contrast}
                        onChange={e=>set("high_contrast", e.target.checked)}
                      />
                      <span>High contrast</span>
                    </label>
                  </div>
                  
                  <h3>TTS</h3>
                  <div className="grid-3">
                  <label>TTS voice
                      <select value={form.tts_voice}
                              onChange={e=>set("tts_voice", e.target.value)}>
                        <option value="App voice">App voice</option>
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                      </select>
                    </label>
                    <label>Speaking rate
                      <select
                      value={form.speaking_rate}
                      onChange={e=>set("speaking_rate", e.target.value)}>
                        <option value="0.5">Slow (0.5)</option>
                        <option value="1">Normal (1.0)</option>
                        <option value="1.5">Fast (1.5)</option>
                        </select>
                        </label>
                    <label className="field inline">
                      <input
                        type="checkbox"
                        checked={!!form.word_highlight_enable}
                        onChange={e=>set("word_highlight_enable", e.target.checked)}
                      />
                      <span>Word highlight</span>
                    </label>
                  </div>
                  <div style={{ marginTop: 16 }}>
                    <button className="btn btn-blue" onClick={async () => {
                        try{
                          const voice = form.tts_voice;
                          const rateNum = parseFloat(form.speaking_rate) || 1.0;
                          // Use the same backend + audio pipeline as captions/story
                          const src = await apiTTS("This is a preview of your text-to-speech settings.", {
                            voice,
                            rate: rateNum,
                          });
                          const audio = new Audio(src);
                          audio.playbackRate = (rateNum > 0 && isFinite(rateNum)) ? rateNum : 1.0;
                          audio.play().catch(()=>{});
                        }catch(e){
                          console.error("TTS preview failed", e);
                        }
                      }}> Preview </button>
                          </div>
                </div>
              </>
            )}

            {/* Sticky actions bar */}
            <div className="settings-sticky">
              <div className="spacer" />
              <button className="btn btn-plain" onClick={resetToDefaults} disabled={busy}>
                Reset to defaults
              </button>
              <button className="btn btn-orange" onClick={persistSettings} disabled={busy}>
                Save
              </button>
            </div>

            {msg && <div className="toast">{msg}</div>}
          </div>
        </section>
      </div>
    </div>
  );
}


// Instructions Page
function InstructionsPage(){
  const [idx, setIdx] = React.useState(0);

  // Orange pill for BOTH asset icons and emojis
  const Pill = ({children, size=28, pad=6, title}) => (
    <span
      title={title}
      style={{
        display:'inline-flex',
        alignItems:'center',
        justifyContent:'center',
        width: size + pad*2,
        height: size + pad*2,
        borderRadius: 999,
        background: 'var(--orange)',
        marginRight: 12,
        verticalAlign: '-4px',
        flex: '0 0 auto'
      }}
    >
      {children}
    </span>
  );

  const Ico = ({src, alt, size=28}) => (
    <Pill size={size}><img src={src} alt={alt} width={size} height={size} /></Pill>
  );
  const Emo = ({char, size=26}) => (
    <Pill size={size}><span aria-hidden style={{fontSize: size}}>{char}</span></Pill>
  );

  // Text layout helpers to avoid "clustered" look
  const Body = ({children}) => (
    <div className="instr-body-inner" style={{
      maxWidth: 820,          // keeps lines readable
      margin: '0 auto',
      textAlign: 'center'
    }}>
      {children}
    </div>
  );

  const List = ({children}) => (
    <ul style={{
      listStyle:'none',
      margin: '1rem auto 1.25rem',
      padding: 0,
      maxWidth: 760,
      textAlign: 'left',
      display: 'grid',
      rowGap: '14px'          // consistent spacing between items
    }}>
      {children}
    </ul>
  );

  const ListItem = ({children}) => (
    <li style={{
      display:'flex',
      alignItems:'flex-start',
      lineHeight: 1.85
    }}>
      {children}
    </li>
  );

  // Roomy typography
  const textStyle = {
    fontSize: '1.3rem',
    lineHeight: 1.9,
    letterSpacing: '0.2px'
  };
  const h3Style = { fontSize: '1.45rem', margin: '1.2rem 0 0.6rem', letterSpacing:'0.3px' };

  const sections = [
    {
      key: 'welcome',
      title: 'WELCOME TO PICTERACTIVE!',
      content: (
        <Body>
          <div style={textStyle}>
            <p style={{margin:'0 0 16px'}}>Picteractive lets you <b>draw</b>, <b>describe</b>, and <b>create stories</b> </p> <p>from your imagination — all in one place!</p>
            <p style={{margin:'0 0 8px'}}>Follow these simple steps to explore every page and have fun learning.</p>
          </div>
        </Body>
      )
    },
    {
      key: 'whatsthis',
      title: '“WHAT’S THIS?” PAGE',
      content: (
        <Body>
          <div style={textStyle}>
            <h3 style={h3Style}>Purpose</h3>
            <p style={{margin:'0 0 14px'}}>Find out what’s inside your picture with a friendly description.</p>

            <h3 style={h3Style}>How to Use</h3>
            <List>
              <ListItem>
                <Ico src={uploadIcon} alt="Upload"/><b>Upload Image</b> or <Ico src={cameraIcon} alt="Camera"/> <b>Take Photo</b> to add a picture.
              </ListItem>
              <ListItem>
                <Ico src={sparklesIcon} alt="AI"/><b>Generate Description</b> - Writes detailed caption for you.
              </ListItem>
              <ListItem>
                <Ico src={cropIcon} alt="Crop"/>
                <b>Region Select</b> to crop and describe a selected part of an image. 
              </ListItem>
              <ListItem>
                <Ico src={translateIcon} alt="Translate"/><b>Translate</b> the caption into English, Chinese, Malay, or Tamil.
              </ListItem>
              <ListItem>
                <Ico src={speakIcon} alt="Speaker"/><b>Listen</b> to the caption with Text-to-Speech.
              </ListItem>
              <ListItem>
                <Emo char="👁️"/><p><b>COLOUR BLINDNESS</b> Simulates colour blindness by adjusting the image for each vision condition.</p>
              </ListItem>
            </List>
          </div>
        </Body>
      )
    },
    {
      key: 'quiz',
      title: 'QUIZ PAGE',
      content: (
        <Body>
          <div style={textStyle}>
            <h3 style={h3Style}>Purpose</h3>
            <p style={{margin:'0 0 14px'}}>Play a short quiz based on your caption to learn new words.</p>

            <h3 style={h3Style}>How it Works</h3>
            <List>
              <ListItem>Picteractive creates <b>3 kid-friendly questions</b> with choices <b>A/B/C</b>.</ListItem>
              <ListItem>Choose the best answer to earn <b>points</b> and improve your understanding.</ListItem>
              <ListItem>Quizzes are fun and help build your vocabulary!</ListItem>
            </List>
          </div>
        </Body>
      )
    },
    {
      key: 'draw',
      title: 'DRAW PAGE',
      content: (
        <Body>
          <div style={textStyle}>
            <h3 style={h3Style}>Purpose</h3>
            <p style={{margin:'0 0 14px'}}>Create <b>three scenes</b> that can become a story.</p>

            <h3 style={h3Style}>Tools</h3>
            <List>
              <ListItem><Emo char="🖌️"/> <b>Brush</b> — draw freely.</ListItem>
              <ListItem><Emo char="⟳"/> <b>Eraser / Clear</b> — fix or start over.</ListItem>
              <ListItem><Emo char="🎨"/> <b>Color & Size</b> — choose any color and brush width.</ListItem>
              <ListItem><Emo char="🪣"/> <b>Fill</b> — quickly color large areas.</ListItem>
            </List>

            <h3 style={h3Style}>Next Step</h3>
            <List>
              <ListItem>When all 3 frames are done, press <b>Generate Story</b> to continue.</ListItem>
            </List>
          </div>
        </Body>
      )
    },
    {
      key: 'story',
      title: 'STORY TIME PAGE',
      content: (
        <Body>
          <div style={textStyle}>
            <h3 style={h3Style}>Purpose</h3>
            <p style={{margin:'0 0 14px'}}>Turn your drawings into a short, fun story with a title.</p>

            <h3 style={h3Style}>What You Can Do</h3>
            <List>
              <ListItem>Read the story or press <Ico src={speakIcon} alt="Speaker"/> <b>Play</b> to hear it aloud.</ListItem>
              <ListItem>Tap <Ico src={editIcon} alt="Edit"/> <b>Edit</b> to change the title or words.</ListItem>
              <ListItem>Press <Ico src={saveIcon} alt="Save"/> <b>Save</b> to store the final story with your drawings.</ListItem>
            </List>
          </div>
        </Body>
      )
    },
    {
      key: 'achievements',
      title: 'ACHIEVEMENTS PAGE',
      content: (
        <Body>
          <div style={textStyle}>
            <h3 style={h3Style}>Purpose</h3>
            <p style={{margin:'0 0 14px'}}>Celebrate your progress and stay motivated!</p>
            <List>
              <ListItem><Emo char="🏆"/> <b>Badges</b> for milestones (captions, quizzes, stories).</ListItem>
              <ListItem><Emo char="🔥"/> <b>Streak Days</b> for learning regularly.</ListItem>
              <ListItem><Emo char="⭐"/> <b>Points</b> grow as you create and explore.</ListItem>
            </List>
          </div>
        </Body>
      )
    },
    {
      key: 'settings',
      title: 'SETTINGS PAGE',
      content: (
        <Body>
          <div style={textStyle}>
            <h3 style={h3Style}>Purpose</h3>
            <p style={{margin:'0 0 14px'}}>Manage your profile and make Picteractive easy on your eyes and ears.</p>

            <h3 style={h3Style}>Profile Settings</h3>
            <List>
              <ListItem>Update your <b>name</b>, <b>email</b>, or <b>password</b>.</ListItem>
              <ListItem>All changes save automatically and apply across your account.</ListItem>
            </List>
            
            <h3 style={h3Style}>Accessibility Settings</h3>
            <List>
              <ListItem><b>Dyslexia-friendly fonts</b> .</ListItem>
              <ListItem><b>High Contrast</b> mode.</ListItem>
              <ListItem><Ico src={speakIcon} alt="TTS" /> <b>Voice & Speed</b> for narration.</ListItem>
            </List>

            <p style={{margin:'6px 0 0'}}>
              <i>Your preferences stay saved and apply throughout the app.</i>
            </p>
          </div>
        </Body>
      )
    },
    {
      key: 'account',
      title: 'ACCOUNT & LOGIN',
      content: (
        <Body>
          <div style={textStyle}>
            <h3 style={h3Style}>Purpose</h3>
            <p style={{margin:'0 0 14px'}}>Keep your work and progress saved safely.</p>

            <h3 style={h3Style}>Steps</h3>
            <List>
              <ListItem><b>Register</b> with your username, email, and password.</ListItem>
              <ListItem><b>Login</b> to access your saved stories and achievements.</ListItem>
              <ListItem>Use your profile menu to <b>Log Out</b> anytime.</ListItem>
            </List>
          </div>
        </Body>
      )
    },
    {
      key: 'tips',
      title: 'TIPS FOR BEST EXPERIENCE',
      content: (
        <Body>
          <div style={textStyle}>
            <List>
              <ListItem>Use clear drawings and bright colors for better captions.</ListItem>
              <ListItem>Turn on sound to enjoy <Ico src={speakIcon} alt="TTS"/> <b>Text-to-Speech</b>.</ListItem>
              <ListItem>Try <Ico src={translateIcon} alt="Translate"/> <b>Translate</b> to learn new words.</ListItem>
              <ListItem>Come back daily to earn streaks and badges!</ListItem>
            </List>
          </div>
        </Body>
      )
    }
  ];

  const cur = sections[idx];
  const canPrev = idx > 0;
  const canNext = idx < sections.length - 1;

  return (
    <div className="min-h-screen" style={{ background:'var(--forest)' }}>
      <NavBar />
      <section className="instr-wrap">
        <h1 className="instr-title">INSTRUCTIONS</h1>

        {/* Bigger, easy-to-read white card */}
        <div className="instr-card" style={{ fontSize:'1.25rem', padding:'2.2rem 3.2rem' }}>
          {/* Section title: bigger, centered, orange */}
          <div className="instr-head" style={{ textAlign:'center', marginBottom:'1.1rem' }}>
            <span
              className="instr-section"
              style={{
                display:'inline-block',
                fontWeight:800,
                fontSize:'1.8rem',
                color:'var(--orange)',
                letterSpacing:'0.4px'
              }}
            >
              {cur.title}
            </span>
          </div>

          <div className="instr-body" style={{ marginBottom:'2.2rem' }}>
            {cur.content}
          </div>

          <div className="instr-nav">
            <button className="btn btn-plain" onClick={()=>setIdx(i=>Math.max(0, i-1))} disabled={!canPrev}>← BACK</button>
            <div className="instr-step" style={{ fontSize:'1.1rem' }}>{idx+1} / {sections.length}</div>
            <button className="btn btn-orange" onClick={()=>setIdx(i=>Math.min(sections.length-1, i+1))} disabled={!canNext}>NEXT →</button>
          </div>
        </div>
      </section>
    </div>
  );
}


export default function App(){
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/draw" element={<ProtectedRoute><DrawPage /></ProtectedRoute>} />
          <Route path="/story" element={<ProtectedRoute><StoryPage /></ProtectedRoute>} />
          <Route path="/whats-this" element={<ProtectedRoute><WhatsThisV2 /></ProtectedRoute>} />
          <Route path="/quiz" element={<ProtectedRoute><QuizPage /></ProtectedRoute>} />
          <Route path="/instructions" element={<ProtectedRoute><InstructionsPage /></ProtectedRoute>} />
          <Route path="/achievements" element={<ProtectedRoute><AchievementsPage /></ProtectedRoute>} />
          <Route path="/settings" element={<ProtectedRoute><SettingsPage /></ProtectedRoute>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
