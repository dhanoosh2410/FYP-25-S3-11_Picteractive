from __future__ import annotations
"""
Flickr8k-only captioner (PyTorch):
- Builds a CLIP image-embedding index over Flickr8k images
- At inference, retrieves nearest images and *uses their Flickr8k captions*
- Region-based cropping supported
- Returns multi-sentence paragraph via simple stitching

Env/Paths:
  FLICKR8K_IMAGES_DIR  (default: server/data/Images)
  FLICKR8K_TOKENS_FILE (default: server/data/captions.txt)
  INDEX_CACHE_PATH     (default: <model_root>/flickr8k_index.npz)
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from PIL import Image
import os, re, json, random, time, math
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from transformers import CLIPModel, CLIPProcessor

# ---------------- text helpers ----------------

def _clean_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    if s and s[-1] not in ".!?":
        s += "."
    return s

def _simplify_vocab(s: str) -> str:
    repl = {
        "automobile": "car",
        "canine": "dog",
        "feline": "cat",
        "residential": "house",
        "individual": "person",
        "adjacent": "next to",
        "beneath": "under",
    }
    words = s.split()
    out = []
    for w in words:
        lw = w.lower().strip(",.")
        out.append(repl.get(lw, w))
    return " ".join(out)

def _dedupe_similar(lines: List[str], jacc_th: float = 0.65) -> List[str]:
    def jac(a, b):
        A, B = set(a.lower().split()), set(b.lower().split())
        if not A or not B: return 0.0
        return len(A & B) / len(A | B)
    out: List[str] = []
    for s in lines:
        s = s.strip()
        if not s: continue
        if all(jac(s, t) < jacc_th for t in out):
            out.append(s)
    return out

def _fuse_to_paragraph(sentences: List[str]) -> str:
    if not sentences: return ""
    parts = []
    for i, s in enumerate(sentences):
        s = _clean_text(s)
        if i == 0:
            parts.append(f"In the picture, {s[0].lower()+s[1:]}")
        elif i == len(sentences) - 1 and len(sentences) > 2:
            parts.append(f"Finally, {s[0].lower()+s[1:]}")
        else:
            parts.append(f"{random.choice(['Next,', 'Also,', 'Then,'])} {s[0].lower()+s[1:]}")
    text = " ".join(parts)
    return _clean_text(_simplify_vocab(text))

# ---------------- dataset & indexing ----------------

# --- replace the whole function in showtellpyTorch.py ---
def _read_flickr8k_tokens(token_path: Path) -> Dict[str, List[str]]:
    """
    Reads Flickr8k captions from either:
      1) tab-separated: 'image.jpg#idx<TAB>caption' OR 'image.jpg<TAB>caption'
      2) csv:           'image.jpg,caption' (first line 'image,caption' is skipped)
    Returns: { "image.jpg": [cap1, cap2, ...] }
    """
    caps: Dict[str, List[str]] = {}

    with open(token_path, "r", encoding="utf-8") as f:
        # Peek first line to detect delimiter
        first = f.readline()
        f.seek(0)

        if "\t" in first:
            # TSV reader
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                left, cap = line.split("\t", 1)
                img = left.split("#", 1)[0].strip()
                cap = cap.strip()
                if img and cap:
                    caps.setdefault(img, []).append(cap)
            return caps

        # CSV reader (comma)
        import csv
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            # skip header if present
            if row[0].strip().lower() == "image" and row[1].strip().lower() == "caption":
                continue
            img = row[0].split("#", 1)[0].strip()
            cap = row[1].strip()
            if img and cap:
                caps.setdefault(img, []).append(cap)

    return caps

class _ImagePathDataset(Dataset):
    def __init__(self, img_paths: List[Path], preprocess):
        self.paths = img_paths
        self.preprocess = preprocess
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        im = Image.open(p).convert("RGB")
        return self.preprocess(im), p.name

# ---------------- main captioner ----------------

@dataclass
class _Index:
    emb: np.ndarray           # (N, D) float32
    names: List[str]          # image file names aligned with emb
    caps: Dict[str, List[str]]# mapping name -> captions

class ShowTellCaptioner:
    """
    Flickr8k-only retrieval captioner.
    On first run, builds an embedding index for all images under FLICKR8K_IMAGES_DIR.
    """
    def __init__(self, model_root: Path | str, timeout_s: float = 60.0,
                 k_neighbors: int = 5, batch_size: int = 32):
        self.model_root = Path(model_root)
        self.timeout_s = float(timeout_s)
        self.k = int(k_neighbors)
        self.batch = int(batch_size)

        # paths
        self.images_dir = Path(os.getenv("FLICKR8K_IMAGES_DIR", "server/data/Images"))
        self.tokens_file = Path(os.getenv("FLICKR8K_TOKENS_FILE", "server/data/captions.txt"))
        self.index_cache = Path(os.getenv("INDEX_CACHE_PATH", str(self.model_root / "flickr8k_index.npz")))

        self.ready = False
        self.err: Optional[str] = None

        # models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip: Optional[CLIPModel] = None
        self.proc: Optional[CLIPProcessor] = None
        self.index: Optional[_Index] = None

        self._startup()

    # ---------- lifecycle ----------

    def _startup(self) -> None:
        try:
            if not self.images_dir.exists():
                raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
            if not self.tokens_file.exists():
                raise FileNotFoundError(f"Captions file not found: {self.tokens_file}")

            # load CLIP
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            # build / load index
            self.index = self._load_or_build_index()
            self.ready = True
            self.err = None
        except Exception as e:
            self.ready = False
            self.err = f"init_failed: {e}"

    def _load_or_build_index(self) -> _Index:
        caps = _read_flickr8k_tokens(self.tokens_file)
        img_paths = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        names = [p.name for p in img_paths]

        if self.index_cache.exists():
            data = np.load(self.index_cache, allow_pickle=True)
            emb = data["emb"].astype(np.float32)
            names_cached = data["names"].tolist()
            # If cache matches the current set, reuse it
            if names_cached == names:
                return _Index(emb=emb, names=names_cached, caps=caps)

        # build embeddings
        assert self.clip is not None and self.proc is not None
        preprocess = self.proc.feature_extractor  # internal use by processor
        # wrap into torchvision-like transform for dataset
        # The CLIPProcessor will be used at batch time; here we just keep PIL
        ds = _ImagePathDataset(img_paths, preprocess=lambda x: x)
        dl = DataLoader(ds, batch_size=self.batch, shuffle=False, num_workers=0, collate_fn=lambda b: b)

        embs: List[np.ndarray] = []
        self.clip.eval()
        with torch.no_grad():
            for batch in tqdm(dl, desc="Indexing Flickr8k", total=len(dl)):
                # batch is list of tuples: (PIL_image, name)
                images = [im for im, _name in batch]
                inputs = self.proc(images=images, return_tensors="pt").to(self.device)
                img_feat = self.clip.get_image_features(**inputs)  # (B, D)
                img_feat = torch.nn.functional.normalize(img_feat, p=2, dim=-1)
                embs.append(img_feat.detach().cpu().numpy().astype(np.float32))
        emb = np.concatenate(embs, axis=0)

        os.makedirs(self.index_cache.parent, exist_ok=True)
        np.savez_compressed(self.index_cache, emb=emb, names=np.array(names, dtype=object))
        return _Index(emb=emb, names=names, caps=caps)

    # ---------- internal utils ----------

    def _embed(self, pil: Image.Image) -> np.ndarray:
        assert self.clip is not None and self.proc is not None
        self.clip.eval()
        with torch.no_grad():
            inputs = self.proc(images=pil, return_tensors="pt").to(self.device)
            feat = self.clip.get_image_features(**inputs)
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat.detach().cpu().numpy().astype(np.float32)[0]  # (D,)

    def _nearest(self, q: np.ndarray, topk: int) -> List[int]:
        assert self.index is not None
        # cosine sim since all vectors are L2-normalized
        sims = self.index.emb @ q  # (N,)
        idx = np.argpartition(-sims, kth=min(topk, len(sims)-1))[:topk]
        return idx[np.argsort(-sims[idx])].tolist()

    def _pick_captions(self, neighbor_names: List[str], max_caps: int = 5) -> List[str]:
        assert self.index is not None
        caps_out: List[str] = []
        for name in neighbor_names:
            caps = self.index.caps.get(name, [])
            # keep 1–2 per neighbour to avoid repetition
            random.shuffle(caps)
            for c in caps[:2]:
                caps_out.append(_clean_text(c))
                if len(caps_out) >= max_caps:
                    break
            if len(caps_out) >= max_caps:
                break
        # dedupe and lightly shorten
        caps_out = _dedupe_similar(caps_out)
        short = []
        for s in caps_out:
            words = s.split()
            if len(words) > 18:
                s = " ".join(words[:18]) + "..."
            short.append(_simplify_vocab(s))
        return short

    # ---------- public API ----------

    def caption(self, image: Image.Image, region: Optional[Dict[str, int]] = None, **_) -> str:
        """
        Returns a single sentence *from Flickr8k* by retrieving the most similar image
        and taking one of its reference captions.
        """
        if not self.ready or self.index is None:
            raise RuntimeError(self.err or "captioner unavailable")

        pil = image
        if region:
            try:
                x = max(0, int(region.get("x", 0))); y = max(0, int(region.get("y", 0)))
                w = max(1, int(region.get("w", 1))); h = max(1, int(region.get("h", 1)))
                pil = image.crop((x, y, x + w, y + h))
            except Exception:
                pil = image

        q = self._embed(pil)
        top = self._nearest(q, topk=max(1, self.k))
        nn_names = [self.index.names[i] for i in top]  # type: ignore
        caps = self._pick_captions(nn_names, max_caps=1)
        return caps[0] if caps else ""

    def describe(
        self,
        image: Image.Image,
        region: Optional[Dict[str, int]] = None,
        mode: str = "paragraph",
        n_candidates: int = 4,
    ) -> Tuple[List[str], str]:
        """
        Returns 2–4 short sentences (from Flickr8k neighbours) and a stitched paragraph.
        """
        if not self.ready or self.index is None:
            raise RuntimeError(self.err or "captioner unavailable")

        pil = image
        if region:
            try:
                x = max(0, int(region.get("x", 0))); y = max(0, int(region.get("y", 0)))
                w = max(1, int(region.get("w", 1))); h = max(1, int(region.get("h", 1)))
                pil = image.crop((x, y, x + w, y + h))
            except Exception:
                pil = image

        q = self._embed(pil)
        top = self._nearest(q, topk=max(3, self.k))
        nn_names = [self.index.names[i] for i in top]  # type: ignore

        # collect a few captions from the nearest images
        # cap count guided by n_candidates
        want = 2 + min(2, max(0, n_candidates - 2))  # 2..4
        sents = self._pick_captions(nn_names, max_caps=want)

        if not sents:
            # fallback: one nearest caption
            sents = [self.caption(pil)]

        para = _fuse_to_paragraph(sents) if mode == "paragraph" else ""
        return sents, para
