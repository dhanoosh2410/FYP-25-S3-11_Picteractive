# server/story_gen.py  — ultra-minimal multimodal story generator (GPT-4o mini)
from __future__ import annotations
import base64, io, os, json, re
from typing import List, Tuple, Optional
from PIL import Image

# OpenAI >= 1.0.0
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Please `pip install openai>=1.0.0`") from e


def _data_url_from_pil(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL Image to a data URL (base64 PNG by default)."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _parse_json_like(s: str) -> Tuple[str, List[str], str]:
    """
    Accepts strict JSON or JSON-ish output with minor deviations.
    Returns (title, [p1,p2,p3], moral).
    """
    # Try clean JSON first
    try:
        obj = json.loads(s)
        title = (obj.get("title") or "").strip()
        panels = obj.get("panels") or []
        moral = (obj.get("moral") or "").strip()
        panels = [str(p).strip() for p in panels][:3]
        while len(panels) < 3:
            panels.append("")
        return title, panels, moral
    except Exception:
        pass

    # Try line-based fallback: Title:/Panel 1:/Panel 2:/Panel 3:
    title = ""
    panels = ["", "", ""]
    moral = ""
    for line in s.splitlines():
        line = line.strip()
        up = line.upper()
        if up.startswith("TITLE:"):
            title = line.split(":", 1)[1].strip()
        elif up.startswith("PANEL 1:"):
            panels[0] = line.split(":", 1)[1].strip()
        elif up.startswith("PANEL 2:"):
            panels[1] = line.split(":", 1)[1].strip()
        elif up.startswith("PANEL 3:"):
            panels[2] = line.split(":", 1)[1].strip()
        elif up.startswith("MORAL:"):
            moral = line.split(":", 1)[1].strip()
    return title, panels, moral


class StoryGenerator:
    """
    Minimal API:
      sg = StoryGenerator()
      title, panels = sg.generate_from_images([img1, img2, img3])
    """

    def __init__(self) -> None:
        self.model = os.getenv("STORY_OPENAI_MODEL", "gpt-4o-mini")
        self.model_name = self.model
        self._mode = "openai"
        self.ready = False
        self.err: Optional[str] = None
        self._last_images: Optional[List[Image.Image]] = None
        self._check_env()
        self.client = OpenAI()
        self.ready = True

    def _check_env(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        provider = (os.getenv("STORY_PROVIDER") or "").strip().lower()
        if provider != "openai":
            # Not hard failing, but strongly nudging the correct setting
            raise RuntimeError("Set STORY_PROVIDER=openai to enable GPT-4o mini")

    def generate_from_images(self, images: List[Image.Image], mood: str = "friendly") -> Tuple[str, List[str], str]:
        """
        images: list of exactly 3 PIL Images (panel1, panel2, panel3)
        returns: (title, [panel1, panel2, panel3], moral)
        """
        if not images or len(images) != 3:
            raise ValueError("Exactly three images are required")

        data_urls = [_data_url_from_pil(img, "PNG") for img in images]

        system_msg = (
            "You are a children's storyteller. "
            "Look ONLY at the three images the user provides (Panel 1, Panel 2, Panel 3). "
            "Write a very short, vivid 3-panel story for kids: one sentence per panel. "
            "No meta talk about images. Keep it concrete and visual. "
            "Output STRICT JSON with keys: title (string), panels (array of 3 strings), moral (string)."
        )

        user_content = [
            {"type": "text", "text": f"Create a short kids' story from these three panels in a {mood} tone."},
            {"type": "image_url", "image_url": {"url": data_urls[0]}},
            {"type": "image_url", "image_url": {"url": data_urls[1]}},
            {"type": "image_url", "image_url": {"url": data_urls[2]}},
            {"type": "text", "text": (
                "Return ONLY JSON, no backticks, no extra text. Example format:\n"
                '{ "title": "A Day at the Park", '
                '"panels": ["First…", "Then…", "Finally…"] }'
            )},
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.6,
            max_tokens=400,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ],
        )

        text = (resp.choices[0].message.content or "").strip()
        title, panels, moral = _parse_json_like(text)

        # minimal sanity: ensure 3 strings
        panels = [(panels[i] if i < len(panels) and isinstance(panels[i], str) else "").strip() for i in range(3)]
        return title.strip(), panels, moral.strip()

    # --- Compatibility layer with main.py ---
    def build_scenes(self, images: List[Image.Image], labels: List[List[str]]):
        """
        Save images for later generation and return minimal scene/delta structures
        expected by main.py. Labels are passed through; captions left empty here.
        """
        if not images or len(images) != 3:
            raise ValueError("Exactly three images are required")
        self._last_images = images
        scenes = []
        for i in range(3):
            scenes.append({
                "caption": "",
                "labels": (labels[i] if i < len(labels) else []),
            })
        # two deltas between 3 panels; keep empty
        deltas = [{"added": [], "removed": []}, {"added": [], "removed": []}]
        return scenes, deltas

    def generate_from_scenes(self, scenes, deltas, mood: str = "friendly") -> Tuple[str, List[str], str]:
        """
        Use the last provided images (via build_scenes) to generate a story.
        Returns (title, panels[3], moral).
        """
        if not self._last_images:
            raise RuntimeError("No images available; call build_scenes first")
        return self.generate_from_images(self._last_images, mood=mood)
