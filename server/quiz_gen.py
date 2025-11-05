# server/quiz_gen.py
from __future__ import annotations
"""
Caption-aware quiz generator with optional GPT-4o mini.

What it does
- Turns a short image caption into EXACTLY N (default 3) kid-friendly MCQs.
- Primary path (when configured): OpenAI Chat Completions with gpt-4o-mini.
- Fallbacks: FLAN-T5-small (text2text) or GPT-2 style local generation.
- If no LLM is available, uses a grounded dynamic generator.

Output shape per item:
{
  "question": str,
  "options": [str, str, str],
  "answer_index": 0|1|2
}

Env toggles
- OPENAI_API_KEY           -> enables OpenAI path if present and QUIZ_PROVIDER=openai
- QUIZ_PROVIDER=openai     -> select OpenAI path
- QUIZ_OPENAI_MODEL        -> defaults to "gpt-4o-mini"
- QUIZGEN_MODEL=flan|gpt2  -> selects HF fallback preference (default flan)
"""

from typing import List, Optional
from pathlib import Path
import os, re, random

# ---------- Optional OpenAI client ----------
try:
    from openai import OpenAI  # openai>=1.x
    _HAS_OPENAI = True
except Exception:
    OpenAI = None
    _HAS_OPENAI = False

def _use_openai_for_quiz() -> bool:
    prov = (os.getenv("QUIZ_PROVIDER") or "").strip().lower()
    return bool(os.getenv("OPENAI_API_KEY")) and prov == "openai" and _HAS_OPENAI

def _openai_client():
    return OpenAI()  # pulls OPENAI_API_KEY from env

# ---------- Robust parsers for the LLM output ----------
_OPTION_RE   = re.compile(r'^([A-Ca-c])[\)\.:\-]\s*(.+)$')
_QUESTION_RE = re.compile(r'^(\d+)[\)\.:\-]\s*(.+)$')
_ANSWER_RE   = re.compile(r'^(?:Answer|Correct)\s*[:\-]\s*([A-Ca-c])\b')

ANIMALS = {"dog","cat","bird","rabbit","horse","cow","sheep","goat","duck","fish"}
PERSON_WORDS = {"man","woman","boy","girl","child","person","people"}

FRUITS = {
    "apple","banana","orange","mango","grape","strawberry","pineapple",
    "watermelon","lemon","lime","pear","peach","cherry","tomato","coconut",
    "papaya","guava","kiwi","plum","pomegranate","avocado"
}
VEHICLES = {"car","bus","bicycle","train","boat","airplane","truck","motorcycle","van","ship"}
FURNITURE = {"chair","table","bed","sofa","lamp"}
BIRDS = {"bird","eagle","owl","penguin","parrot","peacock","duck","chicken","seagull","sparrow","pigeon","crow","flamingo"}
FISH  = {"fish","whale","dolphin","goldfish","salmon","shark","ray"}
PLACES = {"park","beach","kitchen","classroom","forest","street","playground","garden","farm","room","table"}

def _llm_kind() -> str:
    v = (os.getenv("QUIZGEN_MODEL") or "").strip().lower()
    return "gpt2" if v in {"gpt2","distilgpt2"} else "flan"  # default flan

class QuizGenerator:
    def __init__(self, model_root: Path | None = None, shared_pipe=None):
        self.err: Optional[str] = None
        self._pipe = shared_pipe
        self._device = -1
        self._rng = random.Random(1337)
        self.kind = _llm_kind()
        self.ready = shared_pipe is not None
        if not self.ready:
            self._try_load(model_root)

    # ---------------- model loading ----------------
    def _try_load(self, model_root: Path | None) -> None:
        """
        Load HF fallback models. If OpenAI is selected, we mark ready and skip heavy loads.
        """
        try:
            # If using OpenAI, no local pipeline is needed
            if _use_openai_for_quiz():
                self._pipe = "openai"
                self.kind = "openai"
                self.ready = True
                self.err = None
                return

            os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
            os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
            os.environ.setdefault("USE_TF", "0")
            os.environ.setdefault("USE_FLAX", "0")

            if self._pipe is not None:
                self.ready = True
                self.err = None
                return

            import torch  # type: ignore
            from transformers import pipeline  # type: ignore

            self._device = 0 if torch.cuda.is_available() else -1

            if self.kind == "flan":
                # small + fast; override with QUIZGEN_FLAN_MODEL if needed
                model_name = os.getenv("QUIZGEN_FLAN_MODEL", "google/flan-t5-small")
                self._pipe = pipeline("text2text-generation", model=model_name, device=self._device)
            else:
                # GPT-2 family fallback
                candidates: list[dict] = []
                if model_root is not None:
                    local = (Path(model_root) / "gpt2").resolve()
                    if local.exists():
                        candidates.append({"model": str(local), "local_files_only": True})
                candidates.append({"model": "gpt2"})
                candidates.append({"model": "distilgpt2"})

                last_err: Exception | None = None
                from transformers import pipeline as _pipe  # type: ignore
                for opts in candidates:
                    try:
                        self._pipe = _pipe("text-generation", device=self._device, framework="pt", **opts)
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        continue
                if last_err is not None and self._pipe is None:
                    raise last_err

            self.ready = True
            self.err = None
        except Exception as e:
            self.ready = False
            self.err = ("quizgen_load_failed: " + str(e)
                        + "; install transformers/torch or switch model.")

    # ---------------- public API ----------------
    def generate(self, caption: str, num_questions: int = 3) -> List[dict]:
        caption = (caption or "").strip()
        if not caption:
            raise ValueError("empty_caption")

        expected = max(1, min(3, int(num_questions or 3)))

        # Precompute facts for grounding (used by all paths)
        facts = self._extract_facts(caption)

        # OpenAI path (preferred when configured)
        if self.ready and self._pipe == "openai" and _use_openai_for_quiz():
            try:
                return self._openai_questions(caption, expected, facts)
            except Exception:
                # fall back to HF or dynamic
                pass

        # HF fallback path
        if self.ready and self._pipe is not None:
            try:
                return self._hf_llm_questions(caption, expected, facts)
            except Exception:
                return self._dynamic_questions(caption, expected, facts)

        # Dynamic fallback
        return self._dynamic_questions(caption, expected, facts)

    # ---------------- OpenAI path ----------------
    def _openai_questions(self, caption: str, expected: int, facts: dict) -> List[dict]:
        client = _openai_client()
        model = os.getenv("QUIZ_OPENAI_MODEL", "gpt-4o-mini")
        prompt_caption = caption[:240]
        hint_line = self._facts_hint_line(facts)

        system = (
            "You turn a short image caption into EXACTLY "
            f"{expected} kid-friendly MCQs. Each item has:\n"
            "1) <question>\nA) <choice>\nB) <choice>\nC) <choice>\nAnswer: <A/B/C>\n"
            "Rules: one clear sentence per question; 3 options only; one correct answer; "
            "no explanations."
        )
        user = (
            f'Caption: "{prompt_caption}".\n{hint_line}\n'
            "Write the quiz now in the exact format."
        )

        resp = client.chat.completions.create(
            model=model,
            temperature=0.6,
            max_tokens=400,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        parsed = self._parse_output(text, expected)

        if len(parsed) >= expected:
            # Randomize options but preserve correctness
            for item in parsed:
                opts = item["options"][:3]
                correct = opts[item["answer_index"]]
                self._rng.shuffle(opts)
                item["options"] = opts
                item["answer_index"] = opts.index(correct)
            return parsed[:expected]

        # If parsing failed, revert to dynamic generation
        return self._dynamic_questions(caption, expected, facts)

    # ---------------- HF LLM path (FLAN/GPT-2) ----------------
    def _hf_llm_questions(self, caption: str, expected: int, facts: dict) -> List[dict]:
        from transformers import set_seed  # type: ignore

        set_seed(self._rng.randint(0, 10_000_000))
        prompt_caption = caption[:240]
        hint_line = self._facts_hint_line(facts)

        if self.kind == "flan":
            # Instruction for text2text models (FLAN)
            prompt = (
                f'Caption: "{prompt_caption}".\n'
                f"{hint_line}\n"
                f"Write exactly {expected} SHORT, kid-friendly multiple-choice questions about this caption.\n"
                "Rules:\n"
                "- Each question is one sentence, clear and simple.\n"
                "- Provide exactly three answer choices labeled A), B), C).\n"
                "- Put the correct letter on a separate line like: Answer: B\n"
                "- Do not add explanations or extra text.\n\n"
                "Example format:\n"
                "1) What is happening?\n"
                "A) Option one\n"
                "B) Option two\n"
                "C) Option three\n"
                "Answer: B\n\n"
                "Now the quiz:\n"
            )
            eos_token_id = getattr(getattr(self._pipe, "tokenizer", None), "eos_token_id", None)
            out = self._pipe(
                prompt,
                max_new_tokens=220,
                do_sample=True,
                temperature=0.7,
                top_p=0.92,
                repetition_penalty=1.1,
                return_full_text=False,
                eos_token_id=eos_token_id,
            )
            text = (out[0].get("generated_text") or out[0].get("text") or "") if isinstance(out, list) and out else ""
        else:
            # GPT-2 style (text-generation)
            prompt = (
                "Create multiple-choice questions from the caption.\n"
                f'Caption: "{prompt_caption}"\n'
                f"{hint_line}\n"
                f"Write exactly {expected} questions. For each question use this exact format:\n"
                "1) <question>\n"
                "A) <choice>\n"
                "B) <choice>\n"
                "C) <choice>\n"
                "Answer: <A/B/C>\n\n"
                "Now the quiz:\n"
            )
            eos_token_id = getattr(getattr(self._pipe, "tokenizer", None), "eos_token_id", None)
            out = self._pipe(
                prompt,
                max_new_tokens=260,
                do_sample=True,
                temperature=0.8,
                top_p=0.92,
                repetition_penalty=1.1,
                eos_token_id=eos_token_id,
            )
            text = (out[0].get("generated_text") or out[0].get("text") or "") if isinstance(out, list) and out else ""

        parsed = self._parse_output(str(text), expected)

        if len(parsed) >= expected:
            for item in parsed:
                opts = item["options"][:3]
                correct = opts[item["answer_index"]]
                self._rng.shuffle(opts)
                item["options"] = opts
                item["answer_index"] = opts.index(correct)
            return parsed[:expected]

        return self._dynamic_questions(caption, expected, facts)

    # ---------------- parsing ----------------
    def _parse_output(self, raw: str, expected: int) -> List[dict]:
        lines = [line.strip() for line in raw.splitlines()]
        items: List[dict] = []
        current: Optional[dict] = None

        for line in lines:
            if not line:
                continue

            q_match = _QUESTION_RE.match(line)
            if q_match:
                if current and len(current.get("options", [])) >= 3 and current.get("answer") is not None:
                    items.append(current)
                current = {"question": q_match.group(2).strip(), "options": [], "answer": None}
                continue

            if current is None:
                continue

            opt_match = _OPTION_RE.match(line)
            if opt_match:
                if len(current["options"]) < 3:
                    current["options"].append(opt_match.group(2).strip())
                continue

            ans_match = _ANSWER_RE.match(line)
            if ans_match:
                try:
                    current["answer"] = "ABC".index(ans_match.group(1).upper())
                except ValueError:
                    current["answer"] = None

        if current and len(current.get("options", [])) >= 3 and current.get("answer") is not None:
            items.append(current)

        cleaned: List[dict] = []
        for entry in items:
            opts = [self._clean_option(o) for o in entry.get("options", [])][:3]
            ans = entry.get("answer")
            if len(opts) != 3 or ans is None or not (0 <= ans < 3):
                continue
            cleaned.append({
                "question": self._clean_question(entry.get("question", "")),
                "options": opts,
                "answer_index": int(ans),
            })
            if len(cleaned) == expected:
                break

        return cleaned

    # ---------------- dynamic fallback (caption-aware, randomized) ----------------
    def _dynamic_questions(self, caption: str, expected: int, facts: dict) -> List[dict]:
        rng = self._rng
        cap = (caption or "").strip().rstrip(".")
        who = facts.get("who")
        animal = facts.get("animal")
        where_raw = facts.get("where_raw")
        where = facts.get("where")
        action = facts.get("action")
        objects_list: List[str] = list(facts.get("objects", []) or [])

        objects_list = [o.strip().lower() for o in objects_list if isinstance(o, str) and o.strip()]
        seen = set(); objects_list = [o for o in objects_list if not (o in seen or seen.add(o))]

        def _uniq3(correct: str, wrongs: List[str]) -> List[str]:
            seen = set(); opts = []
            for o in [correct] + list(wrongs):
                k = (o or "").strip().lower()
                if not k or k in seen:
                    continue
                seen.add(k); opts.append(o)
                if len(opts) == 3: break
            fillers = ["Something else", "Not sure", "I forget"]
            for f in fillers:
                if len(opts) == 3: break
                if f.lower() not in seen:
                    seen.add(f.lower()); opts.append(f)
            return opts[:3]

        used_q = set()
        def _norm_q(q: str) -> str:
            return re.sub(r"\s+", " ", (q or "").strip().rstrip(" ?!.")).lower()

        def add(q: str, correct: str, wrongs: List[str]) -> None:
            key = _norm_q(q)
            if not key or key in used_q:
                return
            opts = _uniq3(correct, wrongs)
            if (correct or "").strip() and all((o.strip().lower() != correct.strip().lower()) for o in opts):
                opts[0] = correct
            rng.shuffle(opts)
            questions.append({
                "question": q if q.strip().endswith("?") else (q.strip() + "?"),
                "options": opts,
                "answer_index": opts.index(next(o for o in opts if o.strip().lower() == correct.strip().lower())) if correct.strip() else 0,
            })
            used_q.add(key)

        questions: List[dict] = []

        if objects_list:
            present = objects_list[:]
            correct = present[rng.randrange(len(present))].capitalize()
            pool = list((ANIMALS | FRUITS | VEHICLES | FURNITURE | BIRDS | FISH) - set(present))
            rng.shuffle(pool)
            wrongs = [w.capitalize() for w in pool[:2]] if len(pool) >= 2 else ["Robot", "Spaceship"]
            add("Which of these is in the picture", correct, wrongs)

        if where_raw or where:
            place_answer = self._normalize_where(where or where_raw or "a familiar place")
            distract = [p for p in ["Under the ocean", "On the Moon", "Inside a volcano", "In space"] if place_answer.lower() not in p.lower()]
            wrongs = distract[:2] if len(distract) >= 2 else ["Under the ocean", "On the Moon"]
            add("Where is this happening?", place_answer, wrongs)

        if action:
            subj = (who or animal or (objects_list[0].capitalize() if objects_list else "character"))
            correct = action.capitalize()
            action_pool = ["Sleeping", "Running", "Drawing", "Reading", "Dancing", "Singing", "Jumping", "Playing"]
            action_pool = [a for a in action_pool if a.lower() != action.lower()]
            rng.shuffle(action_pool)
            wrongs = action_pool[:2] if len(action_pool) >= 2 else ["Sleeping", "Running"]
            add(f"What is the {subj} doing?", correct, wrongs)

        if len(questions) < expected:
            if objects_list:
                up = [o for o in objects_list[:3]]
                pretty = ", ".join(w.capitalize() for w in up)
                add("What things does the picture show?", pretty, ["Only stars in space", "Nothing at all"])
            else:
                correct = (cap[:1].upper() + cap[1:] + ".") if cap else "A simple scene."
                add("What is shown in the picture?", correct, ["A rocket in space.", "Nothing at all."])

        if len(questions) < expected and len(objects_list) >= 2:
            present = objects_list[:]
            pool = list((ANIMALS | FRUITS | VEHICLES | FURNITURE | BIRDS | FISH) - set(present))
            rng.shuffle(pool)
            not_there = (pool[0] if pool else "spaceship").capitalize()
            wrongs = [present[0].capitalize(), present[1].capitalize()]
            add("Which of these is NOT in the picture?", not_there, wrongs)

        if len(questions) < expected:
            cats = {"Fruits": FRUITS, "Animals": ANIMALS, "Vehicles": VEHICLES}
            counts = {k: 0 for k in cats}
            for o in objects_list:
                lo = o.lower()
                for cname, cset in cats.items():
                    if lo in cset: counts[cname] += 1
            if counts and max(counts.values()) >= 2:
                best_cat = max(counts.items(), key=lambda kv: kv[1])[0]
                other = [c for c in cats.keys() if c != best_cat]
                self._rng.shuffle(other)
                add("These things are mostly what?", best_cat, other[:2])

        while len(questions) < expected:
            if objects_list:
                pool = list(objects_list); rng.shuffle(pool)
                correct = pool[0].capitalize()
                distract = list((ANIMALS | FRUITS | VEHICLES | FURNITURE | BIRDS | FISH) - {pool[0]})
                rng.shuffle(distract)
                wrongs = [distract[0].capitalize(), distract[1].capitalize()] if len(distract) >= 2 else ["Robot", "Spaceship"]
                phr = rng.choice(["Which item do you see", "Pick something you can spot", "Which is present in the picture"])
                add(phr, correct, wrongs)
            else:
                add("How does the scene feel", "Calm and friendly.", ["Very scary.", "As loud as a concert."])

        return questions[:expected]

    # ---------------- helpers ----------------
    @staticmethod
    def _clean_option(text: str) -> str:
        cleaned = text.strip().rstrip(" .")
        return cleaned if cleaned else "Option"

    @staticmethod
    def _clean_question(text: str) -> str:
        cleaned = text.strip().rstrip("?")
        if not cleaned:
            return "What is happening in the caption?"
        if not cleaned.endswith("?"):
            cleaned += "?"
        return cleaned

    def _normalize_where(self, where: str) -> str:
        w = (where or "").strip()
        w = re.sub(r"\s{2,}", " ", w)
        return w[:1].upper() + w[1:] if w else "A familiar place"

    def _extract_facts(self, caption: str) -> dict:
        """
        Lightweight cues from a short caption (no heavy NLP deps):
        - who / animal
        - action (first -ing verb)
        - where (preposition + object)
        - object (first noun-ish token not in stoplist)
        - objects: from trailing hints like "Objects: a, b, c" or "Labels: ..."
        """
        tl = (caption or "").strip().lower()
        tokens = [w.strip(".,!?:;") for w in tl.split() if w]

        who = next((w for w in tokens if w in PERSON_WORDS), None)
        animal = next((
            w[:-1] if (w.endswith("s") and w[:-1] in ANIMALS) else w
            for w in tokens if (w in ANIMALS or (w.endswith("s") and w[:-1] in ANIMALS))
        ), None)
        action = next((w for w in tokens if w.endswith("ing")), None)

        where_raw = None
        for prep in ["around","near","beside","by","on","in","at","under","inside"]:
            m = re.search(rf"\b{prep}\s+([a-z0-9' \-]+)", tl)
            if m:
                where_raw = f"{prep} " + m.group(1).strip()
                break

        stop = PERSON_WORDS | ANIMALS | {"a","an","the","and","of","to","with","without","while","on","in","at","near","by","around","beside"}
        obj = next((w for w in tokens if w not in stop and not w.endswith("ing")), None)

        where = where_raw
        if where and len(where.split()) > 6:
            where = " ".join(where.split()[:6])

        objects_list: List[str] = []
        try:
            m = re.search(r"\b(?:objects?|labels?)\s*[:\-]\s*(.+)$", tl, flags=re.I)
            if m:
                tail = m.group(1)
                parts = re.split(r",|\band\b|/|\|", tail)
                objects_list = [p.strip().strip(" .!") for p in parts if p and p.strip()]
        except Exception:
            objects_list = []

        return {
            "who": who,
            "animal": animal,
            "action": action,
            "where_raw": where_raw,
            "where": where,
            "object": obj,
            "objects": objects_list,
        }

    def _facts_hint_line(self, facts: dict) -> str:
        who = facts.get("who") or ("someone" if facts.get("who") else None)
        act = facts.get("action")
        whr = facts.get("where_raw")
        objs: List[str] = list(facts.get("objects") or [])

        hints = []
        if who:        hints.append(f"Person: {who}")
        if facts.get("animal"): hints.append(f"Animal: {facts['animal']}")
        if act:        hints.append(f"Action: {act}")
        if whr:        hints.append(f"Place: {whr}")
        if objs:
            short = ", ".join([o for o in objs[:5]])
            hints.append(f"Objects: {short}")
        return "Hints: " + ", ".join(hints) if hints else "Hints: Keep questions simple (who/what, where, how it feels)."


__all__ = ["QuizGenerator"]
