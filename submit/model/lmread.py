"""Read a Hangul ship name back into the Latin name a ship register carries, with a public pretrained
instruction-tuned language model stored in model/lm/ (Hugging Face format).

A record may give the Korean translation of the name (바다 공주 for SEA PRINCESS) instead of its sound; the
rankers, trained only on transliterations, cannot place such a name, but a language model can read it back.
Each name is read on its own with beam search (no sampling, no batching), so a name always gets the same
readings whatever else the test file holds. Needs a GPU: without one the reader stays off and nothing changes.
The few-shot examples (model/lm_shots.json, from tools/build_lm_shots.py) are train records.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

PROMPTS = {
    "sound": ("You convert the Korean (Hangul) spelling of a ship's name back to the ship's name as written in an "
              "international ship register, in Latin capital letters. Keep numbers. Answer with the name only."),
    "meaning": ("You convert the Korean (Hangul) spelling of a ship's name back to the ship's name as written in an "
                "international ship register, in Latin capital letters. The Korean may spell the name's sound or "
                "translate its words. Keep numbers. Answer with the name only."),
}


def load_shots(path: Path) -> list[tuple[str, str]]:
    with open(path, encoding="utf-8") as fh:
        return [(s["kor"], s["eng"]) for s in json.load(fh)]


def messages(name: str, shots: list[tuple[str, str]], system: str) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for kor, eng in shots:
        out += [{"role": "user", "content": kor}, {"role": "assistant", "content": eng}]
    return out + [{"role": "user", "content": name}]


def clean(text: str) -> str:
    text = text.strip().splitlines()[0] if text.strip() else ""
    return re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z\-\.\s/&']", " ", text)).strip().upper()


class Reader:
    def __init__(self, path: Path, shots: list[tuple[str, str]], prompt: str = "meaning", beams: int = 4,
                 returns: int = 3, max_new: int = 24, device: str = "cuda"):
        self.ok, self.cache, self.shots, self.system = False, {}, shots, PROMPTS[prompt]
        self.beams, self.returns, self.max_new, self.device = beams, min(returns, beams), max_new, device
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            return
        if device == "cuda" and not torch.cuda.is_available():
            return
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.bfloat16).to(device).eval()
        self.ok = True

    def read(self, name: str) -> list[str]:
        """Up to `returns` distinct readings, best first."""
        if not self.ok:
            return []
        if name not in self.cache:
            torch, tok = self.torch, self.tok
            ids = tok.apply_chat_template(messages(name, self.shots, self.system), add_generation_prompt=True,
                                          return_tensors="pt").to(self.device)
            with torch.no_grad():
                gen = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=self.max_new,
                                          do_sample=False, num_beams=self.beams, num_return_sequences=self.returns,
                                          early_stopping=True, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            reads = [clean(tok.decode(g[ids.shape[1]:], skip_special_tokens=True)) for g in gen]
            self.cache[name] = [r for r in dict.fromkeys(reads) if r]
        return self.cache[name]
