"""Shared, row-independent matching code (used by training and by script.py).

Every prediction looks at one query row, the official vessel catalog and
lookup tables built from the official train set. Nothing depends on other
evaluation rows, so a row gets the same prediction whether it is scored
alone, in a subset, or in any order.
"""

from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein, OSA

# ---------------------------------------------------------------------------
# Basic text normalisation
# ---------------------------------------------------------------------------

HANGUL_RE = re.compile("[가-힣]")
PREFIX_RE = re.compile(r"^(?:M\s*/\s*[VT]|M\.\s*[VT]\.?|MV|MT)\s+")
PAREN_RE = re.compile(r"\s*\([^)]*\)?")
NO_RE = re.compile(r"\bNO\b\.?\s*(?=\d)")
ROMAN = {"II": "2", "III": "3", "IV": "4", "VI": "6", "VII": "7", "VIII": "8", "IX": "9"}


def is_hangul(value) -> bool:
    return isinstance(value, str) and bool(HANGUL_RE.search(value))


JAMO_RE = re.compile("[\u1100-\u11ff]")


def hangul_text(value) -> bool:
    """Hangul syllables, or with JAMO_KEYS also loose jamo, in a name or word: read it as Hangul."""
    return isinstance(value, str) and bool(HANGUL_RE.search(value) or (JAMO_KEYS and JAMO_RE.search(value)))


def upper(value) -> str:
    if not isinstance(value, str):
        return ""
    return unicodedata.normalize("NFKC", value).upper().replace("\xa0", " ").strip()


def alnum(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum())


def strip_prefix(name: str) -> str:
    """Drop ship-type prefixes such as 'M/V ' that never appear in the catalog, and with PAREN_STRIP
    any bracketed remark such as '(EX-SEA STAR)'."""
    if PAREN_STRIP and "(" in name:
        name = PAREN_RE.sub(" ", name).strip() or name
    return PREFIX_RE.sub("", name)


def tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^0-9A-Z가-힣]+", name) if t]


def numbers_of(name: str) -> tuple[str, ...]:
    """Digit groups in a name, with roman numerals mapped to digits."""
    out = re.findall(r"\d+", name)
    for tok in tokens(name):
        if tok in ROMAN:
            out.append(ROMAN[tok])
    return tuple(sorted(str(int(x)) for x in out))


def letter_tokens(name: str) -> list[str]:
    """Alphabetic tokens of a Latin name, without numbers, roman numerals or 'NO.'."""
    name = NO_RE.sub(" ", name)
    out = []
    for tok in tokens(name):
        if tok in ROMAN or tok.isdigit():
            continue
        tok = re.sub(r"\d+", "", tok)
        if tok:
            out.append(tok)
    return out


def positive_float(value) -> float:
    if isinstance(value, str):
        # "12,345", " 12345 " or "12345 GT" read as text
        value = re.sub(r"[^0-9.eE+-]+$", "", re.sub(r"[,\s]", "", value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) and number > 0 else math.nan


def collapse(s: str) -> str:
    out = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------------------
# English -> coarse phonetic key (shared alphabet with Hangul keys)
# ---------------------------------------------------------------------------

VOWELS = set("AEIOU")


def _rules(th: str, ch: str, pinyin: bool = False, keep_r: bool = False):
    soft = [] if pinyin else [
        (re.compile(r"C(?=[EIY])"), "S"),
        (re.compile(r"G(?=[EIY])"), "S"),
    ]
    pin = [
        (re.compile(r"Q"), "S"),
        (re.compile(r"X"), "S"),
        (re.compile(r"C(?!H)"), "S"),
    ] if pinyin else []
    # Korean adds an 'i' after a final /dʒ/, /tʃ/ or /ʃ/ (bridge -> 브리지, beach -> 비치)
    final_i = [] if ch == "K" else [
        (re.compile(r"(?<=[AEIOU].)AGE$|(?<=[AEIOU]..)AGE$"), "IJI"),
        (re.compile(r"D?GE$"), "JI"),
        (re.compile(r"T?CH$"), "CHI"),
        (re.compile(r"SH$"), "SHI"),
    ]
    if pinyin:
        # Korean spelling of pinyin finals (장인, 톈진, 선전, 위안, 중)
        vowels = [
            (re.compile(r"YUE"), "UE"),
            (re.compile(r"YU|(?<=[JQX])U(?!E)"), "UI"),
            (re.compile(r"ONG"), "UNG"),
            (re.compile(r"(?<![IUY])E(?!I)"), "O"),
            (re.compile(r"[IY]AN(?!G)"), "IEN"),
            (re.compile(r"(?<=[JQ])I(?=[AEOU])"), ""),
            (re.compile(r"NG"), "N"),
        ]
    else:
        # vowel spellings that depend on the letters around them
        vowels = [
            # short U before two consonants (bulk -> 벌크), but not before an onset cluster (supreme -> 슈프림)
            (re.compile(r"(?<=[^AEIOU])U(?=(?![PBTDKG]R)[^AEIOUY]{2}|[^AEIOUY]$)"), "O"),
            (re.compile(r"IGH"), "AI"),
            (re.compile(r"^([^AEIOU]*)A([^AEIOU])E$"), r"\1EI\2E"),
            (re.compile(r"^([^AEIOU]*)I([^AEIOU])E$"), r"\1AI\2E"),
            (re.compile(r"^([^AEIOU]+)Y$"), r"\1AI"),
            # silent final E lengthens E too (supreme -> 슈프림, serene -> 세린)
            (re.compile(r"(?<=[^AEIOU])E(?=[^AEIOUWXYR]E$)"), "I"),
            (re.compile(r"AY"), "EI"),
        ] + ([] if keep_r else [
            (re.compile(r"EER(?![AEIOUY])"), "IO"),
            (re.compile(r"ER(?![AEIOUY])"), "O"),
        ]) + [
            (re.compile(r"EE"), "I"),
            (re.compile(r"OO"), "U"),
        ] + ([] if ch == "K" else [
            # sea -> 씨, eagle -> 이글, journey -> 저니
            (re.compile(r"EA"), "I"),
            (re.compile(r"EY$"), "I"),
        ])
        if keep_r:
            # continental spelling: R is sounded (아르페지오, 마르코) and GGI is soft
            vowels.insert(0, (re.compile(r"GG(?=[EIY])"), "J"))
    drop_r = [] if keep_r else [(re.compile(r"(?<=[AEIOU])R(?![AEIOUY])"), "")]
    # silent initial H (honour -> 아너, hour -> 아워)
    silent = [(re.compile(r"^H(?=ONOU?R|ONEST|OUR|EIR)"), "")]
    return silent + vowels + final_i + pin + [
        (re.compile(r"PH"), "F"),
        (re.compile(r"^GH"), "K"),
        (re.compile(r"GH(?=T|$)"), ""),
        (re.compile(r"CK"), "K"),
        (re.compile(r"SCH"), "S"),
        (re.compile(r"TCH"), "S"),
        (re.compile(r"ZH"), "S"),
        (re.compile(r"SH"), "S"),
        (re.compile(r"CH"), ch),
        (re.compile(r"TH"), th),
        (re.compile(r"QU"), "KU"),
        (re.compile(r"Q"), "K"),
        (re.compile(r"^X"), "S"),
        (re.compile(r"X"), "KS"),
        (re.compile(r"WH"), "U"),
    ] + soft + [
        (re.compile(r"C"), "K"),
        (re.compile(r"G"), "K"),
        (re.compile(r"[JZ]"), "S"),
        (re.compile(r"[VFB]"), "P"),
        (re.compile(r"D"), "T"),
    ] + drop_r + [
        (re.compile(r"R"), "L"),
        (re.compile(r"W"), "U"),
        (re.compile(r"Y"), "I"),
        (re.compile(r"(?<=[AEIOU])H(?![AEIOU])"), ""),
    ]


_ENG_RULES = (_rules("T", "S"), _rules("S", "K"), _rules("T", "S", pinyin=True), _rules("T", "S", keep_r=True))
N_ALT = len(_ENG_RULES)


@lru_cache(maxsize=None)
def english_word_key(word: str, alt: int = 0) -> str:
    final_e = len(word) >= 3 and word.endswith("E") and word[-2] not in VOWELS
    for pattern, repl in _ENG_RULES[alt]:
        word = pattern.sub(repl, word)
    if final_e and word.endswith("E"):
        word = word[:-1]
    word = collapse(word)
    word = re.sub(r"NK(?![AEIOU])", "N", word)
    return collapse(word)


def english_key(text: str, alt: int = 0) -> str:
    return collapse("".join(english_word_key(w, alt) for w in letter_tokens(upper(text))))


def consonants(key: str) -> str:
    return collapse("".join(ch for ch in key if ch not in VOWELS))


def consonants_raw(key: str) -> str:
    """Consonant skeleton that keeps repeats separated by a vowel (L-E-L stays LL)."""
    return "".join(ch for ch in key if ch not in VOWELS)


def vowel_skeleton(key: str) -> str:
    """Key with every vowel run reduced to one placeholder (얼라이언스 ~ ALLIANCE)."""
    return re.sub("[AEIOU]+", "A", key)


# ---------------------------------------------------------------------------
# Hangul -> the same coarse phonetic alphabet
# ---------------------------------------------------------------------------

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"

CHO_CLS = ["K", "K", "N", "T", "T", "L", "M", "P", "P", "S", "S", "", "S", "S", "S", "K", "T", "P", "H"]
JUNG_CLS = ["A", "A", "IA", "IE", "O", "E", "IO", "IE", "O", "UA", "UE", "UE", "IO", "U", "UO",
            "UE", "UI", "IU", "", "I", "I"]
JONG_CLS = ["", "K", "K", "K", "N", "N", "N", "T", "L", "L", "M", "L", "L", "L", "P", "L", "M",
            "P", "P", "T", "T", "N", "T", "T", "K", "T", "P", "T"]

# Revised Romanization, for Korean-origin words (성호 -> SEONGHO, 우진 -> UJIN)
RR_CHO = ["G", "KK", "N", "D", "TT", "R", "M", "B", "PP", "S", "SS", "", "J", "JJ", "CH", "K", "T", "P", "H"]
RR_JUNG = ["A", "AE", "YA", "YAE", "EO", "E", "YEO", "YE", "O", "WA", "WAE", "OE", "YO", "U", "WO",
           "WE", "WI", "YU", "EU", "UI", "I"]
RR_JONG = ["", "K", "K", "K", "N", "N", "N", "T", "L", "K", "M", "L", "L", "L", "P", "L", "M",
           "P", "P", "T", "T", "NG", "T", "T", "K", "T", "P", "T"]

# Korean spellings of Latin letter names (for acronyms such as 케이엠티씨 -> KMTC)
LETTER_NAMES = sorted({
    "에이": "A", "비": "B", "씨": "C", "시": "C", "디": "D", "이": "E", "에프": "F",
    "지": "G", "쥐": "G", "에이치": "H", "에치": "H", "아이": "I", "제이": "J", "케이": "K",
    "엘": "L", "엠": "M", "엔": "N", "앤": "N", "오": "O", "피": "P", "큐": "Q",
    "알": "R", "아르": "R", "에스": "S", "티": "T", "유": "U", "브이": "V",
    "더블유": "W", "엑스": "X", "와이": "Y", "제트": "Z", "젯": "Z",
}.items(), key=lambda kv: -len(kv[0]))
KOREAN_NUMBER_WORDS = {"원": "1", "투": "2", "쓰리": "3", "포": "4", "파이브": "5"}


def _syllables(text: str):
    """(initial, medial, final) indices per syllable; with JAMO_KEYS also one triple per loose jamo
    (a damaged record such as 그ᅳ랜드 or 에ᅵᄋ스), its missing parts -1 (final 0)."""
    for ch in text:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            cho, rest = divmod(code, 588)
            jung, jong = divmod(rest, 28)
            yield cho, jung, jong
        elif JAMO_KEYS:
            o = ord(ch)
            if 0x1100 <= o <= 0x1112:
                yield o - 0x1100, -1, 0
            elif 0x1161 <= o <= 0x1175:
                yield -1, o - 0x1161, 0
            elif 0x11A8 <= o <= 0x11C2:
                yield -1, -1, o - 0x11A7


def hangul_key(text: str, eo: str = "O") -> str:
    """Coarse key; eo='A' reads ㅓ as an unstressed English vowel (얼라이언스 -> ALLIANCE)."""
    return collapse("".join((CHO_CLS[c] if c >= 0 else "") + (eo if v == 4 else JUNG_CLS[v] if v >= 0 else "")
                            + JONG_CLS[j] for c, v, j in _syllables(text)))


def romanize(text: str) -> str:
    return "".join((RR_CHO[c] if c >= 0 else "") + (RR_JUNG[v] if v >= 0 else "") + RR_JONG[j]
                   for c, v, j in _syllables(text))


def romanized_key(text: str) -> str:
    """Phonetic key of the romanized form; final ㅇ is a plain nasal, ㄱ is hard."""
    parts = []
    for c, v, j in _syllables(text):
        parts.append(("K" if c == 0 else RR_CHO[c] if c > 0 else "") + (RR_JUNG[v] if v >= 0 else "")
                     + ("N" if j == 21 else RR_JONG[j]))
    return english_key("".join(parts))


# Korean romanization spellings collapsed (RR and older systems: Heung-A Young ~ 흥아영 = heungayeong)
_KR_RULES = [(re.compile(p), r) for p, r in [
    (r"[^A-Z]", ""), (r"EO", "O"), (r"EU", "U"), (r"OU", "O"), (r"AE", "E"), (r"OE", "E"), (r"SH", "S"),
    (r"CH", "S"), (r"[JZ]", "S"), (r"G", "K"), (r"[BVF]", "P"), (r"D", "T"), (r"R", "L"), (r"W", "U"), (r"Y", "I"),
]]


def kr_key(text: str) -> str:
    text = upper(text)
    for pattern, repl in _KR_RULES:
        text = pattern.sub(repl, text)
    return collapse(text)


def spell_acronym(token: str) -> str | None:
    """Return the Latin acronym if the Hangul token is only letter names."""
    out, i = [], 0
    while i < len(token):
        for kor, lat in LETTER_NAMES:
            if token.startswith(kor, i):
                out.append(lat)
                i += len(kor)
                break
        else:
            return None
    return "".join(out) if len(out) >= 2 else None


def strip_korean_suffix(name: str) -> str:
    """Remove the ship suffix '호' and turn '제N호' markers into plain numbers."""
    name = re.sub(r"제\s*(\d+)\s*호", r" \1 ", name)
    name = re.sub(r"제\s*(\d+)", r" \1 ", name)
    name = re.sub(r"(\d+)\s*호", r" \1 ", name)
    name = name.strip()
    if len(name) > 2 and name.endswith("호"):
        name = name[:-1]
    return re.sub(r"\s+", " ", name).strip()


def korean_tokens(name: str) -> tuple[list[str], tuple[str, ...]]:
    """Word tokens (digits removed) and numbers of a Hangul ship name."""
    base = strip_korean_suffix(upper(name))
    toks = [t for t in re.split(r"[\s\-\.\(\)/&']+", re.sub(r"(\d+)", r" \1 ", base)) if t]
    nums, words = [], []
    for idx, tok in enumerate(toks):
        if tok.isdigit():
            nums.append(tok)
        elif tok in KOREAN_NUMBER_WORDS and idx == len(toks) - 1 and idx > 0:
            nums.append(KOREAN_NUMBER_WORDS[tok])
        elif tok in ROMAN and idx > 0:
            nums.append(ROMAN[tok])
        else:
            words.append(tok)
    return words, tuple(sorted(str(int(x)) for x in nums))


def token_readings(tok: str) -> list[str]:
    """Phonetic keys a single token may correspond to."""
    if not hangul_text(tok):
        return [english_key(tok)]
    out = [hangul_key(tok), romanized_key(tok)]
    acro = spell_acronym(tok)
    if acro:
        out.append(english_key(acro))
    else:
        for cut in range(len(tok) - 1, 1, -1):
            acro = spell_acronym(tok[:cut])
            if acro:
                out.append(collapse(english_key(acro) + hangul_key(tok[cut:])))
                break
    return list(dict.fromkeys(out))


def dictionary_part(tables: dict, tok: str, exclude: str | None) -> str | None:
    """English reading of a Hangul token learned from other train vessels."""
    entry = tables.get("kdict", {}).get(tok)
    if not entry:
        return None
    best, best_n = None, 0
    for eng, vids in sorted(entry.items()):
        n = len(vids) - (1 if exclude is not None and exclude in vids else 0)
        if n > best_n:
            best, best_n = eng, n
    return best


RUN_RE = re.compile(r"(.)\1+")


def _loose(text: str) -> int:
    return len(JAMO_RE.findall(unicodedata.normalize("NFC", text)))


def repair_jamo(name: str, depth: int = 3, beam: int = 6, keep: int = 3) -> list[str]:
    """Names a jamo-level typo may have come from: up to `depth` jamo left out or neighbours swapped back
    (a doubled jamo's copy is the cheapest to drop) until every jamo composes again, fewest edits first."""
    start = unicodedata.normalize("NFD", name)
    if not _loose(start):
        return []
    frontier, seen, done = [(0.0, start)], {start}, []
    for _ in range(depth):
        nxt = []
        for cost, s in frontier:
            for i in range(len(s)):
                if JAMO_RE.match(s[i]):
                    twin = (i > 0 and s[i - 1] == s[i]) or (i + 1 < len(s) and s[i + 1] == s[i])
                    nxt.append((cost + (0.5 if twin else 1.0), s[:i] + s[i + 1:]))
                if i + 1 < len(s) and s[i] != s[i + 1] and (JAMO_RE.match(s[i]) or JAMO_RE.match(s[i + 1])):
                    nxt.append((cost + 1.0, s[:i] + s[i + 1] + s[i] + s[i + 2:]))
        scored = []
        for cost, s in nxt:
            if s not in seen:
                seen.add(s)
                scored.append((_loose(s), cost, s))
        scored.sort(key=lambda x: (x[0], x[1]))
        done += [(c, s) for n, c, s in scored if n == 0]
        frontier = [(c, s) for n, c, s in scored if n > 0][:beam]
        if len(done) >= keep or not frontier:
            break
    done.sort(key=lambda x: x[0])
    out = []
    for _, s in done:
        t = unicodedata.normalize("NFC", s)
        if t not in out and HANGUL_RE.search(t):
            out.append(t)
    return out[:keep]


# Korean 2-set keyboard: the key under each jamo (shifted keys give the tense consonants and ㅒ ㅖ), compound
# vowels and finals typed as two keys
_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = ("", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ",
         "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")
_IME_KEY = dict(zip("ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔㅁㄴㅇㄹㅎㅗㅓㅏㅣㅋㅌㅊㅍㅠㅜㅡㅃㅉㄸㄲㅆㅒㅖ", "QWERTYUIOPASDFGHJKLZXCVBNMQWERTOP"))
_IME_KEY.update({"ㅘ": "HK", "ㅙ": "HO", "ㅚ": "HL", "ㅝ": "NJ", "ㅞ": "NP", "ㅟ": "NL", "ㅢ": "ML", "ㄳ": "RT",
                 "ㄵ": "SW", "ㄶ": "SG", "ㄺ": "FR", "ㄻ": "FA", "ㄼ": "FQ", "ㄽ": "FT", "ㄾ": "FX", "ㄿ": "FV", "ㅀ": "FG",
                 "ㅄ": "QT", "": ""})
# conjoining jamo (what NFKC makes of the compatibility letters) back to the compatibility letters
_IME_CONJ = {**{chr(0x1100 + i): c for i, c in enumerate(_CHO)}, **{chr(0x1161 + i): c for i, c in enumerate(_JUNG)},
             **{chr(0x11A7 + i): c for i, c in enumerate(_JONG) if i}}
IME_RE = re.compile("[가-힣\u1100-\u11ff\u3131-\u318e]")
LOOSE_RE = re.compile("[\u1100-\u11ff\u3131-\u318e]")


def ime_keys(name: str) -> str:
    """The keys that type `name` on the Korean 2-set keyboard (히ㅐ퍈 녀ㅜ -> GLOVIS SUN); other characters stay."""
    out = []
    for ch in name:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            s = o - 0xAC00
            out.append(_IME_KEY[_CHO[s // 588]] + _IME_KEY[_JUNG[s % 588 // 28]] + _IME_KEY[_JONG[s % 28]])
        else:
            c = _IME_CONJ.get(ch, ch)
            out.append(_IME_KEY.get(c, c))
    return "".join(out)


_IME_JAMO = {v: k for k, v in _IME_KEY.items() if len(v) == 1 and k not in "ㅃㅉㄸㄲㅆㅒㅖ"}
_IME_PAIRS = {v: k for k, v in _IME_KEY.items() if len(v) == 2}  # "HK" -> ㅘ, "RT" -> ㄳ


def _ime_pair(jamo: str, key: str, pool) -> str | None:
    """The compound vowel or final that `jamo` followed by `key` makes (ㅗ + K -> ㅘ), if it is in `pool`."""
    x = _IME_PAIRS.get(_IME_KEY[jamo] + key) if len(_IME_KEY[jamo]) == 1 else None
    return x if x is not None and x in pool else None


def ime_typed(text: str) -> str:
    """What a 2-set Korean IME composes when `text` is typed with the Hangul layout on (GLOVIS SUN -> 히ㅐ퍈 녀ㅜ),
    the inverse of ime_keys; for stress tests."""
    out, syl = [], []  # syl: the jamo of the syllable being composed, as [cho, jung, jong] (None = empty)

    def flush():
        if syl:
            cho, jung, jong = syl
            if cho and jung:
                out.append(chr(0xAC00 + (_CHO.index(cho) * 21 + _JUNG.index(jung)) * 28 + _JONG.index(jong or "")))
            else:
                out.append((cho or "") + (jung or ""))
            syl.clear()

    for ch in text.upper():
        j = _IME_JAMO.get(ch)
        if j is None:
            flush()
            out.append(ch)
        elif j in _JUNG:
            if syl and syl[2]:  # the final consonant (or its second half) starts the next syllable
                first, last = (_IME_KEY[syl[2]][0], _IME_KEY[syl[2]][1]) if len(_IME_KEY[syl[2]]) == 2 else (None, _IME_KEY[syl[2]])
                syl[2] = _IME_JAMO[first] if first else None
                flush()
                syl.extend([_IME_JAMO[last], j, None])
            elif syl and syl[0] and not syl[1]:
                syl[1] = j
            elif syl and syl[1] and _ime_pair(syl[1], ch, _JUNG):
                syl[1] = _ime_pair(syl[1], ch, _JUNG)
            else:
                flush()
                syl.extend([None, j, None])
        else:
            if syl and syl[0] and syl[1] and not syl[2] and j in _JONG:
                syl[2] = j
            elif syl and syl[2] and _ime_pair(syl[2], ch, _JONG):
                syl[2] = _ime_pair(syl[2], ch, _JONG)
            else:
                flush()
                syl.extend([j, None, None])
    flush()
    return "".join(out)


def ime_accept(query_an: str, f: np.ndarray, loose: bool) -> bool:
    """Keep the keyboard reading of a record when it reads like English (a fifth or more of its letters vowels;
    real Hangul names read as keys have almost none) and its best match spells nearly the same name at a
    compatible tonnage: one edit per four letters, or per two and a half when some letters compose into no
    syllable (loose jamo: 99% of such typed names have one, real Hangul names none); exact under five letters."""
    letters = [c for c in query_an if "A" <= c <= "Z"]
    if len(query_an) < 4 or not math.isfinite(f[COL["e_osa"]]) or sum(c in "AEIOU" for c in letters) < 0.15 * len(letters):
        return False
    allowed = 0 if len(query_an) < 5 else math.floor(len(query_an) * (0.4 if loose else 0.25))
    ton = f[COL["min_alr"]]
    return bool((not math.isfinite(ton) or ton <= 0.3) and f[COL["e_osa"]] <= allowed)


def korean_variants(name: str, tables: dict | None = None, exclude: str | None = None):
    """Whole-name phonetic keys for a Hangul ship name plus token-level info; with HANGUL_DEDUP also the
    keys of the name with doubled characters collapsed, and its numbers when they differ (alt_numbers);
    with JAMO_REPAIR a record with loose jamo is read as its likeliest repaired spelling, keeping the keys of
    the record itself and of the other repairs."""
    out = _korean_variants(name, tables, exclude)
    if JAMO_REPAIR and JAMO_KEYS and JAMO_RE.search(name):
        fixed = repair_jamo(name)
        if fixed:
            main = _korean_variants(fixed[0], tables, exclude)
            extra = out["keys"] + [k for f in fixed[1:] for k in _korean_variants(f, tables, exclude)["keys"]]
            main["keys"] = list(dict.fromkeys(main["keys"] + extra))
            out = main
    out["alt_numbers"] = None
    if HANGUL_DEDUP:
        collapsed = RUN_RE.sub(r"\1", name)
        if collapsed != name:
            alt = _korean_variants(collapsed, tables, exclude)
            out["keys"] = list(dict.fromkeys(out["keys"] + alt["keys"]))
            if alt["numbers"] != out["numbers"]:
                out["alt_numbers"] = alt["numbers"]
    if HANGUL_SWAP:
        chars, extra = list(name), []
        for k in range(min(len(chars), 16) - 1):
            if HANGUL_RE.match(chars[k]) and HANGUL_RE.match(chars[k + 1]) and chars[k] != chars[k + 1]:
                swapped = "".join(chars[:k] + [chars[k + 1], chars[k]] + chars[k + 2:])
                extra += _korean_variants(swapped, tables, exclude)["keys"][:2]
        out["keys"] = list(dict.fromkeys(out["keys"] + extra))
    return out


def _korean_variants(name: str, tables: dict | None = None, exclude: str | None = None):
    words, numbers = korean_tokens(name)
    readings = [token_readings(t) for t in words]
    variants = []
    for pick in range(3):
        parts = [r[min(pick, len(r) - 1)] for r in readings]
        variants.append(collapse("".join(parts)))
    # mixed: best of acronym reading for letter-name tokens, plain for the rest
    variants.append(collapse("".join(r[-1] if spell_acronym(w) else r[0] for w, r in zip(words, readings))))
    # schwa: ㅓ written for an unstressed English vowel
    schwa = [hangul_key(w, eo="A") if hangul_text(w) else r[0] for w, r in zip(words, readings)]
    variants.append(collapse("".join(schwa)))
    for s, r in zip(schwa, readings):
        if s not in r:
            r.append(s)
    dict_parts = []
    if tables is not None:
        mapped = []
        for w, r in zip(words, readings):
            eng = dictionary_part(tables, w, exclude) if hangul_text(w) else None
            dict_parts.append(eng)
            mapped.append(english_key(eng) if eng else r[0])
        if any(dict_parts):
            variants.append(collapse("".join(mapped)))
    return {
        "keys": list(dict.fromkeys(v for v in variants if v)),
        "numbers": numbers,
        "words": words,
        "readings": readings,
        "dict_parts": dict_parts,
        "acros": [a for a in (spell_acronym(w) for w in words) if a],
    }


def align_tokens(kor_words: list[str], eng_tokens: list[str]) -> list[tuple[str, str, float]]:
    """Monotone alignment of Hangul words to English token spans (1:1, 1:2, 2:1)."""
    n, m = len(kor_words), len(eng_tokens)
    if n == 0 or m == 0:
        return []

    def sim(k: str, e: str) -> float:
        ek = [english_key(e, alt) for alt in range(N_ALT)]
        return max(Levenshtein.normalized_similarity(r, x) for r in token_readings(k) for x in ek)

    neg = -1e9
    best = [[neg] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if best[i][j] == neg:
                continue
            moves = []
            if i < n and j < m:
                moves.append((1, 1, sim(kor_words[i], eng_tokens[j])))
            if i < n and j + 1 < m:
                moves.append((1, 2, sim(kor_words[i], eng_tokens[j] + " " + eng_tokens[j + 1])))
            if i + 1 < n and j < m:
                moves.append((2, 1, sim(kor_words[i] + kor_words[i + 1], eng_tokens[j])))
            if i < n:
                moves.append((1, 0, -0.3))
            if j < m:
                moves.append((0, 1, -0.3))
            for di, dj, s in moves:
                if best[i][j] + s > best[i + di][j + dj]:
                    best[i + di][j + dj] = best[i][j] + s
                    back[i + di][j + dj] = (i, j, di, dj, s)
    pairs = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, di, dj, s = back[i][j]
        if di and dj:
            pairs.append(("".join(kor_words[pi:pi + di]), " ".join(eng_tokens[pj:pj + dj]), s))
        i, j = pi, pj
    return pairs[::-1]


# ---------------------------------------------------------------------------
# Lookup tables learned from the official train set
# ---------------------------------------------------------------------------


def fit_tables(train: pd.DataFrame, catalog: pd.DataFrame) -> dict:
    """Type/flag compatibility counts, a Hangul word dictionary and exact lookups."""
    cat = catalog.set_index("vessel_id")
    ctype = train["target_vessel_id"].map(cat["vessel_type"]).fillna("").astype(str)
    cflag = train["target_vessel_id"].map(cat["flag_name"]).fillna("").astype(str)
    qtype = train["vessel_type"].fillna("").astype(str)
    qflag = train["flag_name"].fillna("").astype(str)
    tables = {
        "type_pairs": {f"{a}\t{b}": n for (a, b), n in Counter(zip(qtype, ctype)).items()},
        "type_q": dict(Counter(qtype)),
        "flag_pairs": {f"{a}\t{b}": n for (a, b), n in Counter(zip(qflag, cflag)).items()},
        "flag_q": dict(Counter(qflag)),
        "call_sign": {},
        "kdict": {},
    }
    for cs, vid in zip(train["call_sign"], train["target_vessel_id"]):
        if isinstance(cs, str) and cs:
            votes = tables["call_sign"].setdefault(cs, {})
            votes[vid] = votes.get(vid, 0) + 1
    kdict: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    seen = set()
    for name, vid in zip(train["vessel_name"], train["target_vessel_id"]):
        if not is_hangul(name) or (name, vid) in seen:
            continue
        seen.add((name, vid))
        words, _ = korean_tokens(name)
        eng = letter_tokens(upper(cat.at[vid, "vessel_name"]))
        for kor, en, score in align_tokens(words, eng):
            if score >= 0.45 and HANGUL_RE.search(kor):
                kdict[kor][en].add(vid)
    tables["kdict"] = {k: {e: sorted(v) for e, v in d.items()} for k, d in kdict.items()}
    return tables


# ---------------------------------------------------------------------------
# Catalog index, candidate retrieval and pair features
# ---------------------------------------------------------------------------

FEATURES = [
    "is_korean", "n_cand",
    "gt_alr", "igt_alr", "min_alr", "log_ratio", "gt_exact", "igt_exact", "gt_missing",
    "cgt_missing", "rg_rank", "rg_gap",
    "type_lp", "type_prior", "flag_lp", "flag_prior", "flag_missing",
    "name_count", "num_state", "cand_tokens",
    "e_lev", "e_osa", "e_sim", "e_jw", "e_tsr", "e_tset", "e_partial", "e_exact",
    "e_len_diff", "e_first", "e_last", "e_osa_rank", "e_osa_gap", "e_n_best", "e_key_sim",
    "k_sim", "k_csim", "k_partial", "k_dist", "k_len_ratio", "k_sim_rank", "k_sim_gap",
    "k_csim_rank", "k_pre_rank", "k_pre", "k_tok_mean", "k_tok_min", "k_tok_cover",
    "k_dict_cov", "k_dict_hit", "k_dict_miss", "k_n_words", "k_rsim", "k_rsim_rank", "k_tok_first",
    "k_vsim", "k_vsim_rank", "k_acro_n", "k_acro_hit", "k_krsim", "k_krsim_rank",
    "e_idf_share", "e_idf_max", "e_last_tok",
]
COL = {f: i for i, f in enumerate(FEATURES)}


def _rank_desc(values: np.ndarray) -> np.ndarray:
    """0 for the largest value; tied values share the best rank."""
    return np.searchsorted(np.sort(-values), -values, side="left").astype(np.float32)


def _rank_asc(values: np.ndarray) -> np.ndarray:
    return _rank_desc(-values)


def _num_state(qnums: tuple, cnums: tuple, alt: tuple | None = None) -> int:
    if not qnums and not cnums:
        return 0
    if qnums == cnums:
        return 1
    if qnums and cnums:
        # 5: the numbers agree once doubled digits are collapsed (제22 for 제2)
        return 5 if alt is not None and alt == cnums else 2
    return 3 if qnums else 4


# whole-catalog phonetic search for Hangul names: "always", or only when the record has no tonnage
WHOLE_NAME_SEARCH = "no_tonnage"
# also keep Hangul candidates with exactly the recorded tonnage that the name search ranked out
EXACT_TONNAGE = True
# Latin names: also retrieve catalog names sharing a word, nearest tonnage first
LATIN_WORDS = True
# ... and names with a word that starts with a record word (KENSINGTON cut to KENSIN, PAC for PACIFIC)
LATIN_PREFIX = True
# Hangul records with gross tonnage but no international tonnage: search further up (domestic figures are smaller)
DOMESTIC_GT = True
# Latin names: drop bracketed remarks; also retrieve by word-order-free spelling; edit cutoff grows with length
PAREN_STRIP = False
LATIN_SORTED = False
LATIN_RELCUT = False
# Latin names with more edits than train shows (letters wrong, added or missing): OSA distance (a swap
# counts once) with a cutoff growing with length, the best fuzzy matches always, and the names most like
# the record's among vessels within LATIN_TON_WIN of its tonnage
LATIN_WIDE = False
LATIN_TON_WIN = 0.25
# relative tonnage window for Hangul candidates (0.12 up to v8) and how many of its best name matches to keep
HANGUL_WIN = 0.12
HANGUL_LIMIT = 80
# ... chosen by name evidence minus this much per unit of |log tonnage ratio| (0: name evidence only)
HANGUL_PRE_TON = 0.0
# ... plus this many vessels whose name best contains the record's (a word left out), by partial match
HANGUL_PARTIAL_EXTRA = 0
# ... plus the HANGUL_OUTER_N best name matches between that window and a wider one (tonnage far off)
HANGUL_OUTER = 0.0
HANGUL_OUTER_N = 0
# Hangul records with doubled characters (흥흥아영, 제22흥아): also match the name with every run of one
# character collapsed, and count numbers that agree after collapsing as a separate kind of agreement
HANGUL_DEDUP = False
# ... and with each pair of neighbouring syllables swapped (먼샤 for 샤먼), first two keys of each
HANGUL_SWAP = False
# loose jamo in a record (a typo at the jamo level that no longer composes, 그ᅳ랜드): read each jamo with its
# sound class instead of dropping it, so a doubled jamo collapses away and a swapped or lost one costs one edit
JAMO_KEYS = False
# ... and also read such a record as the spelling it most likely came from (repair_jamo): its words then
# hit the learned dictionary, letter-name acronyms and numbers again
JAMO_REPAIR = False
# a Latin name typed with the Korean keyboard layout still on (GLOVIS SUN -> 히ㅐ퍈 녀ㅜ): script.py also reads such
# a record as the keys that typed it and takes that reading's best match when it spells a catalog name
IME_DECODE = False


# retrieval switches the rankers were trained with; train.py --final saves them to model/settings.json
SETTINGS = ("WHOLE_NAME_SEARCH", "EXACT_TONNAGE", "LATIN_WORDS", "LATIN_PREFIX", "DOMESTIC_GT", "PAREN_STRIP",
            "LATIN_SORTED", "LATIN_RELCUT", "LATIN_WIDE", "LATIN_TON_WIN", "HANGUL_WIN", "HANGUL_LIMIT", "HANGUL_PRE_TON",
            "HANGUL_PARTIAL_EXTRA", "HANGUL_OUTER", "HANGUL_OUTER_N", "HANGUL_DEDUP",
            "HANGUL_SWAP", "JAMO_KEYS", "JAMO_REPAIR", "IME_DECODE")


def current_settings() -> dict:
    return {k: globals()[k] for k in SETTINGS}


def apply_settings(path) -> None:
    """Set the retrieval switches from a settings file, if there is one (older models have none)."""
    try:
        with open(path, encoding="utf-8") as fh:
            values = json.load(fh)
    except FileNotFoundError:
        return
    for k, v in values.items():
        if k in SETTINGS:
            globals()[k] = v


class CatalogIndex:
    def __init__(self, catalog: pd.DataFrame, tables: dict) -> None:
        cat = catalog.reset_index(drop=True)
        self.ids = cat["vessel_id"].astype(str).to_numpy()
        self.id_to_index = {v: i for i, v in enumerate(self.ids)}
        self.up = [upper(v) for v in cat["vessel_name"]]
        self.base = [strip_prefix(u) for u in self.up]
        self.an = [alnum(b) for b in self.base]
        self.an_sorted = ["".join(sorted(alnum(t) for t in b.split())) for b in self.base]
        self.ltoks = [letter_tokens(b) for b in self.base]
        # word keys per spelling variant; a name's key joins its word keys (same as english_key)
        word_keys = [[[english_word_key(t, alt) for t in toks] for alt in range(N_ALT)] for toks in self.ltoks]
        self.keys = [[collapse("".join(wk[alt])) for wk in word_keys] for alt in range(N_ALT)]
        self.key = self.keys[0]
        self.cks = [[consonants(k) for k in keys] for keys in self.keys]
        self.rks = [[consonants_raw(k) for k in keys] for keys in self.keys]
        self.vks = [[vowel_skeleton(k) for k in keys] for keys in self.keys]
        self.krk = [kr_key(b) for b in self.base]
        # per word, its distinct keys over all spelling variants; plus keys of adjacent word pairs
        self.tok_groups = [
            [list(dict.fromkeys(wk[alt][j] for alt in range(N_ALT))) for j in range(len(wk[0]))]
            for wk in word_keys
        ]
        self.tok_pool = []
        for wk in word_keys:
            flat = [k for row in wk for k in row]
            pairs = [row[a] + row[a + 1] for row in wk for a in range(len(row) - 1)]
            self.tok_pool.append(list(dict.fromkeys(flat + pairs)))
        self.tok_first = [[k[:1] for k in wk[0]] for wk in word_keys]
        # sound of each name word -> vessels, for records that leave words out (글로비스 세이프티 -> 세이프티)
        word_post: dict[str, set] = defaultdict(set)
        cons_post: dict[str, set] = defaultdict(set)
        for i, wk in enumerate(word_keys):
            for k in {k for alt_keys in wk for k in alt_keys}:
                if len(k) >= 3:
                    word_post[k].add(i)
                if len(consonants(k)) >= 3:
                    cons_post[consonants(k)].add(i)
        self.word_list = list(word_post)
        self.word_post = [np.fromiter(word_post[k], dtype=np.int64) for k in self.word_list]
        self.cons_post = {c: np.fromiter(v, dtype=np.int64) for c, v in cons_post.items() if len(v) <= 2000}
        self._word_cache: dict[str, list] = {}
        self.tok_sets = [set(toks) for toks in self.ltoks]
        self.nums = [numbers_of(b) for b in self.base]
        self.ntok = np.array([len(t) for t in self.ltoks], dtype=np.float32)
        self.gt = np.array([positive_float(v) for v in cat["gross_tonnage"]], dtype=np.float64)
        self.type = cat["vessel_type"].fillna("").astype(str).to_numpy()
        self.flag = cat["flag_name"].fillna("").astype(str).to_numpy()
        counts = Counter(self.an)
        self.name_count = np.array([counts[a] for a in self.an], dtype=np.float32)
        type_counts, flag_counts, n = Counter(self.type), Counter(self.flag), len(cat)
        self.type_prior = np.array([math.log(type_counts[t] / n) for t in self.type], dtype=np.float32)
        self.flag_prior = np.array([math.log(flag_counts[f] / n) for f in self.flag], dtype=np.float32)
        self.tables = tables
        self._name_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        valid = np.flatnonzero(np.isfinite(self.gt))
        order = np.argsort(self.gt[valid], kind="stable")
        self.gt_sorted_idx = valid[order]
        self.gt_sorted = self.gt[self.gt_sorted_idx]
        self.an_by_gt = [self.an[i] for i in self.gt_sorted_idx]
        postings: dict[str, list[int]] = defaultdict(list)
        for i, toks in enumerate(self.ltoks):
            for t in set(toks):
                postings[t].append(i)
        self.tok_df = {t: len(v) for t, v in postings.items()}
        self.postings = {t: np.array(v) for t, v in postings.items() if len(t) >= 3}
        self.tok_sorted = sorted(self.postings)
        self.short_postings = {t: np.array(v) for t, v in postings.items() if len(t) == 2}

    def idf(self, token: str) -> float:
        return math.log((len(self.ids) + 1) / (self.tok_df.get(token, 0) + 1))

    def acronym_hits(self, acros: list[str], cand: np.ndarray) -> np.ndarray:
        """How many spelled-out acronyms appear as a word of each candidate's name."""
        hits = np.zeros(len(cand), dtype=np.float32)
        for a in acros:
            post = (self.postings if len(a) >= 3 else self.short_postings).get(a)
            if post is not None:
                hits += np.isin(cand, post)
        return hits

    # -- compatibility log-probabilities ------------------------------------
    def cond_lp(self, prefix: str, query_value: str, cand_values: np.ndarray) -> np.ndarray:
        pairs = self.tables[f"{prefix}_pairs"]
        total = self.tables[f"{prefix}_q"].get(query_value, 0)
        out = np.empty(len(cand_values), dtype=np.float32)
        cache: dict[str, float] = {}
        for j, value in enumerate(cand_values):
            if value not in cache:
                cnt = pairs.get(f"{query_value}\t{value}", 0)
                cache[value] = math.log((cnt + 0.1) / (total + 5.0))
            out[j] = cache[value]
        return out

    # -- candidate retrieval ------------------------------------------------
    def _gt_window(self, values: list[float], tol: float, hi_tol: float | None = None) -> np.ndarray:
        """Catalog vessels whose tonnage puts each recorded value within -tol..+tol of it (hi_tol: how
        far below the catalog figure a record may be instead)."""
        found = []
        for v in values:
            if math.isfinite(v):
                lo = np.searchsorted(self.gt_sorted, v / (1 + tol), side="left")
                hi = np.searchsorted(self.gt_sorted, v / (1 - (tol if hi_tol is None else hi_tol)), side="right")
                found.append(self.gt_sorted_idx[lo:hi])
        if not found:
            return np.arange(len(self.ids))
        return np.unique(np.concatenate(found))

    def _min_alr(self, cand: np.ndarray, gt: float, igt: float) -> np.ndarray:
        cgt = self.gt[cand]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.fmin(np.abs(np.log(gt / cgt)), np.abs(np.log(igt / cgt)))

    def _min_rg(self, cand: np.ndarray, gt: float, igt: float) -> np.ndarray:
        cgt = self.gt[cand]
        with np.errstate(invalid="ignore", divide="ignore"):
            r1 = np.abs(gt - cgt) / cgt
            r2 = np.abs(igt - cgt) / cgt
        return np.fmin(r1, r2)

    def english_candidates(self, base: str, gt: float, igt: float, limit: int = 40) -> np.ndarray:
        an = alnum(base)
        if LATIN_WIDE:
            cut = max(4, round(0.3 * len(an)))
            dist = process.cdist([an], self.an, scorer=OSA.distance, score_cutoff=cut, dtype=np.int32, workers=1)[0]
        else:
            cut = max(4, len(an) // 4) if LATIN_RELCUT else 4
            dist = process.cdist([an], self.an, scorer=Levenshtein.distance, score_cutoff=cut,
                                 dtype=np.int32, workers=1)[0]
        cand = np.flatnonzero(dist <= cut)
        if len(cand) > limit:
            cand = cand[np.argsort(dist[cand], kind="stable")[:limit]]
        if LATIN_SORTED and len(base.split()) >= 2:
            # the same words in another order (COURAGE GLOVIS)
            key = "".join(sorted(alnum(t) for t in base.split()))
            d2 = process.cdist([key], self.an_sorted, scorer=Levenshtein.distance, score_cutoff=3,
                               dtype=np.int32, workers=1)[0]
            more = np.flatnonzero(d2 <= 3)
            cand = np.concatenate([cand, more[np.argsort(d2[more], kind="stable")[:20]]])
        if LATIN_WIDE or len(cand) < 5:
            extra = process.extract(an, self.an, scorer=fuzz.ratio, limit=10)
            cand = np.concatenate([cand, np.array([e[2] for e in extra], dtype=np.int64)])
        if LATIN_WIDE:
            # several letters wrong and tonnage a little off: the most similar names near the recorded tonnage
            for v in dict.fromkeys(x for x in (gt, igt) if math.isfinite(x)):
                lo = np.searchsorted(self.gt_sorted, v / (1 + LATIN_TON_WIN), side="left")
                hi = np.searchsorted(self.gt_sorted, v / (1 - LATIN_TON_WIN), side="right")
                if hi > lo:
                    sim = process.cdist([an], self.an_by_gt[lo:hi], scorer=fuzz.ratio, dtype=np.float32, workers=1)[0]
                    cand = np.concatenate([cand, self.gt_sorted_idx[lo:hi][np.argsort(-sim, kind="stable")[:15]]])
        # renamed vessels: same tonnage and a shared distinctive word
        if math.isfinite(gt) or math.isfinite(igt):
            near = self._gt_window([gt, igt], 0.002)
            shared = []
            for t in set(letter_tokens(base)):
                post = self.postings.get(t)
                if post is not None and len(post) <= 3000:
                    shared.append(post)
            if shared:
                near_shared = np.intersect1d(near, np.unique(np.concatenate(shared)))
                cand = np.concatenate([cand, near_shared[:20]])
            exact = self._gt_window([gt, igt], 0.0)
            cand = np.concatenate([cand, exact[:15]])
        if LATIN_WORDS:
            # a record may leave words out (HYUNDAI SUPREME -> SUPREME): names sharing a word, nearest tonnage first
            posts = [self.postings[t] for t in set(letter_tokens(base))
                     if t in self.postings and len(self.postings[t]) <= 5000]
            if LATIN_PREFIX:
                for t in set(letter_tokens(base)):
                    if len(t) >= 3 and (t not in self.postings or len(t) <= 5):
                        lo = bisect.bisect_left(self.tok_sorted, t)
                        hi = bisect.bisect_left(self.tok_sorted, t + "~")
                        longer = [self.postings[w] for w in self.tok_sorted[lo:hi] if w != t]
                        if longer and sum(len(x) for x in longer) <= 5000:
                            posts.append(np.unique(np.concatenate(longer)))
            if posts:
                pool, shared = np.unique(np.concatenate(posts), return_counts=True)
                gap = np.nan_to_num(self._min_alr(pool, gt, igt), nan=9.0)
                cand = np.concatenate([cand, pool[np.lexsort((gap, -shared))[:20]]])
        return np.unique(cand)

    def _korean_pre(self, kv: dict, cand: np.ndarray, gt: float, igt: float) -> np.ndarray:
        """Coarse retrieval score: phonetic similarity and number/acronym agreement."""
        best = np.zeros(len(cand), dtype=np.float32)
        for keys, cks in zip(self.keys, self.cks):
            pk = [keys[i] for i in cand]
            pc = [cks[i] for i in cand]
            for key in kv["keys"]:
                s1 = process.cdist([key], pk, scorer=Levenshtein.normalized_similarity, dtype=np.float32)[0]
                s2 = process.cdist([consonants(key)], pc, scorer=Levenshtein.normalized_similarity,
                                   dtype=np.float32)[0]
                best = np.maximum(best, s1 + s2)
        qn, alt = kv["numbers"], kv.get("alt_numbers")
        bonus = np.array([0.15 if (qn and self.nums[i] == qn) else 0.1 if (alt and self.nums[i] == alt) else
                          (-0.15 if qn and self.nums[i] and self.nums[i] != qn else 0.0)
                          for i in cand], dtype=np.float32)
        bonus += 0.3 * self.acronym_hits(kv["acros"], cand)
        # name evidence only: tonnage has its own features, and evaluation tonnage may be off
        return best + bonus

    def name_candidates(self, kv: dict, gt: float, igt: float, limit: int = 40, wide: int = 120) -> np.ndarray:
        """Best phonetic matches in the whole catalog, whatever the recorded tonnage says."""
        key = tuple(kv["keys"])
        if not key:  # e.g. "제1호": nothing to spell
            return np.zeros(0, dtype=np.int64)
        hit = self._name_cache.get(key)
        if hit is None:
            best = np.zeros(len(self.ids), dtype=np.float32)
            for keys in self.keys:
                sim = process.cdist(kv["keys"], keys, scorer=Levenshtein.normalized_similarity,
                                    dtype=np.float32, workers=-1)
                best = np.maximum(best, sim.max(axis=0))
            top = np.argpartition(-best, wide)[:wide]
            hit = (top, best[top])
            self._name_cache[key] = hit
        top, score = hit
        # equal names are common (동명 선박): among ties prefer the nearer tonnage
        rg = self._min_rg(top, gt, igt)
        adj = score - 0.01 * np.minimum(np.nan_to_num(rg, nan=3.0), 3.0)
        return top[np.argsort(-adj, kind="stable")[:limit]]

    def word_candidates(self, kv: dict, gt: float, igt: float, limit: int = 30) -> np.ndarray:
        """Vessels with a name word that sounds like one of the record's words, nearest tonnage first."""
        idx, sim = [], []
        for readings in kv["readings"]:
            for r in readings:
                if len(r) < 3:
                    continue
                hits = self._word_cache.get(r)
                if hits is None:
                    hits = [(score, self.word_post[j]) for _, score, j in
                            process.extract(r, self.word_list, scorer=Levenshtein.normalized_similarity,
                                            score_cutoff=0.8, limit=8)]
                    post = self.cons_post.get(consonants(r)) if len(consonants(r)) >= 3 else None
                    if post is not None:
                        hits.append((0.8, post))
                    self._word_cache[r] = hits
                for score, post in hits:
                    idx.append(post)
                    sim.append(np.full(len(post), score, dtype=np.float32))
        if not idx:
            return np.zeros(0, dtype=np.int64)
        idx, sim = np.concatenate(idx), np.concatenate(sim)
        first = np.argsort(-sim, kind="stable")
        idx, keep = np.unique(idx[first], return_index=True)
        sim = sim[first][keep]
        gap = np.nan_to_num(self._min_alr(idx, gt, igt), nan=9.0)
        return idx[np.lexsort((gap, -sim))[:limit]]

    def korean_candidates(self, kv: dict, gt: float, igt: float, limit: int = 80):
        names = self.word_candidates(kv, gt, igt)
        if WHOLE_NAME_SEARCH == "always" or not (math.isfinite(gt) or math.isfinite(igt)):
            names = np.union1d(names, self.name_candidates(kv, gt, igt))
        if math.isfinite(gt) or math.isfinite(igt):
            # gross tonnage alone may be the domestic figure, up to ~45% below the international one in the catalog
            domestic = DOMESTIC_GT and math.isfinite(gt) and not math.isfinite(igt)
            pool = self._gt_window([gt, igt], HANGUL_WIN, hi_tol=0.45 if domestic else None)
            pre = self._korean_pre(kv, pool, gt, igt)
            pick = pre - HANGUL_PRE_TON * np.nan_to_num(self._min_alr(pool, gt, igt), nan=1.0) if HANGUL_PRE_TON else pre
            top = np.argsort(-pick, kind="stable")[:limit]
            if HANGUL_PARTIAL_EXTRA:
                part = np.zeros(len(pool), dtype=np.float32)
                keys0 = [self.key[i] for i in pool]
                for key in kv["keys"]:
                    if len(key) >= 4:
                        part = np.maximum(part, process.cdist([key], keys0, scorer=fuzz.partial_ratio, dtype=np.float32)[0])
                top = np.union1d(top, np.argsort(-part, kind="stable")[:HANGUL_PARTIAL_EXTRA])
                top = top[np.argsort(-pre[top], kind="stable")]
            if EXACT_TONNAGE:
                # same recorded tonnage, however different the name (renamed or abbreviated records)
                cgt = self.gt[pool]
                same = np.flatnonzero((cgt == gt) | (cgt == igt))
                same = same[np.argsort(-pre[same], kind="stable")[:20]]
                top = np.union1d(top, same)
                top = top[np.argsort(-pre[top], kind="stable")]
            cand, pre = pool[top], pre[top]
            if HANGUL_OUTER_N and HANGUL_OUTER > HANGUL_WIN:
                outer = np.setdiff1d(self._gt_window([gt, igt], HANGUL_OUTER, hi_tol=min(0.6, HANGUL_OUTER + 0.2) if domestic else None), pool)
                if len(outer):
                    opre = self._korean_pre(kv, outer, gt, igt)
                    best = np.argsort(-opre, kind="stable")[:HANGUL_OUTER_N]
                    cand, pre = np.concatenate([cand, outer[best]]), np.concatenate([pre, opre[best]])
                    order = np.argsort(-pre, kind="stable")
                    cand, pre = cand[order], pre[order]
            extra = np.setdiff1d(names, cand)
        else:
            cand, pre, extra = names[:0], np.zeros(0, dtype=np.float32), names
        if len(extra):
            cand = np.concatenate([cand, extra])
            pre = np.concatenate([pre, self._korean_pre(kv, extra, gt, igt)])
            order = np.argsort(-pre, kind="stable")
            cand, pre = cand[order], pre[order]
        return cand, pre

    # -- features -------------------------------------------------------------
    def query_candidates(self, row: dict):
        name = upper(row.get("vessel_name"))
        gt = positive_float(row.get("gross_tonnage"))
        igt = positive_float(row.get("international_gross_tonnage"))
        if hangul_text(name):
            exclude = row.get("target_vessel_id") if isinstance(row.get("target_vessel_id"), str) else None
            kv = korean_variants(name, self.tables, exclude)
            cand, pre = self.korean_candidates(kv, gt, igt, limit=HANGUL_LIMIT)
            return cand, {"korean": True, "kv": kv, "pre": pre}
        base = strip_prefix(name)
        return self.english_candidates(base, gt, igt), {"korean": False}

    def features(self, row: dict, cand: np.ndarray, info: dict) -> np.ndarray:
        m = len(cand)
        F = np.full((m, len(FEATURES)), np.nan, dtype=np.float32)
        if not m:  # nothing retrieved (e.g. an unreadable name and no tonnage); the caller falls back
            return F
        name = upper(row.get("vessel_name"))
        gt = positive_float(row.get("gross_tonnage"))
        igt = positive_float(row.get("international_gross_tonnage"))
        qtype = row.get("vessel_type") if isinstance(row.get("vessel_type"), str) else ""
        qflag = row.get("flag_name") if isinstance(row.get("flag_name"), str) else ""
        cgt = self.gt[cand]
        with np.errstate(invalid="ignore", divide="ignore"):
            # symmetric in both directions: x/6 must look as far off as x*6
            r1 = np.abs(np.log(gt / cgt))
            r2 = np.abs(np.log(igt / cgt))
            lr = np.log(np.fmax(gt, igt) / cgt)
        minrg = np.fmin(r1, r2)
        F[:, COL["is_korean"]] = float(info["korean"])
        F[:, COL["n_cand"]] = m
        F[:, COL["gt_alr"]] = r1
        F[:, COL["igt_alr"]] = r2
        F[:, COL["min_alr"]] = minrg
        F[:, COL["log_ratio"]] = lr
        F[:, COL["gt_exact"]] = (gt == cgt).astype(np.float32)
        F[:, COL["igt_exact"]] = (igt == cgt).astype(np.float32)
        F[:, COL["gt_missing"]] = float(not math.isfinite(gt) and not math.isfinite(igt))
        F[:, COL["cgt_missing"]] = (~np.isfinite(cgt)).astype(np.float32)
        filled = np.nan_to_num(minrg, nan=9.0)
        F[:, COL["rg_rank"]] = _rank_asc(filled)
        F[:, COL["rg_gap"]] = filled - filled.min()
        F[:, COL["type_lp"]] = self.cond_lp("type", qtype, self.type[cand])
        F[:, COL["type_prior"]] = self.type_prior[cand]
        F[:, COL["flag_lp"]] = self.cond_lp("flag", qflag, self.flag[cand])
        F[:, COL["flag_prior"]] = self.flag_prior[cand]
        F[:, COL["flag_missing"]] = float(qflag == "")
        F[:, COL["name_count"]] = self.name_count[cand]
        F[:, COL["cand_tokens"]] = self.ntok[cand]
        qnums = info["kv"]["numbers"] if info["korean"] else numbers_of(strip_prefix(name))
        alt = info["kv"].get("alt_numbers") if info["korean"] else None
        F[:, COL["num_state"]] = [_num_state(qnums, self.nums[i], alt) for i in cand]
        if info["korean"]:
            self._korean_features(F, cand, info)
        else:
            self._english_features(F, cand, strip_prefix(name))
        return F

    def _english_features(self, F, cand, base):
        an = alnum(base)
        names = [self.an[i] for i in cand]
        bases = [self.base[i] for i in cand]
        osa = process.cdist([an], names, scorer=OSA.distance, dtype=np.float32)[0]
        F[:, COL["e_lev"]] = process.cdist([an], names, scorer=Levenshtein.distance, dtype=np.float32)[0]
        F[:, COL["e_osa"]] = osa
        F[:, COL["e_sim"]] = process.cdist([an], names, scorer=fuzz.ratio, dtype=np.float32)[0] / 100
        F[:, COL["e_jw"]] = process.cdist([an], names, scorer=JaroWinkler.similarity, dtype=np.float32)[0]
        F[:, COL["e_tsr"]] = process.cdist([base], bases, scorer=fuzz.token_sort_ratio, dtype=np.float32)[0] / 100
        F[:, COL["e_tset"]] = process.cdist([base], bases, scorer=fuzz.token_set_ratio, dtype=np.float32)[0] / 100
        F[:, COL["e_partial"]] = process.cdist([an], names, scorer=fuzz.partial_ratio, dtype=np.float32)[0] / 100
        F[:, COL["e_exact"]] = np.array([an == x for x in names], dtype=np.float32)
        F[:, COL["e_len_diff"]] = np.array([len(x) - len(an) for x in names], dtype=np.float32)
        F[:, COL["e_first"]] = np.array([bool(an) and bool(x) and an[0] == x[0] for x in names], dtype=np.float32)
        F[:, COL["e_last"]] = np.array([bool(an) and bool(x) and an[-1] == x[-1] for x in names], dtype=np.float32)
        F[:, COL["e_osa_rank"]] = _rank_asc(osa)
        F[:, COL["e_osa_gap"]] = osa - osa.min()
        F[:, COL["e_n_best"]] = float((osa == osa.min()).sum())
        F[:, COL["e_key_sim"]] = process.cdist([english_key(base)], [self.key[i] for i in cand],
                                               scorer=Levenshtein.normalized_similarity, dtype=np.float32)[0]
        # renamed vessels keep a distinctive word, so weight shared words by rarity
        qtoks = letter_tokens(base)
        idf = {t: self.idf(t) for t in set(qtoks)}
        total = sum(idf.values())
        share = np.zeros(len(cand), dtype=np.float32)
        best = np.zeros(len(cand), dtype=np.float32)
        last = np.zeros(len(cand), dtype=np.float32)
        for j, i in enumerate(cand):
            shared = [idf[t] for t in idf if t in self.tok_sets[i]]
            if shared:
                share[j] = sum(shared) / total
                best[j] = max(shared)
            last[j] = float(bool(qtoks) and bool(self.ltoks[i]) and qtoks[-1] == self.ltoks[i][-1])
        F[:, COL["e_idf_share"]] = share
        F[:, COL["e_idf_max"]] = best
        F[:, COL["e_last_tok"]] = last

    def _korean_features(self, F, cand, info):
        kv = info["kv"]
        m = len(cand)
        sim = np.zeros(m, dtype=np.float32)
        csim = np.zeros(m, dtype=np.float32)
        part = np.zeros(m, dtype=np.float32)
        dist = np.full(m, 99, dtype=np.float32)
        lenr = np.zeros(m, dtype=np.float32)
        rsim = np.zeros(m, dtype=np.float32)
        vsim = np.zeros(m, dtype=np.float32)
        for keys, cks, rks, vks in zip(self.keys, self.cks, self.rks, self.vks):
            ck = [keys[i] for i in cand]
            cc = [cks[i] for i in cand]
            cr = [rks[i] for i in cand]
            cv = [vks[i] for i in cand]
            klen = np.array([max(len(k), 1) for k in ck], dtype=np.float32)
            for key in kv["keys"]:
                s = process.cdist([key], ck, scorer=Levenshtein.normalized_similarity, dtype=np.float32)[0]
                c = process.cdist([consonants(key)], cc, scorer=Levenshtein.normalized_similarity, dtype=np.float32)[0]
                r = process.cdist([consonants_raw(key)], cr, scorer=Levenshtein.normalized_similarity, dtype=np.float32)[0]
                v = process.cdist([vowel_skeleton(key)], cv, scorer=Levenshtein.normalized_similarity, dtype=np.float32)[0]
                rsim = np.maximum(rsim, r)
                vsim = np.maximum(vsim, v)
                p = process.cdist([key], ck, scorer=fuzz.partial_ratio, dtype=np.float32)[0] / 100
                d = process.cdist([key], ck, scorer=Levenshtein.distance, dtype=np.float32)[0]
                lenr = np.where(s > sim, len(key) / klen, lenr)
                sim = np.maximum(sim, s)
                csim = np.maximum(csim, c)
                part = np.maximum(part, p)
                dist = np.minimum(dist, d)
        F[:, COL["k_sim"]] = sim
        F[:, COL["k_csim"]] = csim
        F[:, COL["k_rsim"]] = rsim
        F[:, COL["k_rsim_rank"]] = _rank_desc(rsim)
        F[:, COL["k_vsim"]] = vsim
        F[:, COL["k_vsim_rank"]] = _rank_desc(vsim)
        rr = kr_key("".join(romanize(w) if hangul_text(w) else w for w in kv["words"]))
        krsim = process.cdist([rr], [self.krk[i] for i in cand], scorer=Levenshtein.normalized_similarity,
                              dtype=np.float32)[0]
        F[:, COL["k_krsim"]] = krsim
        F[:, COL["k_krsim_rank"]] = _rank_desc(krsim)
        F[:, COL["k_acro_n"]] = len(kv["acros"])
        F[:, COL["k_acro_hit"]] = self.acronym_hits(kv["acros"], cand)
        F[:, COL["k_partial"]] = part
        F[:, COL["k_dist"]] = dist
        F[:, COL["k_len_ratio"]] = lenr
        F[:, COL["k_sim_rank"]] = _rank_desc(sim)
        F[:, COL["k_sim_gap"]] = sim.max() - sim
        F[:, COL["k_csim_rank"]] = _rank_desc(csim)
        F[:, COL["k_pre_rank"]] = _rank_desc(np.asarray(info["pre"], dtype=np.float32))
        F[:, COL["k_pre"]] = info["pre"]
        # token-level agreement: each Hangul word against the candidate's words
        words, readings, dparts = kv["words"], kv["readings"], kv["dict_parts"]
        n_words = len(words)
        F[:, COL["k_n_words"]] = n_words
        dict_tokens = [p.split() for p in dparts if p]
        F[:, COL["k_dict_cov"]] = (sum(1 for p in dparts if p) / n_words) if n_words else 0.0
        tok_mean = np.zeros(m, dtype=np.float32)
        tok_min = np.zeros(m, dtype=np.float32)
        tok_cover = np.zeros(m, dtype=np.float32)
        tok_first = np.zeros(m, dtype=np.float32)
        q_first = [("V" if r[0][:1] in VOWELS else r[0][:1]) for r in readings if r and r[0]]
        dhit = np.zeros(m, dtype=np.float32)
        dmiss = np.zeros(m, dtype=np.float32)
        for j, i in enumerate(cand):
            pool = self.tok_pool[i]
            if n_words and pool:
                scores = []
                used = set()
                for r in readings:
                    sc = process.extract(r[0], pool, scorer=Levenshtein.normalized_similarity, limit=1)
                    best_s, best_k = sc[0][1], sc[0][0]
                    for alt in r[1:]:
                        sc = process.extract(alt, pool, scorer=Levenshtein.normalized_similarity, limit=1)
                        if sc[0][1] > best_s:
                            best_s, best_k = sc[0][1], sc[0][0]
                    scores.append(best_s)
                    used.add(best_k)
                tok_mean[j] = float(np.mean(scores))
                tok_min[j] = float(np.min(scores))
                groups = self.tok_groups[i]
                covered = sum(1 for g in groups if any(k in u for k in g for u in used))
                tok_cover[j] = covered / max(len(groups), 1)
                c_first = [("V" if f in VOWELS else f) for f in self.tok_first[i] if f]
                if q_first and c_first:
                    tok_first[j] = sum(1 for a, b in zip(q_first, c_first) if a == b) / max(len(q_first), len(c_first))
            if dict_tokens:
                cset = self.tok_sets[i]
                hits = sum(1 for toks in dict_tokens if all(t in cset for t in toks))
                dhit[j] = hits
                dmiss[j] = len(dict_tokens) - hits
        F[:, COL["k_tok_mean"]] = tok_mean
        F[:, COL["k_tok_min"]] = tok_min
        F[:, COL["k_tok_cover"]] = tok_cover
        F[:, COL["k_tok_first"]] = tok_first
        F[:, COL["k_dict_hit"]] = dhit
        F[:, COL["k_dict_miss"]] = dmiss


def row_signature(row: dict) -> tuple:
    """Rows with identical inputs get identical candidates and features."""
    return tuple(str(row.get(c)) for c in ("vessel_name", "gross_tonnage", "international_gross_tonnage",
                                           "flag_name", "vessel_type", "target_vessel_id"))


def build_matrix(index: CatalogIndex, rows: list[dict]):
    """Candidates and features for many rows (computed once per unique input)."""
    cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    out_cands, out_feats = [], []
    for row in rows:
        sig = row_signature(row)
        if sig not in cache:
            cand, info = index.query_candidates(row)
            cache[sig] = (cand, index.features(row, cand, info))
        cand, feats = cache[sig]
        out_cands.append(cand)
        out_feats.append(feats)
    return out_cands, out_feats


def lookup_prediction(tables: dict, row: dict) -> str | None:
    """Exact call-sign lookup against the official train set."""
    cs = row.get("call_sign")
    if isinstance(cs, str) and cs in tables["call_sign"]:
        votes = tables["call_sign"][cs]
        return max(sorted(votes), key=lambda v: votes[v])
    return None


def weighted_accuracy(names, truth, pred) -> dict:
    kor = np.array([is_hangul(n) for n in names])
    ok = np.asarray(truth) == np.asarray(pred)
    hk = float(ok[kor].mean()) if kor.any() else float("nan")
    he = float(ok[~kor].mean()) if (~kor).any() else float("nan")
    return {"korean": hk, "non_korean": he, "score": 0.7 * hk + 0.3 * he}
