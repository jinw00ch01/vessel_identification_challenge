"""Shared, row-independent matching code (used by training and by script.py).

Every prediction looks at one query row, the official vessel catalog and
lookup tables built from the official train set. Nothing depends on other
evaluation rows, so a row gets the same prediction whether it is scored
alone, in a subset, or in any order.
"""

from __future__ import annotations

from collections import Counter, defaultdict
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
NO_RE = re.compile(r"\bNO\b\.?\s*(?=\d)")
ROMAN = {"II": "2", "III": "3", "IV": "4", "VI": "6", "VII": "7", "VIII": "8", "IX": "9"}


def is_hangul(value) -> bool:
    return isinstance(value, str) and bool(HANGUL_RE.search(value))


def upper(value) -> str:
    if not isinstance(value, str):
        return ""
    return unicodedata.normalize("NFKC", value).upper().replace("\xa0", " ").strip()


def alnum(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum())


def strip_prefix(name: str) -> str:
    """Drop ship-type prefixes such as 'M/V ' that never appear in the catalog."""
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


def _rules(th: str, ch: str, pinyin: bool = False):
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
    return [
        # vowel spellings that depend on the letters around them
        (re.compile(r"(?<=[^AEIOU])U(?=[^AEIOUY]{2}|[^AEIOUY]$)"), "O"),
        (re.compile(r"IGH"), "AI"),
        (re.compile(r"^([^AEIOU]*)A([^AEIOU])E$"), r"\1EI\2E"),
        (re.compile(r"^([^AEIOU]*)I([^AEIOU])E$"), r"\1AI\2E"),
        (re.compile(r"^([^AEIOU]+)Y$"), r"\1AI"),
        (re.compile(r"AY"), "EI"),
        (re.compile(r"EER(?![AEIOUY])"), "IO"),
        (re.compile(r"ER(?![AEIOUY])"), "O"),
        (re.compile(r"EE"), "I"),
        (re.compile(r"OO"), "U"),
    ] + final_i + pin + [
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
        (re.compile(r"(?<=[AEIOU])R(?![AEIOUY])"), ""),
        (re.compile(r"R"), "L"),
        (re.compile(r"W"), "U"),
        (re.compile(r"Y"), "I"),
        (re.compile(r"(?<=[AEIOU])H(?![AEIOU])"), ""),
    ]


_ENG_RULES = (_rules("T", "S"), _rules("S", "K"), _rules("T", "S", pinyin=True))
N_ALT = len(_ENG_RULES)


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
    for ch in text:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            cho, rest = divmod(code, 588)
            jung, jong = divmod(rest, 28)
            yield cho, jung, jong


def hangul_key(text: str) -> str:
    return collapse("".join(CHO_CLS[c] + JUNG_CLS[v] + JONG_CLS[j] for c, v, j in _syllables(text)))


def romanize(text: str) -> str:
    return "".join(RR_CHO[c] + RR_JUNG[v] + RR_JONG[j] for c, v, j in _syllables(text))


def romanized_key(text: str) -> str:
    """Phonetic key of the romanized form; final ㅇ is a plain nasal, ㄱ is hard."""
    parts = []
    for c, v, j in _syllables(text):
        parts.append(("K" if c == 0 else RR_CHO[c]) + RR_JUNG[v] + ("N" if j == 21 else RR_JONG[j]))
    return english_key("".join(parts))


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
    if not HANGUL_RE.search(tok):
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


def korean_variants(name: str, tables: dict | None = None, exclude: str | None = None):
    """Whole-name phonetic keys for a Hangul ship name plus token-level info."""
    words, numbers = korean_tokens(name)
    readings = [token_readings(t) for t in words]
    variants = []
    for pick in range(3):
        parts = [r[min(pick, len(r) - 1)] for r in readings]
        variants.append(collapse("".join(parts)))
    # mixed: best of acronym reading for letter-name tokens, plain for the rest
    variants.append(collapse("".join(r[-1] if spell_acronym(w) else r[0] for w, r in zip(words, readings))))
    dict_parts = []
    if tables is not None:
        mapped = []
        for w, r in zip(words, readings):
            eng = dictionary_part(tables, w, exclude) if HANGUL_RE.search(w) else None
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
    "gt_rg", "igt_rg", "min_rg", "log_ratio", "gt_exact", "igt_exact", "gt_missing",
    "cgt_missing", "rg_rank", "rg_gap",
    "type_lp", "type_prior", "flag_lp", "flag_prior", "flag_missing",
    "name_count", "num_state", "cand_tokens",
    "e_lev", "e_osa", "e_sim", "e_jw", "e_tsr", "e_tset", "e_partial", "e_exact",
    "e_len_diff", "e_first", "e_last", "e_osa_rank", "e_osa_gap", "e_n_best", "e_key_sim",
    "k_sim", "k_csim", "k_partial", "k_dist", "k_len_ratio", "k_sim_rank", "k_sim_gap",
    "k_csim_rank", "k_pre_rank", "k_pre", "k_tok_mean", "k_tok_min", "k_tok_cover",
    "k_dict_cov", "k_dict_hit", "k_dict_miss", "k_n_words", "k_rsim", "k_rsim_rank", "k_tok_first",
]
COL = {f: i for i, f in enumerate(FEATURES)}


def _rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values))
    return ranks


def _rank_asc(values: np.ndarray) -> np.ndarray:
    return _rank_desc(-values)


def _num_state(qnums: tuple, cnums: tuple) -> int:
    if not qnums and not cnums:
        return 0
    if qnums == cnums:
        return 1
    if qnums and cnums:
        return 2
    return 3 if qnums else 4


class CatalogIndex:
    def __init__(self, catalog: pd.DataFrame, tables: dict) -> None:
        cat = catalog.reset_index(drop=True)
        self.ids = cat["vessel_id"].astype(str).to_numpy()
        self.id_to_index = {v: i for i, v in enumerate(self.ids)}
        self.up = [upper(v) for v in cat["vessel_name"]]
        self.base = [strip_prefix(u) for u in self.up]
        self.an = [alnum(b) for b in self.base]
        self.ltoks = [letter_tokens(b) for b in self.base]
        self.keys = [[english_key(b, alt) for b in self.base] for alt in range(N_ALT)]
        self.key = self.keys[0]
        self.cks = [[consonants(k) for k in keys] for keys in self.keys]
        self.rks = [[consonants_raw(k) for k in keys] for keys in self.keys]
        self.tok_keys = [
            list(dict.fromkeys(english_word_key(t, alt) for alt in range(N_ALT) for t in toks))
            for toks in self.ltoks
        ]
        self.tok_first = [
            [english_word_key(t)[:1] for t in toks] for toks in self.ltoks
        ]
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
        valid = np.flatnonzero(np.isfinite(self.gt))
        order = np.argsort(self.gt[valid], kind="stable")
        self.gt_sorted_idx = valid[order]
        self.gt_sorted = self.gt[self.gt_sorted_idx]
        postings: dict[str, list[int]] = defaultdict(list)
        for i, toks in enumerate(self.ltoks):
            for t in set(toks):
                if len(t) >= 3:
                    postings[t].append(i)
        self.postings = {t: np.array(v) for t, v in postings.items()}

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
    def _gt_window(self, values: list[float], tol: float) -> np.ndarray:
        found = []
        for v in values:
            if math.isfinite(v):
                lo = np.searchsorted(self.gt_sorted, v / (1 + tol), side="left")
                hi = np.searchsorted(self.gt_sorted, v / (1 - tol), side="right")
                found.append(self.gt_sorted_idx[lo:hi])
        if not found:
            return np.arange(len(self.ids))
        return np.unique(np.concatenate(found))

    def _min_rg(self, cand: np.ndarray, gt: float, igt: float) -> np.ndarray:
        cgt = self.gt[cand]
        with np.errstate(invalid="ignore", divide="ignore"):
            r1 = np.abs(gt - cgt) / cgt
            r2 = np.abs(igt - cgt) / cgt
        return np.fmin(r1, r2)

    def english_candidates(self, base: str, gt: float, igt: float, limit: int = 40) -> np.ndarray:
        an = alnum(base)
        dist = process.cdist([an], self.an, scorer=Levenshtein.distance, score_cutoff=4,
                             dtype=np.int32, workers=1)[0]
        cand = np.flatnonzero(dist <= 4)
        if len(cand) > limit:
            cand = cand[np.argsort(dist[cand], kind="stable")[:limit]]
        if len(cand) < 5:
            extra = process.extract(an, self.an, scorer=fuzz.ratio, limit=10)
            cand = np.concatenate([cand, np.array([e[2] for e in extra], dtype=np.int64)])
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
        return np.unique(cand)

    def korean_candidates(self, kv: dict, gt: float, igt: float, limit: int = 80):
        pool = self._gt_window([gt, igt], 0.12)
        best = np.zeros(len(pool), dtype=np.float32)
        for keys, cks in zip(self.keys, self.cks):
            pk = [keys[i] for i in pool]
            pc = [cks[i] for i in pool]
            for key in kv["keys"]:
                s1 = process.cdist([key], pk, scorer=Levenshtein.normalized_similarity, dtype=np.float32)[0]
                s2 = process.cdist([consonants(key)], pc, scorer=Levenshtein.normalized_similarity,
                                   dtype=np.float32)[0]
                best = np.maximum(best, s1 + s2)
        qn = kv["numbers"]
        bonus = np.array([0.15 if (qn and self.nums[i] == qn) else (-0.15 if qn and self.nums[i] and self.nums[i] != qn else 0.0)
                          for i in pool], dtype=np.float32)
        rg = self._min_rg(pool, gt, igt)
        pre = best + bonus - 0.5 * np.nan_to_num(rg, nan=0.5)
        order = np.argsort(-pre, kind="stable")[:limit]
        return pool[order], pre[order]

    # -- features -------------------------------------------------------------
    def query_candidates(self, row: dict):
        name = upper(row.get("vessel_name"))
        gt = positive_float(row.get("gross_tonnage"))
        igt = positive_float(row.get("international_gross_tonnage"))
        if is_hangul(name):
            exclude = row.get("target_vessel_id") if isinstance(row.get("target_vessel_id"), str) else None
            kv = korean_variants(name, self.tables, exclude)
            cand, pre = self.korean_candidates(kv, gt, igt)
            return cand, {"korean": True, "kv": kv, "pre": pre}
        base = strip_prefix(name)
        return self.english_candidates(base, gt, igt), {"korean": False}

    def features(self, row: dict, cand: np.ndarray, info: dict) -> np.ndarray:
        m = len(cand)
        F = np.full((m, len(FEATURES)), np.nan, dtype=np.float32)
        name = upper(row.get("vessel_name"))
        gt = positive_float(row.get("gross_tonnage"))
        igt = positive_float(row.get("international_gross_tonnage"))
        qtype = row.get("vessel_type") if isinstance(row.get("vessel_type"), str) else ""
        qflag = row.get("flag_name") if isinstance(row.get("flag_name"), str) else ""
        cgt = self.gt[cand]
        with np.errstate(invalid="ignore", divide="ignore"):
            r1 = np.abs(gt - cgt) / cgt
            r2 = np.abs(igt - cgt) / cgt
            lr = np.log(np.fmax(gt, igt) / cgt)
        minrg = np.fmin(r1, r2)
        F[:, COL["is_korean"]] = float(info["korean"])
        F[:, COL["n_cand"]] = m
        F[:, COL["gt_rg"]] = r1
        F[:, COL["igt_rg"]] = r2
        F[:, COL["min_rg"]] = minrg
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
        F[:, COL["num_state"]] = [_num_state(qnums, self.nums[i]) for i in cand]
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

    def _korean_features(self, F, cand, info):
        kv = info["kv"]
        m = len(cand)
        sim = np.zeros(m, dtype=np.float32)
        csim = np.zeros(m, dtype=np.float32)
        part = np.zeros(m, dtype=np.float32)
        dist = np.full(m, 99, dtype=np.float32)
        lenr = np.zeros(m, dtype=np.float32)
        rsim = np.zeros(m, dtype=np.float32)
        for keys, cks, rks in zip(self.keys, self.cks, self.rks):
            ck = [keys[i] for i in cand]
            cc = [cks[i] for i in cand]
            cr = [rks[i] for i in cand]
            klen = np.array([max(len(k), 1) for k in ck], dtype=np.float32)
            for key in kv["keys"]:
                s = process.cdist([key], ck, scorer=Levenshtein.normalized_similarity, dtype=np.float32)[0]
                c = process.cdist([consonants(key)], cc, scorer=Levenshtein.normalized_similarity, dtype=np.float32)[0]
                r = process.cdist([consonants_raw(key)], cr, scorer=Levenshtein.normalized_similarity, dtype=np.float32)[0]
                rsim = np.maximum(rsim, r)
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
        F[:, COL["k_partial"]] = part
        F[:, COL["k_dist"]] = dist
        F[:, COL["k_len_ratio"]] = lenr
        F[:, COL["k_sim_rank"]] = _rank_desc(sim)
        F[:, COL["k_sim_gap"]] = sim.max() - sim
        F[:, COL["k_csim_rank"]] = _rank_desc(csim)
        F[:, COL["k_pre_rank"]] = np.arange(m, dtype=np.float32)
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
            tkeys = self.tok_keys[i]
            pair_keys = [a + b for a, b in zip(tkeys, tkeys[1:])]
            pool = tkeys + pair_keys
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
                tok_cover[j] = sum(1 for k in tkeys if k in used or any(k in u for u in used)) / max(len(tkeys), 1)
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
