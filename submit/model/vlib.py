"""Shared, row-independent matching code (used by training and by script.py).

Every prediction looks at one query row, the official vessel catalog and
lookup tables built from the official train set. Nothing depends on other
evaluation rows, so a row gets the same prediction whether it is scored
alone, in a subset, or in any order.
"""

from __future__ import annotations

from collections import Counter
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


def letters_only(name: str) -> str:
    """Name without numbers, roman numerals and 'NO.' markers."""
    name = NO_RE.sub(" ", name)
    toks = [t for t in tokens(name) if t not in ROMAN and not t.isdigit()]
    return " ".join(re.sub(r"\d+", "", t) for t in toks).strip()


def positive_float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) and number > 0 else math.nan


# ---------------------------------------------------------------------------
# Hangul -> coarse phonetic key
# ---------------------------------------------------------------------------

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
        "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

CHO_CLS = {"ㄱ": "K", "ㄲ": "K", "ㅋ": "K", "ㄷ": "T", "ㄸ": "T", "ㅌ": "T",
           "ㅂ": "P", "ㅃ": "P", "ㅍ": "P", "ㅅ": "S", "ㅆ": "S", "ㅈ": "S",
           "ㅉ": "S", "ㅊ": "S", "ㄹ": "L", "ㅁ": "M", "ㄴ": "N", "ㅎ": "H", "ㅇ": ""}
JUNG_CLS = {"ㅏ": "A", "ㅐ": "E", "ㅑ": "IA", "ㅒ": "IE", "ㅓ": "O", "ㅔ": "E",
            "ㅕ": "IO", "ㅖ": "IE", "ㅗ": "O", "ㅘ": "UA", "ㅙ": "UE", "ㅚ": "UE",
            "ㅛ": "IO", "ㅜ": "U", "ㅝ": "UO", "ㅞ": "UE", "ㅟ": "UI", "ㅠ": "IU",
            "ㅡ": "", "ㅢ": "I", "ㅣ": "I"}
JONG_CLS = {"": "", "ㄱ": "K", "ㄲ": "K", "ㄳ": "K", "ㄴ": "N", "ㄵ": "N", "ㄶ": "N",
            "ㄷ": "T", "ㄹ": "L", "ㄺ": "L", "ㄻ": "M", "ㄼ": "L", "ㄽ": "L", "ㄾ": "L",
            "ㄿ": "P", "ㅀ": "L", "ㅁ": "M", "ㅂ": "P", "ㅄ": "P", "ㅅ": "T", "ㅆ": "T",
            "ㅇ": "N", "ㅈ": "T", "ㅊ": "T", "ㅋ": "K", "ㅌ": "T", "ㅍ": "P", "ㅎ": "T"}
VOWELS = set("AEIOU")

# Korean spellings of Latin letter names, longest first (used for acronyms
# such as 케이엠티씨 -> KMTC or 에스엠 -> SM).
LETTER_NAMES = sorted({
    "에이": "A", "비": "B", "씨": "C", "시": "C", "디": "D", "이": "E", "에프": "F",
    "지": "G", "쥐": "G", "에이치": "H", "에치": "H", "아이": "I", "제이": "J", "케이": "K",
    "엘": "L", "엠": "M", "엔": "N", "앤": "N", "오": "O", "피": "P", "큐": "Q",
    "알": "R", "아르": "R", "에스": "S", "티": "T", "유": "U", "브이": "V",
    "더블유": "W", "엑스": "X", "와이": "Y", "제트": "Z", "젯": "Z",
}.items(), key=lambda kv: -len(kv[0]))
KOREAN_NUMBER_WORDS = {"원": "1", "투": "2", "쓰리": "3", "포": "4", "파이브": "5"}


def collapse(s: str) -> str:
    out = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def hangul_key(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            cho, rest = divmod(code, 588)
            jung, jong = divmod(rest, 28)
            out.append(CHO_CLS[CHO[cho]])
            out.append(JUNG_CLS[JUNG[jung]])
            out.append(JONG_CLS[JONG[jong]])
    return collapse("".join(out))


def strip_korean_suffix(name: str) -> str:
    """Remove the ship suffix '호' and turn '제N호' markers into plain numbers."""
    name = re.sub(r"제\s*(\d+)\s*호", r" \1 ", name)
    name = re.sub(r"제\s*(\d+)", r" \1 ", name)
    name = re.sub(r"(\d+)\s*호", r" \1 ", name)
    name = name.strip()
    if len(name) > 2 and name.endswith("호"):
        name = name[:-1]
    return re.sub(r"\s+", " ", name).strip()


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


def korean_variants(name: str) -> list[tuple[str, tuple[str, ...]]]:
    """Candidate Latin-ish readings of a Hangul ship name.

    Returns (phonetic key, numbers) pairs: the plain transliteration key plus
    variants where leading tokens spelled as letter names become acronyms.
    """
    base = strip_korean_suffix(name)
    nums = list(re.findall(r"\d+", base))
    toks = [t for t in re.split(r"[\s\-\.]+", re.sub(r"(\d+)", r" \1 ", base)) if t]
    plain_parts, acro_parts = [], []
    for idx, tok in enumerate(toks):
        if tok.isdigit():
            continue
        if tok in KOREAN_NUMBER_WORDS and idx == len(toks) - 1 and idx > 0:
            nums.append(KOREAN_NUMBER_WORDS[tok])
            continue
        if HANGUL_RE.search(tok):
            key = hangul_key(tok)
            acro = spell_acronym(tok)
            plain_parts.append(key)
            acro_parts.append(english_key(acro) if acro else key)
        else:
            key = english_key(tok)
            plain_parts.append(key)
            acro_parts.append(key)
    numbers = tuple(sorted(str(int(x)) for x in nums))
    variants = [(collapse("".join(plain_parts)), numbers)]
    acro_key = collapse("".join(acro_parts))
    if acro_key != variants[0][0]:
        variants.append((acro_key, numbers))
    # Also try the acronym split off: a leading letter-name prefix glued to a word
    if toks and HANGUL_RE.search(toks[0]) and spell_acronym(toks[0]) is None:
        for cut in range(len(toks[0]) - 1, 1, -1):
            acro = spell_acronym(toks[0][:cut])
            if acro:
                rest = hangul_key(toks[0][cut:])
                tail = "".join(acro_parts[1:])
                variants.append((collapse(english_key(acro) + rest + tail), numbers))
                break
    return variants


# ---------------------------------------------------------------------------
# English -> the same coarse phonetic alphabet
# ---------------------------------------------------------------------------

_ENG_RULES = [
    (re.compile(r"PH"), "F"),
    (re.compile(r"^GH"), "K"),
    (re.compile(r"GH"), ""),
    (re.compile(r"CK"), "K"),
    (re.compile(r"SCH"), "S"),
    (re.compile(r"TCH"), "S"),
    (re.compile(r"ZH"), "S"),
    (re.compile(r"SH"), "S"),
    (re.compile(r"CH"), "S"),
    (re.compile(r"TH"), "T"),
    (re.compile(r"QU"), "KU"),
    (re.compile(r"Q"), "K"),
    (re.compile(r"^X"), "S"),
    (re.compile(r"X"), "KS"),
    (re.compile(r"WH"), "U"),
    (re.compile(r"EE|EA|IE(?=.)"), "I"),
    (re.compile(r"OO"), "U"),
    (re.compile(r"C(?=[EIY])"), "S"),
    (re.compile(r"C"), "K"),
    (re.compile(r"G(?=[EIY])"), "S"),
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


def english_word_key(word: str) -> str:
    word = collapse(word)
    final_e = len(word) > 3 and word.endswith("E") and word[-2] not in VOWELS
    for pattern, repl in _ENG_RULES:
        word = pattern.sub(repl, word)
    if final_e and word.endswith("E"):
        word = word[:-1]
    word = re.sub(r"NK(?![AEIOU])", "N", word)
    return collapse(word)


def english_key(text: str) -> str:
    words = re.findall(r"[A-Z]+", letters_only(upper(text)))
    return collapse("".join(english_word_key(w) for w in words))


def consonants(key: str) -> str:
    return collapse("".join(ch for ch in key if ch not in VOWELS))


# ---------------------------------------------------------------------------
# Lookup tables learned from the official train set
# ---------------------------------------------------------------------------


def fit_tables(train: pd.DataFrame, catalog: pd.DataFrame) -> dict:
    """Type/flag compatibility counts and exact lookups from train rows."""
    cat = catalog.set_index("vessel_id")
    ctype = train["target_vessel_id"].map(cat["vessel_type"]).fillna("").astype(str)
    cflag = train["target_vessel_id"].map(cat["flag_name"]).fillna("").astype(str)
    qtype = train["vessel_type"].fillna("").astype(str)
    qflag = train["flag_name"].fillna("").astype(str)
    type_pairs = Counter(zip(qtype, ctype))
    flag_pairs = Counter(zip(qflag, cflag))
    tables = {
        "type_pairs": {f"{a}\t{b}": n for (a, b), n in type_pairs.items()},
        "type_q": dict(Counter(qtype)),
        "flag_pairs": {f"{a}\t{b}": n for (a, b), n in flag_pairs.items()},
        "flag_q": dict(Counter(qflag)),
        "call_sign": {},
        "name": {},
    }
    for cs, vid in zip(train["call_sign"], train["target_vessel_id"]):
        if isinstance(cs, str) and cs:
            tables["call_sign"].setdefault(cs, {})
            tables["call_sign"][cs][vid] = tables["call_sign"][cs].get(vid, 0) + 1
    for name, vid in zip(train["vessel_name"], train["target_vessel_id"]):
        key = alnum(upper(name))
        if key:
            tables["name"].setdefault(key, {})
            tables["name"][key][vid] = tables["name"][key].get(vid, 0) + 1
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
    "k_csim_rank", "k_pre_rank",
]


def _rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values))
    return ranks


def _rank_asc(values: np.ndarray) -> np.ndarray:
    return _rank_desc(-values)


class CatalogIndex:
    def __init__(self, catalog: pd.DataFrame, tables: dict) -> None:
        cat = catalog.reset_index(drop=True)
        self.ids = cat["vessel_id"].astype(str).to_numpy()
        self.id_to_index = {v: i for i, v in enumerate(self.ids)}
        self.up = [upper(v) for v in cat["vessel_name"]]
        self.base = [strip_prefix(u) for u in self.up]
        self.an = [alnum(b) for b in self.base]
        self.key = [english_key(b) for b in self.base]
        self.ck = [consonants(k) for k in self.key]
        self.nums = [numbers_of(b) for b in self.base]
        self.ntok = np.array([len(tokens(b)) for b in self.base], dtype=np.float32)
        self.gt = np.array([positive_float(v) for v in cat["gross_tonnage"]], dtype=np.float64)
        self.type = cat["vessel_type"].fillna("").astype(str).to_numpy()
        self.flag = cat["flag_name"].fillna("").astype(str).to_numpy()
        counts = Counter(self.an)
        self.name_count = np.array([counts[a] for a in self.an], dtype=np.float32)
        type_counts = Counter(self.type)
        flag_counts = Counter(self.flag)
        n = len(cat)
        self.type_prior = np.array([math.log(type_counts[t] / n) for t in self.type], dtype=np.float32)
        self.flag_prior = np.array([math.log(flag_counts[f] / n) for f in self.flag], dtype=np.float32)
        self.tables = tables
        self.valid_gt = np.flatnonzero(np.isfinite(self.gt))
        order = np.argsort(self.gt[self.valid_gt], kind="stable")
        self.gt_sorted_idx = self.valid_gt[order]
        self.gt_sorted = self.gt[self.gt_sorted_idx]

    # -- compatibility log-probabilities ------------------------------------
    def _cond_lp(self, prefix: str, query_value: str, cand_values: np.ndarray) -> np.ndarray:
        pairs = self.tables[f"{prefix}_pairs"]
        total = self.tables[f"{prefix}_q"].get(query_value, 0)
        k = 50.0
        out = np.empty(len(cand_values), dtype=np.float32)
        cache: dict[str, float] = {}
        for j, value in enumerate(cand_values):
            if value not in cache:
                cnt = pairs.get(f"{query_value}\t{value}", 0)
                cache[value] = math.log((cnt + 0.1) / (total + 0.1 * k))
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

    def english_candidates(self, an: str, limit: int = 40) -> np.ndarray:
        dist = process.cdist([an], self.an, scorer=Levenshtein.distance, score_cutoff=4,
                             dtype=np.int32, workers=1)[0]
        cand = np.flatnonzero(dist <= 4)
        if len(cand) > limit:
            cand = cand[np.argsort(dist[cand], kind="stable")[:limit]]
        if len(cand) < 5:
            extra = process.extract(an, self.an, scorer=fuzz.ratio, limit=10)
            cand = np.unique(np.concatenate([cand, np.array([e[2] for e in extra], dtype=np.int64)]))
        return cand

    def korean_candidates(self, variants, gt: float, igt: float, limit: int = 60):
        pool = self._gt_window([gt, igt], 0.12)
        keys = [self.key[i] for i in pool]
        cks = [self.ck[i] for i in pool]
        best = np.zeros(len(pool), dtype=np.float32)
        for key, _ in variants:
            s1 = process.cdist([key], keys, scorer=Levenshtein.normalized_similarity,
                               dtype=np.float32, workers=1)[0]
            s2 = process.cdist([consonants(key)], cks, scorer=Levenshtein.normalized_similarity,
                               dtype=np.float32, workers=1)[0]
            best = np.maximum(best, s1 + s2)
        rg = self._min_rg(pool, gt, igt)
        pre = best - 0.5 * np.nan_to_num(rg, nan=0.5)
        order = np.argsort(-pre, kind="stable")[:limit]
        return pool[order], pre[order]

    def _min_rg(self, cand: np.ndarray, gt: float, igt: float) -> np.ndarray:
        cgt = self.gt[cand]
        with np.errstate(invalid="ignore", divide="ignore"):
            r1 = np.abs(gt - cgt) / cgt
            r2 = np.abs(igt - cgt) / cgt
        return np.fmin(r1, r2)

    # -- features -------------------------------------------------------------
    def query_candidates(self, row: dict):
        name = upper(row.get("vessel_name"))
        gt = positive_float(row.get("gross_tonnage"))
        igt = positive_float(row.get("international_gross_tonnage"))
        if is_hangul(name):
            variants = korean_variants(name)
            cand, pre = self.korean_candidates(variants, gt, igt)
            return cand, {"korean": True, "variants": variants, "pre": pre}
        an = alnum(strip_prefix(name))
        cand = self.english_candidates(an)
        return cand, {"korean": False}

    def features(self, row: dict, cand: np.ndarray, info: dict) -> np.ndarray:
        m = len(cand)
        F = np.full((m, len(FEATURES)), np.nan, dtype=np.float32)
        col = {f: i for i, f in enumerate(FEATURES)}
        name = upper(row.get("vessel_name"))
        gt = positive_float(row.get("gross_tonnage"))
        igt = positive_float(row.get("international_gross_tonnage"))
        qtype = row.get("vessel_type") if isinstance(row.get("vessel_type"), str) else ""
        qflag = row.get("flag_name") if isinstance(row.get("flag_name"), str) else ""
        cgt = self.gt[cand]
        with np.errstate(invalid="ignore", divide="ignore"):
            r1 = np.abs(gt - cgt) / cgt
            r2 = np.abs(igt - cgt) / cgt
            lr = np.log(np.fmax(gt, igt) / cgt) if math.isfinite(gt) else np.full(m, np.nan)
        minrg = np.fmin(r1, r2)
        F[:, col["is_korean"]] = float(info["korean"])
        F[:, col["n_cand"]] = m
        F[:, col["gt_rg"]] = r1
        F[:, col["igt_rg"]] = r2
        F[:, col["min_rg"]] = minrg
        F[:, col["log_ratio"]] = lr
        F[:, col["gt_exact"]] = (gt == cgt).astype(np.float32)
        F[:, col["igt_exact"]] = (igt == cgt).astype(np.float32)
        F[:, col["gt_missing"]] = float(not math.isfinite(gt) and not math.isfinite(igt))
        F[:, col["cgt_missing"]] = (~np.isfinite(cgt)).astype(np.float32)
        filled = np.nan_to_num(minrg, nan=9.0)
        F[:, col["rg_rank"]] = _rank_asc(filled)
        F[:, col["rg_gap"]] = filled - filled.min()
        F[:, col["type_lp"]] = self._cond_lp("type", qtype, self.type[cand])
        F[:, col["type_prior"]] = self.type_prior[cand]
        F[:, col["flag_lp"]] = self._cond_lp("flag", qflag, self.flag[cand])
        F[:, col["flag_prior"]] = self.flag_prior[cand]
        F[:, col["flag_missing"]] = float(qflag == "")
        F[:, col["name_count"]] = self.name_count[cand]
        F[:, col["cand_tokens"]] = self.ntok[cand]

        if info["korean"]:
            qnums = info["variants"][0][1]
        else:
            qnums = numbers_of(strip_prefix(name))
        state = np.empty(m, dtype=np.float32)
        for j, i in enumerate(cand):
            cn = self.nums[i]
            if not qnums and not cn:
                state[j] = 0
            elif qnums == cn:
                state[j] = 1
            elif qnums and cn:
                state[j] = 2
            elif qnums:
                state[j] = 3
            else:
                state[j] = 4
        F[:, col["num_state"]] = state

        if info["korean"]:
            keys = [self.key[i] for i in cand]
            cks = [self.ck[i] for i in cand]
            sim = np.zeros(m, dtype=np.float32)
            csim = np.zeros(m, dtype=np.float32)
            part = np.zeros(m, dtype=np.float32)
            dist = np.full(m, 99, dtype=np.float32)
            lenr = np.zeros(m, dtype=np.float32)
            for key, _ in info["variants"]:
                s = process.cdist([key], keys, scorer=Levenshtein.normalized_similarity, dtype=np.float32)[0]
                c = process.cdist([consonants(key)], cks, scorer=Levenshtein.normalized_similarity, dtype=np.float32)[0]
                p = process.cdist([key], keys, scorer=fuzz.partial_ratio, dtype=np.float32)[0] / 100
                d = process.cdist([key], keys, scorer=Levenshtein.distance, dtype=np.float32)[0]
                better = s > sim
                lenr = np.where(better, np.array([len(key) / max(len(k), 1) for k in keys], dtype=np.float32), lenr)
                sim = np.maximum(sim, s)
                csim = np.maximum(csim, c)
                part = np.maximum(part, p)
                dist = np.minimum(dist, d)
            F[:, col["k_sim"]] = sim
            F[:, col["k_csim"]] = csim
            F[:, col["k_partial"]] = part
            F[:, col["k_dist"]] = dist
            F[:, col["k_len_ratio"]] = lenr
            F[:, col["k_sim_rank"]] = _rank_desc(sim)
            F[:, col["k_sim_gap"]] = sim.max() - sim
            F[:, col["k_csim_rank"]] = _rank_desc(csim)
            F[:, col["k_pre_rank"]] = np.arange(m, dtype=np.float32)
        else:
            base = strip_prefix(name)
            an = alnum(base)
            names = [self.an[i] for i in cand]
            bases = [self.base[i] for i in cand]
            lev = process.cdist([an], names, scorer=Levenshtein.distance, dtype=np.float32)[0]
            osa = process.cdist([an], names, scorer=OSA.distance, dtype=np.float32)[0]
            F[:, col["e_lev"]] = lev
            F[:, col["e_osa"]] = osa
            F[:, col["e_sim"]] = process.cdist([an], names, scorer=fuzz.ratio, dtype=np.float32)[0] / 100
            F[:, col["e_jw"]] = process.cdist([an], names, scorer=JaroWinkler.similarity, dtype=np.float32)[0]
            F[:, col["e_tsr"]] = process.cdist([base], bases, scorer=fuzz.token_sort_ratio, dtype=np.float32)[0] / 100
            F[:, col["e_tset"]] = process.cdist([base], bases, scorer=fuzz.token_set_ratio, dtype=np.float32)[0] / 100
            F[:, col["e_partial"]] = process.cdist([an], names, scorer=fuzz.partial_ratio, dtype=np.float32)[0] / 100
            F[:, col["e_exact"]] = np.array([an == x for x in names], dtype=np.float32)
            F[:, col["e_len_diff"]] = np.array([len(x) - len(an) for x in names], dtype=np.float32)
            F[:, col["e_first"]] = np.array([bool(an) and bool(x) and an[0] == x[0] for x in names], dtype=np.float32)
            F[:, col["e_last"]] = np.array([bool(an) and bool(x) and an[-1] == x[-1] for x in names], dtype=np.float32)
            F[:, col["e_osa_rank"]] = _rank_asc(osa)
            F[:, col["e_osa_gap"]] = osa - osa.min()
            F[:, col["e_n_best"]] = float((osa == osa.min()).sum())
            qkey = english_key(base)
            F[:, col["e_key_sim"]] = process.cdist([qkey], [self.key[i] for i in cand],
                                                   scorer=Levenshtein.normalized_similarity, dtype=np.float32)[0]
        return F


def row_signature(row: dict) -> tuple:
    """Rows with identical inputs get identical candidates and features."""
    return tuple(str(row.get(c)) for c in ("vessel_name", "gross_tonnage", "international_gross_tonnage",
                                           "flag_name", "vessel_type"))


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
