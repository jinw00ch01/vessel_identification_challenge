"""Cross-validate the matcher and train the final ranker.

Usage:
    python tools/train.py --cv            # grouped 5-fold CV (unseen vessels)
    python tools/train.py --final         # train on all rows, write submit/model/
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import sys
import time
import unicodedata

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submit" / "model"))
import vlib  # noqa: E402

DATA = ROOT / "data"
MODEL_DIR = ROOT / "submit" / "model"
WORK = ROOT / "work"

PARAMS = {
    "objective": "lambdarank",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 4,
    "seed": 0,
}
ROUNDS = 400
KOREAN_WEIGHT = 4.0
# rankers averaged at inference: all features, and one without the crude pre-score
VIEWS = ["none", "k_pre+k_pre_rank"]
# copies of Hangul rows with perturbed tonnage; v4 used 2 and scored lower on Public, so off by default
AUG_COPIES = 0
AUG_MODE = "aug"
AUG_NAME = 1
# Hangul name copies: "dropfirst" leaves out the first word; "mix" also drops a random word or swaps a jamo
NAME_AUG = "mix"
# share of multi-word Latin rows copied without their first word; only the Latin rankers train on them
LATIN_NAME_FRAC = 0.3
LATIN_MODELS = True
# share of Hangul rows copied with international tonnage blank (gross tonnage alone, often the domestic figure)
NOIGT_FRAC = 0.0
# extra copies with a broad mixture of name damage the train records never show (word order, extra
# tokens, several edits, syllables changed, words dropped): Hangul rows each get one, and this share of
# Latin rows gets one (Latin copies train only the Latin rankers)
BROAD_HANGUL = 0
BROAD_LATIN_FRAC = 0.0
# copies with heavier spelling damage and looser tonnage than train shows (the Public probes point to records
# with several letters or syllables wrong and tonnage off by up to ~15%): this many per Hangul row, and one
# for this share of Latin rows (Latin copies train only the Latin rankers)
NOISY_HANGUL = 0
NOISY_LATIN_FRAC = 0.0


def load():
    train = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(DATA / "vessel_catalog.csv", encoding="utf-8-sig")
    return train, catalog


def refresh_tables(index, rows, cands, feats):
    """Recompute the type/flag columns with the index's current tables."""
    ti, fi = vlib.COL["type_lp"], vlib.COL["flag_lp"]
    out = []
    for row, cand, f in zip(rows, cands, feats):
        f = f.copy()
        qtype = row.get("vessel_type") if isinstance(row.get("vessel_type"), str) else ""
        qflag = row.get("flag_name") if isinstance(row.get("flag_name"), str) else ""
        f[:, ti] = index.cond_lp("type", qtype, index.type[cand])
        f[:, fi] = index.cond_lp("flag", qflag, index.flag[cand])
        out.append(f)
    return out


def stack(index, rows, cands, feats, with_labels=True, korean_weight=KOREAN_WEIGHT):
    X = np.vstack(feats)
    groups = np.array([len(c) for c in cands])
    kor = np.array([vlib.is_hangul(r.get("vessel_name")) for r in rows])
    w = np.repeat(np.where(kor, korean_weight, 1.0), groups)
    y = None
    if with_labels:
        y = np.concatenate([
            (cand == index.id_to_index[row["target_vessel_id"]]).astype(np.int8)
            for row, cand in zip(rows, cands)
        ])
    return X, y, groups, w


def argmax_ids(index, cands, scores):
    out, offset = [], 0
    for cand in cands:
        local = scores[offset: offset + len(cand)]
        out.append(index.ids[cand[int(np.argmax(local))]] if len(cand) else index.ids[0])
        offset += len(cand)
    return out


SURE_GAP = 5.0  # tools/probe.py hangul_sure default: rows below it count as unsure


def row_gaps(cands, scores):
    """Best minus second-best score per row (inf with one candidate), as the probes compute it."""
    out, offset = [], 0
    for cand in cands:
        local = scores[offset: offset + len(cand)]
        offset += len(cand)
        out.append(float(np.ptp(np.partition(local, -2)[-2:])) if len(local) > 1 else np.inf)
    return np.array(out)


def train_model(X, y, groups, w, rounds=ROUNDS, params=PARAMS):
    ds = lgb.Dataset(X, y, weight=w, group=groups if params["objective"] != "binary" else None,
                     feature_name=vlib.FEATURES, free_raw_data=False)
    return lgb.train(params, ds, num_boost_round=rounds)


def run_cv(args):
    train, catalog = load()
    t0 = time.time()
    index = vlib.CatalogIndex(catalog, vlib.fit_tables(train, catalog))
    print(f"index built in {time.time() - t0:.1f}s", flush=True)
    rows = train.to_dict("records")
    kor = train["vessel_name"].map(vlib.is_hangul).to_numpy()
    eng_idx = np.flatnonzero(~kor)
    kor_idx = np.flatnonzero(kor)
    t0 = time.time()
    e_cands, e_feats = vlib.build_matrix(index, [rows[i] for i in eng_idx])
    print(f"latin features in {time.time() - t0:.1f}s", flush=True)

    cands = [None] * len(rows)
    feats = [None] * len(rows)
    for k, i in enumerate(eng_idx):
        cands[i], feats[i] = e_cands[k], e_feats[k]

    results = []
    for split_seed in args.split_seeds:
        results.append(cv_once(args, train, catalog, index, rows, kor, kor_idx, cands, feats, split_seed))
    if len(results) > 1:
        df = pd.DataFrame(results)
        print("mean over split seeds:", df.mean().round(5).to_dict(), "std:", df.std().round(5).to_dict())


def fold_splits(train, split_seed):
    if split_seed < 0:
        yield from GroupKFold(n_splits=5).split(train, groups=train["target_vessel_id"])
        return
    vessels = train["target_vessel_id"].unique()
    rng = np.random.default_rng(split_seed)
    fold_of = dict(zip(rng.permutation(vessels), np.arange(len(vessels)) % 5))
    f = train["target_vessel_id"].map(fold_of).to_numpy()
    for k in range(5):
        yield np.flatnonzero(f != k), np.flatnonzero(f == k)


def perturb(df, rng, mode):
    """Hangul rows with unreliable tonnage, as the evaluation records may have it."""
    df = df.copy()
    if mode in ("notype", "noflag"):
        # 일부 정보의 누락: train never lacks a vessel type
        df["vessel_type" if mode == "notype" else "flag_name"] = None
        return df
    if mode in ("badtype", "badflag"):
        # the field holds another record's value (a wrong but plausible label)
        col = "vessel_type" if mode == "badtype" else "flag_name"
        df[col] = df[col].to_numpy()[rng.permutation(len(df))]
        return df
    if mode == "swapton":
        # both tonnage fields taken from another record: tonnage that says nothing about the vessel
        perm = rng.permutation(len(df))
        df[["gross_tonnage", "international_gross_tonnage"]] = df[["gross_tonnage", "international_gross_tonnage"]].to_numpy()[perm]
        return df
    if mode == "nameonly":
        # only the name kept: what the name alone is worth (tools/probe.py latin_nameonly does the same)
        df[["gross_tonnage", "international_gross_tonnage", "vessel_type", "flag_name"]] = None
        return df
    n = len(df)
    gt = df["gross_tonnage"].to_numpy(float)
    igt = df["international_gross_tonnage"].to_numpy(float)
    if mode == "miss":
        f1 = f2 = np.full(n, np.nan)
    elif re.fullmatch(r"u\d+", mode):
        # uniform relative error up to the given percent, the same for both fields
        f1 = f2 = rng.uniform(1 - int(mode[1:]) / 100, 1 + int(mode[1:]) / 100, n)
    elif mode == "heavy":
        f1 = f2 = np.exp(rng.choice([-1.0, 1.0], n) * rng.uniform(0.1, 0.6, n))
    elif mode in ("noigt", "nogt"):
        # one of the two fields left blank; for Korean-flag vessels gross_tonnage is the (smaller) domestic figure
        f1 = np.full(n, np.nan) if mode == "nogt" else np.ones(n)
        f2 = np.full(n, np.nan) if mode == "noigt" else np.ones(n)
    elif mode == "aug":
        # training mixture: a quarter blank, the rest off by up to x/÷1.8, fields usually off together
        f1 = np.exp(rng.choice([-1.0, 1.0], n) * rng.uniform(0.03, 0.6, n))
        other = np.exp(rng.choice([-1.0, 1.0], n) * rng.uniform(0.03, 0.6, n))
        f2 = np.where(rng.random(n) < 0.7, f1, other)
        blank = rng.random(n) < 0.25
        f1, f2 = np.where(blank, np.nan, f1), np.where(blank, np.nan, f2)
    elif mode == "aug2":
        # training mixture: a fifth blank, the rest off by a half-normal log factor (sd 0.25, at most x/÷2.7)
        lr1 = np.minimum(np.abs(rng.normal(0.0, 0.25, n)), 1.0)
        lr2 = np.minimum(np.abs(rng.normal(0.0, 0.25, n)), 1.0)
        f1 = np.exp(rng.choice([-1.0, 1.0], n) * lr1)
        f2 = np.where(rng.random(n) < 0.7, f1, np.exp(rng.choice([-1.0, 1.0], n) * lr2))
        blank = rng.random(n) < 0.2
        f1, f2 = np.where(blank, np.nan, f1), np.where(blank, np.nan, f2)
    else:
        raise ValueError(mode)
    df["gross_tonnage"] = np.round(gt * f1)
    df["international_gross_tonnage"] = np.round(igt * f2)
    return df


# jamo swaps that Hangul spellings of one foreign name commonly differ by
_JUNG_SWAP = {1: 5, 5: 1, 4: 8, 8: 4, 6: 12, 12: 6}            # ㅐ/ㅔ, ㅓ/ㅗ, ㅕ/ㅛ
_CHO_SWAP = {0: 15, 15: 0, 3: 16, 16: 3, 7: 17, 17: 7, 9: 10, 10: 9, 12: 14, 14: 12}  # ㄱ/ㅋ ㄷ/ㅌ ㅂ/ㅍ ㅅ/ㅆ ㅈ/ㅊ
NAME_MODES = ("spell", "spell3", "dropsyl", "nospace", "dropword", "dropfirst", "droplast", "initial", "trunc",
              "randname", "garbage", "lspell2", "lspell4", "lgarbage",
              "lgen2", "lgen3", "lgen4", "wordswap", "hsyl2", "hsyl3", "addtoken", "hspell2", "hspell3",
              "jgen1", "jgen2", "jgen3", "jgen4", "ime", "eng")


def perturb_name(df, rng, mode):
    """Hangul names written differently: one jamo swapped, spaces removed, or a word left out."""
    df = df.copy()
    if mode == "randname":
        # record carries another vessel's name (renamed vessel whose old name now belongs to someone else)
        names = df["vessel_name"].to_numpy()
        vids = df["target_vessel_id"].to_numpy()
        perm = rng.permutation(len(df))
        for _ in range(10):  # never the vessel's own name: take the next row's pick instead
            same = np.flatnonzero(vids[perm] == vids)
            if not len(same):
                break
            perm[same] = perm[(same + 1) % len(df)]
        df["vessel_name"] = names[perm]
        return df
    out = []
    for name in df["vessel_name"]:
        if mode == "nospace":
            name = name.replace(" ", "")
        elif mode in ("dropword", "dropfirst", "droplast"):
            # dropfirst: the leading (often fleet or company) word left out, e.g. 글로비스 세이프티 -> 세이프티
            words = name.split()
            if len(words) >= 2:
                del words[{"dropfirst": 0, "droplast": -1}.get(mode, rng.integers(len(words)))]
                name = " ".join(words)
        elif mode == "initial":
            # leading word cut to its first letter: PACIFIC GRACE -> P GRACE
            words = name.split()
            if len(words) >= 2:
                name = " ".join([words[0][0]] + words[1:])
        elif mode == "trunc":
            # fixed-width record fields: the name cut after 60% of its characters
            name = name[:max(3, math.ceil(0.6 * len(name)))].rstrip()
        elif mode in ("spell", "spell3"):
            for _ in range(3 if mode == "spell3" else 1):
                spots = [i for i, ch in enumerate(name) if 0 <= ord(ch) - 0xAC00 < 11172
                         and ((ord(ch) - 0xAC00) // 588 in _CHO_SWAP or (ord(ch) - 0xAC00) % 588 // 28 in _JUNG_SWAP)]
                if spots:
                    i = spots[rng.integers(len(spots))]
                    cho, rest = divmod(ord(name[i]) - 0xAC00, 588)
                    jung, jong = divmod(rest, 28)
                    if jung in _JUNG_SWAP and (cho not in _CHO_SWAP or rng.random() < 0.5):
                        jung = _JUNG_SWAP[jung]
                    elif cho in _CHO_SWAP:
                        cho = _CHO_SWAP[cho]
                    name = name[:i] + chr(0xAC00 + cho * 588 + jung * 28 + jong) + name[i + 1:]
        elif mode == "dropsyl":
            # one syllable of a word of three or more left out (얼라이언스 -> 얼라언스)
            spots = [i for w in re.finditer(r"[가-힣]{3,}", name) for i in range(w.start(), w.end())]
            if spots:
                i = spots[rng.integers(len(spots))]
                name = name[:i] + name[i + 1:]
        elif mode in ("lspell2", "lspell4"):
            # Latin name with 2 or 4 random edits (substitution, insertion, deletion or transposition)
            for _ in range(int(mode[-1])):
                i = int(rng.integers(max(len(name), 1)))
                op = int(rng.integers(4))
                ch = chr(65 + int(rng.integers(26)))
                if op == 0:
                    name = name[:i] + ch + name[i + 1:]
                elif op == 1:
                    name = name[:i] + ch + name[i:]
                elif op == 2 and len(name) > 3:
                    name = name[:i] + name[i + 1:]
                elif i + 1 < len(name):
                    name = name[:i] + name[i + 1] + name[i] + name[i + 2:]
        elif mode in ("lgen2", "lgen3", "lgen4"):
            # the train noise generator's own edits, more of them: a letter doubled, two neighbours
            # swapped or a character (space included) left out
            for _ in range(int(mode[-1])):
                i = int(rng.integers(max(len(name) - 1, 1)))
                op = int(rng.integers(3))
                if op == 0:
                    name = name[:i] + name[i] + name[i:]
                elif op == 1 and i + 1 < len(name):
                    name = name[:i] + name[i + 1] + name[i] + name[i + 2:]
                elif len(name) > 3:
                    name = name[:i] + name[i + 1:]
        elif mode in ("jgen1", "jgen2", "jgen3", "jgen4"):
            # the generator's edits applied to the jamo of a Hangul name (a jamo doubled, two neighbours
            # swapped or one left out), recomposed where it still composes: 그랜드 -> 그ᅳ랜드, 에이스 -> 에ᅵᄋ스
            name = unicodedata.normalize("NFD", name)
            for _ in range(int(mode[-1])):
                i = int(rng.integers(max(len(name) - 1, 1)))
                op = int(rng.integers(3))
                if op == 0:
                    name = name[:i] + name[i] + name[i:]
                elif op == 1 and i + 1 < len(name):
                    name = name[:i] + name[i + 1] + name[i] + name[i + 2:]
                elif len(name) > 3:
                    name = name[:i] + name[i + 1:]
            name = unicodedata.normalize("NFC", name)
        elif mode == "ime":
            # a Latin name typed with the Korean keyboard layout left on: GLOVIS SUN -> 히ㅐ퍈 녀ㅜ
            name = vlib.ime_typed(name)
        elif mode == "eng":
            # a Hangul name typed with the English layout left on: 글로비스 선 -> RMFFHQLTM TJS
            name = vlib.ime_keys(name)
        elif mode == "wordswap":
            # word order reversed (COURAGE GLOVIS, 커리지 글로비스)
            words = name.split()
            if len(words) >= 2:
                name = " ".join(words[::-1])
        elif mode in ("hsyl2", "hsyl3"):
            # two or three Hangul syllables replaced by a random syllable of the same initial consonant
            for _ in range(int(mode[-1])):
                spots = [i for i, ch in enumerate(name) if "가" <= ch <= "힣"]
                if spots:
                    i = spots[rng.integers(len(spots))]
                    cho = (ord(name[i]) - 0xAC00) // 588
                    name = name[:i] + chr(0xAC00 + cho * 588 + 28 * int(rng.integers(21)) + int(rng.integers(2)) * 4) + name[i + 1:]
        elif mode in ("hspell2", "hspell3"):
            # Hangul name with 2 or 3 random syllable edits (any syllable substituted or inserted, one left
            # out, or two neighbours swapped), the Hangul counterpart of lspell
            for _ in range(int(mode[-1])):
                i = int(rng.integers(max(len(name), 1)))
                op = int(rng.integers(4))
                ch = chr(0xAC00 + 28 * int(rng.integers(399)) + int(rng.integers(2)) * 4)
                if op == 0 and "가" <= name[i:i + 1] <= "힣":
                    name = name[:i] + ch + name[i + 1:]
                elif op == 1:
                    name = name[:i] + ch + name[i:]
                elif op == 2 and len(name) > 2:
                    name = name[:i] + name[i + 1:]
                elif i + 1 < len(name):
                    name = name[:i] + name[i + 1] + name[i] + name[i + 2:]
        elif mode == "addtoken":
            # a prefix or suffix the train names never carry
            if vlib.is_hangul(name):
                name = [lambda x: x + "호", lambda x: "엠브이 " + x, lambda x: x + " 선박"][int(rng.integers(3))](name)
            else:
                name = [lambda x: "MV " + x, lambda x: "M.T. " + x, lambda x: x + " (EX-" + x.split()[0] + " STAR)"][int(rng.integers(3))](name)
        elif mode == "lgarbage":
            # Latin name useless: every letter replaced at random
            name = "".join(chr(65 + int(rng.integers(26))) if ch.isalpha() else ch for ch in name)
        elif mode == "garbage":
            # name useless: every Hangul syllable replaced at random, so only tonnage, type and flag are left
            name = "".join(chr(0xAC00 + 28 * int(rng.integers(399))) if "가" <= ch <= "힣" else ch for ch in name)
        else:
            raise ValueError(mode)
        out.append(name)
    df["vessel_name"] = out
    return df


def compose(df, rng, mode):
    """Stress mode, or several joined by '+' applied in turn (e.g. lgen3+nameonly)."""
    for part in mode.split("+"):
        df = (perturb_name if part in NAME_MODES else perturb)(df, rng, part)
    return df


# (mode, weight, multi-word names only); single-word names draw only from the modes that apply to them
HANGUL_BROAD = [("hsyl2", 0.2, False), ("hsyl3", 0.1, False), ("wordswap", 0.15, True), ("droplast", 0.1, True),
                ("trunc", 0.1, False), ("addtoken", 0.1, False), ("dropsyl", 0.1, False), ("spell3", 0.05, False),
                ("dropword", 0.1, True)]
LATIN_BROAD = [("wordswap", 0.15, True), ("lgen2", 0.15, False), ("lgen3", 0.1, False), ("lspell2", 0.1, False),
               ("addtoken", 0.1, False), ("dropfirst", 0.1, True), ("droplast", 0.1, True), ("trunc", 0.1, False),
               ("initial", 0.05, True), ("lspell4", 0.05, False)]


def broad_copy(df, rng, mixture):
    """One damaged copy of each row, the damage drawn from the mixture."""
    if df.empty:
        return df
    multi = df["vessel_name"].str.split().str.len().to_numpy() >= 2
    choice = np.empty(len(df), dtype=object)
    for is_multi in (True, False):
        rows = np.flatnonzero(multi == is_multi)
        modes = [(m, w) for m, w, needs_multi in mixture if is_multi or not needs_multi]
        w = np.array([w for _, w in modes])
        choice[rows] = [modes[k][0] for k in rng.choice(len(modes), len(rows), p=w / w.sum())]
    return pd.concat([perturb_name(df[choice == m], rng, m) for m in dict.fromkeys(choice) if (choice == m).any()])


# name damage for the noisy copies: (mode, weight, multi-word names only). "spell" holds letter- and
# syllable-level edits, "word" leaves words out or shortens the name, "both" mixes the two halves
NOISY_NAME = "spell"
NOISY_MIXES = {
    "hangul": {
        "spell": [("hsyl2", 0.2, False), ("hsyl3", 0.1, False), ("hspell2", 0.2, False), ("spell3", 0.15, False),
                  ("dropsyl", 0.15, False), ("keep", 0.2, False)],
        "word": [("dropfirst", 0.25, True), ("droplast", 0.25, True), ("dropword", 0.1, True), ("trunc", 0.15, False),
                 ("dropsyl", 0.1, False), ("keep", 0.15, False)],
    },
    "latin": {
        "spell": [("lspell2", 0.25, False), ("lspell4", 0.25, False), ("lgen3", 0.2, False), ("lgen4", 0.2, False),
                  ("keep", 0.1, False)],
        "word": [("dropfirst", 0.25, True), ("droplast", 0.25, True), ("dropword", 0.1, True), ("trunc", 0.15, False),
                 ("initial", 0.1, True), ("lgen2", 0.05, False), ("keep", 0.1, False)],
    },
}
TONNAGE_NOISY = [("keep", 0.35), ("u10", 0.3), ("u20", 0.35)]
TONNAGE_NOISY_HANGUL = None  # tonnage mixture for the Hangul copies (None: the same as TONNAGE_NOISY)
# vessel type / flag damage in the noisy copies (keep only: off, and no random draws)
FIELDS_NOISY = [("keep", 1.0)]


def noisy_mix(script: str) -> list:
    mixes = NOISY_MIXES[script]
    if NOISY_NAME == "both":
        return [(m, w / 2, multi) for half in ("spell", "word") for m, w, multi in mixes[half]]
    return mixes[NOISY_NAME]


def noisy_copy(df, rng, name_mix, tonnage_mix=None):
    """One copy of each row with a name damage and a tonnage damage drawn per row from the mixtures
    (single-word names draw only from the modes that apply to them)."""
    if df.empty:
        return df
    multi = df["vessel_name"].str.split().str.len().to_numpy() >= 2
    names = np.empty(len(df), dtype=object)
    for is_multi in (True, False):
        rows = np.flatnonzero(multi == is_multi)
        modes = [(m, w) for m, w, needs_multi in name_mix if is_multi or not needs_multi]
        w = np.array([w for _, w in modes])
        names[rows] = [modes[k][0] for k in rng.choice(len(modes), len(rows), p=w / w.sum())]
    tonnage_mix = TONNAGE_NOISY if tonnage_mix is None else tonnage_mix
    w = np.array([w for _, w in tonnage_mix])
    tonnage = np.array([m for m, _ in tonnage_mix], dtype=object)[rng.choice(len(tonnage_mix), len(df), p=w / w.sum())]
    parts = []
    for nm in dict.fromkeys(names):
        for tm in dict.fromkeys(tonnage):
            sub = df[(names == nm) & (tonnage == tm)]
            if len(sub):
                sub = sub if nm == "keep" else perturb_name(sub, rng, nm)
                parts.append(sub if tm == "keep" else perturb(sub, rng, tm))
    out = pd.concat(parts)
    if any(m != "keep" for m, _ in FIELDS_NOISY):
        w = np.array([w for _, w in FIELDS_NOISY])
        fields = np.array([m for m, _ in FIELDS_NOISY], dtype=object)[rng.choice(len(FIELDS_NOISY), len(out), p=w / w.sum())]
        out = pd.concat([out[fields == fm] if fm == "keep" else perturb(out[fields == fm], rng, fm)
                         for fm in dict.fromkeys(fields)])
    return out


def augment(df, rng, copies, mode="aug", name_copies=0):
    """Extra training queries: copies of the Hangul rows with perturbed tonnage, copies with the name
    written differently (NAME_AUG; half of those with perturbed tonnage too when copies > 0), and
    copies of a share of the multi-word Latin rows without their first word (LATIN_NAME_FRAC)."""
    kor = df[df["vessel_name"].map(vlib.is_hangul)]
    parts = [perturb(kor, rng, mode).assign(sample_id=lambda d, c=c: d["sample_id"] + f"_aug{c}")
             for c in range(copies)]
    multi = kor[kor["vessel_name"].str.split().str.len() >= 2]
    for c in range(name_copies):
        if NAME_AUG == "dropfirst":
            part = perturb_name(multi, rng, "dropfirst")
        elif NAME_AUG == "mix2":
            # wider mixture: multi-word names lose their first, last or a random word, a syllable,
            # or get three jamo swaps; single-word names lose a syllable or get the jamo swaps
            u = rng.random(len(kor))
            is_multi = kor.index.isin(multi.index)
            groups = [(is_multi & (u < 0.3), "dropfirst"), (is_multi & (u >= 0.3) & (u < 0.5), "droplast"),
                      (is_multi & (u >= 0.5) & (u < 0.6), "dropword"),
                      ((is_multi & (u >= 0.6) & (u < 0.8)) | (~is_multi & (u < 0.5)), "dropsyl"),
                      ((is_multi & (u >= 0.8)) | (~is_multi & (u >= 0.5)), "spell3")]
            part = pd.concat([perturb_name(kor[g], rng, m) for g, m in groups if g.any()])
        else:
            # mix: multi-word names lose their first (1/2) or a random word (1/4) or get a jamo swap (1/4);
            # single-word names get a jamo swap
            u = rng.random(len(kor))
            is_multi = kor.index.isin(multi.index)
            groups = [(is_multi & (u < 0.5), "dropfirst"), (is_multi & (u >= 0.5) & (u < 0.75), "dropword"),
                      ((is_multi & (u >= 0.75)) | ~is_multi, "spell")]
            part = pd.concat([perturb_name(kor[g], rng, m) for g, m in groups if g.any()])
        noisy = rng.random(len(part)) < (0.5 if copies else 0.0)
        part = pd.concat([part[~noisy], perturb(part[noisy], rng, mode)])
        parts.append(part.assign(sample_id=lambda d, c=c: d["sample_id"] + f"_name{c}"))
    if LATIN_NAME_FRAC:
        lat = df[~df["vessel_name"].map(vlib.is_hangul) & (df["vessel_name"].str.split().str.len() >= 2)]
        lat = lat[rng.random(len(lat)) < LATIN_NAME_FRAC]
        parts.append(perturb_name(lat, rng, "dropfirst").assign(sample_id=lambda d: d["sample_id"] + "_lname"))
    for c in range(BROAD_HANGUL):
        parts.append(broad_copy(kor, rng, HANGUL_BROAD).assign(sample_id=lambda d, c=c: d["sample_id"] + f"_broad{c}"))
    if BROAD_LATIN_FRAC:
        lat = df[~df["vessel_name"].map(vlib.is_hangul)]
        lat = lat[rng.random(len(lat)) < BROAD_LATIN_FRAC]
        parts.append(broad_copy(lat, rng, LATIN_BROAD).assign(sample_id=lambda d: d["sample_id"] + "_b_lname"))
    for c in range(NOISY_HANGUL):
        parts.append(noisy_copy(kor, rng, noisy_mix("hangul"), TONNAGE_NOISY_HANGUL)
                     .assign(sample_id=lambda d, c=c: d["sample_id"] + f"_noisy{c}"))
    if NOISY_LATIN_FRAC:
        lat = df[~df["vessel_name"].map(vlib.is_hangul)]
        lat = lat[rng.random(len(lat)) < NOISY_LATIN_FRAC]
        parts.append(noisy_copy(lat, rng, noisy_mix("latin")).assign(sample_id=lambda d: d["sample_id"] + "_n_lname"))
    if NOIGT_FRAC:
        part = kor[kor["international_gross_tonnage"].notna() & (rng.random(len(kor)) < NOIGT_FRAC)]
        parts.append(part.assign(international_gross_tonnage=np.nan, sample_id=lambda d: d["sample_id"] + "_noigt"))
    return pd.concat(parts, ignore_index=True) if parts else kor.iloc[:0]


def cv_once(args, train, catalog, index, rows, kor, kor_idx, cands, feats, split_seed):
    params = dict(PARAMS, objective=args.objective)
    rng = np.random.default_rng(1000 + split_seed)
    oof = np.empty(len(train), dtype=object)
    margin = np.full(len(train), np.nan)
    gap = np.full(len(train), np.nan)  # best minus second-best score, known without the label
    recall = np.zeros(len(train), dtype=bool)
    stress = {mode: [] for mode in args.stress}
    stress_lat = {mode: [] for mode in args.stress_latin}
    s_recall = defaultdict(list)  # target among the candidates, per stress mode
    s_unsure = defaultdict(list)  # ranker gap below SURE_GAP, per stress mode
    dump = []  # with --dump-stress: one line per stressed row (mode, record, target, prediction)
    # stress draws use their own generators, so adding a stress mode never changes the augmented training rows
    rng_kor = np.random.default_rng(3000 + split_seed)
    rng_lat = np.random.default_rng(2000 + split_seed)
    sel = lambda idx, seq: [seq[i] for i in idx]  # noqa: E731
    for fold, (tr_idx, va_idx) in enumerate(fold_splits(train, split_seed)):
        if args.max_folds and fold >= args.max_folds:
            break
        t0 = time.time()
        index.tables = vlib.fit_tables(train.iloc[tr_idx], catalog)
        k_cands, k_feats = vlib.build_matrix(index, [rows[i] for i in kor_idx])
        fc, ff = list(cands), list(feats)
        for k, i in enumerate(kor_idx):
            fc[i], ff[i] = k_cands[k], k_feats[k]
        ff = refresh_tables(index, rows, fc, ff)
        tr_rows, tr_c, tr_f = sel(tr_idx, rows), sel(tr_idx, fc), sel(tr_idx, ff)
        aug = augment(train.iloc[tr_idx], rng, args.aug, args.aug_mode, args.aug_name).to_dict("records")
        sets = model_sets(tr_rows, tr_c, tr_f, aug, vlib.build_matrix(index, aug) if aug else ([], []), args.latin_models)
        models = {}
        for rows_for, parts in sets.items():
            Xtr, ytr, gtr, wtr = stack(index, *parts, korean_weight=args.kw)
            models[rows_for] = [(train_model(blank(Xtr, d), ytr, gtr, wtr, rounds=args.rounds, params=params), d)
                                for d in parse_views(args)]

        def score(feats_, rows_):
            X = np.vstack(feats_)
            korean = np.repeat([vlib.is_hangul(r.get("vessel_name")) for r in rows_], [len(f) for f in feats_])
            total, count = np.zeros(len(X)), np.zeros(len(X))
            for rows_for, ms in models.items():
                mask = row_mask(rows_for, korean)
                for m, d in ms:
                    total[mask] += m.predict(blank(X[mask], d))
                    count[mask] += 1
            return total / count

        scores = score(sel(va_idx, ff), sel(va_idx, rows))
        pred = argmax_ids(index, sel(va_idx, fc), scores)
        oof[va_idx] = pred
        offset = 0
        for i in va_idx:
            local = scores[offset: offset + len(fc[i])]
            offset += len(fc[i])
            hit = fc[i] == index.id_to_index[rows[i]["target_vessel_id"]]
            recall[i] = hit.any()
            if len(local) > 1:
                top2 = np.partition(local, -2)[-2:]
                gap[i] = top2[1] - top2[0]
            # target score minus the best other candidate (negative = wrong)
            if hit.any() and (~hit).any():
                margin[i] = local[hit].max() - local[~hit].max()
        va_kor = [i for i in va_idx if kor[i]]
        if "twin" in args.stress:
            # 동명 선박: an unrelated catalog vessel renamed to each held-out target's name
            targets = train["target_vessel_id"].iloc[va_kor].unique()
            others = np.setdiff1d(np.arange(len(catalog)), [index.id_to_index[t] for t in targets])
            donors = rng_kor.choice(others, size=len(targets), replace=False)
            cat2 = catalog.copy()
            names = catalog.set_index("vessel_id")["vessel_name"]
            cat2.loc[cat2.index[donors], "vessel_name"] = names.loc[targets].to_numpy()
            index2 = vlib.CatalogIndex(cat2, index.tables)
            st = train.iloc[va_kor].to_dict("records")
            s_c, s_f = vlib.build_matrix(index2, st)
            s_pred = argmax_ids(index2, s_c, score(s_f, st))
            stress["twin"] += [p == r["target_vessel_id"] for p, r in zip(s_pred, st)]
            s_recall["k_twin"] += [index.id_to_index[r["target_vessel_id"]] in c for c, r in zip(s_c, st)]
        if "nodict" in args.stress:
            # new vocabulary: none of the record's Hangul words were seen in train
            saved = index.tables
            index.tables = dict(saved, kdict={})
            st = train.iloc[va_kor].to_dict("records")
            s_c, s_f = vlib.build_matrix(index, st)
            index.tables = saved
            s_pred = argmax_ids(index, s_c, score(s_f, st))
            stress["nodict"] += [p == r["target_vessel_id"] for p, r in zip(s_pred, st)]
            s_recall["k_nodict"] += [index.id_to_index[r["target_vessel_id"]] in c for c, r in zip(s_c, st)]
        for mode in args.stress:
            if mode in ("twin", "nodict"):
                continue
            st = compose(train.iloc[va_kor], rng_kor, mode).to_dict("records")
            s_c, s_f = vlib.build_matrix(index, st)
            s_scores = score(s_f, st)
            s_pred = argmax_ids(index, s_c, s_scores)
            stress[mode] += [p == r["target_vessel_id"] for p, r in zip(s_pred, st)]
            s_unsure[f"k_{mode}"] += list(row_gaps(s_c, s_scores) < SURE_GAP)
            s_recall[f"k_{mode}"] += [index.id_to_index[r["target_vessel_id"]] in c for c, r in zip(s_c, st)]
            if args.dump_stress:
                dump += stress_rows(f"k_{mode}", index, st, s_c, s_scores, s_pred)
        va_lat = [i for i in va_idx if not kor[i]]
        if args.stress_latin and args.latin_cap and len(va_lat) > args.latin_cap:
            va_lat = sorted(rng_lat.choice(va_lat, args.latin_cap, replace=False).tolist())
        for mode in args.stress_latin:
            st = compose(train.iloc[va_lat], rng_lat, mode).to_dict("records")
            s_c, s_f = vlib.build_matrix(index, st)
            s_scores = score(s_f, st)
            s_pred = argmax_ids(index, s_c, s_scores)
            stress_lat[mode] += [p == r["target_vessel_id"] for p, r in zip(s_pred, st)]
            s_unsure[f"e_{mode}"] += list(row_gaps(s_c, s_scores) < SURE_GAP)
            s_recall[f"e_{mode}"] += [index.id_to_index[r["target_vessel_id"]] in c for c, r in zip(s_c, st)]
            if args.dump_stress:
                dump += stress_rows(f"e_{mode}", index, st, s_c, s_scores, s_pred)
        if args.verbose:
            m = vlib.weighted_accuracy(train["vessel_name"].iloc[va_idx], train["target_vessel_id"].iloc[va_idx], pred)
            print(f"fold {fold}: {m} ({time.time() - t0:.0f}s)", flush=True)
    done = np.array([o is not None for o in oof])  # every row, or with --max-folds only the folds run
    m = vlib.weighted_accuracy(train["vessel_name"][done], train["target_vessel_id"][done], oof[done])
    m.update({f"k_{mode}": float(np.mean(v)) for mode, v in stress.items()})
    m.update({f"e_{mode}": float(np.mean(v)) for mode, v in stress_lat.items()})
    m.update({f"{k}_recall": round(float(np.mean(v)), 4) for k, v in s_recall.items()})
    # probe fingerprint per mode: shares of (correct and unsure, correct and sure, wrong), as probe_sure measures
    for k, uns in s_unsure.items():
        ok = np.array((stress if k.startswith("k_") else stress_lat)[k[2:]], dtype=bool)
        uns = np.array(uns, dtype=bool)
        m[f"{k}_print"] = (round(float((ok & uns).mean()), 3), round(float((ok & ~uns).mean()), 3), round(float((~ok).mean()), 3))
    for nm, msk in (("k_clean", kor & done), ("e_clean", ~kor & done)):
        ok = (oof == train["target_vessel_id"].to_numpy())[msk]
        uns = (np.nan_to_num(gap, nan=np.inf) < SURE_GAP)[msk]
        m[f"{nm}_print"] = (round(float((ok & uns).mean()), 3), round(float((ok & ~uns).mean()), 3), round(float((~ok).mean()), 3))
    print(f"split {split_seed}: recall k={recall[kor & done].mean():.4f} e={recall[~kor & done].mean():.4f} CV {m}", flush=True)
    WORK.mkdir(exist_ok=True)
    out = train.copy()
    out["pred"] = oof
    out["margin"] = margin
    out["gap"] = gap
    out.to_csv(WORK / f"oof_{args.tag}{split_seed}.csv", index=False)
    if dump:
        pd.DataFrame(dump).to_csv(WORK / f"stress_{args.tag}{split_seed}.csv", index=False)
    if args.verbose:
        model = next(iter(models.values()))[0][0]
        imp = pd.Series(model.feature_importance("gain"), index=vlib.FEATURES).sort_values(ascending=False)
        print(imp.head(30).round(0).to_string())
    return m


def stress_rows(mode, index, rows, cands, scores, preds):
    """Per-row stress outcome for error analysis: the damaged record, its target, the pick and the target's rank."""
    out, offset = [], 0
    for row, cand, pred in zip(rows, cands, preds):
        local = scores[offset: offset + len(cand)]
        offset += len(cand)
        hit = np.flatnonzero(cand == index.id_to_index[row["target_vessel_id"]])
        rank = int((local > local[hit[0]]).sum()) if len(hit) else -1
        out.append({"mode": mode, "vessel_name": row["vessel_name"], "gross_tonnage": row["gross_tonnage"],
                    "international_gross_tonnage": row["international_gross_tonnage"],
                    "target": row["target_vessel_id"], "pred": pred, "rank": rank, "n_cand": len(cand)})
    return out


def model_sets(rows, cands, feats, aug, aug_matrix, latin_models):
    """Training queries per ranker set: one set for every row, or with latin_models a set for the
    Hangul rows that leaves out the Latin name copies and a set for the Latin rows that keeps them."""
    a_c, a_f = aug_matrix
    everything = (rows + aug, cands + a_c, feats + a_f)
    if not latin_models:
        return {"all": everything}
    keep = [k for k, r in enumerate(aug) if not str(r["sample_id"]).endswith("_lname")]
    without = (rows + [aug[k] for k in keep], cands + [a_c[k] for k in keep], feats + [a_f[k] for k in keep])
    return {"hangul": without, "latin": everything}


def row_mask(rows_for: str, korean: np.ndarray) -> np.ndarray:
    return {"all": np.ones_like(korean), "hangul": korean, "latin": ~korean}[rows_for]


def parse_views(args) -> list[list[str]]:
    """One ranker per view; a view lists the features it leaves out ('none' = all)."""
    views = []
    for v in args.views:
        drop = [] if v == "none" else v.split("+")
        unknown = [f for f in drop + args.drop if f not in vlib.COL]
        if unknown:
            raise SystemExit(f"unknown features: {unknown}")
        views.append(drop + args.drop)
    return views


def blank(X, drop):
    """Copy of X with the dropped feature columns set to missing."""
    if not drop:
        return X
    X = X.copy()
    X[:, [vlib.COL[f] for f in drop]] = np.nan
    return X


def run_final(args):
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    train, catalog = load()
    tables = vlib.fit_tables(train, catalog)
    index = vlib.CatalogIndex(catalog, tables)
    rows = train.to_dict("records")
    cands, feats = vlib.build_matrix(index, rows)
    aug = augment(train, np.random.default_rng(1000), args.aug, args.aug_mode, args.aug_name).to_dict("records")
    sets = model_sets(rows, cands, feats, aug, vlib.build_matrix(index, aug) if aug else ([], []), args.latin_models)
    for old in model_dir.glob("ranker*.txt"):
        old.unlink()
    views = []
    for rows_for, parts in sets.items():
        X, y, g, w = stack(index, *parts, korean_weight=args.kw)
        for k, drop in enumerate(parse_views(args)):
            name = f"ranker_{k}.txt" if rows_for == "all" else f"ranker_{rows_for[0]}{k}.txt"
            model = train_model(blank(X, drop), y, g, w, rounds=args.rounds, params=dict(PARAMS, objective=args.objective))
            model.save_model(str(model_dir / name))
            views.append({"file": name, "drop": drop, "rows": rows_for})
    with open(model_dir / "views.json", "w", encoding="utf-8") as fh:
        json.dump(views, fh, indent=1)
    with open(model_dir / "tables.json", "w", encoding="utf-8") as fh:
        json.dump(tables, fh, ensure_ascii=False)
    with open(model_dir / "settings.json", "w", encoding="utf-8") as fh:
        json.dump(vlib.current_settings(), fh, indent=1)
    print(f"saved {len(views)} ranker(s), tables and settings {vlib.current_settings()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--split-seeds", type=int, nargs="+", default=[-1])
    ap.add_argument("--kw", type=float, default=KOREAN_WEIGHT)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--objective", default=PARAMS["objective"])
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--drop", nargs="*", default=[], help="features every view leaves out")
    ap.add_argument("--aug", type=int, default=AUG_COPIES, help="perturbed-tonnage copies per Hangul train row")
    ap.add_argument("--aug-mode", default=AUG_MODE, help="perturb() mode for the augmented copies")
    ap.add_argument("--aug-name", type=int, default=AUG_NAME, help="Hangul copies with the name written differently (see NAME_AUG)")
    ap.add_argument("--stress", nargs="*", default=["miss", "u15", "heavy"],
                    help="CV also scores Hangul validation rows with tonnage perturbed this way")
    ap.add_argument("--stress-latin", nargs="*", default=[],
                    help="the same for the non-Hangul validation rows")
    ap.add_argument("--latin-cap", type=int, default=0, help="Latin stress rows per fold (0 = all)")
    ap.add_argument("--tag", default="", help="prefix for the OOF file name")
    ap.add_argument("--max-folds", type=int, default=0, help="stop after this many folds (0 = all five); quick A/B")
    ap.add_argument("--model-dir", default=str(MODEL_DIR), help="where --final writes the rankers, tables and settings")
    ap.add_argument("--dump-stress", action="store_true", help="also write work/stress_<tag><seed>.csv, one line per stressed row")
    ap.add_argument("--threads", type=int, default=PARAMS["num_threads"])
    ap.add_argument("--views", nargs="+", default=VIEWS,
                    help="rankers to average; each is 'none' or features to leave out joined by '+'")
    ap.add_argument("--whole-name", default=vlib.WHOLE_NAME_SEARCH, choices=["always", "no_tonnage"])
    ap.add_argument("--exact", action=argparse.BooleanOptionalAction, default=vlib.EXACT_TONNAGE,
                    help="add Hangul candidates with exactly the recorded tonnage")
    ap.add_argument("--name-aug", default=NAME_AUG, choices=["dropfirst", "mix", "mix2"])
    ap.add_argument("--latin-words", action=argparse.BooleanOptionalAction, default=vlib.LATIN_WORDS,
                    help="Latin word-level retrieval")
    ap.add_argument("--latin-prefix", action=argparse.BooleanOptionalAction, default=vlib.LATIN_PREFIX,
                    help="with --latin-words, also words a record word starts")
    ap.add_argument("--aug-latin-name", type=float, default=LATIN_NAME_FRAC,
                    help="share of multi-word Latin rows copied without their first word")
    ap.add_argument("--latin-models", action=argparse.BooleanOptionalAction, default=LATIN_MODELS,
                    help="separate rankers for Latin rows, the only ones trained on the Latin name copies")
    ap.add_argument("--domestic", action=argparse.BooleanOptionalAction, default=vlib.DOMESTIC_GT,
                    help="Hangul records with gross tonnage only: search up to ~1.8x above it")
    ap.add_argument("--paren", action=argparse.BooleanOptionalAction, default=vlib.PAREN_STRIP,
                    help="drop bracketed remarks from Latin names")
    ap.add_argument("--latin-sorted", action=argparse.BooleanOptionalAction, default=vlib.LATIN_SORTED,
                    help="also retrieve Latin names with the same words in another order")
    ap.add_argument("--latin-relcut", action=argparse.BooleanOptionalAction, default=vlib.LATIN_RELCUT,
                    help="Latin edit-distance retrieval cutoff grows with name length")
    ap.add_argument("--broad-hangul", type=int, default=BROAD_HANGUL, help="broad-damage copies per Hangul row")
    ap.add_argument("--broad-latin", type=float, default=BROAD_LATIN_FRAC, help="share of Latin rows with a broad-damage copy")
    ap.add_argument("--noisy-hangul", type=int, default=NOISY_HANGUL, help="heavy-spelling/loose-tonnage copies per Hangul row")
    ap.add_argument("--noisy-latin", type=float, default=NOISY_LATIN_FRAC,
                    help="share of Latin rows with a heavy-spelling/loose-tonnage copy")
    ap.add_argument("--hangul-jamo", type=float, default=0.0,
                    help="weight of jamo-level generator edits (jgen2, jgen3) added to the Hangul spelling mixture")
    ap.add_argument("--jamo-keys", action=argparse.BooleanOptionalAction, default=vlib.JAMO_KEYS,
                    help="read loose jamo of a damaged Hangul record by sound instead of dropping them")
    ap.add_argument("--hangul-gen", type=float, default=0.0,
                    help="weight of the train noise generator's edits (lgen2-4: a character doubled, neighbours "
                         "swapped or one left out) added to the Hangul spelling mix of the noisy copies")
    ap.add_argument("--noisy-name", default=NOISY_NAME, choices=["spell", "word", "both"],
                    help="name damage in the noisy copies: letter/syllable edits, words left out, or both")
    ap.add_argument("--noisy-tonnage", default=",".join(f"{m}:{w}" for m, w in TONNAGE_NOISY),
                    help="tonnage damage mixture of the noisy copies, e.g. keep:0.35,u10:0.3,u20:0.35")
    ap.add_argument("--noisy-tonnage-hangul", default="",
                    help="tonnage mixture for the Hangul copies only, same format (default: --noisy-tonnage)")
    ap.add_argument("--noisy-fields", default=",".join(f"{m}:{w}" for m, w in FIELDS_NOISY),
                    help="vessel type/flag damage of the noisy copies, e.g. keep:0.8,badtype:0.1,notype:0.1")
    ap.add_argument("--latin-wide", action=argparse.BooleanOptionalAction, default=vlib.LATIN_WIDE,
                    help="Latin retrieval for heavier spelling damage (OSA cutoff by length, tonnage-window fuzzy matches)")
    ap.add_argument("--latin-ton-win", type=float, default=vlib.LATIN_TON_WIN)
    ap.add_argument("--hangul-win", type=float, default=vlib.HANGUL_WIN, help="relative tonnage window for Hangul candidates")
    ap.add_argument("--hangul-limit", type=int, default=vlib.HANGUL_LIMIT, help="best name matches kept from that window")
    ap.add_argument("--hangul-pre-ton", type=float, default=vlib.HANGUL_PRE_TON,
                    help="pick the Hangul window's candidates by name evidence minus this x |log tonnage ratio|")
    ap.add_argument("--hangul-partial", type=int, default=vlib.HANGUL_PARTIAL_EXTRA,
                    help="extra Hangul candidates whose name best contains the record's")
    ap.add_argument("--hangul-outer", type=float, default=vlib.HANGUL_OUTER,
                    help="wider tonnage window for a few more Hangul candidates (tonnage far off)")
    ap.add_argument("--hangul-outer-n", type=int, default=vlib.HANGUL_OUTER_N, help="best name matches kept from it")
    ap.add_argument("--hangul-dedup", action=argparse.BooleanOptionalAction, default=vlib.HANGUL_DEDUP,
                    help="also match Hangul names with doubled characters collapsed (흥흥아영 -> 흥아영)")
    ap.add_argument("--hangul-swap", action=argparse.BooleanOptionalAction, default=vlib.HANGUL_SWAP,
                    help="also match Hangul names with two neighbouring syllables swapped (먼샤 -> 샤먼)")
    ap.add_argument("--aug-noigt", type=float, default=NOIGT_FRAC,
                    help="share of Hangul rows copied with the international tonnage blank")
    a = ap.parse_args()
    PARAMS["num_threads"] = a.threads
    vlib.WHOLE_NAME_SEARCH = a.whole_name
    vlib.EXACT_TONNAGE = a.exact
    NAME_AUG = a.name_aug
    vlib.LATIN_WORDS = a.latin_words
    vlib.LATIN_PREFIX = a.latin_prefix
    LATIN_NAME_FRAC = a.aug_latin_name
    vlib.DOMESTIC_GT = a.domestic
    NOIGT_FRAC = a.aug_noigt
    BROAD_HANGUL = a.broad_hangul
    vlib.PAREN_STRIP = a.paren
    vlib.LATIN_SORTED = a.latin_sorted
    vlib.LATIN_RELCUT = a.latin_relcut
    BROAD_LATIN_FRAC = a.broad_latin
    NOISY_HANGUL = a.noisy_hangul
    NOISY_LATIN_FRAC = a.noisy_latin
    NOISY_NAME = a.noisy_name
    if a.hangul_jamo:
        NOISY_MIXES["hangul"]["spell"] = NOISY_MIXES["hangul"]["spell"] + [(m, a.hangul_jamo / 2, False) for m in ("jgen2", "jgen3")]
    vlib.JAMO_KEYS = a.jamo_keys
    if a.hangul_gen:
        NOISY_MIXES["hangul"]["spell"] = NOISY_MIXES["hangul"]["spell"] + [(m, a.hangul_gen / 3, False) for m in ("lgen2", "lgen3", "lgen4")]
    TONNAGE_NOISY = [(m, float(w)) for m, w in (part.split(":") for part in a.noisy_tonnage.split(","))]
    if a.noisy_tonnage_hangul:
        TONNAGE_NOISY_HANGUL = [(m, float(w)) for m, w in (part.split(":") for part in a.noisy_tonnage_hangul.split(","))]
    FIELDS_NOISY = [(m, float(w)) for m, w in (part.split(":") for part in a.noisy_fields.split(","))]
    vlib.LATIN_WIDE = a.latin_wide
    vlib.LATIN_TON_WIN = a.latin_ton_win
    vlib.HANGUL_WIN = a.hangul_win
    vlib.HANGUL_LIMIT = a.hangul_limit
    vlib.HANGUL_PRE_TON = a.hangul_pre_ton
    vlib.HANGUL_PARTIAL_EXTRA = a.hangul_partial
    vlib.HANGUL_OUTER = a.hangul_outer
    vlib.HANGUL_OUTER_N = a.hangul_outer_n
    vlib.HANGUL_DEDUP = a.hangul_dedup
    vlib.HANGUL_SWAP = a.hangul_swap
    if a.cv:
        run_cv(a)
    if a.final:
        run_final(a)
