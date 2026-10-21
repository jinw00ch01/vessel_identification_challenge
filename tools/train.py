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
    n = len(df)
    gt = df["gross_tonnage"].to_numpy(float)
    igt = df["international_gross_tonnage"].to_numpy(float)
    if mode == "miss":
        f1 = f2 = np.full(n, np.nan)
    elif mode == "u15":
        f1 = f2 = rng.uniform(0.85, 1.15, n)
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
NAME_MODES = ("spell", "spell3", "dropsyl", "nospace", "dropword", "dropfirst", "droplast", "initial", "trunc")


def perturb_name(df, rng, mode):
    """Hangul names written differently: one jamo swapped, spaces removed, or a word left out."""
    df = df.copy()
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
        else:
            raise ValueError(mode)
        out.append(name)
    df["vessel_name"] = out
    return df


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
    if NOIGT_FRAC:
        part = kor[kor["international_gross_tonnage"].notna() & (rng.random(len(kor)) < NOIGT_FRAC)]
        parts.append(part.assign(international_gross_tonnage=np.nan, sample_id=lambda d: d["sample_id"] + "_noigt"))
    return pd.concat(parts, ignore_index=True) if parts else kor.iloc[:0]


def cv_once(args, train, catalog, index, rows, kor, kor_idx, cands, feats, split_seed):
    params = dict(PARAMS, objective=args.objective)
    rng = np.random.default_rng(1000 + split_seed)
    oof = np.empty(len(train), dtype=object)
    margin = np.full(len(train), np.nan)
    recall = np.zeros(len(train), dtype=bool)
    stress = {mode: [] for mode in args.stress}
    stress_lat = {mode: [] for mode in args.stress_latin}
    s_recall = defaultdict(list)  # target among the candidates, per stress mode
    # stress draws use their own generators, so adding a stress mode never changes the augmented training rows
    rng_kor = np.random.default_rng(3000 + split_seed)
    rng_lat = np.random.default_rng(2000 + split_seed)
    sel = lambda idx, seq: [seq[i] for i in idx]  # noqa: E731
    for fold, (tr_idx, va_idx) in enumerate(fold_splits(train, split_seed)):
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
            pert = perturb_name if mode in NAME_MODES else perturb
            st = pert(train.iloc[va_kor], rng_kor, mode).to_dict("records")
            s_c, s_f = vlib.build_matrix(index, st)
            s_pred = argmax_ids(index, s_c, score(s_f, st))
            stress[mode] += [p == r["target_vessel_id"] for p, r in zip(s_pred, st)]
            s_recall[f"k_{mode}"] += [index.id_to_index[r["target_vessel_id"]] in c for c, r in zip(s_c, st)]
        va_lat = [i for i in va_idx if not kor[i]]
        for mode in args.stress_latin:
            pert = perturb_name if mode in NAME_MODES else perturb
            st = pert(train.iloc[va_lat], rng_lat, mode).to_dict("records")
            s_c, s_f = vlib.build_matrix(index, st)
            s_pred = argmax_ids(index, s_c, score(s_f, st))
            stress_lat[mode] += [p == r["target_vessel_id"] for p, r in zip(s_pred, st)]
            s_recall[f"e_{mode}"] += [index.id_to_index[r["target_vessel_id"]] in c for c, r in zip(s_c, st)]
        if args.verbose:
            m = vlib.weighted_accuracy(train["vessel_name"].iloc[va_idx], train["target_vessel_id"].iloc[va_idx], pred)
            print(f"fold {fold}: {m} ({time.time() - t0:.0f}s)", flush=True)
    m = vlib.weighted_accuracy(train["vessel_name"], train["target_vessel_id"], oof)
    m.update({f"k_{mode}": float(np.mean(v)) for mode, v in stress.items()})
    m.update({f"e_{mode}": float(np.mean(v)) for mode, v in stress_lat.items()})
    m.update({f"{k}_recall": round(float(np.mean(v)), 4) for k, v in s_recall.items()})
    print(f"split {split_seed}: recall k={recall[kor].mean():.4f} e={recall[~kor].mean():.4f} CV {m}", flush=True)
    WORK.mkdir(exist_ok=True)
    out = train.copy()
    out["pred"] = oof
    out["margin"] = margin
    out.to_csv(WORK / f"oof_{args.tag}{split_seed}.csv", index=False)
    if args.verbose:
        model = next(iter(models.values()))[0][0]
        imp = pd.Series(model.feature_importance("gain"), index=vlib.FEATURES).sort_values(ascending=False)
        print(imp.head(30).round(0).to_string())
    return m


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
    train, catalog = load()
    tables = vlib.fit_tables(train, catalog)
    index = vlib.CatalogIndex(catalog, tables)
    rows = train.to_dict("records")
    cands, feats = vlib.build_matrix(index, rows)
    aug = augment(train, np.random.default_rng(1000), args.aug, args.aug_mode, args.aug_name).to_dict("records")
    sets = model_sets(rows, cands, feats, aug, vlib.build_matrix(index, aug) if aug else ([], []), args.latin_models)
    for old in MODEL_DIR.glob("ranker*.txt"):
        old.unlink()
    views = []
    for rows_for, parts in sets.items():
        X, y, g, w = stack(index, *parts, korean_weight=args.kw)
        for k, drop in enumerate(parse_views(args)):
            name = f"ranker_{k}.txt" if rows_for == "all" else f"ranker_{rows_for[0]}{k}.txt"
            model = train_model(blank(X, drop), y, g, w, rounds=args.rounds, params=dict(PARAMS, objective=args.objective))
            model.save_model(str(MODEL_DIR / name))
            views.append({"file": name, "drop": drop, "rows": rows_for})
    with open(MODEL_DIR / "views.json", "w", encoding="utf-8") as fh:
        json.dump(views, fh, indent=1)
    with open(MODEL_DIR / "tables.json", "w", encoding="utf-8") as fh:
        json.dump(tables, fh, ensure_ascii=False)
    print(f"saved {len(views)} ranker(s) and tables")


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
    ap.add_argument("--tag", default="", help="prefix for the OOF file name")
    ap.add_argument("--threads", type=int, default=PARAMS["num_threads"])
    ap.add_argument("--views", nargs="+", default=VIEWS,
                    help="rankers to average; each is 'none' or features to leave out joined by '+'")
    ap.add_argument("--whole-name", default=vlib.WHOLE_NAME_SEARCH, choices=["always", "no_tonnage"])
    ap.add_argument("--exact", action=argparse.BooleanOptionalAction, default=vlib.EXACT_TONNAGE,
                    help="add Hangul candidates with exactly the recorded tonnage")
    ap.add_argument("--name-aug", default=NAME_AUG, choices=["dropfirst", "mix"])
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
    if a.cv:
        run_cv(a)
    if a.final:
        run_final(a)
