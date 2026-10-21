"""Cross-validate the matcher and train the final ranker.

Usage:
    python tools/train.py --cv            # grouped 5-fold CV (unseen vessels)
    python tools/train.py --final         # train on all rows, write submit/model/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
# train rows never have bad tonnage on Hangul records, evaluation records do
AUG_COPIES = 2


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
    n = len(df)
    gt = df["gross_tonnage"].to_numpy(float)
    igt = df["international_gross_tonnage"].to_numpy(float)
    if mode == "miss":
        f1 = f2 = np.full(n, np.nan)
    elif mode == "u15":
        f1 = f2 = rng.uniform(0.85, 1.15, n)
    elif mode == "heavy":
        f1 = f2 = np.exp(rng.choice([-1.0, 1.0], n) * rng.uniform(0.1, 0.6, n))
    elif mode == "aug":
        # training mixture: a quarter blank, the rest off by up to x/÷1.8, fields usually off together
        f1 = np.exp(rng.choice([-1.0, 1.0], n) * rng.uniform(0.03, 0.6, n))
        other = np.exp(rng.choice([-1.0, 1.0], n) * rng.uniform(0.03, 0.6, n))
        f2 = np.where(rng.random(n) < 0.7, f1, other)
        blank = rng.random(n) < 0.25
        f1, f2 = np.where(blank, np.nan, f1), np.where(blank, np.nan, f2)
    else:
        raise ValueError(mode)
    df["gross_tonnage"] = np.round(gt * f1)
    df["international_gross_tonnage"] = np.round(igt * f2)
    return df


def augment(df, rng, copies):
    """Extra training queries: copies of the Hangul rows with perturbed tonnage."""
    kor = df[df["vessel_name"].map(vlib.is_hangul)]
    if not copies or kor.empty:
        return kor.iloc[:0]
    parts = [perturb(kor, rng, "aug").assign(sample_id=lambda d, c=c: d["sample_id"] + f"_aug{c}")
             for c in range(copies)]
    return pd.concat(parts, ignore_index=True)


def cv_once(args, train, catalog, index, rows, kor, kor_idx, cands, feats, split_seed):
    params = dict(PARAMS, objective=args.objective)
    rng = np.random.default_rng(1000 + split_seed)
    oof = np.empty(len(train), dtype=object)
    margin = np.full(len(train), np.nan)
    recall = np.zeros(len(train), dtype=bool)
    stress = {mode: [] for mode in args.stress}
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
        aug = augment(train.iloc[tr_idx], rng, args.aug).to_dict("records")
        if aug:
            a_c, a_f = vlib.build_matrix(index, aug)
            tr_rows, tr_c, tr_f = tr_rows + aug, tr_c + a_c, tr_f + a_f
        Xtr, ytr, gtr, wtr = stack(index, tr_rows, tr_c, tr_f, korean_weight=args.kw)
        models = [(train_model(blank(Xtr, d), ytr, gtr, wtr, rounds=args.rounds, params=params), d)
                  for d in parse_views(args)]

        def score(feats_):
            X = np.vstack(feats_)
            return sum(m.predict(blank(X, d)) for m, d in models)

        scores = score(sel(va_idx, ff))
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
        for mode in args.stress:
            st = perturb(train.iloc[va_kor], rng, mode).to_dict("records")
            s_c, s_f = vlib.build_matrix(index, st)
            s_pred = argmax_ids(index, s_c, score(s_f))
            stress[mode] += [p == r["target_vessel_id"] for p, r in zip(s_pred, st)]
        if args.verbose:
            m = vlib.weighted_accuracy(train["vessel_name"].iloc[va_idx], train["target_vessel_id"].iloc[va_idx], pred)
            print(f"fold {fold}: {m} ({time.time() - t0:.0f}s)", flush=True)
    m = vlib.weighted_accuracy(train["vessel_name"], train["target_vessel_id"], oof)
    m.update({f"k_{mode}": float(np.mean(v)) for mode, v in stress.items()})
    print(f"split {split_seed}: recall k={recall[kor].mean():.4f} e={recall[~kor].mean():.4f} CV {m}", flush=True)
    WORK.mkdir(exist_ok=True)
    out = train.copy()
    out["pred"] = oof
    out["margin"] = margin
    out.to_csv(WORK / f"oof_{split_seed}.csv", index=False)
    if args.verbose:
        model = models[0][0]
        imp = pd.Series(model.feature_importance("gain"), index=vlib.FEATURES).sort_values(ascending=False)
        print(imp.head(30).round(0).to_string())
    return m


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
    rows += augment(train, np.random.default_rng(1000), args.aug).to_dict("records")
    cands, feats = vlib.build_matrix(index, rows)
    X, y, g, w = stack(index, rows, cands, feats)
    for old in MODEL_DIR.glob("ranker*.txt"):
        old.unlink()
    views = []
    for k, drop in enumerate(parse_views(args)):
        model = train_model(blank(X, drop), y, g, w, rounds=args.rounds, params=dict(PARAMS, objective=args.objective))
        model.save_model(str(MODEL_DIR / f"ranker_{k}.txt"))
        views.append({"file": f"ranker_{k}.txt", "drop": drop})
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
    ap.add_argument("--stress", nargs="*", default=["miss", "u15", "heavy"],
                    help="CV also scores Hangul validation rows with tonnage perturbed this way")
    ap.add_argument("--threads", type=int, default=PARAMS["num_threads"])
    ap.add_argument("--views", nargs="+", default=VIEWS,
                    help="rankers to average; each is 'none' or features to leave out joined by '+'")
    a = ap.parse_args()
    PARAMS["num_threads"] = a.threads
    if a.cv:
        run_cv(a)
    if a.final:
        run_final(a)
