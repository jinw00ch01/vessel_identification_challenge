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
    "objective": "binary",
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


def stack(index, rows, cands, feats, with_labels=True):
    X = np.vstack(feats)
    groups = np.array([len(c) for c in cands])
    kor = np.array([vlib.is_hangul(r.get("vessel_name")) for r in rows])
    w = np.repeat(np.where(kor, KOREAN_WEIGHT, 1.0), groups)
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

    gkf = GroupKFold(n_splits=5)
    oof = np.empty(len(train), dtype=object)
    recall = np.zeros(len(train), dtype=bool)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, groups=train["target_vessel_id"])):
        t0 = time.time()
        index.tables = vlib.fit_tables(train.iloc[tr_idx], catalog)
        k_cands, k_feats = vlib.build_matrix(index, [rows[i] for i in kor_idx])
        fc, ff = list(cands), list(feats)
        for k, i in enumerate(kor_idx):
            fc[i], ff[i] = k_cands[k], k_feats[k]
        ff = refresh_tables(index, rows, fc, ff)
        sel = lambda idx, seq: [seq[i] for i in idx]  # noqa: E731
        Xtr, ytr, gtr, wtr = stack(index, sel(tr_idx, rows), sel(tr_idx, fc), sel(tr_idx, ff))
        model = train_model(Xtr, ytr, gtr, wtr)
        Xva, _, _, _ = stack(index, sel(va_idx, rows), sel(va_idx, fc), sel(va_idx, ff), with_labels=False)
        pred = argmax_ids(index, sel(va_idx, fc), model.predict(Xva))
        oof[va_idx] = pred
        for i in va_idx:
            recall[i] = index.id_to_index[rows[i]["target_vessel_id"]] in set(fc[i])
        m = vlib.weighted_accuracy(train["vessel_name"].iloc[va_idx], train["target_vessel_id"].iloc[va_idx], pred)
        print(f"fold {fold}: {m} ({time.time() - t0:.0f}s)", flush=True)
    print(f"recall korean={recall[kor].mean():.4f} non_korean={recall[~kor].mean():.4f}")
    m = vlib.weighted_accuracy(train["vessel_name"], train["target_vessel_id"], oof)
    print("CV (unseen vessels):", m)
    WORK.mkdir(exist_ok=True)
    out = train.copy()
    out["pred"] = oof
    out.to_csv(WORK / "oof.csv", index=False)
    imp = pd.Series(model.feature_importance("gain"), index=vlib.FEATURES).sort_values(ascending=False)
    print(imp.head(30).round(0).to_string())


def run_final(args):
    train, catalog = load()
    tables = vlib.fit_tables(train, catalog)
    index = vlib.CatalogIndex(catalog, tables)
    rows = train.to_dict("records")
    cands, feats = vlib.build_matrix(index, rows)
    X, y, g, w = stack(index, rows, cands, feats)
    model = train_model(X, y, g, w)
    model.save_model(str(MODEL_DIR / "ranker.txt"))
    with open(MODEL_DIR / "tables.json", "w", encoding="utf-8") as fh:
        json.dump(tables, fh, ensure_ascii=False)
    print("saved model and tables")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    if a.cv:
        run_cv(a)
    if a.final:
        run_final(a)
