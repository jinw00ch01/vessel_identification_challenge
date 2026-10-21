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


def load():
    train = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(DATA / "vessel_catalog.csv", encoding="utf-8-sig")
    return train, catalog


def featurize(index: vlib.CatalogIndex, df: pd.DataFrame):
    rows = df.to_dict("records")
    cands, feats = vlib.build_matrix(index, rows)
    return rows, cands, feats


def refresh_tables(index, rows, cands, feats, tables):
    """Recompute the train-derived columns for a given set of tables."""
    index.tables = tables
    ti = vlib.FEATURES.index("type_lp")
    fi = vlib.FEATURES.index("flag_lp")
    out = []
    for row, cand, f in zip(rows, cands, feats):
        f = f.copy()
        qtype = row.get("vessel_type") if isinstance(row.get("vessel_type"), str) else ""
        qflag = row.get("flag_name") if isinstance(row.get("flag_name"), str) else ""
        f[:, ti] = index._cond_lp("type", qtype, index.type[cand])
        f[:, fi] = index._cond_lp("flag", qflag, index.flag[cand])
        out.append(f)
    return out


def stack(index, rows, cands, feats, with_labels=True):
    X = np.vstack(feats)
    groups = np.array([len(c) for c in cands])
    y = None
    if with_labels:
        y = np.concatenate([
            (cand == index.id_to_index[row["target_vessel_id"]]).astype(np.int8)
            for row, cand in zip(rows, cands)
        ])
    return X, y, groups


def argmax_ids(index, cands, scores):
    out, offset = [], 0
    for cand in cands:
        local = scores[offset: offset + len(cand)]
        out.append(index.ids[cand[int(np.argmax(local))]] if len(cand) else index.ids[0])
        offset += len(cand)
    return out


def train_model(X, y, groups, rounds=ROUNDS, params=PARAMS):
    ds = lgb.Dataset(X, y, group=groups if params["objective"] != "binary" else None,
                     feature_name=vlib.FEATURES, free_raw_data=False)
    return lgb.train(params, ds, num_boost_round=rounds)


def run_cv(args):
    train, catalog = load()
    t0 = time.time()
    index = vlib.CatalogIndex(catalog, vlib.fit_tables(train, catalog))
    print(f"index built in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    rows, cands, feats = featurize(index, train)
    print(f"features in {time.time() - t0:.1f}s, rows={sum(len(c) for c in cands)}", flush=True)

    target_idx = np.array([index.id_to_index[t] for t in train["target_vessel_id"]])
    recall = np.array([t in set(c) for t, c in zip(target_idx, cands)])
    kor = train["vessel_name"].map(vlib.is_hangul).to_numpy()
    print(f"recall korean={recall[kor].mean():.4f} non_korean={recall[~kor].mean():.4f}")

    gkf = GroupKFold(n_splits=5)
    oof = np.empty(len(train), dtype=object)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, groups=train["target_vessel_id"])):
        tables = vlib.fit_tables(train.iloc[tr_idx], catalog)
        f2 = refresh_tables(index, rows, cands, feats, tables)
        sel = lambda idx, seq: [seq[i] for i in idx]  # noqa: E731
        Xtr, ytr, gtr = stack(index, sel(tr_idx, rows), sel(tr_idx, cands), sel(tr_idx, f2))
        model = train_model(Xtr, ytr, gtr)
        Xva, _, _ = stack(index, sel(va_idx, rows), sel(va_idx, cands), sel(va_idx, f2), with_labels=False)
        pred = argmax_ids(index, sel(va_idx, cands), model.predict(Xva))
        oof[va_idx] = pred
        m = vlib.weighted_accuracy(train["vessel_name"].iloc[va_idx], train["target_vessel_id"].iloc[va_idx], pred)
        print(f"fold {fold}: {m}", flush=True)
    m = vlib.weighted_accuracy(train["vessel_name"], train["target_vessel_id"], oof)
    print("CV (unseen vessels):", m)
    WORK.mkdir(exist_ok=True)
    out = train.copy()
    out["pred"] = oof
    out.to_csv(WORK / "oof.csv", index=False)
    imp = pd.Series(model.feature_importance("gain"), index=vlib.FEATURES).sort_values(ascending=False)
    print(imp.head(25).round(0).to_string())


def run_final(args):
    train, catalog = load()
    tables = vlib.fit_tables(train, catalog)
    index = vlib.CatalogIndex(catalog, tables)
    rows, cands, feats = featurize(index, train)
    X, y, g = stack(index, rows, cands, feats)
    model = train_model(X, y, g)
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
