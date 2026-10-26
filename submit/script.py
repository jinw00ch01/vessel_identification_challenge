"""DACON inference entry point: match each port-call record to a catalog vessel.

Reads data/test.csv and data/vessel_catalog.csv, writes output/submission.csv.
Each row is predicted from its own values, the official catalog and lookup
tables built from the official train set only.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import unicodedata

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "model"))
import vlib  # noqa: E402


def resolve(relative: str) -> Path:
    for root in (BASE, Path.cwd()):
        path = root / relative
        if path.exists():
            return path
    raise FileNotFoundError(relative)


def rank_scores(X: np.ndarray, korean: np.ndarray) -> np.ndarray:
    """Average of the rankers listed in views.json, each on its own feature view and on the rows
    it was trained for (all rows, or only Hangul or only Latin names)."""
    with open(BASE / "model" / "views.json", encoding="utf-8") as fh:
        views = json.load(fh)
    total, count = np.zeros(len(X)), np.zeros(len(X))
    for view in views:
        mask = {"all": np.ones_like(korean), "hangul": korean, "latin": ~korean}[view.get("rows", "all")]
        if not mask.any():
            continue
        booster = lgb.Booster(model_file=str(BASE / "model" / view["file"]))
        Xv = X[mask]
        Xv[:, [vlib.COL[f] for f in view["drop"]]] = np.nan
        total[mask] += booster.predict(Xv)
        count[mask] += 1
    return total / np.maximum(count, 1)


def predict(test: pd.DataFrame, catalog: pd.DataFrame) -> list[str]:
    with open(BASE / "model" / "tables.json", encoding="utf-8") as fh:
        tables = json.load(fh)
    vlib.apply_settings(BASE / "model" / "settings.json")
    test = test.copy()
    for col in ("vessel_name", "vessel_type", "flag_name"):
        if col in test:
            # decomposed Hangul (NFD, e.g. from a macOS export) would otherwise not count as Hangul
            test[col] = test[col].map(lambda v: unicodedata.normalize("NFC", v) if isinstance(v, str) else v)
    rows = test.to_dict("records")
    known = set(catalog["vessel_id"].astype(str))
    # A call sign seen in train decides the row on its own; rank the rest.
    preds = [vlib.lookup_prediction(tables, row) for row in rows]
    preds = [p if p in known else None for p in preds]
    todo = [k for k, p in enumerate(preds) if p is None]
    if not todo:
        return preds
    index = vlib.CatalogIndex(catalog, tables)
    cands, feats = vlib.build_matrix(index, [rows[k] for k in todo])
    korean = np.repeat([vlib.hangul_text(rows[k].get("vessel_name")) for k in todo], [len(c) for c in cands])
    scores = rank_scores(np.vstack(feats), korean) if sum(len(c) for c in cands) else np.zeros(0)
    offset = 0
    best = {}  # row -> position of its pick among its candidates and the pick's score
    for k, cand in zip(todo, cands):
        local = scores[offset: offset + len(cand)]
        offset += len(cand)
        if len(cand):
            best[k] = (int(np.argmax(local)), float(local.max()))
        preds[k] = index.ids[cand[best[k][0]]] if len(cand) else index.ids[0]
    if vlib.IME_DECODE:
        # a Latin name typed with the Korean layout left on (히ㅐ퍈 = GLOVIS): read it as the keys that typed it
        alt = [k for k in todo if isinstance(rows[k].get("vessel_name"), str) and vlib.IME_RE.search(rows[k]["vessel_name"])]
        arows = [dict(rows[k], vessel_name=vlib.ime_keys(vlib.upper(rows[k]["vessel_name"]))) for k in alt]
        a_cands, a_feats = vlib.build_matrix(index, arows)
        n = sum(len(c) for c in a_cands)
        a_scores = rank_scores(np.vstack(a_feats), np.zeros(n, dtype=bool)) if n else np.zeros(0)
        offset = 0
        for k, row, cand, F in zip(alt, arows, a_cands, a_feats):
            local = a_scores[offset: offset + len(cand)]
            offset += len(cand)
            if len(cand):
                j = int(np.argmax(local))
                loose = bool(vlib.LOOSE_RE.search(rows[k]["vessel_name"]))  # letters left out of any syllable
                if vlib.ime_accept(vlib.alnum(vlib.strip_prefix(vlib.upper(row["vessel_name"]))), F[j], loose):
                    preds[k] = index.ids[cand[j]]
    if vlib.IME_ENCODE:
        # a Hangul name typed with the English layout left on (RMFFHQLTM TJS = 글로비스 선): read the Hangul it types
        pos = {k: i for i, k in enumerate(todo)}
        alt, hrows = [], []
        for k in todo:
            name = rows[k].get("vessel_name")
            if isinstance(name, str) and not vlib.IME_RE.search(name):
                typed = vlib.ime_typed(vlib.upper(name))
                if vlib.HANGUL_RE.search(typed) and not vlib.LOOSE_RE.search(typed):
                    alt.append(k)
                    hrows.append(dict(rows[k], vessel_name=typed))
        h_cands, h_feats = vlib.build_matrix(index, hrows)
        n = sum(len(c) for c in h_cands)
        h_scores = rank_scores(np.vstack(h_feats), np.ones(n, dtype=bool)) if n else np.zeros(0)
        offset = 0
        for k, cand, F in zip(alt, h_cands, h_feats):
            local = h_scores[offset: offset + len(cand)]
            offset += len(cand)
            if len(cand):
                j = int(np.argmax(local))
                an = vlib.alnum(vlib.strip_prefix(vlib.upper(rows[k]["vessel_name"])))
                lj, ltop = best.get(k, (None, -np.inf))  # the Latin reading's pick, if it had any candidate
                off = feats[pos[k]][lj, vlib.COL["e_osa"]] / max(len(an), 1) if lj is not None else float("nan")
                if vlib.ime_encode_accept(off, float(local[j]) - ltop, F[j]):
                    preds[k] = index.ids[cand[j]]
    return preds


def main() -> None:
    start = time.time()
    test = pd.read_csv(resolve("data/test.csv"), encoding="utf-8-sig")
    catalog = pd.read_csv(resolve("data/vessel_catalog.csv"), encoding="utf-8-sig")
    preds = predict(test, catalog)
    result = pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": preds})
    # Write next to script.py and, if different, under the working directory.
    targets = list(dict.fromkeys([BASE / "output" / "submission.csv", Path.cwd() / "output" / "submission.csv"]))
    written, error = 0, None
    for path in targets:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(path, index=False, encoding="utf-8")
            written += 1
        except OSError as exc:
            error = exc
    if not written:
        raise error
    print(f"Saved {len(result)} predictions in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
