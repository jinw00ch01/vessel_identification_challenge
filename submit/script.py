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
    korean = np.repeat([vlib.is_hangul(rows[k].get("vessel_name")) for k in todo], [len(c) for c in cands])
    scores = rank_scores(np.vstack(feats), korean) if sum(len(c) for c in cands) else np.zeros(0)
    offset = 0
    for k, cand in zip(todo, cands):
        local = scores[offset: offset + len(cand)]
        offset += len(cand)
        preds[k] = index.ids[cand[int(np.argmax(local))]] if len(cand) else index.ids[0]
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
