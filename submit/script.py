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


def predict(test: pd.DataFrame, catalog: pd.DataFrame) -> list[str]:
    with open(BASE / "model" / "tables.json", encoding="utf-8") as fh:
        tables = json.load(fh)
    booster = lgb.Booster(model_file=str(BASE / "model" / "ranker.txt"))
    index = vlib.CatalogIndex(catalog, tables)
    rows = test.to_dict("records")
    cands, feats = vlib.build_matrix(index, rows)
    sizes = [len(c) for c in cands]
    scores = booster.predict(np.vstack(feats)) if sum(sizes) else np.zeros(0)
    preds, offset = [], 0
    for row, cand in zip(rows, cands):
        local = scores[offset: offset + len(cand)]
        offset += len(cand)
        hit = vlib.lookup_prediction(tables, row)
        if hit is not None and hit in index.id_to_index:
            preds.append(hit)
        elif len(cand):
            preds.append(index.ids[cand[int(np.argmax(local))]])
        else:
            preds.append(index.ids[0])
    return preds


def main() -> None:
    start = time.time()
    test = pd.read_csv(resolve("data/test.csv"), encoding="utf-8-sig")
    catalog = pd.read_csv(resolve("data/vessel_catalog.csv"), encoding="utf-8-sig")
    preds = predict(test, catalog)
    result = pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": preds})
    targets = {BASE / "output" / "submission.csv", Path.cwd() / "output" / "submission.csv"}
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False, encoding="utf-8")
    print(f"Saved {len(result)} predictions in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
