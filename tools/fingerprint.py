"""What a finished submit zip does on damaged train rows, in the terms the Public probes measure.

For each stress mode (tools/train.py names, joined with '+'), on a sample of Hangul or Latin train rows
with unseen call signs: accuracy, the share of misses whose target ranked second (the rank2 probe),
recall (target among the candidates) and the share of correct rows whose recorded tonnage equals the
target's (the tonexact probes). The rankers saw these rows in training, so a model trained on copies of
a damage type looks better here than it is; compare such models with a grouped CV (--cv --max-folds 1).

Usage: python tools/fingerprint.py ZIP --modes clean jgen3 lgen4+u15 [--n 300] [--script latin]
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unicodedata
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("zip")
ap.add_argument("--modes", nargs="+", required=True)
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--script", default="hangul", choices=["hangul", "latin"])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

box = Path(tempfile.mkdtemp(prefix="fingerprint_"))
with zipfile.ZipFile(a.zip) as zf:
    zf.extractall(box)
sys.path.insert(0, str(box))
sys.path.insert(0, str(box / "model"))
script = importlib.import_module("script")
vlib = sys.modules["vlib"]
sys.path.insert(0, str(ROOT / "tools"))
import train as T  # noqa: E402  (its vlib import resolves to the zip's copy already loaded)

train = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig")
catalog = pd.read_csv(ROOT / "data" / "vessel_catalog.csv", encoding="utf-8-sig")
is_h = train["vessel_name"].map(vlib.is_hangul)
pool = train[is_h if a.script == "hangul" else ~is_h]
rows = pool.sample(n=min(a.n, len(pool)), random_state=a.seed).reset_index(drop=True).assign(call_sign="CS_UNSEEN")
with open(box / "model" / "tables.json", encoding="utf-8") as fh:
    tables = json.load(fh)
vlib.apply_settings(box / "model" / "settings.json")
index = vlib.CatalogIndex(catalog, tables)
route = getattr(vlib, "hangul_text", vlib.is_hangul)

for mode in a.modes:
    st = rows if mode == "clean" else T.compose(rows, np.random.default_rng(11 + a.seed), mode)
    recs = st.to_dict("records")
    for r in recs:  # as script.py reads them
        if isinstance(r["vessel_name"], str):
            r["vessel_name"] = unicodedata.normalize("NFC", r["vessel_name"])
    cands, feats = vlib.build_matrix(index, recs)
    korean = np.repeat([route(r["vessel_name"]) for r in recs], [len(c) for c in cands])
    scores = script.rank_scores(np.vstack(feats), korean)
    off = ok = second = found = exact = 0
    for r, cand in zip(recs, cands):
        local = scores[off: off + len(cand)]
        off += len(cand)
        target = index.id_to_index[r["target_vessel_id"]]
        order = cand[np.argsort(-local, kind="stable")]
        found += target in cand
        if len(order) and order[0] == target:
            ok += 1
            g = index.gt[target]
            exact += g in (vlib.positive_float(r["gross_tonnage"]), vlib.positive_float(r["international_gross_tonnage"]))
        elif len(order) > 1 and order[1] == target:
            second += 1
    n, miss = len(recs), len(recs) - ok
    print(f"{Path(a.zip).stem} {a.script} {mode:18s} acc {ok / n:.3f}  rank2/miss {second / max(miss, 1):.2f}  "
          f"recall {found / n:.3f}  exact&ok/ok {exact / max(ok, 1):.2f}  (n={n})", flush=True)
