"""Build diagnostic submissions from an existing submit zip.

A probe keeps the base model's predictions for some rows and gives every other row a
catalog vessel that is never a target (a small crew boat), so the Public score isolates
one part of the test set:

    latin_only   0.3 x non-Hangul accuracy; with the base score this also gives Hangul accuracy
    lookup_only  rows decided by a call sign seen in train, nothing else
    hangul_sure  ranked Hangul rows whose best score beats the second by less than --gap get the
                 dummy: shows whether the misses are rows the ranker was unsure about
    latin_sure   the same for the other rows
    latin_nameonly  non-Hangul rows lose tonnage, type and flag before prediction (Hangul rows unchanged):
                 compared with CV under the same blanking, shows whether test names are like train names
    hangul_nameonly the same for Hangul rows
    latin_exact  Latin rows whose name is exactly a catalog name and whose prediction carries that name
    hangul_tonexact / latin_tonexact  rows whose recorded tonnage equals the predicted vessel's tonnage
    tonexact_exact  hangul_tonexact for the Hangul rows and latin_exact for the rest, in one submission
    rank2        every ranked row gets the ranker's second choice (rows with one candidate get the dummy):
                 counts the misses whose target was runner-up, for Hangul and Latin in one submission
    notonnage_fix  each row is also predicted with both tonnage fields blank; that prediction is kept where it
                 differs from the full one, the dummy elsewhere: counts the misses that blanking the recorded
                 tonnage would fix (high = the record's tonnage misleads the ranker), Hangul and Latin at once
    hangul_tonexact_only  hangul_tonexact with every Latin row given the dummy: one count, no alias

and one variant that is a model change rather than a probe:

    nolookup     call signs ignored, every row ranked (tests whether train call signs mislead)

Usage:
    python tools/probe.py --base /mnt/project-files/submissions/submit_v6.zip --mode latin_only --name p1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DUMMY = "VESSEL_65034"  # crew boat Ocean Ranger, 263 GT: never a train target

ANCHORS = {
    "lookup_only": (
        "    preds = [p if p in known else None for p in preds]\n",
        f"    preds = [p if p in known else None for p in preds]\n"
        f"    return [p if p is not None else {DUMMY!r} for p in preds]  # probe: lookup only\n",
    ),
    "latin_only": (
        "    preds = predict(test, catalog)\n",
        "    preds = predict(test, catalog)\n"
        f"    preds = [{DUMMY!r} if vlib.is_hangul(n) else p for n, p in zip(test[\"vessel_name\"], preds)]  # probe\n",
    ),
    "hangul_sure": (
        "        preds[k] = index.ids[cand[int(np.argmax(local))]] if len(cand) else index.ids[0]\n",
        "        preds[k] = index.ids[cand[int(np.argmax(local))]] if len(cand) else index.ids[0]\n"
        "        if vlib.is_hangul(rows[k].get(\"vessel_name\")) and len(cand) > 1 and np.ptp(np.partition(local, -2)[-2:]) < {gap}:\n"
        f"            preds[k] = {DUMMY!r}  # probe: unsure Hangul rows\n",
    ),
    "latin_sure": (
        "        preds[k] = index.ids[cand[int(np.argmax(local))]] if len(cand) else index.ids[0]\n",
        "        preds[k] = index.ids[cand[int(np.argmax(local))]] if len(cand) else index.ids[0]\n"
        "        if not vlib.is_hangul(rows[k].get(\"vessel_name\")) and len(cand) > 1 and np.ptp(np.partition(local, -2)[-2:]) < {gap}:\n"
        f"            preds[k] = {DUMMY!r}  # probe: unsure Latin rows\n",
    ),
    "latin_nameonly": (
        "    catalog = pd.read_csv(resolve(\"data/vessel_catalog.csv\"), encoding=\"utf-8-sig\")\n",
        "    catalog = pd.read_csv(resolve(\"data/vessel_catalog.csv\"), encoding=\"utf-8-sig\")\n"
        "    latin = ~test[\"vessel_name\"].map(vlib.is_hangul)  # probe: Latin rows keep only their name\n"
        "    test.loc[latin, [\"gross_tonnage\", \"international_gross_tonnage\", \"vessel_type\", \"flag_name\"]] = np.nan\n",
    ),
    "hangul_nameonly": (
        "    catalog = pd.read_csv(resolve(\"data/vessel_catalog.csv\"), encoding=\"utf-8-sig\")\n",
        "    catalog = pd.read_csv(resolve(\"data/vessel_catalog.csv\"), encoding=\"utf-8-sig\")\n"
        "    hangul = test[\"vessel_name\"].map(vlib.is_hangul)  # probe: Hangul rows keep only their name\n"
        "    test.loc[hangul, [\"gross_tonnage\", \"international_gross_tonnage\", \"vessel_type\", \"flag_name\"]] = np.nan\n",
    ),
    "latin_exact": (
        "    preds = predict(test, catalog)\n",
        "    preds = predict(test, catalog)\n"
        "    # probe: only Latin rows whose name is exactly a catalog name and whose prediction carries that name\n"
        "    cat_an = dict(zip(catalog[\"vessel_id\"].astype(str), [vlib.alnum(vlib.strip_prefix(vlib.upper(v))) for v in catalog[\"vessel_name\"]]))\n"
        "    known_an = set(cat_an.values())\n"
        "    q_an = [vlib.alnum(vlib.strip_prefix(vlib.upper(v))) for v in test[\"vessel_name\"]]\n"
        f"    preds = [p if (not vlib.is_hangul(n) and q and q in known_an and cat_an.get(p) == q) else {DUMMY!r}\n"
        "             for n, q, p in zip(test[\"vessel_name\"], q_an, preds)]\n",
    ),
    "hangul_tonexact": (
        "    preds = predict(test, catalog)\n",
        "    preds = predict(test, catalog)\n"
        "    # probe: only Hangul rows whose gross or international tonnage equals the predicted vessel's tonnage\n"
        "    cat_gt = dict(zip(catalog[\"vessel_id\"].astype(str), [vlib.positive_float(v) for v in catalog[\"gross_tonnage\"]]))\n"
        "    gts = [(vlib.positive_float(a), vlib.positive_float(b)) for a, b in zip(test[\"gross_tonnage\"], test[\"international_gross_tonnage\"])]\n"
        f"    preds = [p if (vlib.is_hangul(n) and np.isfinite(cat_gt.get(p, np.nan)) and cat_gt[p] in g) else {DUMMY!r}\n"
        "             for n, g, p in zip(test[\"vessel_name\"], gts, preds)]\n",
    ),
    "latin_tonexact": (
        "    preds = predict(test, catalog)\n",
        "    preds = predict(test, catalog)\n"
        "    # probe: only Latin rows whose gross or international tonnage equals the predicted vessel's tonnage\n"
        "    cat_gt = dict(zip(catalog[\"vessel_id\"].astype(str), [vlib.positive_float(v) for v in catalog[\"gross_tonnage\"]]))\n"
        "    gts = [(vlib.positive_float(a), vlib.positive_float(b)) for a, b in zip(test[\"gross_tonnage\"], test[\"international_gross_tonnage\"])]\n"
        f"    preds = [p if (not vlib.is_hangul(n) and np.isfinite(cat_gt.get(p, np.nan)) and cat_gt[p] in g) else {DUMMY!r}\n"
        "             for n, g, p in zip(test[\"vessel_name\"], gts, preds)]\n",
    ),
    "tonexact_exact": (
        "    preds = predict(test, catalog)\n",
        "    preds = predict(test, catalog)\n"
        "    # probe: Hangul rows whose tonnage equals the predicted vessel's, and Latin rows whose name is exactly a\n"
        "    # catalog name carried by the prediction (two counts from one submission)\n"
        "    cat_gt = dict(zip(catalog[\"vessel_id\"].astype(str), [vlib.positive_float(v) for v in catalog[\"gross_tonnage\"]]))\n"
        "    gts = [(vlib.positive_float(a), vlib.positive_float(b)) for a, b in zip(test[\"gross_tonnage\"], test[\"international_gross_tonnage\"])]\n"
        "    cat_an = dict(zip(catalog[\"vessel_id\"].astype(str), [vlib.alnum(vlib.strip_prefix(vlib.upper(v))) for v in catalog[\"vessel_name\"]]))\n"
        "    known_an = set(cat_an.values())\n"
        "    q_an = [vlib.alnum(vlib.strip_prefix(vlib.upper(v))) for v in test[\"vessel_name\"]]\n"
        "    keep = [(np.isfinite(cat_gt.get(p, np.nan)) and cat_gt[p] in g) if vlib.is_hangul(n) else (bool(q) and q in known_an and cat_an.get(p) == q)\n"
        "            for n, g, q, p in zip(test[\"vessel_name\"], gts, q_an, preds)]\n"
        f"    preds = [p if k else {DUMMY!r} for k, p in zip(keep, preds)]\n",
    ),
    "rank2": (
        "        preds[k] = index.ids[cand[int(np.argmax(local))]] if len(cand) else index.ids[0]\n",
        "        order = np.argsort(-local, kind=\"stable\")  # probe: runner-up (order[0] is the argmax)\n"
        f"        preds[k] = index.ids[cand[int(order[1])]] if len(cand) > 1 else {DUMMY!r}\n",
    ),
    "notonnage_fix": (
        "    preds = predict(test, catalog)\n",
        "    preds = predict(test, catalog)\n"
        "    # probe: the prediction without the recorded tonnage, where it differs from the full one\n"
        "    blank = test.copy()\n"
        "    blank[[\"gross_tonnage\", \"international_gross_tonnage\"]] = np.nan\n"
        "    alt = predict(blank, catalog)\n"
        f"    preds = [q if q != p else {DUMMY!r} for p, q in zip(preds, alt)]\n",
    ),
    "hangul_tonexact_only": (
        "    preds = predict(test, catalog)\n",
        "    preds = predict(test, catalog)\n"
        "    # probe: only Hangul rows whose gross or international tonnage equals the predicted vessel's tonnage\n"
        "    cat_gt = dict(zip(catalog[\"vessel_id\"].astype(str), [vlib.positive_float(v) for v in catalog[\"gross_tonnage\"]]))\n"
        "    gts = [(vlib.positive_float(a), vlib.positive_float(b)) for a, b in zip(test[\"gross_tonnage\"], test[\"international_gross_tonnage\"])]\n"
        f"    preds = [p if (vlib.is_hangul(n) and np.isfinite(cat_gt.get(p, np.nan)) and cat_gt[p] in g) else {DUMMY!r}\n"
        "             for n, g, p in zip(test[\"vessel_name\"], gts, preds)]\n",
    ),
    "nolookup": (
        "    preds = [vlib.lookup_prediction(tables, row) for row in rows]\n",
        "    preds = [None for row in rows]  # variant: call signs ignored, every row ranked\n",
    ),
}


def build(base: Path, mode: str, name: str, gap: float = 2.0) -> Path:
    old, new = ANCHORS[mode]
    new = new.replace("{gap}", repr(gap))
    out = ROOT / "dist" / f"submit_{name}.zip"
    assert len(out.name) <= 30, f"{out.name}: upload file names are kept to 30 characters"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(base) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "script.py":
                text = data.decode("utf-8")
                assert text.count(old) == 1, f"anchor for {mode} not found exactly once"
                data = text.replace(old, new).encode("utf-8")
            dst.writestr(info.filename, data)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--mode", choices=sorted(ANCHORS), required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--gap", type=float, default=2.0, help="hangul_sure: score gap below which a row counts as unsure")
    a = ap.parse_args()
    build(a.base, a.mode, a.name, a.gap)
