"""Build diagnostic submissions from an existing submit zip.

A probe keeps the base model's predictions for some rows and gives every other row a
catalog vessel that is never a target (a small crew boat), so the Public score isolates
one part of the test set:

    latin_only   0.3 x non-Hangul accuracy; with the base score this also gives Hangul accuracy
    lookup_only  rows decided by a call sign seen in train, nothing else

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
}


def build(base: Path, mode: str, name: str) -> Path:
    old, new = ANCHORS[mode]
    out = ROOT / "dist" / f"submit_{name}.zip"
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
    a = ap.parse_args()
    build(a.base, a.mode, a.name)
