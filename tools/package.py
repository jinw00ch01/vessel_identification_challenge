"""Build submit.zip and smoke-test it the way the DACON server runs it.

Usage:
    python tools/package.py --name v1          # writes dist/submit_v1.zip
    python tools/package.py --name v1 --smoke  # also runs it on a fake server dir
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "submit"
DIST = ROOT / "dist"
WORK = ROOT / "work"


def build_zip(name: str) -> Path:
    DIST.mkdir(exist_ok=True)
    path = DIST / f"submit_{name}.zip"
    files = [SUBMIT / "script.py", SUBMIT / "requirements.txt"]
    files += sorted(p for p in (SUBMIT / "model").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.relative_to(SUBMIT).as_posix())
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    with zipfile.ZipFile(path) as zf:
        print(" ", sorted(zf.namelist()))
    return path


def run_box(zip_path: Path, sample: pd.DataFrame, label: str) -> None:
    """Unzip into a fresh dir with data/ and output/, run script.py, score it."""
    box = WORK / "smoke" / label
    if box.exists():
        shutil.rmtree(box)
    box.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(box)
    (box / "data").mkdir()
    (box / "output").mkdir()
    sample.drop(columns=["target_vessel_id"]).to_csv(box / "data" / "test.csv", index=False, encoding="utf-8-sig")
    shutil.copy(ROOT / "data" / "vessel_catalog.csv", box / "data" / "vessel_catalog.csv")
    t0 = time.time()
    proc = subprocess.run([sys.executable, "script.py"], cwd=box, capture_output=True, text=True)
    print(proc.stdout[-2000:], proc.stderr[-4000:])
    print(f"[{label}] runtime {time.time() - t0:.1f}s for {len(sample)} rows")
    sub = pd.read_csv(box / "output" / "submission.csv")
    assert list(sub.columns) == ["sample_id", "target_vessel_id"], sub.columns
    assert len(sub) == len(sample) and sub["sample_id"].tolist() == sample["sample_id"].tolist()
    sys.path.insert(0, str(box / "model"))
    import vlib
    print(f"[{label}] in-sample score:",
          vlib.weighted_accuracy(sample["vessel_name"], sample["target_vessel_id"], sub["target_vessel_id"]))


def smoke(zip_path: Path, n_rows: int = 3000) -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig")
    sample = train.sample(n=min(n_rows, len(train)), random_state=7).reset_index(drop=True)
    sample = sample.assign(sample_id=[f"TEST_{i:05d}" for i in range(len(sample))])
    # train rows: every call sign is known, so the lookup answers them
    run_box(zip_path, sample, "lookup")
    # unseen call signs force every row through retrieval and the ranker
    run_box(zip_path, sample.assign(call_sign="CS_UNSEEN"), "ranker")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    z = build_zip(a.name)
    if a.smoke:
        smoke(z)
