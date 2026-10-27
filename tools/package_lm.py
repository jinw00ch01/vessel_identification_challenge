"""Build a submission zip that adds the language-model reader (LM_READ) to an existing submission zip.

Run on a machine that has the model weights (Hugging Face cache) and a CUDA GPU, from a checkout of this repo:
    python tools/package_lm.py --base submit_v15m.zip --model Qwen/Qwen2.5-1.5B-Instruct --name v16q \
        --data open_data --work lm/build --out-dir .
Takes the base zip's rankers and tables, this checkout's script.py, vlib.py and lmread.py, few-shot train records
(tools/build_lm_shots.py) and the model files (model/lm/), sets LM_READ in settings.json, then smoke-tests the
package on train rows (one full run, then a reversed half run that must give the same predictions row by row) and
writes <out-dir>/submit_<name>.zip (weights stored, the rest deflated).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_lm_shots  # noqa: E402


def model_dir(model: str) -> Path:
    path = Path(model)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, local_files_only=True))


def smoke(pkg: Path, data: Path, work: Path, rows: int) -> None:
    train = pd.read_csv(data / "train.csv", encoding="utf-8-sig")
    hangul = train.vessel_name.str.contains("[가-힣]", na=False)
    test = pd.concat([train[hangul].sample(rows, random_state=0), train[~hangul].sample(rows, random_state=0)])
    truth = test.set_index("sample_id").target_vessel_id
    test = test.drop(columns=["target_vessel_id"]).assign(call_sign="CS_SMOKE")  # no call-sign lookup
    preds = {}
    for label, part in (("full", test), ("half reversed", test.iloc[::-2])):
        run = work / f"smoke_{label.split()[0]}"
        shutil.rmtree(run, ignore_errors=True)
        (run / "data").mkdir(parents=True)
        part.to_csv(run / "data" / "test.csv", index=False, encoding="utf-8")
        shutil.copy(data / "vessel_catalog.csv", run / "data" / "vessel_catalog.csv")
        t0 = time.time()
        subprocess.run([sys.executable, str(pkg / "script.py")], cwd=run, check=True)
        out = pd.read_csv(run / "output" / "submission.csv").set_index("sample_id").target_vessel_id
        assert len(out) == len(part) and out.notna().all(), "submission rows missing"
        preds[label] = out
        is_h = part.set_index("sample_id").vessel_name.str.contains("[가-힣]", na=False)
        ok = out == truth[out.index]
        print(f"smoke {label}: {len(part)} rows in {time.time() - t0:.0f}s, right: Hangul {ok[is_h].mean():.3f}, "
              f"Latin {ok[~is_h].mean():.3f}")
    common = preds["half reversed"].index
    same = (preds["full"][common] == preds["half reversed"]).mean()
    print(f"same prediction for a row alone in a smaller, reordered file: {same:.4f}")
    assert same == 1.0, "predictions depend on the other rows"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="existing submission zip")
    ap.add_argument("--model", required=True, help="model directory or Hugging Face id (cached)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True, help="folder with train.csv and vessel_catalog.csv")
    ap.add_argument("--work", default="lm/build")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--prompt", default="meaning")
    ap.add_argument("--smoke-rows", type=int, default=150)
    a = ap.parse_args()
    zip_name = f"submit_{a.name}.zip"
    assert len(zip_name) <= 30, f"{zip_name} is longer than 30 characters"
    work, data = Path(a.work), Path(a.data)
    pkg = work / a.name
    shutil.rmtree(pkg, ignore_errors=True)
    pkg.mkdir(parents=True)
    with zipfile.ZipFile(a.base) as zf:
        zf.extractall(pkg)
    if not (pkg / "script.py").exists():  # zip with one top folder
        top = next(p for p in pkg.iterdir() if (p / "script.py").exists())
        for p in top.iterdir():
            shutil.move(str(p), pkg / p.name)
        top.rmdir()
    shutil.copy(ROOT / "submit" / "script.py", pkg / "script.py")
    for f in ("vlib.py", "lmread.py"):
        shutil.copy(ROOT / "submit" / "model" / f, pkg / "model" / f)
    sys.argv = ["build_lm_shots", "--data", str(data), "--out", str(pkg / "model" / "lm_shots.json")]
    build_lm_shots.main()
    src = model_dir(a.model)
    (pkg / "model" / "lm").mkdir()
    for f in src.iterdir():
        if f.is_file() and not f.name.startswith("."):
            shutil.copy(f, pkg / "model" / "lm" / f.name)  # follows cache symlinks
    with open(pkg / "model" / "settings.json", encoding="utf-8") as fh:
        settings = json.load(fh)
    settings.update(LM_READ=True, LM_MARGIN=a.margin, LM_PROMPT=a.prompt)
    with open(pkg / "model" / "settings.json", "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=1)
    with open(pkg / "requirements.txt", "w", encoding="utf-8") as fh:
        fh.write("# Uses the evaluation image preinstalled packages: numpy, pandas, lightgbm, rapidfuzz, torch, transformers.\n")
    print("package:", pkg, "model files:", sorted(p.name for p in (pkg / "model" / "lm").iterdir()))
    smoke(pkg, data, work, a.smoke_rows)
    out = Path(a.out_dir) / zip_name
    with zipfile.ZipFile(out, "w") as zf:
        for p in sorted(pkg.rglob("*")):
            rel = p.relative_to(pkg)
            if p.is_file() and "__pycache__" not in rel.parts and rel.parts[0] != "output":  # smoke runs write output/
                stored = p.suffix in (".safetensors", ".bin")
                zf.write(p, rel.as_posix(), compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
    md5 = hashlib.md5()
    with open(out, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            md5.update(chunk)
    print(f"wrote {out}: {out.stat().st_size:,} bytes, md5 {md5.hexdigest()}")


if __name__ == "__main__":
    main()
