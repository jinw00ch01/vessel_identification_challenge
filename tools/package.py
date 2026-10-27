"""Build submit.zip and smoke-test it the way the DACON server runs it.

Usage:
    python tools/package.py --name v1          # writes dist/submit_v1.zip
    python tools/package.py --name v1 --smoke  # also runs it on a fake server dir
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import unicodedata
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "submit"
DIST = ROOT / "dist"
WORK = ROOT / "work"


def build_zip(name: str, model_dir: Path = SUBMIT / "model") -> Path:
    DIST.mkdir(exist_ok=True)
    path = DIST / f"submit_{name}.zip"
    assert len(path.name) <= 30, f"{path.name}: upload file names are kept to 30 characters"
    # code from submit/, trained files (rankers, tables, settings) from model_dir
    files = {"script.py": SUBMIT / "script.py", "requirements.txt": SUBMIT / "requirements.txt",
             "model/vlib.py": SUBMIT / "model" / "vlib.py"}
    for p in sorted(model_dir.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".py":
            files[f"model/{p.relative_to(model_dir).as_posix()}"] = p
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, f in files.items():
            zf.write(f, arc)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    with zipfile.ZipFile(path) as zf:
        print(" ", sorted(zf.namelist()))
    return path


def run_box(zip_path: Path, sample: pd.DataFrame, label: str, score: bool = True) -> None:
    """Unzip into a fresh dir with data/ and output/, run script.py, score it."""
    box = WORK / "smoke" / label
    if box.exists():
        shutil.rmtree(box)
    box.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(box)
    (box / "data").mkdir()
    (box / "output").mkdir()
    sample.drop(columns=["target_vessel_id"], errors="ignore").to_csv(box / "data" / "test.csv", index=False, encoding="utf-8-sig")
    shutil.copy(ROOT / "data" / "vessel_catalog.csv", box / "data" / "vessel_catalog.csv")
    t0 = time.time()
    proc = subprocess.run([sys.executable, "script.py"], cwd=box, capture_output=True, text=True)
    print(proc.stdout[-2000:], proc.stderr[-4000:])
    print(f"[{label}] runtime {time.time() - t0:.1f}s for {len(sample)} rows")
    sub = pd.read_csv(box / "output" / "submission.csv")
    assert list(sub.columns) == ["sample_id", "target_vessel_id"], sub.columns
    assert len(sub) == len(sample) and sub["sample_id"].tolist() == sample["sample_id"].tolist()
    assert sub["target_vessel_id"].notna().all()
    if not score:
        return
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
    # malformed rows must not stop the run: blank or number-only names, blank or bad numbers
    odd = sample.head(12).assign(call_sign="CS_UNSEEN").copy()
    odd["vessel_name"] = [None, "", "123", "호", "제1호", "2호", "M/V", "오슬로(OSLO)", "HMM 오슬로", "ㄱ", "가스오너", "DILAM"]
    # text in a number column (pandas 3 refuses it in a float column)
    odd[["gross_tonnage", "international_gross_tonnage"]] = odd[["gross_tonnage", "international_gross_tonnage"]].astype(object)
    odd.loc[odd.index[8:], ["gross_tonnage", "international_gross_tonnage"]] = [[None, None], [0, -5], ["abc", None], [None, 1e9]]
    odd.loc[odd.index[::3], ["flag_name", "vessel_type", "arrival_year"]] = None
    run_box(zip_path, odd, "odd", score=False)
    # decomposed Hangul (NFD) must be matched like the composed names: the score should equal the ranker run's
    kor = sample[sample["vessel_name"].map(vlib_is_hangul)].head(200).assign(call_sign="CS_UNSEEN")
    kor["vessel_name"] = kor["vessel_name"].map(lambda v: unicodedata.normalize("NFD", v))
    run_box(zip_path, kor, "nfd")
    # jamo-level typos that no longer compose (그ᅳ랜드): the run must not stop, and with JAMO_KEYS the score stays high
    rng = random.Random(3)
    jam = sample[sample["vessel_name"].map(vlib_is_hangul)].head(200).assign(call_sign="CS_UNSEEN")
    jam["vessel_name"] = [jamo_typos(v, rng, 2) for v in jam["vessel_name"]]
    run_box(zip_path, jam, "jamo")
    # Latin names typed with the Korean layout on (GLOVIS SUN -> 히ㅐ퍈 녀ㅜ): the run must not stop, and with
    # IME_DECODE the score stays high
    sys.path.insert(0, str(SUBMIT / "model"))
    import vlib
    ime = sample[~sample["vessel_name"].map(vlib_is_hangul)].head(200).assign(call_sign="CS_UNSEEN")
    ime["vessel_name"] = ime["vessel_name"].map(vlib.ime_typed)
    run_box(zip_path, ime, "ime")
    # and Hangul names typed with the English layout on (글로비스 선 -> RMFFHQLTM TJS): with IME_ENCODE the score
    # stays high (these rows score with the Latin weight)
    eng = sample[sample["vessel_name"].map(vlib_is_hangul)].head(200).assign(call_sign="CS_UNSEEN")
    eng["vessel_name"] = eng["vessel_name"].map(vlib.ime_keys)
    run_box(zip_path, eng, "eng")
    # Latin names with a Hangul marker (GLOVIS SUN호, 엠브이 GLOVIS SUN, (GLOVIS SUN) 글로비스 선): they count as Hangul
    # rows; with MIXED_LATIN the Latin rankers read the Latin name and the score stays high
    mix = sample[~sample["vessel_name"].map(vlib_is_hangul)].head(300).assign(call_sign="CS_UNSEEN")
    mix["vessel_name"] = [(f"{v}호", f"엠브이 {v}", f"({v}) 선박")[i % 3] for i, v in enumerate(mix["vessel_name"])]
    run_box(zip_path, mix, "mixed")


def jamo_typos(name: str, rng: random.Random, k: int) -> str:
    s = unicodedata.normalize("NFD", name)
    for _ in range(k):
        i, op = rng.randrange(max(len(s) - 1, 1)), rng.randrange(3)
        if op == 0:
            s = s[:i] + s[i] + s[i:]
        elif op == 1 and i + 1 < len(s):
            s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
        elif len(s) > 3:
            s = s[:i] + s[i + 1:]
    return unicodedata.normalize("NFC", s)


def vlib_is_hangul(value) -> bool:
    return isinstance(value, str) and any("가" <= ch <= "힣" for ch in value)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model-dir", type=Path, default=SUBMIT / "model", help="trained files to package (default submit/model)")
    a = ap.parse_args()
    z = build_zip(a.name, a.model_dir)
    if a.smoke:
        smoke(z)
