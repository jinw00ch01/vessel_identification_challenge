"""Few-shot examples for the language-model reader (submit/model/lm_shots.json): ten train records' Hangul names
with their vessels' catalog names in capitals. The records are named by sample id, so no train data sits in the code.

Usage: python tools/build_lm_shots.py [--data data] [--out submit/model/lm_shots.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# picked by hand for variety: plain words, an acronym spelled out, 호, a number, a hyphen, a Thai name
SAMPLES = ["TRAIN_00236", "TRAIN_00013", "TRAIN_02109", "TRAIN_00057", "TRAIN_01802", "TRAIN_00613", "TRAIN_00333",
           "TRAIN_02746", "TRAIN_00263", "TRAIN_02060"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "submit" / "model" / "lm_shots.json"))
    a = ap.parse_args()
    train = pd.read_csv(Path(a.data) / "train.csv", encoding="utf-8-sig").set_index("sample_id")
    catalog = pd.read_csv(Path(a.data) / "vessel_catalog.csv", encoding="utf-8-sig").set_index("vessel_id")
    shots = [{"kor": train.vessel_name[s], "eng": catalog.vessel_name[train.target_vessel_id[s]].upper(),
              "vessel_id": train.target_vessel_id[s]} for s in SAMPLES]
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(shots, fh, ensure_ascii=False, indent=1)
    print(f"wrote {len(shots)} shots to {a.out}")


if __name__ == "__main__":
    main()
