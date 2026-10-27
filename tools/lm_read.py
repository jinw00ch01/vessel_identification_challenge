"""Read the names in one or more names files (uid, vessel_name) with the submission's language-model reader
(submit/model/lmread.py), one name at a time as at inference.

Usage (on a machine with a CUDA GPU; keep HF_HOME inside the project folder):
    python tools/lm_read.py --model Qwen/Qwen2.5-1.5B-Instruct --names lm/lm_names.csv --out lm/out_qwen15m.csv
--model is a local model directory or a Hugging Face id; --prompt picks one of lmread.PROMPTS. Writes uid, read1,
read2, read3, seconds (time spent on the name; 0 when an earlier row had the same name).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submit" / "model"))
import lmread  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shots", default=str(ROOT / "submit" / "model" / "lm_shots.json"))
    ap.add_argument("--prompt", default="meaning", choices=sorted(lmread.PROMPTS))
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    reader = lmread.Reader(a.model, lmread.load_shots(Path(a.shots)), prompt=a.prompt, beams=a.beams)
    if not reader.ok:
        raise SystemExit("the reader needs torch, transformers and a CUDA GPU")
    rows = []
    for path in a.names:
        with open(path, encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    rows = rows[: a.limit or None]
    start = time.time()
    with open(a.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "read1", "read2", "read3", "seconds"])
        for i, r in enumerate(rows):
            t0, new = time.time(), r["vessel_name"] not in reader.cache
            reads = (reader.read(r["vessel_name"]) + ["", "", ""])[:3]
            w.writerow([r["uid"], *reads, f"{time.time() - t0:.2f}" if new else "0"])
            if i % 100 == 0:
                print(f"{i}/{len(rows)} {r['vessel_name']} -> {reads[0]}  ({time.time() - start:.0f}s)", flush=True)
    print(f"wrote {a.out}: {len(rows)} rows, {len(reader.cache)} names in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
