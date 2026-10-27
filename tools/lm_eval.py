"""Score language-model readings of Hangul ship names (tools/lm_read.py) against the Hangul pipeline.

For each row: the Hangul pipeline's pick, and the Latin rankers' pick for each LM reading (the best-scoring one as
in script.py's LM_READ), under exact, +-15% and blank tonnage. Prints accuracy per set and the LM_MARGIN rule.
Usage: python tools/lm_eval.py --zip ZIPDIR --data DATADIR --names lm/lm_names.csv [--extra more.csv ...]
           --tag qwen15m=lm/out_qwen15m.csv [--tag other=a.csv,b.csv ...] [--out-dir lm]
  ZIPDIR is an unzipped submission (script.py, model/) whose rankers are used.
Sets behind --names (rebuilt here from train.csv and the catalog, rows N00000-N01847): clean / hspell2 / hsyl2 /
dropfirst = the first Hangul record of each of the 382 train vessels with a Hangul name, clean or damaged (the
rankers were trained on these vessels, so the Hangul pipeline looks better there than it is); llm / llm_hspell2 =
hand-written Hangul for 160 catalog vessels never in train. --extra files (uid, vessel_name, target_vessel_id[, set])
add rows for other catalog vessels, with type, flag and tonnage from the catalog.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

LLM_IDS = """14454 56279 17400 56319 52613 7651 40641 51086 44584 63759 64155 69700 43712 24634 65228 51618 20197 67027
61619 56271 45634 25946 17658 54490 14901 66248 45875 71193 5014 33400 9448 43789 55226 70253 42782 63518 13210 59898
68858 7279 64631 36685 64363 47513 58263 3703 28600 74492 63379 32282 56247 36370 35811 55476 18949 72197 41034 16602
45706 1184 8124 60463 78450 31566 37478 69538 4843 2630 47145 6750 73854 60642 397 71493 65236 6114 29158 9433 22285
15701 4392 29053 56596 15308 52751 69417 50568 27345 51681 44154 59910 25264 23240 55810 30846 33605 71381 51422 42030
43444 35824 73939 35884 16048 57480 75609 55094 16806 49421 41149 52093 57979 31633 3461 73568 20259 66791 20052 13386
29776 7566 28981 76537 33083 6522 8624 51833 16072 21015 27854 44890 7772 6318 74935 3520 14647 53920 44397 34206
36356 60223 36018 9420 71171 12143 47972 77110 69624 62401 28879 7535 11981 60949 27801 27680 50685 42909 32214 11536
43484""".split()
ROOT = Path(__file__).resolve().parents[1]
MARGINS = (99, 8, 6, 4, 3, 2, 1, 0, -1)  # 99 = never take the LM pick


class Tee:
    """Print to the console and to a UTF-8 log file."""

    def __init__(self, path):
        self.file, self.out = open(path, "w", encoding="utf-8"), sys.stdout

    def write(self, text):
        self.file.write(text)
        self.file.flush()
        try:
            self.out.write(text)
        except UnicodeEncodeError:
            self.out.write(text.encode("ascii", "replace").decode())

    def flush(self):
        self.out.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--shots", default=str(ROOT / "submit" / "model" / "lm_shots.json"))
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--per-set", type=int, default=0, help="keep only the first N rows of each set (quick test)")
    a = ap.parse_args()
    out_dir, zipdir, datadir = Path(a.out_dir), Path(a.zip), Path(a.data)
    tags = dict(t.split("=", 1) for t in a.tag)
    sys.stdout = Tee(out_dir / f"lm_eval_{'_'.join(tags)}.txt")
    sys.path[:0] = [str(zipdir), str(zipdir / "model")]
    import vlib
    import script

    script.BASE = zipdir
    vlib.apply_settings(zipdir / "model" / "settings.json")
    vlib.HANJA_READ = False
    assert len(LLM_IDS) == 160 and sum(map(int, LLM_IDS)) == 6353606, "vessel id list damaged"
    catalog = pd.read_csv(datadir / "vessel_catalog.csv", encoding="utf-8-sig")
    train = pd.read_csv(datadir / "train.csv", encoding="utf-8-sig")
    with open(zipdir / "model" / "tables.json", encoding="utf-8") as fh:
        index = vlib.CatalogIndex(catalog, json.load(fh))
    cat = catalog.set_index("vessel_id")
    with open(a.shots, encoding="utf-8") as fh:
        shot_vids = {s["vessel_id"] for s in json.load(fh)}

    # record-style labels for catalog-made rows: the Korean label train records use most for the catalog's English one
    m = train.merge(catalog[["vessel_id", "vessel_type", "flag_name"]].rename(columns={"vessel_type": "ct", "flag_name": "cf"}),
                    left_on="target_vessel_id", right_on="vessel_id")
    tlab = m.groupby("ct").vessel_type.agg(lambda x: x.value_counts().index[0]).to_dict()
    flab = m.dropna(subset=["flag_name"]).groupby("cf").flag_name.agg(lambda x: x.value_counts().index[0]).to_dict()

    def from_catalog(vids):
        x = cat.loc[list(vids), ["flag_name", "vessel_type", "gross_tonnage"]].rename_axis("target_vessel_id").reset_index()
        x["flag_name"], x["vessel_type"] = x.flag_name.map(flab), x.vessel_type.map(tlab)
        x["international_gross_tonnage"] = x.gross_tonnage
        return x

    cols = ["target_vessel_id", "gross_tonnage", "international_gross_tonnage", "flag_name", "vessel_type"]
    rep = train[train.vessel_name.map(vlib.is_hangul)].groupby("target_vessel_id").head(1).reset_index(drop=True)
    llm = from_catalog(f"VESSEL_{int(i):05d}" for i in LLM_IDS)
    ev = pd.concat([rep[cols].assign(set=s) for s in ("clean", "hspell2", "hsyl2", "dropfirst")]
                   + [llm[cols].assign(set=s) for s in ("llm", "llm_hspell2")], ignore_index=True)
    ev["uid"] = [f"N{i:05d}" for i in range(len(ev))]
    names = pd.read_csv(a.names).set_index("uid").vessel_name
    assert len(ev) == len(names) == 1848, (len(ev), len(names))
    ev["vessel_name"] = ev.uid.map(names).to_numpy()
    assert (ev.vessel_name[ev.set == "clean"].to_numpy() == rep.vessel_name.to_numpy()).all(), \
        "train.csv is not the one lm_names.csv was made from"
    for path in a.extra:
        x = pd.read_csv(path)
        y = from_catalog(x.target_vessel_id)
        y["uid"], y["vessel_name"], y["set"] = x.uid.to_numpy(), x.vessel_name.to_numpy(), x.get("set", pd.Series(["extra"] * len(x))).to_numpy()
        ev = pd.concat([ev, y[cols + ["set", "uid", "vessel_name"]]], ignore_index=True)
    ev = ev[~ev.target_vessel_id.isin(shot_vids)].reset_index(drop=True)
    if a.per_set:
        ev = ev.groupby("set", sort=False).head(a.per_set).reset_index(drop=True)
    sets = list(dict.fromkeys(ev.set))
    fair = ev.set.isin([s for s in sets if s not in ("clean", "hspell2", "hsyl2", "dropfirst")]).to_numpy()
    print("rows", len(ev), ev.set.value_counts(sort=False).to_dict())

    def picks(rows, korean):
        cands, feats = vlib.build_matrix(index, rows)
        n = sum(len(c) for c in cands)
        sc = script.rank_scores(np.vstack(feats), np.repeat(korean, [len(c) for c in cands])) if n else np.zeros(0)
        out, off = [], 0
        for c in cands:
            loc = sc[off: off + len(c)]
            off += len(c)
            out.append((index.ids[c[int(np.argmax(loc))]], float(loc.max())) if len(c) else (None, -np.inf))
        return out

    rng = np.random.default_rng(5)
    base_rows = {}
    for ton in ("exact", "u15", "blank"):
        gt = ev.gross_tonnage.to_numpy(float).copy()
        igt = ev.international_gross_tonnage.to_numpy(float).copy()
        if ton == "u15":
            f = 1 + rng.uniform(-0.15, 0.15, len(ev))
            gt, igt = gt * f, igt * f
        elif ton == "blank":
            gt[:], igt[:] = np.nan, np.nan
        base_rows[ton] = [dict(vessel_name=n, call_sign="CS_UNSEEN", gross_tonnage=g, international_gross_tonnage=i,
                               flag_name=fl, vessel_type=ty)
                          for n, g, i, fl, ty in zip(ev.vessel_name, gt, igt, ev.flag_name, ev.vessel_type)]
    hang = {ton: picks(r, np.array([vlib.hangul_text(x["vessel_name"]) for x in r])) for ton, r in base_rows.items()}
    pd.set_option("display.width", 250)
    key = lambda s: "".join(ch for ch in str(s).upper() if ch.isalnum())  # noqa: E731
    true = ev.target_vessel_id.map(cat.vessel_name).map(key).to_numpy()
    cache, summary, reads = {}, {}, {}
    for tag, files in tags.items():
        lm = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files.split(",")])
        lm = lm.drop_duplicates("uid", keep="last").set_index("uid").reindex(ev.uid).fillna("")
        sec = pd.to_numeric(lm.seconds, errors="coerce")
        sec = sec[sec > 0.01]
        print(f"\n######## {tag}: {len(sec)} names read, {sec.mean():.2f} s per name (median {sec.median():.2f}); "
              f"{int((lm.read1 == '').sum())} rows without a reading")
        hit = {k: lm[k].map(key).to_numpy() == true for k in ("read1", "read2", "read3")}
        exact = pd.DataFrame({"read1": hit["read1"], "any3": hit["read1"] | hit["read2"] | hit["read3"]})
        print("LM reading equals the catalog name (letters and digits only):")
        print(exact.groupby(ev.set.to_numpy(), sort=False).mean().round(3).T.to_string())
        res = []
        for ton, base in base_rows.items():
            # the Latin rankers' pick for each (row, reading), shared by tags that read a name the same way
            need = sorted({(i, t) for k in ("read1", "read2", "read3") for i, t in enumerate(lm[k])
                           if t and (ton, i, t) not in cache})
            for (i, t), p in zip(need, picks([dict(base[i], vessel_name=t) for i, t in need], np.zeros(len(need), dtype=bool))):
                cache[ton, i, t] = p
            lp = [[cache[ton, i, t] if t else (None, -np.inf) for i, t in enumerate(lm[k])] for k in ("read1", "read2", "read3")]
            for i in range(len(ev)):
                y = ev.target_vessel_id[i]
                ok = [lm[k].iat[i] != "" for k in ("read1", "read2", "read3")]
                b = max((k for k in range(3) if ok[k]), key=lambda k: lp[k][i][1], default=0)
                lb_pick, lb_top = lp[b][i] if ok[b] else (None, -np.inf)
                res.append(dict(uid=ev.uid[i], set=ev.set[i], ton=ton, h_pick=hang[ton][i][0], h_top=hang[ton][i][1],
                                lb_pick=lb_pick, lb_top=lb_top, h_ok=hang[ton][i][0] == y, l1_ok=ok[0] and lp[0][i][0] == y,
                                lb_ok=lb_pick == y, any_ok=any(ok[k] and lp[k][i][0] == y for k in range(3)), best=b))
        d = pd.DataFrame(res)
        d.to_csv(out_dir / f"lm_eval_{tag}.csv", index=False)
        print("share right: h = Hangul pipeline, l1 = Latin rankers on reading 1, lb = on the best-scoring reading "
              "(script.py), any = one of 3 readings right, h|lb = either")
        g = d.groupby(["set", "ton"], sort=False)
        t = g[["h_ok", "l1_ok", "lb_ok", "any_ok"]].mean()
        t["h|lb"] = (d.h_ok | d.lb_ok).groupby([d.set, d.ton], sort=False).mean()
        print(t.round(3).to_string())
        agree = d.h_pick == d.lb_pick
        print("\nh and lb agree / right when they agree / when they differ: h right, lb right")
        for s, x in d.groupby("set", sort=False):
            a_ = agree[x.index]
            print(f"  {s:12s} agree {a_.mean():.3f}  right {x.h_ok[a_].mean():.3f}  |  h {x.h_ok[~a_].mean():.3f}"
                  f"  lb {x.lb_ok[~a_].mean():.3f}  (n differ {int((~a_).sum())})")
        gap = (d.lb_top - d.h_top).to_numpy()
        fair_d = np.tile(fair, 3)
        print("\nLM_MARGIN rule: take the LM pick when its score beats the Hangul pick's by more than m (mean over the "
              "3 tonnages); FAIR = sets of vessels the rankers never saw")
        table = {}
        fair_sets = [s for s in sets if s not in ("clean", "hspell2", "hsyl2", "dropfirst")]
        for mm in MARGINS:
            ok = pd.Series(np.where(gap > mm, d.lb_ok, d.h_ok))
            r = ok.groupby(d.set, sort=False).mean()
            r["ALL"], r["FAIR"], r["FAIRSET"] = ok.mean(), ok[fair_d].mean(), r[fair_sets].mean()
            table[f"m={mm}"] = r
            summary[tag, mm] = r["FAIRSET"]
        print(pd.DataFrame(table).T.round(3).to_string())
        x = d[(d.ton == "exact") & fair_d].copy()
        x["name"] = x.uid.map(ev.set_index("uid").vessel_name)
        x["read"] = [lm.loc[u, f"read{b + 1}"] for u, b in zip(x.uid, x.best)]
        x["true"] = x.uid.map(ev.set_index("uid").target_vessel_id).map(cat.vessel_name)
        x["h_name"] = x.h_pick.map(cat.vessel_name)
        x["gap"] = x.lb_top - x.h_top
        for title, sel in (("only lb right", ~x.h_ok & x.lb_ok), ("only h right", x.h_ok & ~x.lb_ok),
                           ("both wrong", ~x.h_ok & ~x.lb_ok)):
            print(f"\nexamples, unseen vessels, exact tonnage, {title} ({int(sel.sum())}): name | LM reading | true | "
                  "Hangul pick | score gap")
            for r in x[sel].head(8).itertuples():
                print(f"  {r.name} | {r.read} | {r.true} | {r.h_name} | {r.gap:.1f}")
        reads[tag] = lm.read1
    extra = ev[~ev.set.isin(["clean", "hspell2", "hsyl2", "dropfirst", "llm", "llm_hspell2"])]
    if len(extra):
        print("\nextra rows: name | true | first reading of " + " / ".join(tags))
        for u, n, v in zip(extra.uid, extra.vessel_name, extra.target_vessel_id):
            print(f"  {n} | {cat.vessel_name[v]} | " + " / ".join(reads[t][u] for t in tags))
    # each set of unseen vessels counts the same; on a tie the larger margin (fewer changes) wins
    best = max(summary, key=lambda tm: (round(summary[tm], 4), tm[1]))
    print(f"\nFAIRSET (mean over the sets of unseen vessels): " + ", ".join(f"{t} m={m}: {v:.4f}" for (t, m), v in summary.items() if m in (99, 2, 0)))
    print(f"RECOMMEND tag={best[0]} margin={best[1]} FAIRSET={summary[best]:.4f} (no LM: {summary[best[0], 99]:.4f})")


if __name__ == "__main__":
    main()
