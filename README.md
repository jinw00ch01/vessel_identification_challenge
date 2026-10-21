# Vessel identification challenge (Dacon 236771)

Matches each port-call record to one `vessel_id` in the official vessel catalog.

## Layout

- `submit/` is exactly what goes into `submit.zip` (`script.py`, `requirements.txt`, `model/`).
  - `submit/model/vlib.py` holds all normalisation, candidate retrieval and pair features, shared by training and inference.
- `tools/train.py` runs grouped cross-validation (`--cv --split-seeds 1 2 3`) and trains the final rankers (`--final`, writes `submit/model/ranker_*.txt`, `views.json`, `tables.json`).
- `tools/package.py --name vN --smoke` builds `dist/submit_vN.zip` and runs it in a fake server directory three times: as is (call-sign lookup answers train rows), with unseen call signs (every row goes through the ranker), and on malformed rows (blank or number-only names, bad tonnage) that must not stop the run.
- `data/` (not committed) must contain the official `train.csv` and `vessel_catalog.csv`.

## Approach

1. Latin names: uppercase, drop `M/V`-style prefixes, keep alphanumerics, retrieve catalog names within Levenshtein distance 4. Renamed vessels are also retrieved by same tonnage plus a shared word, and shared words are weighted by how rare they are in the catalog.
2. Hangul names: strip `호` / `제N호`, turn spelled-out letters into acronyms (케이엠티씨 -> KMTC) and match them as whole catalog words (제이에스 -> JS), and map both Hangul and English names to a coarse shared phonetic key. Several readings are tried per side:
   - Hangul: plain sounds, Revised Romanization (for Korean-origin words such as 성호 -> Seongho), and ㅓ read as an unstressed English vowel (얼라이언스 -> Alliance).
   - English: spelling rules with TH/CH alternatives (sea -> 씨, eagle -> 이글), pinyin with Korean conventions for Chinese names (장인 -> Jiangyin, 톈진 -> Tianjin, 선전 -> Shenzhen), and a continental reading that keeps R (아르페지오 -> Arpeggio).
   - A Hangul-word -> English-word dictionary learned from train (each train row uses it without its own vessel).
   Candidates come from within 12% gross tonnage and, whatever the tonnage says, from the 40 best phonetic matches in the whole catalog.
3. Train Hangul records always carry the right tonnage, but evaluation records may not (the overview lists tonnage errors and missing values). Training therefore adds two copies of every Hangul train row with the tonnage blanked or scaled by a random factor up to 1.8x either way, and CV also scores held-out Hangul rows with blank (`miss`), +-15% (`u15`) and heavy (`heavy`) tonnage errors.
4. Two LightGBM LambdaRank models score each candidate with name, tonnage, type and flag compatibility features, one with every feature and one without the coarse retrieval score; their scores are averaged and the top candidate wins. Tonnage gaps are absolute log ratios, so a record at a sixth of the catalog tonnage looks as far off as one at six times it, and the coarse Hangul retrieval score holds name evidence only.
5. A call sign seen in train maps straight to its vessel. Those rows skip retrieval entirely, so inference is fast when test vessels were seen in train.

Every row is predicted only from its own values, the catalog and train-derived tables.

## Results log

| Version | Grouped CV (unseen vessels) | Hangul accuracy with tonnage blank / +-15% / heavy | Public |
|---|---|---|---|
| v1 | 0.9855 (Hangul 0.9798, Latin 0.9987) | | 0.7947 |
| v2 | 0.9956 (Hangul 0.9941, Latin 0.9992) | | 0.8017 |
| v3 | 0.9957 (Hangul 0.9941, Latin 0.9995), mean of 3 random grouped splits | | 0.8080 |
| v4 | 0.9950 (Hangul 0.9933, Latin 0.9990), split seed 1 | 0.986 / 0.992 / 0.986 | pending |
| v5 | 0.9962 (Hangul 0.9950, Latin 0.9991), split seed 1 | 0.987 / 0.994 / 0.988 | pending |

The Public scores of v1 and v2 were far below CV. Train Hangul rows never miss tonnage and are never more than 9.3% off, so grouped CV alone overrated the tonnage-window retrieval. On one held-out fold, the tonnage-window pipeline kept 0.989 Hangul accuracy on clean rows but fell to 0.947 with tonnage blank and 0.809 with +-15% errors. v4 is the first version trained and checked against unreliable tonnage. The same pipeline without the augmented copies scores 0.9980 (Hangul 0.9975) on clean CV but 0.976 / 0.988 / 0.950 under the three tonnage stresses.
