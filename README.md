# Vessel identification challenge (Dacon 236771)

Matches each port-call record to one `vessel_id` in the official vessel catalog.

## Layout

- `submit/` is exactly what goes into `submit.zip` (`script.py`, `requirements.txt`, `model/`).
  - `submit/model/vlib.py` holds all normalisation, candidate retrieval and pair features, shared by training and inference.
- `tools/train.py` runs grouped cross-validation (`--cv --split-seeds 1 2 3`) and trains the final rankers (`--final`, writes `submit/model/ranker_*.txt`, `views.json`, `tables.json`).
- `tools/package.py --name vN --smoke` builds `dist/submit_vN.zip` and runs it in a fake server directory three times: as is (call-sign lookup answers train rows), with unseen call signs (every row goes through the ranker), and on malformed rows (blank or number-only names, bad tonnage) that must not stop the run.
- `tools/probe.py` turns a submit zip into a diagnostic submission. `latin_only` keeps the predictions for non-Hangul rows and gives Hangul rows a vessel that is never a target, so its Public score is 0.3 x non-Hangul accuracy; with the base zip's own score that also gives the Hangul accuracy. `lookup_only` keeps only the rows a train call sign decides.
- `data/` (not committed) must contain the official `train.csv` and `vessel_catalog.csv`.

## Approach

1. Latin names: uppercase, drop `M/V`-style prefixes, keep alphanumerics, retrieve catalog names within Levenshtein distance 4. Renamed vessels are also retrieved by same tonnage plus a shared word, and shared words are weighted by how rare they are in the catalog. For records that leave words out or shorten them, names sharing a word with the record, or with a word the record word starts (KENSIN -> KENSINGTON), are retrieved too, nearest tonnage first.
2. Hangul names: strip `호` / `제N호`, turn spelled-out letters into acronyms (케이엠티씨 -> KMTC) and match them as whole catalog words (제이에스 -> JS), and map both Hangul and English names to a coarse shared phonetic key. Several readings are tried per side:
   - Hangul: plain sounds, Revised Romanization (for Korean-origin words such as 성호 -> Seongho), and ㅓ read as an unstressed English vowel (얼라이언스 -> Alliance).
   - English: spelling rules with TH/CH alternatives (sea -> 씨, eagle -> 이글), pinyin with Korean conventions for Chinese names (장인 -> Jiangyin, 톈진 -> Tianjin, 선전 -> Shenzhen), and a continental reading that keeps R (아르페지오 -> Arpeggio).
   - A Hangul-word -> English-word dictionary learned from train (each train row uses it without its own vessel).
   Candidates come from within 12% gross tonnage, from catalog names that share a similar-sounding word (a record may leave words out), from vessels with exactly the recorded tonnage however different their name, and, when the record has no tonnage, from the 40 best whole-name phonetic matches in the catalog. A Korean-romanization comparison (흥아영 ~ Heung-A Young) is one of the name features.
3. CV also scores held-out rows under input changes the overview mentions (`--stress` for Hangul rows, `--stress-latin` for the rest): tonnage blank (`miss`), +-15% (`u15`) or heavily off (`heavy`), one jamo swapped (`spell`), spaces removed (`nospace`), a word left out (`dropfirst`, `droplast`, `dropword`), the first word cut to its initial (`initial`), the name cut after 60% of its letters (`trunc`), a catalog vessel renamed to each target's name (`twin`), and no learned Hangul dictionary (`nodict`). It also reports how often the target is still among the candidates (`*_recall`). Training adds one copy of every Hangul row with its name written differently (`--name-aug mix`: first or a random word left out, or a jamo swapped) and copies of 30% of the multi-word Latin rows without their first word (`--aug-latin-name`); copies with perturbed tonnage (`--aug`) are available but off, since they lowered the Public score.
4. LightGBM LambdaRank models score each candidate with name, tonnage, type and flag compatibility features. There are two per row type, one with every feature and one without the coarse retrieval score, and their scores are averaged; the top candidate wins. Latin rows use their own pair (`--latin-models`), the only one trained on the Latin name copies, because those copies cost Hangul accuracy in a shared model. Tonnage gaps are absolute log ratios, so a record at a sixth of the catalog tonnage looks as far off as one at six times it, and the coarse Hangul retrieval score holds name evidence only. Tonnage written as text (`12,345`, `12345 GT`) is still read as a number.
5. A call sign seen in train maps straight to its vessel. Those rows skip retrieval entirely, so inference is fast when test vessels were seen in train.

Every row is predicted only from its own values, the catalog and train-derived tables.

## Results log

| Version | Grouped CV (unseen vessels) | Stress (held-out accuracy) | Public |
|---|---|---|---|
| v1 | 0.9855 (Hangul 0.9798, Latin 0.9987) | | 0.7947 |
| v2 | 0.9956 (Hangul 0.9941, Latin 0.9992) | | 0.8017 |
| v3 | 0.9957 (Hangul 0.9941, Latin 0.9995), mean of 3 random grouped splits | | 0.8080 |
| v4 | 0.9950 (Hangul 0.9933, Latin 0.9990), split seed 1 | Hangul tonnage blank / +-15% / heavy 0.986 / 0.992 / 0.986 | 0.7695 |
| v5 | 0.9962 (Hangul 0.9950, Latin 0.9991), split seed 1 | Hangul tonnage 0.987 / 0.994 / 0.988 | not submitted |
| v6 | 0.9955 (Hangul 0.9938, Latin 0.9993), mean of split seeds 1-3 | Hangul tonnage 0.986 / 0.919 / 0.625; Hangul word dropped first / last 0.82 / 0.71; Latin first word dropped / initial / cut 0.60 / 0.73 / 0.44 | 0.8204 |
| v7 | 0.9966 (Hangul 0.9955, Latin 0.9990), mean of split seeds 1-3 | Hangul word dropped first / last 0.86 / 0.75; Latin first word dropped / initial / cut 0.79 / 0.84 / 0.69 | pending |

The Public scores of v1 and v2 were far below CV. Train Hangul rows never miss tonnage and are never more than 9.3% off, so grouped CV alone overrated the tonnage-window retrieval. On one held-out fold, the tonnage-window pipeline kept 0.989 Hangul accuracy on clean rows but fell to 0.947 with tonnage blank and 0.809 with +-15% errors. v4 is the first version trained and checked against unreliable tonnage. The same pipeline without the augmented copies scores 0.9980 (Hangul 0.9975) on clean CV but 0.976 / 0.988 / 0.950 under the three tonnage stresses.

v4 then scored 0.7695 on Public, below v3. Trusting names over tonnage hurts on the real evaluation records, so their tonnage is probably fine and their names are what differ. v6 drops the tonnage copies, searches the whole catalog by name only when tonnage is missing, and adds word-level retrieval with training copies that leave out the first word of multi-word Hangul names: held-out Hangul rows with the first word removed go from 0.30 to 0.82.

v6 scored 0.8204, above v3, so records with only part of the name do occur, but if all of the gain came from first-word-dropped Hangul names they would be only about 3% of the Hangul rows, and most of the gap to CV is still unexplained. v7 extends the partial-name handling to Latin rows and to other kinds of shortening, and always keeps the vessels with exactly the recorded tonnage as Hangul candidates. Its Latin-only probe (`tools/probe.py --mode latin_only`) splits the Public score into Hangul and Latin accuracy, to show which of the two holds the gap.
