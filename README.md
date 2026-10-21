# Vessel identification challenge (Dacon 236771)

Matches each port-call record to one `vessel_id` in the official vessel catalog.

## Layout

- `submit/` is exactly what goes into `submit.zip` (`script.py`, `requirements.txt`, `model/`).
  - `submit/model/vlib.py` holds all normalisation, candidate retrieval and pair features, shared by training and inference.
- `tools/train.py` runs grouped cross-validation (`--cv`) and trains the final ranker (`--final`).
- `tools/package.py --name vN --smoke` builds `dist/submit_vN.zip` and runs it in a fake server directory.
- `data/` (not committed) must contain the official `train.csv` and `vessel_catalog.csv`.

## Approach

1. Latin names: uppercase, drop `M/V`-style prefixes, keep alphanumerics, retrieve catalog names within Levenshtein distance 4.
2. Hangul names: strip `호` / `제N호`, turn spelled-out letters into acronyms (케이엠티씨 -> KMTC), and map both Hangul and English names to a coarse shared phonetic key. Several readings are tried per side: plain Hangul sounds, Revised Romanization (for Korean-origin words such as 성호 -> Seongho), English spelling rules with TH/CH and pinyin alternatives, and a Hangul-word -> English-word dictionary learned from train (each train row uses the dictionary without its own vessel). Candidates come from within 12% gross tonnage.
3. A LightGBM model scores each candidate with name, tonnage, type and flag compatibility features; the top score wins.
4. A call sign seen in train maps straight to its vessel.

Every row is predicted only from its own values, the catalog and train-derived tables.

## Results log

| Version | Grouped CV (unseen vessels) | Public |
|---|---|---|
| v1 | 0.9855 (Hangul 0.9798, Latin 0.9987) | pending |
| v2 | 0.9956 (Hangul 0.9941, Latin 0.9992) | pending |
