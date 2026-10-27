"""Reading tables for Hangul names written as Korean Hanja readings of Chinese or Japanese ship names
(JIN XING -> 금성, KOSEI MARU -> 행성환): romanised syllable -> Korean readings with weights.

Sources: KANJIDIC2 (EDRDG, CC BY-SA 4.0, bundled in the jamdict-data package) for pinyin, Japanese on/kun
readings and Korean readings per character, plus the hanja package's Hanja -> Hangul table with pypinyin
for Chinese characters outside KANJIDIC2. General character dictionaries only: no vessel data.

Usage: python tools/build_hanja.py work/models/<name>/hanja.json   (needs: pip install jamdict jamdict-data hanja pypinyin)
"""
import json, re, sqlite3, sys
from collections import defaultdict

import jamdict_data, os
import pypinyin
from hanja import table as hanja_table

DB = os.path.join(os.path.dirname(jamdict_data.__file__), "jamdict.db")

KANA = {
    "ア": "A", "イ": "I", "ウ": "U", "エ": "E", "オ": "O", "カ": "KA", "キ": "KI", "ク": "KU", "ケ": "KE", "コ": "KO",
    "サ": "SA", "シ": "SHI", "ス": "SU", "セ": "SE", "ソ": "SO", "タ": "TA", "チ": "CHI", "ツ": "TSU", "テ": "TE",
    "ト": "TO", "ナ": "NA", "ニ": "NI", "ヌ": "NU", "ネ": "NE", "ノ": "NO", "ハ": "HA", "ヒ": "HI", "フ": "FU",
    "ヘ": "HE", "ホ": "HO", "マ": "MA", "ミ": "MI", "ム": "MU", "メ": "ME", "モ": "MO", "ヤ": "YA", "ユ": "YU",
    "ヨ": "YO", "ラ": "RA", "リ": "RI", "ル": "RU", "レ": "RE", "ロ": "RO", "ワ": "WA", "ヰ": "I", "ヱ": "E",
    "ヲ": "O", "ン": "N", "ガ": "GA", "ギ": "GI", "グ": "GU", "ゲ": "GE", "ゴ": "GO", "ザ": "ZA", "ジ": "JI",
    "ズ": "ZU", "ゼ": "ZE", "ゾ": "ZO", "ダ": "DA", "ヂ": "JI", "ヅ": "ZU", "デ": "DE", "ド": "DO", "バ": "BA",
    "ビ": "BI", "ブ": "BU", "ベ": "BE", "ボ": "BO", "パ": "PA", "ピ": "PI", "プ": "PU", "ペ": "PE", "ポ": "PO",
    "ヴ": "VU",
}
SMALL = {"ャ": "YA", "ュ": "YU", "ョ": "YO"}


def kana_romaji(text: str) -> str | None:
    # hiragana -> katakana
    text = "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text)
    out, i = "", 0
    while i < len(text):
        c = text[i]
        if c == "ッ":  # doubled consonant
            nxt = text[i + 1:i + 2]
            r = KANA.get(nxt, "")
            out += r[:1] if r else ""
            i += 1
            continue
        if c == "ー":
            out += out[-1:] if out else ""
            i += 1
            continue
        if c in SMALL:
            if out.endswith("I"):
                base = out[:-1]
                y = SMALL[c]
                out = base + (y[1:] if base.endswith(("SH", "CH", "J")) else y)
            i += 1
            continue
        if c in "ァィゥェォ":
            i += 1
            continue
        r = KANA.get(c)
        if r is None:
            return None
        out += r
        i += 1
    return out


def canon_ja(s: str) -> str:
    """One spelling for Japanese romaji (Hepburn or Kunrei, long vowels marked or not)."""
    s = s.upper()
    for a, b in (("SYA", "SHA"), ("SYU", "SHU"), ("SYO", "SHO"), ("TYA", "CHA"), ("TYU", "CHU"), ("TYO", "CHO"),
                 ("ZYA", "JA"), ("ZYU", "JU"), ("ZYO", "JO"), ("JYA", "JA"), ("JYU", "JU"), ("JYO", "JO"),
                 ("DYA", "JA"), ("DYU", "JU"), ("DYO", "JO")):
        s = s.replace(a, b)
    s = re.sub(r"TU", "TSU", s)
    s = re.sub(r"(?<![SC])HU", "FU", s)
    s = re.sub(r"(?<![SC])SI", "SHI", s)
    s = re.sub(r"(?<!S)TI", "CHI", s)
    s = re.sub(r"[ZD]I", "JI", s)
    s = re.sub(r"DU", "ZU", s)
    s = re.sub(r"M(?=[BPM])", "N", s)
    s = re.sub(r"OU|OO|OH(?![AIUEO])", "O", s)
    s = re.sub(r"UU", "U", s)
    return s


RENDAKU = {"K": "G", "S": "Z", "T": "D", "H": "B", "F": "B", "C": "J"}


def dueum(kor: str) -> set[str]:
    """Initial-sound law variants: 룡 -> 용, 녕 -> 영, 로 -> 노."""
    out = {kor}
    code = ord(kor) - 0xAC00
    if not 0 <= code < 11172:
        return out
    cho, rest = divmod(code, 588)
    jung = rest // 28
    y_vowels = {2, 3, 6, 7, 12, 17, 20}  # ㅑ ㅒ ㅕ ㅖ ㅛ ㅠ ㅣ
    if cho == 5:  # ㄹ
        out.add(chr(0xAC00 + (11 if jung in y_vowels else 2) * 588 + rest))
    if cho == 2 and jung in y_vowels:  # ㄴ
        out.add(chr(0xAC00 + 11 * 588 + rest))
    return out


def main(out_path: str) -> None:
    db = sqlite3.connect(DB)
    rows = db.execute("select c.literal, c.freq, r.r_type, r.value from character c join rm_group g on g.cid=c.ID "
                      "join reading r on r.gid=g.ID where r.r_type in ('pinyin','korean_h','ja_on','ja_kun')").fetchall()
    nanori = db.execute("select c.literal, n.value from character c join nanori n on n.cid=c.ID").fetchall()
    chars = defaultdict(lambda: defaultdict(list))
    freq = {}
    for lit, f, t, v in rows:
        chars[lit][t].append(v)
        freq[lit] = f
    for lit, v in nanori:
        chars[lit]["nanori"].append(v)
    py = defaultdict(lambda: defaultdict(float))
    ja = defaultdict(lambda: defaultdict(float))

    def weight(lit):
        f = freq.get(lit)
        return 3.0 if f not in (None, "") and int(f) <= 2500 else 1.0

    for lit, d in chars.items():
        kors = set()
        for k in d.get("korean_h", []):
            kors |= dueum(k)
        if not kors:
            continue
        w = weight(lit)
        for p in d.get("pinyin", []):
            p = re.sub(r"\d", "", p).upper().replace("Ü", "V")
            for k in kors:
                py[p][k] += w
        for kind, scale in (("ja_on", 1.0), ("ja_kun", 0.5), ("nanori", 0.3)):
            for r in d.get(kind, []):
                r = r.split(".")[0].replace("-", "")
                rom = kana_romaji(r)
                if not rom:
                    continue
                rom = canon_ja(rom)
                forms = {rom}
                if rom[0] in RENDAKU:
                    forms.add(RENDAKU[rom[0]] + rom[1:] if rom[0] not in "CF" else ("J" + rom[2:] if rom.startswith("CH") else "B" + rom[1:]))
                    if rom[0] in "HF":
                        forms.add("P" + rom[1:])
                for fm in forms:
                    for k in kors:
                        ja[fm][k] += w * scale * (1.0 if fm == rom else 0.5)
    # Chinese characters outside KANJIDIC2 (KS X 1001 Hanja with their pinyin)
    for ch, kor in hanja_table.hanja_table.items():
        if len(ch) != 1 or len(kor) != 1 or ch in chars:
            continue
        try:
            ch.encode("euc_kr")
        except UnicodeEncodeError:
            continue
        for p in pypinyin.pinyin(ch, style=pypinyin.Style.NORMAL, heteronym=True)[0]:
            if p.isascii() and p.isalpha():
                for k in dueum(kor):
                    py[p.upper().replace("Ü", "V")][k] += 1.0
    out = {"py": {s: dict(v) for s, v in py.items()}, "ja": {s: dict(v) for s, v in ja.items()}}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    print(f"pinyin syllables {len(py)}, japanese syllables {len(ja)}")
    for s in ("JIN", "XING", "HAI", "MARU", "KO", "SEI", "SHIN", "YANG", "ZHONG"):
        tab = py.get(s) or ja.get(s) or {}
        print(s, sorted(tab.items(), key=lambda kv: -kv[1])[:8])


if __name__ == "__main__":
    main(sys.argv[1])
