"""Romanise the Hindi items: same words as the MMMLU Hindi text, Latin script instead of Devanagari.

Writes data/processed/items_hr.csv (item_id, hr_question, hr_A, hr_B, hr_C, hr_D), used by
`python src/run.py --exp translit`, and once data/translit_check_sample.csv (30 items for a spot check). Text that is not Devanagari (numbers, formulas, English words) is kept.

The scheme follows informal romanised Hindi and is deterministic:
  * every output letter is lower case, so a romanised word can never look like an answer letter A-D;
  * no diacritics: long and short vowels are written alike (aa only for a word-initial आ, as in "aap");
  * the inherent vowel "a" is dropped at the end of a word and in the middle of a word between
    a vowel+consonant and a consonant+vowel (the usual Hindi schwa deletion: "apna", "samajhna").
Run `python src/transliterate.py --show` to print the romanised prompts used in src/run.py.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
        "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d",
        "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
        "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h"}
NUKTA = {"क": "q", "ख": "kh", "ग": "gh", "ज": "z", "फ": "f", "ड": "r", "ढ": "rh"}
VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ए": "e", "ऐ": "ai",
          "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e"}
SIGNS = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o",
         "ौ": "au", "ॉ": "o", "ॅ": "e"}
OTHER = {"ं": "n", "ँ": "n", "ः": "h", "।": ".", "॥": ".", "ॐ": "om", "ऽ": ""}
VIRAMA, NUKTA_SIGN = "्", "़"
WORD = re.compile(r"[ऀ-ॿ]+")


def _word(w: str) -> str:
    # 1. split into units [kind, consonant, vowel]: kind "C" consonant (vowel None = inherent a, "" = virama),
    #    "V" standalone vowel, "N" nasal/visarga sign, "P" punctuation or digit
    units, i = [], 0
    while i < len(w):
        ch = w[i]
        if ch in CONS:
            c = CONS[ch]
            if i + 1 < len(w) and w[i + 1] == NUKTA_SIGN:
                c, i = NUKTA.get(ch, c), i + 1
            v = None
            if i + 1 < len(w) and w[i + 1] in SIGNS:
                v, i = SIGNS[w[i + 1]], i + 1
            elif i + 1 < len(w) and w[i + 1] == VIRAMA:
                v, i = "", i + 1
            units.append(["C", c, v])
        elif ch in VOWELS:
            units.append(["V", "", VOWELS[ch] if ch != "आ" or not units else "a"])  # "aa" only word-initially
        elif ch in ("ं", "ँ", "ः"):
            units.append(["N", "", OTHER[ch]])
        elif ch in OTHER:
            units.append(["P", "", OTHER[ch]])
        elif ch.isdigit():
            units.append(["P", "", str(unicodedata.digit(ch))])
        i += 1                                                      # anything else (e.g. a stray sign) is dropped
    # 2. schwa deletion
    letters = [k for k, u in enumerate(units) if u[0] != "P"]
    if len(letters) > 1 and units[letters[-1]][0] == "C" and units[letters[-1]][2] is None:
        units[letters[-1]][2] = ""                                  # word-final: "uttar", "vikalp"
    for k in range(len(units) - 2, 0, -1):                          # medial, right to left: V C(a) C V
        u, prv, nxt = units[k], units[k - 1], units[k + 1]
        if u[0] == "C" and u[2] is None and nxt[0] == "C" and nxt[2] != "" and (
                prv[0] == "V" or (prv[0] == "C" and prv[2] != "")):
            u[2] = ""                                               # "apna", "samajhna", "karna"
    return "".join(u[1] + ("a" if u[0] == "C" and u[2] is None else u[2]) for u in units)


def romanise(text: str) -> str:
    return WORD.sub(lambda m: _word(m.group()), text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print the romanised prompts used in src/run.py")
    a = ap.parse_args()
    if a.show:
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        import run
        print("INSTRUCTION:", romanise(run.INSTRUCTION["hi"]))
        for p, d in run.PUSHBACK.items():
            print(f"{p}:", romanise(d["hi"]))
        print("REMINDER:", romanise(run.REMINDER["hi"]))
        return
    d = pd.read_csv(ROOT / "data/processed/items_trilingual.csv", dtype=str, keep_default_na=False)
    out = pd.DataFrame({"item_id": d.item_id})
    for f in ["question", "A", "B", "C", "D"]:
        out[f"hr_{f}"] = d[f"hi_{f}"].map(romanise)
    left = out.drop(columns="item_id").apply(lambda s: s.str.contains(r"[ऀ-ॿ]")).to_numpy().sum()
    assert left == 0, f"{left} fields still contain Devanagari"
    out.to_csv(ROOT / "data/processed/items_hr.csv", index=False)
    print(f"wrote data/processed/items_hr.csv ({len(out)} items)")
    check = ROOT / "data/translit_check_sample.csv"   # for a Hindi speaker: is the romanisation readable?
    if not check.exists():
        s = d.sample(30, random_state=2026)[["item_id", "hi_question"]].merge(out[["item_id", "hr_question"]])
        s.assign(readable_1to5="", wrong_words="", comment="").to_csv(check, index=False)
        print(f"wrote {check.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
