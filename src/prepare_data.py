"""Build the item files from the raw MMLU / MMMLU data and the Hinglish translations.

Reads   data/raw/mmlu_en_test.parquet      MMLU test split        (cais/mmlu, config "all")
        data/raw/mmmlu_HI_IN_test.csv      MMMLU Hindi test split (openai/MMMLU, config "HI_IN")
        data/hinglish/hg_batch*.tsv        Hinglish translations of the 520 sampled items
Writes  data/processed/items_trilingual.csv   520 aligned items (English, Hindi, Hinglish)
        data/processed/items_subset.csv       a 208-item subset used by an earlier version (not used any more)
        data/processed/items_meta.json        sampling parameters
        data/processed/hinglish_audit.json    lexical audit of the Hinglish translations
        data/hinglish/validation_sample.csv   50 random items for two human raters (only if it does not exist)

Steps
  1. Check that MMMLU Hindi is row-aligned with MMLU (subject and gold letter on all 14,042 rows).
  2. Keep short items: len(question) + len(all four options) <= 300 English characters.
     Five subjects have fewer than 10 such items and drop out.
  3. Sample 10 items per remaining subject (52 x 10 = 520, seed 2026).
  4. Attach the Hinglish translations and audit them.
  5. Draw a 208-item subset (4 per subject, seed 2026); kept for compatibility, the experiment uses all 520.

Hinglish TSV format: item_id <TAB> question <TAB> A <TAB> B <TAB> C <TAB> D
  "="    option kept identical to English (numbers, formulas, names, technical terms)
  "\\n"  line break inside a question

The existing output files are reproduced byte for byte by this script. The audit needs the English
word list /usr/share/dict/words (present on macOS and most Linux systems).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LETTERS = "ABCD"
SEED = 2026
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Frequent romanised-Hindi words, used by the approximate code-mixing index.
HINDI_ROMAN = set("""
hai hain ho hota hoti hote hua hui hue tha thi the hoga hogi honge ka ki ke ko se mein me par pe tak liye
aur ya lekin agar to jo jab tab kya kaun kaunsa kaunsi kis kise kisi kiske kiska kitna kitni kitne kahan kaise kyun
inmein unmein isme usme yeh woh ye wo is us in un iska uska inka unka apna apni apne aap hum humein humare mujhe main
nahi na sirf bhi hi saath beech baad pehle upar neeche andar bahar taraf jaisa jaise waise jitna utna zyada kam sabse
karo karna karta karti karte kiya kiye kar sakta sakti sakte chahiye jaata jaati jaate jaaye jaayega gaya gayi gaye diya
diye di dete deta deti lena leta lete liya mila mile milta milti wala wali wale waala hona hone kehte kaha kehlata
maan lo maano find batao likho socho dobara phir ek do teen chaar paanch har saare sab koi kuch aisa aise aisi
sahi galat zaroori baare hisaab dwara wajah kaaran matlab sankhya log logon insaan cheez cheezein tarah tareeka
badhta badhti badhna ghatna ghat badh kab kyunki isliye siwaay dono haan nahi jaane sach bilkul thoda bahut
""".split())


# ---------------------------------------------------------------- steps 1-3: aligned English-Hindi sample
def sample_items(per_subject: int = 10, max_chars: int = 300) -> tuple[pd.DataFrame, dict]:
    hi = pd.read_csv(ROOT / "data/raw/mmmlu_HI_IN_test.csv")
    en = pd.read_parquet(ROOT / "data/raw/mmlu_en_test.parquet")
    gold = np.array([LETTERS[a] for a in en.answer])
    assert len(hi) == len(en), "row count differs"
    assert (hi.Subject.values == en.subject.values).all(), "subject misalignment"
    assert (hi.Answer.values == gold).all(), "answer misalignment"

    df = pd.DataFrame({"mmlu_idx": np.arange(len(en)), "subject": en.subject.values, "answer": gold,
                       "en_question": en.question.values})
    for i, L in enumerate(LETTERS):
        df[f"en_{L}"] = en.choices.apply(lambda c: str(c[i])).values
    df["hi_question"] = hi.Question.astype(str).values
    for L in LETTERS:
        df[f"hi_{L}"] = hi[L].astype(str).values
    df["en_chars"] = df.en_question.str.len() + sum(df[f"en_{L}"].str.len() for L in LETTERS)
    eligible = df[df.en_chars <= max_chars]

    rng = np.random.default_rng(SEED)
    parts = []
    for _, g in eligible.groupby("subject", sort=True):
        if len(g) >= per_subject:
            parts.append(df.loc[np.sort(rng.choice(g.index.values, size=per_subject, replace=False))])
    s = pd.concat(parts).reset_index(drop=True)
    # One fixed wrong option per item. Only the free-form pilot used it; the column is kept so that
    # the item file stays identical to the one the reported runs were made with.
    s["suggest_wrong"] = [[L for L in LETTERS if L != a][rng.integers(0, 3)] for a in s.answer]
    s.insert(0, "item_id", [f"q{i:04d}" for i in range(len(s))])

    meta = {"n_items": len(s), "n_subjects": s.subject.nunique(), "per_subject": per_subject,
            "max_chars": max_chars, "seed": SEED,
            "excluded_subjects": sorted(set(df.subject) - set(s.subject)),
            "gold_letter_distribution": s.answer.value_counts().sort_index().to_dict(),
            "mean_en_chars": float(s.en_chars.mean())}
    return s.astype(str), meta


# ---------------------------------------------------------------- step 4: Hinglish
def cmi(text: str, en_words: set) -> tuple[float, int, int]:
    """Approximate Code-Mixing Index (Das & Gambäck 2014) of one text: 100 * (1 - max_lang / n).

    A word counts as English if it is in the English word list and not in HINDI_ROMAN;
    everything else counts as Hindi. This is a heuristic, not a gold language-ID annotation.
    """
    hi = en = 0
    for t in re.findall(r"[A-Za-z']+", text):
        w = t.lower()
        if w not in HINDI_ROMAN and w in en_words and len(w) > 1:
            en += 1
        else:
            hi += 1
    n = hi + en
    return (100.0 * (1 - max(hi, en) / n) if n else 0.0), hi, en


def add_hinglish(base: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = {}
    for f in sorted((ROOT / "data/hinglish").glob("hg_batch*.tsv")):
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split("\t")
            assert len(parts) == 6, f"{f.name}:{ln} has {len(parts)} fields"
            assert parts[0].strip() not in rows, f"duplicate {parts[0]}"
            rows[parts[0].strip()] = parts[1:]
    assert set(rows) == set(base.item_id), "Hinglish translations do not match the sampled items"

    words = Path("/usr/share/dict/words")
    en_words = set(w.strip().lower() for w in words.read_text(errors="ignore").split()) if words.exists() else set()
    if not en_words:
        print("WARNING: /usr/share/dict/words not found; the code-mixing numbers in the audit will be wrong")

    out = base.copy()
    for col in ["hg_question"] + [f"hg_{L}" for L in LETTERS]:
        out[col] = ""
    copied, cmis, ratios, hi_tok, en_tok = 0, [], [], 0, 0
    for i, r in out.iterrows():
        q, *opts = rows[r.item_id]
        q = q.replace("\\n", "\n").strip()
        assert not DEVANAGARI.search(q), r.item_id
        out.at[i, "hg_question"] = q
        for L, o in zip(LETTERS, opts):
            o = o.strip()
            if o == "=":
                o, copied = r[f"en_{L}"], copied + 1
            assert not DEVANAGARI.search(o), (r.item_id, L)
            out.at[i, f"hg_{L}"] = o
        c, h, e = cmi(q, en_words)
        cmis.append(c)
        hi_tok, en_tok = hi_tok + h, en_tok + e
        ratios.append(len(q) / max(1, len(r.en_question)))

    audit = {
        "n_items": len(out),
        "options_identical_to_english": copied, "options_total": 4 * len(out),
        "share_options_identical": round(copied / (4 * len(out)), 3),
        "stems_identical_to_english": int((out.hg_question.str.strip() == out.en_question.str.strip()).sum()),
        "stem_char_length_ratio_hg_over_en": {"mean": round(float(np.mean(ratios)), 3),
                                              "median": round(float(np.median(ratios)), 3)},
        "approx_CMI_stem": {"mean": round(float(np.mean(cmis)), 1), "median": round(float(np.median(cmis)), 1),
                            "share_stems_CMI_gt_0": round(float(np.mean(np.array(cmis) > 0)), 3)},
        "token_share_hindi_vs_english_in_stems": {"hindi": hi_tok, "english": en_tok,
                                                  "hindi_share": round(hi_tok / max(1, hi_tok + en_tok), 3)},
        "note": "CMI uses a dictionary heuristic (English = in /usr/share/dict/words and not a listed Hindi "
                "word); it is an approximation, not a gold language-ID annotation.",
    }
    return out, audit


# ---------------------------------------------------------------- step 5: experiment subset
def subset(items: pd.DataFrame, n_per_subject: int = 4) -> pd.DataFrame:
    return (items.groupby("subject").sample(n=n_per_subject, random_state=SEED)
                 .sort_values("item_id").reset_index(drop=True))


def validation_sheet(items: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """A random sample of items for two raters who judge the Hinglish translation against the English.

    meaning  3 = same meaning, 2 = small loss or shift, 1 = wrong or misleading
    natural  3 = reads like natural Hinglish, 2 = acceptable, 1 = unnatural
    """
    s = items.sample(n=n, random_state=SEED).sort_values("item_id")
    join = lambda r, lang: " | ".join(f"{L}. {r[f'{lang}_{L}']}" for L in LETTERS)
    sheet = pd.DataFrame({"item_id": s.item_id, "subject": s.subject, "en_question": s.en_question,
                          "en_options": s.apply(join, axis=1, lang="en"), "hg_question": s.hg_question,
                          "hg_options": s.apply(join, axis=1, lang="hg")})
    for col in ["rater1_meaning", "rater1_natural", "rater2_meaning", "rater2_natural", "comment"]:
        sheet[col] = ""
    return sheet


def main():
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    base, meta = sample_items()
    items, audit = add_hinglish(base)
    items.to_csv(out / "items_trilingual.csv", index=False)
    # re-read exactly as run.py does, so the subset is the same whichever script creates it
    items = pd.read_csv(out / "items_trilingual.csv", dtype=str, keep_default_na=False)
    sub = subset(items)
    sub.to_csv(out / "items_subset.csv", index=False)
    (out / "items_meta.json").write_text(json.dumps(meta, indent=2))
    (out / "hinglish_audit.json").write_text(json.dumps(audit, indent=2))
    sheet = ROOT / "data/hinglish/validation_sample.csv"
    if not sheet.exists():  # never overwrite ratings that have been filled in
        validation_sheet(items).to_csv(sheet, index=False)
    print(json.dumps(meta, indent=2))
    print(json.dumps(audit, indent=2))
    print(f"wrote {len(items)} items and the {len(sub)}-item subset to {out}")


if __name__ == "__main__":
    main()
