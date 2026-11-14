r"""Analysis: accuracy, flip-to-wrong rates, paired language tests, confidence-adjusted regression.

Reads  results/raw/<model>.csv                  one row per item x language x option order x pushback message
                                                (src/run.py); the main experiment is option order 0 with the
                                                neutral, mild and strong messages
       results/readout_check/<model>.csv        written replies on 65 items (src/check_readout.py)
Writes results/tables/*.csv, results/summary.json    every number quoted in the paper
       paper-springer/tables/*.tex                   the result tables, \input by main.tex
       figures/*.pdf|png, paper-springer/figures/*.pdf
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
RAW, TAB, FIG = ROOT / "results/raw", ROOT / "results/tables", ROOT / "figures"
READOUT = ROOT / "results/readout_check"
PAPER = ROOT / "paper-springer"
PTAB, PFIG = PAPER / "tables", PAPER / "figures"
for d in (TAB, FIG, PTAB, PFIG):
    d.mkdir(parents=True, exist_ok=True)

MODELS = {"llama3.2-3b": "Llama-3.2-3B", "qwen3-4b": "Qwen3-4B", "gemma3-4b": "Gemma-3-4B"}
LANGS = {"en": "English", "hi": "Hindi", "hg": "Hinglish"}
PTYPES = ["neutral", "mild", "strong"]
COLORS = {"en": "#2a78d6", "hi": "#eb6834", "hg": "#1baf7a"}  # validated categorical slots 1-3
MARKERS = {"en": "o", "hi": "s", "hg": "^"}
LETTERS = list("ABCD")
# A turn-1 answer is used only if the model puts at least this much first-token probability on the letters
# A-D. Below it the model wants to start its reply with other text ("Sahi option: ...") and the letter read from
# the probabilities is usually not the answer it would write (see readout_check).
MIN_MASS = 0.5
rng = np.random.default_rng(2026)


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    return proportion_confint(k, n, method="wilson")


def mcnemar_p(b, c):
    """Exact McNemar test from the two discordant counts."""
    return binomtest(b, b + c, 0.5).pvalue if b + c else 1.0


def bootstrap_ci(diff):
    """Percentile 95% CI of the mean of diff (2000 resamples)."""
    n = len(diff)
    boots = [diff[i].mean() for i in (rng.integers(0, n, n) for _ in range(2000))] if n else [np.nan]
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def add_derived(d):
    p0 = d[[f"p0_{L}" for L in LETTERS]].to_numpy()
    # p0_*, p1_* are already normalised over the four letters (run.py), so confidence does not depend on how much
    # probability a model puts on non-letter tokens.
    d["conf0"] = p0.max(axis=1)                        # turn-1 probability of the chosen answer
    d["flip"] = (d.a1 != d.a0).astype(int)             # any change of answer
    d["c2w"] = ((d.correct0 == 1) & (d.correct1 == 0)).astype(int)
    return d


def load() -> pd.DataFrame:
    """All rows of all models, including turn-1 answers below MIN_MASS. main_rows() selects the main experiment."""
    frames = [pd.read_csv(RAW / f"{m}.csv").assign(model=m) for m in MODELS if (RAW / f"{m}.csv").exists()]
    return add_derived(pd.concat(frames, ignore_index=True))


def main_rows(full):
    return full[(full.rot == 0) & full.ptype.isin(PTYPES)].copy()


# ------------------------------------------------------------------ descriptive table
def describe(df):
    rows = []
    for m in [m for m in MODELS if m in df.model.unique()]:
        for L in LANGS:
            d = df[(df.model == m) & (df.lang == L)]
            t1 = d[d.ptype == "neutral"]
            n, k = len(t1), int(t1.correct0.sum())
            lo, hi = wilson(k, n)
            r = {"model": m, "lang": L, "n_items": n, "acc0": k / n, "acc0_lo": lo, "acc0_hi": hi, "n_correct0": k,
                 "letter_mass0_median": t1.mass0.median(), "share_mass0_lt_0.5": (t1.mass0 < 0.5).mean()}
            for p in PTYPES:
                c = d[(d.ptype == p) & (d.correct0 == 1)]
                kk = int(c.c2w.sum())
                lo, hi = wilson(kk, len(c))
                r.update({f"c2w_{p}": kk / len(c) if len(c) else np.nan, f"c2w_{p}_lo": lo, f"c2w_{p}_hi": hi,
                          f"acc1_{p}": d[d.ptype == p].correct1.mean(),
                          f"flip_{p}": d[d.ptype == p].flip.mean()})
                w = d[(d.ptype == p) & (d.correct0 == 0)]
                r[f"w2c_{p}"] = w.correct1.mean() if len(w) else np.nan
                ch = w[w.flip == 1]  # wrong answers that were changed: how many, and how many became right
                r.update({f"flip_wrong_{p}": w.flip.mean() if len(w) else np.nan, f"n_changed_wrong_{p}": len(ch),
                          f"w2c_changed_{p}": ch.correct1.mean() if len(ch) else np.nan})
            rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ answer-letter bias
def letter_bias(df):
    """Which letters does a model answer at turn 1, and how much of its accuracy comes from its favourite letter?"""
    rows = []
    for m in [m for m in MODELS if m in df.model.unique()]:
        for L in LANGS:
            d = df[(df.model == m) & (df.lang == L)]
            t1 = d[d.ptype == "neutral"]
            share = t1.a0.value_counts(normalize=True).reindex(LETTERS, fill_value=0.0)
            top = share.idxmax()
            r = {"model": m, "lang": L, **{f"share_{x}": share[x] for x in LETTERS},
                 "gold_share_top": (t1.gold == top).mean(), "top_letter": top, "top_share": share[top],
                 "acc_gold_top": t1[t1.gold == top].correct0.mean(), "acc_gold_other": t1[t1.gold != top].correct0.mean()}
            for p in ["mild", "strong"]:
                c = d[(d.ptype == p) & (d.correct0 == 1)]
                r.update({f"c2w_{p}_top": c[c.gold == top].c2w.mean(), f"n_{p}_top": int((c.gold == top).sum()),
                          f"c2w_{p}_other": c[c.gold != top].c2w.mean(), f"n_{p}_other": int((c.gold != top).sum()),
                          f"share_flips_to_top_{p}": (c[c.c2w == 1].a1 == top).mean()})
            rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ paired language contrasts
def paired_tests(df):
    """Items answered correctly at turn 1 in BOTH English and language L; compare C->W under the same pushback."""
    rows = []
    for m in [m for m in MODELS if m in df.model.unique()]:
        for p in ["mild", "strong"]:
            w = df[(df.model == m) & (df.ptype == p)].pivot(index="item_id", columns="lang", values=["correct0", "c2w"])
            for L in ["hi", "hg"]:
                both = w[(w[("correct0", "en")] == 1) & (w[("correct0", L)] == 1)]
                fe, fl = both[("c2w", "en")].to_numpy(), both[("c2w", L)].to_numpy()
                n = len(both)
                b, c = int(((fl == 1) & (fe == 0)).sum()), int(((fl == 0) & (fe == 1)).sum())  # discordant pairs
                lo, hi = bootstrap_ci(fl - fe)
                rows.append({"model": m, "ptype": p, "contrast": f"{L}-en", "n_pairs": n,
                             "c2w_en": fe.mean() if n else np.nan, "c2w_L": fl.mean() if n else np.nan,
                             "delta": (fl - fe).mean() if n else np.nan,
                             "ci_lo": lo, "ci_hi": hi, "only_L_flips": b, "only_en_flips": c, "p_mcnemar": mcnemar_p(b, c)})
    t = pd.DataFrame(rows)
    t["p_holm"] = multipletests(t.p_mcnemar, method="holm")[1]
    return t


# ------------------------------------------------------------------ confidence-adjusted regression
def add_reg_cols(d):
    """Add the logit of turn-1 confidence and make English the reference language."""
    d = d.copy()
    d["conf_logit"] = np.log(d.conf0.clip(1e-4, 1 - 1e-4) / (1 - d.conf0.clip(1e-4, 1 - 1e-4)))
    d["lang"] = pd.Categorical(d.lang, categories=["en", "hi", "hg"])
    return d


def reg_data(df):
    """Correct turn-1 answers under mild or strong pushback, with the logit of turn-1 confidence."""
    return add_reg_cols(df[(df.correct0 == 1) & (df.ptype.isin(["mild", "strong"]))])


def fit_or(g, formula, **info):
    """Logistic regression with SEs clustered by item; returns the Hindi and Hinglish odds ratios vs English."""
    try:
        fit = smf.glm(formula, g, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(g.item_id)[0]})
    except Exception as e:  # e.g. perfect separation
        print(f"regression failed for {info}: {e}")
        return [], None
    ci = fit.conf_int()
    rows = []
    for L in ["hi", "hg"]:
        k = f"C(lang)[T.{L}]"
        r = {**info, "lang": L, "n": int(fit.nobs), "n_flips_L": int(g[g.lang == L].c2w.sum()),
             "OR": np.exp(fit.params[k]), "OR_lo": np.exp(ci.loc[k, 0]), "OR_hi": np.exp(ci.loc[k, 1]),
             "p": fit.pvalues[k]}
        if r["n_flips_L"] == 0:  # no flip at all in this language: the odds ratio is 0 and has no usable CI
            r.update({"OR": 0.0, "OR_lo": np.nan, "OR_hi": np.nan, "p": np.nan})
        rows.append(r)
    return rows, fit


def regression(df):
    """Logistic regression of C->W on language (vs English), with and without turn-1 confidence.

    Rows: turn-1-correct answers under mild or strong pushback. SEs clustered by item.
    Pooled model adds model fixed effects.
    """
    d = reg_data(df)
    rows = []
    groups = [(m, d[d.model == m]) for m in MODELS if m in d.model.unique()] + [("pooled", d)]
    for name, g in groups:
        fe = " + C(model)" if name == "pooled" else ""
        for adj, f in [("unadjusted", f"c2w ~ C(lang) + C(ptype){fe}"),
                       ("confidence-adjusted", f"c2w ~ C(lang) + C(ptype) + conf_logit{fe}")]:
            rr, fit = fit_or(g, f, model=name, spec=adj)
            for r in rr:
                del r["n_flips_L"]
                r["OR_conf_per_logit"] = np.exp(fit.params["conf_logit"]) if "conf_logit" in fit.params else np.nan
            rows += rr
    return pd.DataFrame(rows)


def regression_checks(df):
    """How far can the regression of Table 3 be trusted? Four checks.

    by pushback        the same regression for mild and strong pushback separately
    leave one out      the pooled confidence-adjusted model without each model in turn
    confidence scale   the per-model adjusted model with confidence clipped differently or cut into 5 bins
    interactions       Wald tests: does the language effect differ between models / between pushback types?
    """
    d = reg_data(df)
    models = [m for m in MODELS if m in d.model.unique()]
    rows = []
    for m in models:
        for p in ["mild", "strong"]:
            g = d[(d.model == m) & (d.ptype == p)]
            rows += fit_or(g, "c2w ~ C(lang)", check="by pushback", model=m, ptype=p, spec="unadjusted")[0]
            rows += fit_or(g, "c2w ~ C(lang) + conf_logit", check="by pushback", model=m, ptype=p,
                           spec="confidence-adjusted")[0]
    for m in models:
        rows += fit_or(d[d.model != m], "c2w ~ C(lang) + C(ptype) + conf_logit + C(model)", check="leave one out",
                       model=f"pooled without {m}", ptype="mild+strong", spec="confidence-adjusted")[0]
    for m in models:
        g = d[d.model == m].copy()
        for eps in (1e-2, 1e-5):
            c = g.conf0.clip(eps, 1 - eps)
            g["conf_alt"] = np.log(c / (1 - c))
            rows += fit_or(g, "c2w ~ C(lang) + C(ptype) + conf_alt", check="confidence scale", model=m,
                           ptype="mild+strong", spec=f"logit, clipped at {eps:g}")[0]
        g["conf_bin"] = pd.cut(g.conf0, [0, 0.6, 0.9, 0.99, 0.9999, 1.01], right=False).astype(str)
        rows += fit_or(g, "c2w ~ C(lang) + C(ptype) + C(conf_bin)", check="confidence scale", model=m,
                       ptype="mild+strong", spec="5 confidence bins")[0]
    out = pd.DataFrame(rows)

    def wald(formula, term):
        fit = smf.glm(formula, d, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(d.item_id)[0]})
        return float(fit.wald_test_terms(scalar=True).table.loc[term, "pvalue"])
    info = {"p_language_x_model": wald("c2w ~ C(lang) * C(model) + C(ptype) + conf_logit", "C(lang):C(model)"),
            "p_language_x_pushback": wald("c2w ~ C(lang) * C(ptype) + C(model) + conf_logit", "C(lang):C(ptype)"),
            "share_conf_ge_0.9999": {m: d[d.model == m].groupby("lang", observed=True).conf0
                                     .apply(lambda x: round(float((x >= 0.9999).mean()), 3)).to_dict() for m in models}}
    return out, info


# ------------------------------------------------------------------ wording experiment
# What each message asks for. "mild"/"strong" are the main wordings; the others come from results/wording.
REQUEST = {"mild": {"en": "reconsider", "hi": "reconsider", "hg": "check"},
           "mild_2": {"en": "check", "hi": "check", "hg": "reconsider"}}


DOUBT = ["mild", "mild_2", "mild_3", "strong", "strong_2", "strong_3"]


def load_wording(full):
    """All six doubt messages in the original option order."""
    d = full[(full.rot == 0) & full.ptype.isin(DOUBT)].copy()
    d["family"] = d.ptype.str.split("_").str[0]
    return d if d.ptype.nunique() > 2 else None


def paired(d1, d2):
    """Exact McNemar test of C->W between two conditions on items answered correctly in both."""
    a, b = d1.set_index("item_id"), d2.set_index("item_id")
    idx = a.index[(a.correct0 == 1)].intersection(b.index[(b.correct0 == 1)])
    fa, fb = a.loc[idx, "c2w"].to_numpy(), b.loc[idx, "c2w"].to_numpy()
    only_a, only_b = int(((fa == 1) & (fb == 0)).sum()), int(((fa == 0) & (fb == 1)).sum())
    return {"n_pairs": len(idx), "c2w_a": fa.mean() if len(idx) else np.nan, "c2w_b": fb.mean() if len(idx) else np.nan,
            "only_a_flips": only_a, "only_b_flips": only_b,
            "p_mcnemar": mcnemar_p(only_a, only_b)}


def wording(full):
    """C->W for every wording, and the language contrasts that hold the wording constant."""
    d = load_wording(full)
    if d is None:
        return None, None, None
    models = [m for m in MODELS if m in d.model.unique()]
    order = DOUBT
    rates = []
    for m in models:
        for L in LANGS:
            for p in order:
                g = d[(d.model == m) & (d.lang == L) & (d.ptype == p)]
                c = g[g.correct0 == 1]
                w = g[g.correct0 == 0]
                lo, hi = wilson(int(c.c2w.sum()), len(c))
                rates.append({"model": m, "lang": L, "ptype": p, "n_correct": len(c), "c2w": c.c2w.mean(),
                              "c2w_lo": lo, "c2w_hi": hi, "flip_all": g.flip.mean(), "flip_wrong": w.flip.mean()})
    rates = pd.DataFrame(rates)

    tests = []
    for m in models:
        g = d[d.model == m]
        cell = lambda L, p: g[(g.lang == L) & (g.ptype == p)]
        for L in ["hi", "hg"]:
            for p in order:  # same message label in both languages
                tests.append({"model": m, "contrast": f"{L}-en", "kind": "same label", "a": f"{L}:{p}", "b": f"en:{p}",
                              **paired(cell(L, p), cell("en", p))})
        # same request in both languages: English "check" vs Hinglish "check", English "reconsider" vs Hinglish "reconsider"
        for req in ["check", "reconsider"]:
            pe = [p for p in REQUEST if REQUEST[p]["en"] == req][0]
            ph = [p for p in REQUEST if REQUEST[p]["hg"] == req][0]
            tests.append({"model": m, "contrast": "hg-en", "kind": f"same request: {req}", "a": f"hg:{ph}",
                          "b": f"en:{pe}", **paired(cell("hg", ph), cell("en", pe))})
        # within a language: does the request ("check" vs "reconsider") or the instruction (strong_3 has none) matter?
        for L in LANGS:
            pc = [p for p in REQUEST if REQUEST[p][L] == "check"][0]
            pr = [p for p in REQUEST if REQUEST[p][L] == "reconsider"][0]
            tests.append({"model": m, "contrast": L, "kind": "check vs reconsider", "a": f"{L}:{pc}", "b": f"{L}:{pr}",
                          **paired(cell(L, pc), cell(L, pr))})
            tests.append({"model": m, "contrast": L, "kind": "claim only vs claim + instruction", "a": f"{L}:strong_3",
                          "b": f"{L}:strong", **paired(cell(L, "strong_3"), cell(L, "strong"))})
    tests = pd.DataFrame(tests)
    tests["delta"] = tests.c2w_a - tests.c2w_b

    # language effect averaged over the three wordings of each family, with and without confidence
    reg = []
    c = add_reg_cols(d[d.correct0 == 1])
    for m in models:
        for fam in ["mild", "strong"]:
            g = c[(c.model == m) & (c.family == fam)]
            reg += fit_or(g, "c2w ~ C(lang) + C(ptype)", model=m, family=fam, spec="unadjusted")[0]
            reg += fit_or(g, "c2w ~ C(lang) + C(ptype) + conf_logit", model=m, family=fam, spec="confidence-adjusted")[0]
    return rates, tests, pd.DataFrame(reg)


def latex_wording(rates):
    cols = ["mild", "mild_2", "mild_3", "strong", "strong_2", "strong_3"]
    out = [r"\begin{tabular}{llcccccc}", r"\toprule",
           r" & & \multicolumn{3}{c}{Mild (doubt)} & \multicolumn{3}{c}{Strong (``your answer is wrong'')}\\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
           r"Model & Language & M1 & M2 & M3 & S1 & S2 & S3\\", r"\midrule"]
    for i, m in enumerate(rates.model.unique()):
        if i:
            out.append(r"\midrule")
        for j, L in enumerate(LANGS):
            r = rates[(rates.model == m) & (rates.lang == L)].set_index("ptype").c2w
            out.append(f"{MODELS[m] if j == 0 else ''} & {LANGS[L]} & " + " & ".join(f"{r[c]:.3f}" for c in cols) + "\\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PTAB / "wording.tex").write_text("\n".join(out) + "\n")


# ------------------------------------------------------------------ option-order experiment
def load_order(full):
    """Items that were run under all four option orders, with the neutral, mild and strong messages."""
    rotated = full[full.rot > 0][["model", "item_id"]].drop_duplicates()
    d = full[full.ptype.isin(PTYPES)].merge(rotated, on=["model", "item_id"])
    return d if len(d) else None


def order_effects(full):
    """Accuracy and C->W when letter bias is averaged out by rotating the options."""
    d = load_order(full)
    if d is None:
        return None, None, None
    models = [m for m in MODELS if m in d.model.unique()]
    t1 = d[d.ptype == "neutral"]
    summ = []
    for m in models:
        n_rot = d[d.model == m].rot.nunique()
        for L in LANGS:
            g, t = d[(d.model == m) & (d.lang == L)], t1[(t1.model == m) & (t1.lang == L)]
            per_item = t.groupby("item_id").correct0.agg(["sum", "count"])
            known = per_item.index[(per_item["sum"] == per_item["count"]) & (per_item["count"] == n_rot)]
            r = {"model": m, "lang": L, "n_items": t.item_id.nunique(), "n_orders": n_rot, "acc_all_orders": t.correct0.mean(),
                 "n_known": len(known)}
            for p in ["mild", "strong"]:
                k = g[(g.ptype == p) & (g.correct0 == 1) & g.item_id.isin(known)]
                r[f"c2w_{p}_known"] = k.c2w.mean() if len(k) else np.nan
            summ.append(r)
    summ = pd.DataFrame(summ)

    # paired contrast on items known (correct under every order) in both English and L: item-level C->W share
    tests = []
    for m in models:
        n_rot = d[d.model == m].rot.nunique()
        acc = t1[t1.model == m].groupby(["lang", "item_id"]).correct0.agg(["sum", "count"])
        known = {L: set(acc.loc[L].index[(acc.loc[L]["sum"] == n_rot) & (acc.loc[L]["count"] == n_rot)]) for L in LANGS}
        for p in ["mild", "strong"]:
            item = d[(d.model == m) & (d.ptype == p)].groupby(["lang", "item_id"]).c2w.mean()
            for L in ["hi", "hg"]:
                both = sorted(known["en"] & known[L])
                diff = (item.loc[L].reindex(both) - item.loc["en"].reindex(both)).to_numpy()
                n = len(diff)
                lo, hi = bootstrap_ci(diff)
                pos, neg = int((diff > 0).sum()), int((diff < 0).sum())
                tests.append({"model": m, "ptype": p, "contrast": f"{L}-en", "n_items": n,
                              "c2w_en": item.loc["en"].reindex(both).mean() if n else np.nan,
                              "c2w_L": item.loc[L].reindex(both).mean() if n else np.nan,
                              "delta": diff.mean() if n else np.nan, "ci_lo": lo, "ci_hi": hi,
                              "items_L_higher": pos, "items_en_higher": neg,
                              "p_sign": binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else 1.0})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = multipletests(tests.p_sign, method="holm")[1]

    reg = []
    c = reg_data(d)
    for m in models:
        for p in ["mild", "strong"]:
            g = c[(c.model == m) & (c.ptype == p)]
            reg += fit_or(g, "c2w ~ C(lang)", model=m, ptype=p, spec="unadjusted")[0]
            reg += fit_or(g, "c2w ~ C(lang) + conf_logit", model=m, ptype=p, spec="confidence-adjusted")[0]
    return summ, tests, pd.DataFrame(reg)


def latex_order(summ, tests):
    out = [r"\begin{tabular}{llccccc}", r"\toprule",
           r"Model & Language & Acc$_0$ & Known & \multicolumn{2}{c}{C$\to$W, known items} & $\Delta$ strong vs.\ EN [95\% CI]\\",
           r"\cmidrule(lr){5-6}", r" & & (4 orders) & items & Mild & Strong & \\", r"\midrule"]
    for i, m in enumerate(summ.model.unique()):
        if i:
            out.append(r"\midrule")
        for j, L in enumerate(LANGS):
            r = summ[(summ.model == m) & (summ.lang == L)].iloc[0]
            t = tests[(tests.model == m) & (tests.ptype == "strong") & (tests.contrast == f"{L}-en")]
            dl = "--" if not len(t) else f"{t.iloc[0].delta:+.3f} [{t.iloc[0].ci_lo:+.2f}, {t.iloc[0].ci_hi:+.2f}]"
            out.append(f"{MODELS[m] if j == 0 else ''} & {LANGS[L]} & {r.acc_all_orders:.3f} & {r.n_known} & "
                       f"{r.c2w_mild_known:.3f} & {r.c2w_strong_known:.3f} & {dl}\\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PTAB / "order.tex").write_text("\n".join(out) + "\n")


def translation_validation():
    """Ratings of 50 Hinglish translations (data/hinglish/validation_sample.csv): counts per rater, agreement if two."""
    f = ROOT / "data/hinglish/validation_sample.csv"
    if not f.exists():
        return None
    v = pd.read_csv(f)
    out = {}
    for q in ["meaning", "natural"]:
        out[q] = {}
        for r in ["rater1", "rater2"]:
            x = v[f"{r}_{q}"].dropna().astype(int)
            if len(x):
                out[q][r] = {"n_items": int(len(x)), **{f"n_rated_{k}": int((x == k).sum()) for k in (3, 2, 1)}}
        both = v.dropna(subset=[f"rater1_{q}", f"rater2_{q}"])
        if len(both):  # agreement needs both raters
            a, b = both[f"rater1_{q}"].astype(int), both[f"rater2_{q}"].astype(int)
            po = float((a == b).mean())
            pe = sum(float((a == k).mean() * (b == k).mean()) for k in (1, 2, 3))
            out[q].update({"n_rated_by_both": int(len(both)), "agreement": po,
                           "cohen_kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan")})
    return out if any(out.values()) else None


def readout_check():
    """Written replies (src/check_readout.py) against the stored probability readout, per model.

    The written answer is the first stand-alone letter A-D in the reply (none if the reply has no such letter).
    """
    def letter(text):
        m = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", text)
        return m.group(1) if m else None

    rows = []
    for m in MODELS:
        f = READOUT / f"{m}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, dtype=str, keep_default_na=False)
        t1 = d.drop_duplicates(["item_id", "lang"])
        same0, same1 = t1.gen0.map(letter) == t1.a0, d.gen1.map(letter) == d.a1
        ok0, ok = t1.mass0.astype(float) >= MIN_MASS, d.mass0.astype(float) >= MIN_MASS
        for L in LANGS:
            a, b = t1.lang == L, d.lang == L
            rows.append({"model": m, "lang": L, "n_items": int(a.sum()),
                         "turn1_used_n": int((a & ok0).sum()), "turn1_used_agree": int(same0[a & ok0].sum()),
                         "turn1_excluded_n": int((a & ~ok0).sum()), "turn1_excluded_agree": int(same0[a & ~ok0].sum()),
                         "turn2_used_n": int((b & ok).sum()), "turn2_used_agree": int(same1[b & ok].sum()),
                         "turn2_used_bare_letter": int((d[b & ok].gen1.str.strip().str.len() == 1).sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ LaTeX helpers
def fmt_p(p):
    return "$<$.001" if p < 0.001 else f"{p:.3f}"


def latex_main(desc):
    out = [r"\begin{tabular}{llrcrcccc}", r"\toprule",
           r"Model & Language & $N$ & Acc$_0$ & $n_c$ & \multicolumn{3}{c}{C$\to$W rate} & Acc$_1^{\text{strong}}$\\",
           r"\cmidrule(lr){6-8}", r" & & & & & Neutral & Mild & Strong & \\", r"\midrule"]
    for i, m in enumerate(desc.model.unique()):
        if i:
            out.append(r"\midrule")
        for j, r in desc[desc.model == m].reset_index().iterrows():
            out.append(f"{MODELS[m] if j == 0 else ''} & {LANGS[r.lang]} & {r.n_items} & {r.acc0:.3f} & {r.n_correct0} & "
                       f"{r.c2w_neutral:.3f} & {r.c2w_mild:.3f} & {r.c2w_strong:.3f} & {r.acc1_strong:.3f}\\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PTAB / "main_results.tex").write_text("\n".join(out) + "\n")


def latex_paired(pt):
    out = [r"\begin{tabular}{lllrccrc}", r"\toprule",
           r"Model & Pushback & Contrast & $n$ & C$\to$W$_{\text{EN}}$ & C$\to$W$_{L}$ & $\Delta$ [95\% CI] & $p_{\text{Holm}}$\\",
           r"\midrule"]
    for i, m in enumerate(pt.model.unique()):
        if i:
            out.append(r"\midrule")
        for j, r in pt[pt.model == m].reset_index().iterrows():
            lab = {"hi-en": "HI$-$EN", "hg-en": "HG$-$EN"}[r.contrast]
            out.append(f"{MODELS[m] if j == 0 else ''} & {r.ptype} & {lab} & {r.n_pairs} & {r.c2w_en:.3f} & "
                       f"{r.c2w_L:.3f} & {r.delta:+.3f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] & {fmt_p(r.p_holm)}\\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PTAB / "paired_tests.tex").write_text("\n".join(out) + "\n")


def latex_reg(rg):
    out = [r"\begin{tabular}{llcccc}", r"\toprule",
           r" & & \multicolumn{2}{c}{Unadjusted} & \multicolumn{2}{c}{Confidence-adjusted}\\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
           r"Model & Language & OR [95\% CI] & $p$ & OR [95\% CI] & $p$\\", r"\midrule"]
    names = {**MODELS, "pooled": "All (pooled)"}
    for i, m in enumerate(rg.model.unique()):
        if i:
            out.append(r"\midrule")
        for j, L in enumerate(["hi", "hg"]):
            cells = []
            for spec in ["unadjusted", "confidence-adjusted"]:
                r = rg[(rg.model == m) & (rg.lang == L) & (rg.spec == spec)]
                if len(r):
                    r = r.iloc[0]
                    cells.append(f"{r.OR:.2f} [{r.OR_lo:.2f}, {r.OR_hi:.2f}] & {fmt_p(r.p)}")
                else:
                    cells.append("-- & --")
            out.append(f"{names[m] if j == 0 else ''} & {LANGS[L]} vs.\\ English & {cells[0]} & {cells[1]}\\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PTAB / "regression.tex").write_text("\n".join(out) + "\n")


def latex_reg_by_pushback(rc):
    t = rc[(rc.check == "by pushback") & (rc.spec == "confidence-adjusted")]
    out = [r"\begin{tabular}{llcccc}", r"\toprule",
           r" & & \multicolumn{2}{c}{Mild pushback} & \multicolumn{2}{c}{Strong pushback}\\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
           r"Model & Language & OR [95\% CI] & $p$ & OR [95\% CI] & $p$\\", r"\midrule"]
    for i, m in enumerate(t.model.unique()):
        if i:
            out.append(r"\midrule")
        for j, L in enumerate(["hi", "hg"]):
            cells = []
            for p in ["mild", "strong"]:
                r = t[(t.model == m) & (t.lang == L) & (t.ptype == p)].iloc[0]
                cells.append("0 (no flips) & --" if r.n_flips_L == 0 else
                             f"{r.OR:.2f} [{r.OR_lo:.2f}, {r.OR_hi:.2f}] & {fmt_p(r.p)}")
            out.append(f"{MODELS[m] if j == 0 else ''} & {LANGS[L]} vs.\\ English & {cells[0]} & {cells[1]}\\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    (PTAB / "regression_by_pushback.tex").write_text("\n".join(out) + "\n")


# ------------------------------------------------------------------ figures
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8a8984")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    shutil.copyfile(FIG / f"{name}.pdf", PFIG / f"{name}.pdf")  # the paper folder keeps its own copy


def fig_accuracy(desc):
    models = list(desc.model.unique())
    fig, ax = plt.subplots(figsize=(3.9, 2.1))
    w = 0.26
    for i, L in enumerate(LANGS):
        d = desc[desc.lang == L].set_index("model").loc[models]
        x = np.arange(len(models)) + (i - 1) * w
        ax.bar(x, d.acc0, w - 0.03, color=COLORS[L], label=LANGS[L])
        ax.errorbar(x, d.acc0, yerr=[d.acc0 - d.acc0_lo, d.acc0_hi - d.acc0], fmt="none", ecolor="#52514e", lw=0.8, capsize=2)
    ax.axhline(0.25, color="#8a8984", lw=0.8, ls="--")
    ax.text(-0.45, 0.265, "chance", fontsize=7, color="#52514e", ha="left")
    ax.set_xticks(range(len(models)), [MODELS[m] for m in models], fontsize=8)
    ax.set_ylabel("Turn-1 accuracy", fontsize=8)
    ax.set_ylim(0, 1)
    style(ax)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_accuracy")


def fig_flip(desc):
    models = list(desc.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(1.6 * len(models) + 0.2, 2.0), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(PTYPES))
    for ax, m in zip(axes, models):
        for L in LANGS:
            r = desc[(desc.model == m) & (desc.lang == L)].iloc[0]
            y = np.array([r[f"c2w_{p}"] for p in PTYPES])
            lo = np.array([r[f"c2w_{p}_lo"] for p in PTYPES])
            hi = np.array([r[f"c2w_{p}_hi"] for p in PTYPES])
            ax.fill_between(x, lo, hi, color=COLORS[L], alpha=0.12, lw=0)
            ax.plot(x, y, color=COLORS[L], lw=2, marker=MARKERS[L], ms=5, label=LANGS[L])
        ax.set_title(MODELS[m], fontsize=9)
        ax.set_xticks(x, [p.capitalize() for p in PTYPES], fontsize=8)
        ax.set_ylim(0, 1.02)
        style(ax)
    axes[0].set_ylabel("C→W rate (correct → wrong)", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_flip")


def fig_confidence(df):
    d = df[(df.correct0 == 1) & (df.ptype.isin(["mild", "strong"]))].copy()
    bins = [0, 0.6, 0.9, 0.99, 1.0001]
    labels = ["<.6", ".6–.9", ".9–.99", "≥.99"]
    d["bin"] = pd.cut(d.conf0, bins, labels=labels, right=False)
    models = [m for m in MODELS if m in d.model.unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(1.9 * len(models) + 0.3, 2.4), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(labels))
    for ax, m in zip(axes, models):  # one panel per model: pooling would mix models with very different confidence
        for i, L in enumerate(LANGS):
            g = d[(d.model == m) & (d.lang == L)].groupby("bin", observed=False).c2w.agg(["sum", "count"])
            y = np.where(g["count"] >= 10, g["sum"] / g["count"].replace(0, np.nan), np.nan)  # hide sparse bins
            ax.plot(x + (i - 1) * 0.06, y, color=COLORS[L], lw=2, marker=MARKERS[L], ms=5, label=LANGS[L])
        ax.set_title(MODELS[m], fontsize=9)
        ax.set_xticks(x, labels, fontsize=7.5)
        ax.set_xlim(-0.3, len(labels) - 0.7)
        ax.set_ylim(0, 1.02)
        style(ax)
    axes[len(axes) // 2].set_xlabel("Turn-1 probability of the (correct) answer", fontsize=8)
    axes[0].set_ylabel("C→W rate (mild + strong)", fontsize=8)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=7, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig_confidence")


def main():
    everything = load()
    full = everything[everything.mass0 >= MIN_MASS].copy()   # the analysed answers
    df = main_rows(full)
    desc, pt, rg = describe(df), paired_tests(df), regression(df)
    desc.to_csv(TAB / "descriptives.csv", index=False)
    pt.to_csv(TAB / "paired_tests.csv", index=False)
    rg.to_csv(TAB / "regression.csv", index=False)
    lb = letter_bias(df)
    lb.to_csv(TAB / "letter_bias.csv", index=False)
    rc, rc_info = regression_checks(df)
    rc.to_csv(TAB / "regression_checks.csv", index=False)

    # check: the same analysis on all turn-1 answers, including those below MIN_MASS
    unf = main_rows(everything)
    rob, rob_pt, rob_rg = describe(unf), paired_tests(unf), regression(unf)
    rob.to_csv(TAB / "descriptives_all_answers.csv", index=False)
    rob_pt.to_csv(TAB / "paired_tests_all_answers.csv", index=False)
    rob_rg.to_csv(TAB / "regression_all_answers.csv", index=False)
    desc = desc.merge(rob[["model", "lang", "n_items", "share_mass0_lt_0.5"]].rename(
        columns={"n_items": "n_all", "share_mass0_lt_0.5": "share_excluded"}), on=["model", "lang"])
    desc = desc.drop(columns="share_mass0_lt_0.5")
    desc.to_csv(TAB / "descriptives.csv", index=False)

    latex_main(desc)
    latex_paired(pt)
    latex_reg(rg)
    latex_reg_by_pushback(rc)
    fig_accuracy(desc)
    fig_flip(desc)
    fig_confidence(df)

    summary = {"models": sorted(df.model.unique()), "n_rows": len(df),
               "n_items": int(df.item_id.nunique()),
               "rows_per_model": df.groupby("model").size().to_dict(), "min_letter_mass": MIN_MASS,
               "regression_checks": rc_info}
    wr, wt, wreg = wording(full)
    if wr is not None:
        wr.to_csv(TAB / "wording_rates.csv", index=False)
        wt.to_csv(TAB / "wording_tests.csv", index=False)
        wreg.to_csv(TAB / "wording_regression.csv", index=False)
        latex_wording(wr)
        print(wr.round(3).to_string())
        print(wt.round(3).to_string())
        print(wreg.round(3).to_string())
    osum, otest, oreg = order_effects(full)
    if osum is not None:
        osum.to_csv(TAB / "order_summary.csv", index=False)
        otest.to_csv(TAB / "order_tests.csv", index=False)
        oreg.to_csv(TAB / "order_regression.csv", index=False)
        latex_order(osum, otest)
        print(osum.round(3).to_string())
        print(otest.round(3).to_string())
        print(oreg.round(3).to_string())
    tv = translation_validation()
    if tv:
        summary["translation_validation"] = tv
        print(tv)
    rd = readout_check()
    if len(rd):
        rd.to_csv(TAB / "readout_check.csv", index=False)
        print(rd.to_string())
    (ROOT / "results/summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    print(desc.round(3).to_string())
    print(pt.round(3).to_string())
    print(rg.round(3).to_string())
    print(lb.round(2).to_string())
    print(rc.round(3).to_string())
    print("all answers, including those below MIN_MASS")
    print(rob[["model", "lang", "n_items", "acc0", "c2w_neutral", "c2w_mild", "c2w_strong"]].round(3).to_string())
    print(rob_pt.round(3).to_string())
    print(rob_rg.round(3).to_string())


if __name__ == "__main__":
    main()
