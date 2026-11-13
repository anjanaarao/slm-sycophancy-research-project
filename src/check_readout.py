"""Readout check: let the model WRITE its replies and compare them with the probability readout of run.py.

run.py never generates text; it reads the answer from the probabilities of the letter tokens. This
script checks that shortcut on a spread of items (every 8th item of the subset, 26 items): the model
writes its turn-1 reply and, with the stored turn-1 answer in the conversation as in run.py, its
turn-2 reply to each pushback, with greedy decoding. src/analyze.py compares the written letters
with the stored answers.

Usage:  python src/check_readout.py --model gemma3-4b      (needs results/raw/<model>.csv)
Writes: results/readout_check/<model>.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import run as R

OUT = R.ROOT / "results/readout_check"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(R.MODELS))
    ap.add_argument("--model-path", default=None, help="folder with an already downloaded copy of the checkpoint")
    ap.add_argument("--every", type=int, default=8, help="use every N-th item of the subset")
    a = ap.parse_args()

    from mlx_lm import generate, load
    items = R.load_items().to_dict("records")[:: a.every]
    stored = pd.read_csv(R.ROOT / f"results/raw/{a.model}.csv").set_index(["item_id", "lang", "ptype"])
    model, tok = load(a.model_path or R.resolve(*R.MODELS[a.model]))

    def encode(messages):
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, date_string="26 Jul 2024")
        return list(tok.encode(text, add_special_tokens=False))

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, it in enumerate(items, 1):
        for L in R.LANGS:
            turn1 = [{"role": "user", "content": R.question_prompt(it, L)}]
            gen0 = generate(model, tok, prompt=encode(turn1), max_tokens=12)
            a0 = stored.loc[(it["item_id"], L, "neutral"), "a0"]
            for p in R.PTYPES:
                conv = turn1 + [{"role": "assistant", "content": a0}, {"role": "user", "content": R.pushback_prompt(p, L)}]
                gen1 = generate(model, tok, prompt=encode(conv), max_tokens=40)
                s = stored.loc[(it["item_id"], L, p)]
                rows.append({"item_id": it["item_id"], "lang": L, "ptype": p, "gold": it["answer"], "a0": a0,
                             "gen0": gen0, "a1": s.a1, "gen1": gen1, "mass0": s.mass0, "mass1": s.mass1})
        pd.DataFrame(rows).to_csv(OUT / f"{a.model}.csv", index=False)
        print(f"[{a.model}] {k}/{len(items)} items", flush=True)


if __name__ == "__main__":
    main()
