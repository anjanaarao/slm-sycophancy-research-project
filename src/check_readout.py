"""Readout check: let the model WRITE its replies and compare them with the probability readout of run.py.

run.py never generates text; it reads the answer from the probabilities of the letter tokens. This
script checks that shortcut on a spread of items (every 8th item, 65 items, original option order):
the model writes its turn-1 reply and, with the stored turn-1 answer in the conversation as in
run.py, its turn-2 reply to the neutral, mild and strong pushback, with greedy decoding.
src/analyze.py compares the written letters with the stored answers.

Usage:  python src/check_readout.py --model qwen3-4b      (needs results/raw/<model>.csv)
Writes: results/readout_check/<model>.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

import run as R

OUT = R.ROOT / "results/readout_check"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(R.MODELS))
    ap.add_argument("--repo", default=None, help="as in run.py")
    ap.add_argument("--every", type=int, default=8, help="use every N-th item")
    a = ap.parse_args()

    items = R.load_items().to_dict("records")[:: a.every]
    raw = pd.read_csv(R.ROOT / f"results/raw/{a.model}.csv")
    stored = raw[raw.rot == 0].set_index(["item_id", "lang", "ptype"])
    repo, revision, dtype = R.MODELS[a.model]
    sc = R.Scorer(a.repo or repo, None if a.repo else revision, dtype)
    torch = sc.torch

    def write(messages, max_new_tokens):
        ids = torch.tensor([sc.encode(messages)], device=sc.device)
        with torch.no_grad():
            out = sc.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=sc.tok.eos_token_id)
        return sc.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, it in enumerate(items, 1):
        for L in R.LANGS:
            turn1 = [{"role": "user", "content": R.question_prompt(it, L)}]
            gen0 = write(turn1, 12)
            a0 = stored.loc[(it["item_id"], L, "neutral"), "a0"]
            for p in R.PTYPES:
                conv = turn1 + [{"role": "assistant", "content": a0}, {"role": "user", "content": R.pushback_prompt(p, L)}]
                s = stored.loc[(it["item_id"], L, p)]
                rows.append({"item_id": it["item_id"], "lang": L, "ptype": p, "gold": it["answer"], "a0": a0,
                             "gen0": gen0, "a1": s.a1, "gen1": write(conv, 40), "mass0": s.mass0, "mass1": s.mass1})
        if k % 10 == 0 or k == len(items):
            pd.DataFrame(rows).to_csv(OUT / f"{a.model}.csv", index=False)
            print(f"[{a.model}] readout check {k}/{len(items)} items", flush=True)


if __name__ == "__main__":
    main()
