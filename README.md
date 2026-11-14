# Do small LLMs give up correct answers more easily in Hindi and Hinglish?

Minor project: a two-turn "Are you sure?" experiment on 520 MMLU questions in aligned
**English / Hindi / Hinglish**, with three small open models (Llama-3.2-3B, Qwen3-4B, Gemma-3-4B),
seven pushback messages per language and four orders of the answer options.

This repository holds the data, code and raw results of the paper. The paper's LaTeX source is not
included; `src/analyze.py` writes the paper's tables and figures into `paper-springer/`.

## Layout
```
src/prepare_data.py     raw MMLU + MMMLU + Hinglish translations -> item files
src/run.py              the experiment (one model per call, resumable)
src/check_readout.py    the model writes its replies on 65 items, to check the answer readout
src/analyze.py          all statistics, the paper's tables and figures
run_all.sh              runs everything in order
kaggle/kernel.py        the Kaggle notebook script that produced the reported results

data/raw/               MMLU test set (English) and MMMLU Hindi test set, as downloaded
data/hinglish/          hg_batch*.tsv          Hinglish translations of the 520 items
                        src_batch*.txt         the English text the translations were made from
                        validation_sample.csv  50 items for two human raters (ratings to be filled in)
data/processed/         items_trilingual.csv   the 520 aligned items used in the experiment
                        items_meta.json, hinglish_audit.json

results/raw/<model>.csv            one row per item x language x option order x pushback message
results/readout_check/<model>.csv  written replies on 65 items
results/tables/                    CSV version of every number in the paper
results/summary.json               interaction tests, share of answers at the confidence ceiling
figures/                           paper figures (PDF + PNG)
paper-springer/                    created by analyze.py: the paper's LaTeX tables and figure copies
literature/                        arXiv / Crossref metadata for the references
```

## Reproduce
The raw results are included, so the analysis can be rebuilt without running any model:
```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/analyze.py
```
Running the models needs a GPU with about 12 GB of memory.

| Step | Command | Time |
|---|---|---|
| Item files | `python src/prepare_data.py` | seconds |
| Experiment | `python src/run.py --model llama3.2-3b` (also `qwen3-4b`, `gemma3-4b`) | 1.2 / 1.7 / 5.5 h on a Tesla T4, including the readout check |
| Readout check | `python src/check_readout.py --model llama3.2-3b` | minutes |
| Analysis | `python src/analyze.py` | under a minute |

`run.py` skips work that is already in `results/raw/`; delete a model's CSV to run it again.

**On Kaggle.** Create a script notebook from `kaggle/kernel.py`, set `MODEL` at the top, switch on
the GPU and internet, and run it. The notebook clones this repository, runs the experiment and the
readout check, and leaves `<model>.csv` and `<model>_readout_check.csv` in its output. Put them in
`results/raw/<model>.csv` and `results/readout_check/<model>.csv`.

## Models
Original weights in half precision (float16; bfloat16 for Gemma), pinned in `src/run.py`:

| Name | Source | Revision |
|---|---|---|
| `llama3.2-3b` | `unsloth/Llama-3.2-3B-Instruct` (Hugging Face) | `006f5dcd1393c3add266de40994ba96225e9689d` |
| `qwen3-4b` | `Qwen/Qwen3-4B-Instruct-2507` (Hugging Face) | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| `gemma3-4b` | `google/gemma-3-4b-it` (Hugging Face) | `093f9f388b31de276ce2de164bdc2081324b9767` |

- Meta's own Llama repository needs an access request. The `unsloth` copy can be downloaded freely
  and its weight files have the same SHA-256 hashes.
- Gemma needs its licence accepted. The reported Gemma run used the same model from Kaggle Models
  (`google/gemma-3/transformers/gemma-3-4b-it/1`), attached to the notebook (`ATTACHED = True`).
- `--repo <folder>` makes `run.py` and `check_readout.py` load a model from a local folder.

## Data sources
- `data/raw/mmlu_en_test.parquet`: MMLU test split, `cais/mmlu` (config `all`) on the Hugging Face Hub.
- `data/raw/mmmlu_HI_IN_test.csv`: MMMLU Hindi test split, `openai/MMMLU` (config `HI_IN`).
- Hinglish translations: made for this project (see the paper, Sec. 3.1).

## Licence
Code, Hinglish translations and results: MIT (see `LICENSE`). MMLU and MMMLU are redistributed in
`data/raw/` under their own MIT licences; please cite the original datasets when you use them.

## Design in one paragraph
For each item and language, the model answers (turn 1); its answer is put back into the chat and
the user pushes back **in the same language**. The main experiment uses three messages: neutral
("Please state your final answer again"), mild ("Are you sure?...") and strong ("I am confident that
your answer is wrong..."). Two paraphrases of the mild and of the strong message test how much
depends on the wording, and three rotations of the answer options test how much depends on
answer-letter bias. Answers are read from the next-token probabilities of A/B/C/D. The main measure
is the **C→W rate**: of the answers that were correct at turn 1, how many become wrong after
pushback. Languages are compared on the same items (paired McNemar tests) and with a logistic
regression that controls for the model's turn-1 confidence.

**Which answers are analysed.** A turn-1 answer is used only if the model puts at least 0.5 of its
first-token probability on the letters A-D (`MIN_MASS` in `src/analyze.py`). Below that the model
wants to start its reply with other text, and the letter read from the probabilities is usually not
the answer it would write. This removes about half of Llama's Hindi and Hinglish answers and almost
nothing else. The `*_all_answers.csv` tables give the results without this rule.

## Where each number in the paper comes from
| In the paper | File in `results/tables/` |
|---|---|
| Table 1, accuracy, C→W rates, flip and correction rates, excluded answers | `descriptives.csv` |
| Table 2, paired language contrasts | `paired_tests.csv` |
| Table 3, regression | `regression.csv` |
| Table 4, regression by pushback type; leave-one-model-out; confidence-scale checks | `regression_checks.csv` |
| Table 5, wordings | `wording_rates.csv`, `wording_tests.csv`, `wording_regression.csv` |
| Table 6, option orders | `order_summary.csv`, `order_tests.csv`, `order_regression.csv` |
| Answer-letter bias | `letter_bias.csv` |
| Readout check | `readout_check.csv` |
| Results without the letter-probability rule | `descriptives_all_answers.csv`, `paired_tests_all_answers.csv`, `regression_all_answers.csv` |

## Notes
- `run.py` reads answers from letter probabilities and never generates text. `check_readout.py`
  lets each model write its replies on every 8th item and `analyze.py` compares the two.
- Items whose options refer to other options ("both A and B", "all of the above") are not rotated
  (23 of 520).
- The Hinglish audit in `prepare_data.py` uses the English word list `/usr/share/dict/words`
  (present on macOS). On a system without it, the code-mixing numbers in `hinglish_audit.json` change.
- `data/processed/items_subset.csv` is a 208-item subset used by an earlier version of the study.
  It is no longer used.
