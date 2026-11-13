# Do small LLMs give up correct answers more easily in Hindi and Hinglish?

Minor project: a two-turn "Are you sure?" experiment on 208 MMLU questions in aligned
**English / Hindi / Hinglish**, with three small open models (Llama-3.2-3B, Qwen3-4B, Gemma-3-4B).

This repository holds the data, code and raw results of the paper. The paper's LaTeX source is not
included; `src/analyze.py` writes the paper's tables and figures into `paper-springer/`.

## Layout
```
src/prepare_data.py     raw MMLU + MMMLU + Hinglish translations -> item files
src/run.py              the experiment (one model per call, resumable)
src/analyze.py          all statistics, the paper's tables and figures
src/check_readout.py    optional check: the model writes its replies on 26 items (see Notes)
run_all.sh              runs the three scripts in order (~2 h on an M1 Mac)

data/raw/               MMLU test set (English) and MMMLU Hindi test set, as downloaded
data/hinglish/          hg_batch*.tsv   Hinglish translations of the 520 sampled items
                        src_batch*.txt  the English text the translations were made from
data/processed/         items_trilingual.csv  520 aligned items
                        items_subset.csv      the 208 items used (4 per subject, seed 2026)
                        items_meta.json, hinglish_audit.json

results/raw/<model>.csv one row per item x language x pushback type, with A-D probabilities
results/tables/         CSV version of every number in the paper
results/pilot/          free-form pilot (Llama, 139 items), used only to validate the answer readout
results/readout_check/  written replies of each model on 26 items (from check_readout.py)
results/run.log         log of the reported runs (timings)
figures/                paper figures (PDF + PNG)
paper-springer/         created by analyze.py: the paper's LaTeX tables and figure copies
literature/             arXiv / Crossref metadata for the references
```

## Reproduce
Needs an Apple-silicon Mac (the models run with MLX) and Python 3.13.
```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_all.sh
```
Each step can also be run on its own:

| Step | Command | Needs | Time |
|---|---|---|---|
| Item files | `.venv/bin/python src/prepare_data.py` | `data/raw`, `data/hinglish` | seconds |
| Experiment | `.venv/bin/python src/run.py --model llama3.2-3b` (also `qwen3-4b`, `gemma3-4b`) | item files; downloads the model (2-3 GB) on first use | 19 / 43 / 47 min |
| Analysis | `.venv/bin/python src/analyze.py` | `results/raw`, `results/pilot`, `results/readout_check` | seconds |
| Readout check (optional) | `.venv/bin/python src/check_readout.py --model gemma3-4b` | `results/raw`, the model | 15-30 min per model |

The raw results are included, so the analysis and the paper can be rebuilt without running any model.
`run.py` skips work that is already in `results/raw/`; delete a model's CSV to run it again.

What has been checked:
- `prepare_data.py` reproduces the four files in `data/processed/` byte for byte.
- `analyze.py` reproduces every CSV in `results/tables/` and the three LaTeX tables exactly.
- `run.py` is deterministic (no sampling): re-running the first 12 items for Llama gave the stored
  answers and probabilities exactly.

## Models
4-bit MLX checkpoints from the Hugging Face Hub, pinned in `src/run.py`:

| Name | Repository | Revision |
|---|---|---|
| `llama3.2-3b` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | `7f0dc925e0d0afb0322d96f9255cfddf2ba5636e` |
| `qwen3-4b` | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b` |
| `gemma3-4b` | `mlx-community/gemma-3-text-4b-it-4bit` | `4f665a4c50ecfe4ecdc34056ab52fe3e3c4abf9e` |

If a model is already on disk, pass its folder with `--model-path` to skip the download.

## Data sources
- `data/raw/mmlu_en_test.parquet`: MMLU test split, `cais/mmlu` (config `all`) on the Hugging Face Hub.
- `data/raw/mmmlu_HI_IN_test.csv`: MMMLU Hindi test split, `openai/MMMLU` (config `HI_IN`).
- Hinglish translations: made for this project (see the paper, Sec. 3.1).

## Licence
Code, Hinglish translations and results: MIT (see `LICENSE`). MMLU and MMMLU are redistributed in
`data/raw/` under their own MIT licences; please cite the original datasets when you use them.

## Design in one paragraph
For each item and language, the model answers (turn 1); its answer is put back into the chat and
the user pushes back **in the same language** with one of three messages: neutral ("Please state your
final answer again"), mild ("Are you sure?...") or strong ("I am confident that your answer is
wrong..."). Answers are read from the next-token probabilities of A/B/C/D. The main measure is the
**C→W rate**: of the answers that were correct at turn 1, how many become wrong after pushback.
Languages are compared on the same items (paired McNemar tests) and with a logistic regression that
controls for the model's turn-1 confidence.

## Where each number in the paper comes from
| In the paper | File in `results/tables/` |
|---|---|
| Table 1, accuracy, C→W rates, flip and correction rates, probabilities | `descriptives.csv` |
| Table 2, paired language contrasts | `paired_tests.csv` |
| Table 3, regression | `regression.csv` |
| Table 4, regression by pushback type; leave-one-model-out; confidence-scale checks | `regression_checks.csv` |
| Answer-letter bias | `letter_bias.csv` |
| Low-letter-mass robustness check | `descriptives_mass_ge_0.5.csv` |
| Readout checks | `pilot_agreement.csv`, `readout_check.csv` |

Interaction tests and the share of answers at the confidence ceiling are in `results/summary.json`.

## Notes
- `run.py` reads answers from letter probabilities and never generates text. `check_readout.py`
  lets each model write its replies on every 8th item and `analyze.py` compares the two. It is not
  part of `run_all.sh`.
- The Hinglish audit in `prepare_data.py` uses the English word list `/usr/share/dict/words`
  (present on macOS). On a system without it, the code-mixing numbers in `hinglish_audit.json` change.
- `results/pilot/` holds the output of an earlier free-form version of the experiment
  (`freeform_pipeline.ipynb` contains that code). It is not part of the pipeline; `analyze.py` only
  reads its CSV to compute the readout-agreement figures quoted in the paper.
