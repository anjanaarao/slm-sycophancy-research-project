#!/bin/sh
# Rebuild the item files, run the experiment and the readout check for the three models, then compute
# all statistics, tables and figures. Needs a GPU (about 1-2 h per model on a T4); the reported runs
# were made on Kaggle with kaggle/kernel.py. Finished work is skipped, so it is safe to stop and re-run.
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
$PY src/prepare_data.py || exit 1
for m in llama3.2-3b qwen3-4b gemma3-4b; do
  $PY src/run.py --model $m || exit 1
  $PY src/check_readout.py --model $m || exit 1
done
$PY src/analyze.py
