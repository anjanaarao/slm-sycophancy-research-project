#!/bin/sh
# Rebuild the item files, run the experiment for all three models (~2 h in total on an M1 Mac),
# then compute all statistics, tables and figures. Finished work is skipped, so it is safe to re-run.
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python
$PY src/prepare_data.py || exit 1
for m in llama3.2-3b qwen3-4b gemma3-4b; do
  caffeinate -i $PY src/run.py --model $m || exit 1
done
$PY src/analyze.py
