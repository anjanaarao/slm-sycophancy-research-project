"""Kaggle script: run the experiment for one model on a Kaggle GPU and leave the result files in /kaggle/working.

Set MODEL (and ATTACHED = True for a model that is attached to the notebook as a Kaggle Model instead
of downloaded from the Hugging Face Hub), then run with GPU and internet enabled.
"""
import os
import shutil
import subprocess

MODEL = "qwen3-4b"      # llama3.2-3b | qwen3-4b | gemma3-4b
ATTACHED = False        # False: download the Hugging Face repository pinned in src/run.py
                        # True:  use the Kaggle Model attached to this notebook (found under /kaggle/input)
BRANCH = "main"
GIT = "https://github.com/anjanaarao/slm-sycophancy-research-project.git"


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"git clone -q --depth 1 -b {BRANCH} {GIT} /kaggle/working/repo")
os.chdir("/kaggle/working/repo")
sh("git log -1 --format='commit %H' && nvidia-smi --query-gpu=name --format=csv,noheader && pip list 2>/dev/null | grep -i -E '^(torch|transformers|numpy|pandas) '")
extra = ""
if ATTACHED:
    found = sorted(str(p.parent) for p in __import__("pathlib").Path("/kaggle/input").rglob("config.json"))
    assert len(found) == 1, f"expected exactly one attached model, found {found}"
    extra = f" --repo {found[0]}"
sh(f"python src/run.py --model {MODEL}{extra}")
sh(f"python src/check_readout.py --model {MODEL}{extra}")
shutil.copy(f"results/raw/{MODEL}.csv", f"/kaggle/working/{MODEL}.csv")
shutil.copy(f"results/readout_check/{MODEL}.csv", f"/kaggle/working/{MODEL}_readout_check.csv")
shutil.rmtree("/kaggle/working/repo")
print("DONE", flush=True)
