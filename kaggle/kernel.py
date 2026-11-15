"""Kaggle script: run the experiment for one model on a Kaggle GPU and leave the result files in /kaggle/working.

Set MODEL, EXPS (and ATTACHED = True for a model that is attached to the notebook as a Kaggle Model instead
of downloaded from the Hugging Face Hub), then run with GPU and internet enabled.
"""
import os
import shutil
import subprocess
from pathlib import Path

MODEL = "qwen3-4b"      # llama3.2-3b | qwen3-4b | gemma3-4b
EXPS = ["main"]         # any of "main", "cross", "translit" (see src/run.py)
ATTACHED = False        # False: download the Hugging Face repository pinned in src/run.py
                        # True:  use the Kaggle Model attached to this notebook (found under /kaggle/input)
CODE_DATASET = False    # True: take src/ and data/processed/ from an attached Kaggle dataset instead of GitHub
BRANCH = "main"
GIT = "https://github.com/anjanaarao/slm-sycophancy-research-project.git"


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


repo = Path("/kaggle/working/repo")
if CODE_DATASET:
    found = [p.parent for p in Path("/kaggle/input").rglob("run.py")]   # flat dataset: src/*.py + data/processed/*.csv
    assert len(found) == 1, f"expected one attached copy of the code, found {found}"
    for sub, pat in [("src", "*.py"), ("data/processed", "*.csv")]:
        (repo / sub).mkdir(parents=True)
        for f in found[0].glob(pat):
            shutil.copy(f, repo / sub)
    version = "sha256sum src/*.py data/processed/*.csv"
else:
    sh(f"git clone -q --depth 1 -b {BRANCH} {GIT} {repo}")
    version = "git log -1 --format='commit %H'"
os.chdir(repo)
sh(f"{version} && nvidia-smi --query-gpu=name --format=csv,noheader && pip list 2>/dev/null | grep -i -E '^(torch|transformers|numpy|pandas) '")
extra = ""
if ATTACHED:
    found = sorted(str(p.parent) for p in Path("/kaggle/input").rglob("config.json"))
    assert len(found) == 1, f"expected exactly one attached model, found {found}"
    extra = f" --repo {found[0]}"
for exp in EXPS:
    sh(f"python src/run.py --model {MODEL} --exp {exp}{extra}")
    name = MODEL if exp == "main" else f"{MODEL}_{exp}"
    shutil.copy(f"results/raw/{name}.csv", f"/kaggle/working/{name}.csv")
if "main" in EXPS:
    sh(f"python src/check_readout.py --model {MODEL}{extra}")
    shutil.copy(f"results/readout_check/{MODEL}.csv", f"/kaggle/working/{MODEL}_readout_check.csv")
os.chdir("/kaggle/working")
shutil.rmtree(repo)
print("DONE", flush=True)
