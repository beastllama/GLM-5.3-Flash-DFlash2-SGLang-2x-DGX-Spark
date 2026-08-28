#!/usr/bin/env python3
"""Mirror current recipe state to the HF repos. Called by the tuning loop after
each kept rung / milestone. Updates only; never posts discussions."""
import os, shutil
from huggingface_hub import HfApi
tok = open(os.path.expanduser("~/.env.hf")).read().strip().split("=",1)[1]
api = HfApi(token=tok)
base = os.path.expanduser("~/glm53-dflash2-recipe")
# refresh the mirror folder from canonical files
for f in ("RESULTS.md", "LADDER.md"):
    src = os.path.join(base, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(base, "hf_mirror", f))
api.upload_folder(folder_path=os.path.join(base, "hf_mirror"),
                  repo_id="randomllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark",
                  commit_message="sync: latest measured numbers")
print("HF mirror synced")
