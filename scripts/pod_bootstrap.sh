#!/usr/bin/env bash
# Pod bootstrap for suitesmith run 1b. Idempotent; run as root on a fresh Prime pod: bash pod_bootstrap.sh
# Two things are Dom's, done in the pod's web terminal before this runs:
#   ~/.netrc for W&B (machine api.wandb.ai / login user / password <key>)
#   hf auth login  (read token; the step-20 weights are in a private repo)
set -euo pipefail
export PATH="/root/.local/bin:$PATH"
PIN=b2c6fe9aecfdcd83655be3564182e41af4fba515   # last prime-rl commit on CUDA 12.8 (28 Aug 2026); the Lambda pods run driver 570 / CUDA 12.8
echo "== host $(hostname); gpus: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' ';')"
[ -f /root/.netrc ] && grep -q wandb /root/.netrc || { echo "!! /root/.netrc for W&B missing: Dom writes it"; exit 1; }
if [ ! -d /root/prime-rl ]; then
  curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh -o /tmp/install.sh
  (cd /root && bash /tmp/install.sh)
fi
cd /root/prime-rl
git fetch -q --all && git checkout -q "$PIN"
echo "== prime-rl at $(git log --oneline -1 | cut -c1-70)"
uv sync --all-extras 2>&1 | tail -2
if [ ! -d /root/suitesmith ]; then git clone -q https://github.com/dtcolligan/suitesmith.git /root/suitesmith; else (cd /root/suitesmith && git pull -q); fi
echo "== suitesmith at $(cd /root/suitesmith && git log --oneline -1 | cut -c1-70)"
uv pip install -q -e /root/suitesmith
uv run hf auth whoami >/dev/null 2>&1 || { echo "!! not logged in to Hugging Face: Dom runs 'uv run hf auth login' here"; exit 1; }
uv run python -c "import torch, vllm, verifiers, suitesmith; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'gpus; vllm', vllm.__version__, 'verifiers', verifiers.__version__)"
echo "== pulling the step-20 weights"
uv run hf download dtcolligan/suitesmith-qwen3.5-4b-run1-step20 >/dev/null && echo "weights cached"
mkdir -p /root/prime-rl/outputs/run1b-4b
echo "== dry run"
uv run rl @ /root/suitesmith/configs/run1b.toml --dry-run 2>&1 | tail -3
echo "== bootstrap done. Launch: bash /root/suitesmith/scripts/pod_run.sh start"
