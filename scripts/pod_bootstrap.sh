#!/usr/bin/env bash
# Pod bootstrap for suitesmith run 1b. Idempotent; run as root on a fresh Prime pod: bash pod_bootstrap.sh
# One credential is needed and is Dom's to place: the sandbox runtime needs the Prime API key on the pod
#   (either /root/.prime/config.json copied from the laptop, or PRIME_API_KEY exported in the shell that runs pod_run.sh).
# W&B is off and the step-20 weights are a public Hugging Face repo, so nothing else.
set -euo pipefail
export PATH="/root/.local/bin:$PATH"
PIN=b2c6fe9aecfdcd83655be3564182e41af4fba515   # last prime-rl commit on CUDA 12.8 (28 Aug 2026); the Lambda pods run driver 570 / CUDA 12.8
echo "== host $(hostname); gpus: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' ';')"
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
uv run python -c "import torch, vllm, verifiers, suitesmith; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'gpus; vllm', vllm.__version__, 'verifiers', verifiers.__version__)"
echo "== step-20 weights (public HF repo)"
uv run hf download dtcolligan/suitesmith-qwen3.5-4b-run1-step20 >/dev/null && echo "weights cached"
[ -n "${PRIME_API_KEY:-}" ] || [ -f /root/.prime/config.json ] || { echo "!! no Prime credential on the pod (PRIME_API_KEY or /root/.prime/config.json): sandboxes cannot be created; Dom places it, then rerun"; exit 1; }
mkdir -p /root/prime-rl/outputs/run1b-4b
echo "== dry run"
uv run rl @ /root/suitesmith/configs/run1b.toml --dry-run 2>&1 | tail -3
echo "== bootstrap done. Launch: bash /root/suitesmith/scripts/pod_run.sh start"
