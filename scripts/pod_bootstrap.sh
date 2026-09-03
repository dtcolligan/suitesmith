#!/usr/bin/env bash
# Pod bootstrap for suitesmith run 1b. Idempotent; run as root on a fresh Prime pod: bash pod_bootstrap.sh
# No credentials needed: W&B is off and the step-20 weights arrive by scp at /root/weights/step20 (started from the laptop as soon as the pod is ACTIVE).
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
echo "== step-20 weights (scp from the laptop)"
W=/root/weights/step20/model.safetensors
until [ -f "$W" ] && [ "$(stat -c %s "$W")" = "10350019328" ]; do echo "waiting for $W (have $(stat -c %s "$W" 2>/dev/null || echo 0) of 10350019328 bytes)"; sleep 60; done
ls /root/weights/step20 | tr '\n' ' '; echo
mkdir -p /root/prime-rl/outputs/run1b-4b
echo "== dry run"
uv run rl @ /root/suitesmith/configs/run1b.toml --dry-run 2>&1 | tail -3
echo "== bootstrap done. Launch: bash /root/suitesmith/scripts/pod_run.sh start"
