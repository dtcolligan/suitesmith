#!/usr/bin/env bash
# Start, stop, resume or check run 1b on the pod. Never type these live; the stop pattern must not match the operator's own ssh command.
#   pod_run.sh start | stop | resume | status
set -uo pipefail
export PATH="/root/.local/bin:$PATH"
CFG=/root/suitesmith/configs/run1b.toml; LOG=/root/run1b-run.log   # not inside the run dir: prime-rl refuses to start if outputs/<run> holds any file
cd /root/prime-rl
case "${1:-status}" in
  start)  nohup uv run rl @ "$CFG" > "$LOG" 2>&1 & echo "started pid $!, log $LOG" ;;
  resume) nohup uv run rl @ "$CFG" --resume > "$LOG.resume" 2>&1 & echo "resumed pid $!, log $LOG.resume" ;;
  stop)   pkill -f "[u]v run r[l] @" ; pkill -f "[p]rime_rl" ; sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader ;;
  status) pgrep -fl "[u]v run r[l] @" | head -3; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; tail -2 "$LOG" 2>/dev/null; ls /root/prime-rl/outputs/run1b-4b 2>/dev/null ;;
esac
