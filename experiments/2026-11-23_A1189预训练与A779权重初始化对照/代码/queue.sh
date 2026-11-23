#!/usr/bin/env bash
set -Eeuo pipefail
root="$HOME/autodl-tmp/yolo_obb_1122"
here="$(cd "$(dirname "$0")" && pwd)"
workspace="$(dirname "$root")"
py="$HOME/.venvs/yolo26-autodl/bin/python"
stage="$root/a1189_initialization_20261123"
batch=24
workers=12
export YOLO_CONFIG_DIR="$workspace/yolo26-config"
export YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 WANDB_MODE=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$root/runs" "$root/logs"
cd "$workspace"

finish() { printf '%s\n' "$?" > "$root/logs/a1189_initialization_20261123.exit"; }
trap finish EXIT
"$py" -B "$here/prepare.py" "$root"
for mode in official a779_best; do
  run="a1189_yolo26n_obb_960_b24_${mode}_val_v3"
  logfile="$root/logs/$run"
  test ! -e "$root/runs/$run"
  echo "$(date -Is) TRAIN $run"
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then echo 'GPU busy'; exit 4; fi
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,nounits -l 1 > "$logfile.gpu.csv" &
  sampler=$!
  status=0
  "$py" -B "$here/train.py" --init "$mode" --inputs "$stage/inputs.json" --project "$root/runs" \
    --name "$run" --batch "$batch" --workers "$workers" > "$logfile.console.log" 2>&1 || status=$?
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  printf '%s\n' "$status" > "$logfile.train.exit"
  if (( status != 0 )); then tail -n 26 "$logfile.console.log"; exit "$status"; fi
done
for mode in official a779_best; do
  run="a1189_yolo26n_obb_960_b24_${mode}_val_v3"
  echo "$(date -Is) EVAL $run"
  "$py" -B "$here/select_checkpoint.py" "$root/runs/$run" "$stage/A1189" \
    > "$root/logs/$run.eval.log" 2>&1
done
echo "$(date -Is) COMPLETE A1189"
