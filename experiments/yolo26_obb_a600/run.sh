#!/usr/bin/env bash
set -Eeuo pipefail
mode=${1:?usage: run.sh probe or run.sh train BATCH [RUN_NAME]}
root="${A600_ROOT:-$HOME/autodl-tmp/yolo_obb_1122}"
workspace="$(dirname "$root")"
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="${A600_PYTHON:-$HOME/.venvs/yolo26-autodl/bin/python}"
weights="${A600_PRETRAINED:-$workspace/weights/yolo26n-obb.pt}"
mkdir -p "$root/runs" "$root/logs"
export YOLO_CONFIG_DIR="$workspace/yolo26-config"
export YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 WANDB_MODE=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# The AMP check can reuse the locally available stock model in this directory.
cd "$workspace"

run_one() {
  local name=$1 batch=$2 epochs=$3
  local gpu_log="$root/logs/$name.gpu.csv" sampler code=0
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,nounits -l 1 > "$gpu_log" &
  sampler=$!
  "$py" -B "$source_dir/train.py" --data-zip "$root/4911_A600_foreign_YOLO_OBB.zip" \
    --dataset-dir "$root/A600" --weights "$weights" --project "$root/runs" \
    --name "$name" --batch "$batch" --epochs "$epochs" || code=$?
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  return "$code"
}

case "$mode" in
  probe)
    for batch in 24 40; do
      name="a600_probe_b$batch"
      if run_one "$name" "$batch" 2 > "$root/logs/$name.log" 2>&1; then code=0; else code=$?; fi
      printf '%s %s\n' "$batch" "$code" | tee -a "$root/logs/probe.status"
    done
    ;;
  train)
    batch=${2:?batch required}
    name=${3:-a600_yolo26n_obb_960_b${batch}}
    [[ "$batch" =~ ^[1-9][0-9]*$ && "$name" =~ ^[A-Za-z0-9_-]+$ ]]
    if run_one "$name" "$batch" 100; then code=0; else code=$?; fi
    printf '%s\n' "$code" > "$root/logs/$name.train.exit"
    if [[ "$code" == 0 ]]; then
      "$py" -B "$source_dir/evaluate.py" --weights "$root/runs/$name/weights/best.pt" \
        --dataset-dir "$root/A600" --output-dir "$root/runs/$name/val_eval" || code=$?
    fi
    printf '%s\n' "$code" > "$root/logs/$name.exit"
    exit "$code"
    ;;
  *) printf 'Unknown mode: %s\n' "$mode" >&2; exit 2 ;;
esac
