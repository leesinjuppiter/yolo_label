#!/usr/bin/env bash
set -Eeuo pipefail
root="${DATASET_ROOT:-$HOME/autodl-tmp/yolo_obb_1122}"
workspace="$(dirname "$root")"
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="${TRAIN_PYTHON:-$HOME/.venvs/yolo26-autodl/bin/python}"
weights="${TRAIN_PRETRAINED:-$workspace/weights/yolo26n-obb.pt}"
mkdir -p "$root/runs" "$root/logs"
export YOLO_CONFIG_DIR="$workspace/yolo26-config"
export YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 WANDB_MODE=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$workspace"

variant=${1:?usage: run.sh A600 or run.sh A779}
[[ "$variant" == A600 || "$variant" == A779 ]]
name="${variant,,}_yolo26n_obb_960_b24_val_v3"
log="$root/logs/$name"
if [[ -e "$root/runs/$name" ]]; then echo "Run already exists: $name"; exit 3; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then echo 'GPU busy'; exit 4; fi
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,nounits -l 1 > "$log.gpu.csv" &
sampler=$!
code=0
"$py" -B "$source_dir/train.py" --variant "$variant" --revision "$root/class_v3/revision.json" \
  --weights "$weights" --project "$root/runs" --name "$name" --batch 24 || code=$?
kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true
printf '%s\n' "$code" > "$log.train.exit"
if [[ "$code" == 0 ]]; then
  "$py" -B "$source_dir/evaluate.py" --weights "$root/runs/$name/weights/best.pt" \
    --dataset-dir "$root/class_v3/$variant" --output-dir "$root/runs/$name/val_eval" || code=$?
fi
printf '%s\n' "$code" > "$log.exit"
exit "$code"
