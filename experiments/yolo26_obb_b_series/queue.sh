#!/usr/bin/env bash
set -Eeuo pipefail
root="$HOME/autodl-tmp/yolo_obb_1122"
here="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$(cd "$here/../.." && pwd)"
workspace="$(dirname "$root")"
py="$HOME/.venvs/yolo26-autodl/bin/python"
stage="$root/b_series_20261123"
export YOLO_CONFIG_DIR="$workspace/yolo26-config"
export YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 WANDB_MODE=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$root/runs" "$root/logs"
cd "$workspace"

trap 'printf "%s\n" "$?" > "$root/logs/b_series_20261123.exit"' EXIT
"$py" -B "$here/prepare.py" "$root"
"$py" -B "$here/check_evaluator.py" "$root/runs/a1189_yolo26n_obb_960_b24_official_val_v3/checkpoint_eval"

for variant in B600 B1189; do
  run="${variant,,}_yolo26n_obb_960_b24_official_val_v3"
  logfile="$root/logs/$run"
  directory="$root/runs/$run"
  dataset="$stage/$variant"
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then echo 'GPU busy'; exit 4; fi
  if [[ -f "$logfile.train.exit" && $(cat "$logfile.train.exit") == 0 ]]; then
    test -s "$directory/weights/best.pt"
    test -s "$directory/results.csv"
  else
    test ! -e "$directory"
    workers=8
    save_period=-1
    if [[ "$variant" == B1189 ]]; then workers=12; save_period=10; fi
    echo "$(date -Is) TRAIN $run"
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,nounits -l 1 > "$logfile.gpu.csv" &
    sampler=$!
    status=0
    "$py" -B "$here/train.py" --init official --inputs "$dataset/inputs.json" --project "$root/runs" \
      --name "$run" --batch 24 --workers "$workers" --save-period "$save_period" > "$logfile.console.log" 2>&1 || status=$?
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
    printf '%s\n' "$status" > "$logfile.train.exit"
    if (( status != 0 )); then tail -n 26 "$logfile.console.log"; exit "$status"; fi
  fi
  echo "$(date -Is) EVALUATE $run"
  a1189="$root/runs/a1189_yolo26n_obb_960_b24_official_val_v3/checkpoint_eval"
  a_run="$root/runs/a600_yolo26n_obb_960_b24_val_v3"
  options=()
  if [[ "$variant" == B600 ]]; then
    options=(--native-only)
    a_best="$a_run/val_eval/val_operating_points.json"
  else
    a_best="$a1189/best/val_operating_points.json"
  fi
  (
    test -s "$a_best"
    "$py" -B "$here/select_checkpoint.py" "$directory" "$dataset" "${options[@]}"
    "$py" -B "$here/diagnostics.py" --b "$directory/checkpoint_eval/best" --a "$a_best" --dataset "$dataset"
    choice="$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["evaluation_dir"])' "$directory/checkpoint_eval/summary.json")"
    if [[ "$variant" == B1189 ]]; then
      "$py" -B "$here/diagnostics.py" --b "$directory/checkpoint_eval/$choice" --a "$a1189/epoch50/val_operating_points.json" --dataset "$dataset"
      mv -- "$directory/checkpoint_eval/$choice/paired_diagnostics.json" "$directory/checkpoint_eval/$choice/recall_paired_diagnostics.json"
    fi
    if "$py" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["frames_at_max_det"])' "$directory/checkpoint_eval/$choice/val_operating_points.json"; then
      "$py" -B "$here/evaluate.py" --weights "$directory/weights/$choice.pt" \
        --dataset-dir "$dataset" --output-dir "$directory/max_det_1000" --max-det 1000
    fi
  ) > "$logfile.eval.log" 2>&1 || { tail -n 25 "$logfile.eval.log"; exit 1; }
  echo "$(date -Is) COMPLETE $run"
done
