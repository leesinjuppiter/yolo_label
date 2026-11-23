#!/usr/bin/env bash
set -Eeuo pipefail
root="$HOME/autodl-tmp/yolo_obb_1122"
workspace="$(dirname "$root")"
here="$(cd "$(dirname "$0")" && pwd)"
py="$HOME/.venvs/yolo26-autodl/bin/python"
stage="$root/obb_matrix_20261123"
logs="$root/logs"
export PYTHONPATH="$here/../yolo26_obb_b_series"
export YOLO_CONFIG_DIR="$workspace/yolo26-config"
export YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 WANDB_MODE=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$stage/runs" "$stage/calibration" "$logs"
exec 8>"$stage/queue.lock"
flock -n 8 || { echo 'Matrix queue already running'; exit 3; }
trap 'printf "%s\n" "$?" > "$logs/obb_matrix_20261123.exit"' EXIT
printf 'active\n' > "$logs/obb_matrix_20261123.exit"
cd "$workspace"
"$py" -B "$here/audit.py" "$root" "$stage"
for dataset in A600 B600 A1189 B1189; do
  dataset_dir="$("$py" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["datasets"][sys.argv[2]]["path"])' "$stage/inputs.json" "$dataset")"
  for scale in s m l x; do
    name="${dataset,,}_yolo26${scale}_obb_960"
    run="$stage/runs/$name"
    log="$logs/obb_matrix_$name"
    if [[ -s "$run/matrix_eval/test_summary.json" ]]; then
      "$py" -B "$here/summary.py" "$stage"
      continue
    fi
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then echo 'GPU busy'; exit 4; fi
    if [[ ! -f "$run/training.done" ]]; then
      batch=.82
      calibration="$stage/calibration/$scale.batch"
      if [[ -s "$calibration" ]]; then batch="$(cat "$calibration")"; fi
      echo "$(date -Is) TRAIN $name batch=$batch"
      nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,nounits -l 1 >> "$log.gpu.csv" &
      sampler=$!
      status=0
      "$py" -B "$here/train.py" "$stage/inputs.json" "$dataset" "$scale" "$run" --batch "$batch" 2>&1 | tee -a "$log.train.log" || status=$?
      kill "$sampler" 2>/dev/null || true
      wait "$sampler" 2>/dev/null || true
      (( status == 0 )) || exit "$status"
      touch "$run/training.done"
      "$py" -B -c 'import json,sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text().splitlines()[-1])["batch"])' \
        "$run/epoch_performance.jsonl" > "$calibration.part"
      mv -- "$calibration.part" "$calibration"
    fi
    echo "$(date -Is) TEST $name"
    "$py" -B "$here/evaluate_matrix.py" "$run" "$dataset_dir" "$stage/T724" 2>&1 | tee -a "$log.eval.log"
    "$py" -B "$here/summary.py" "$stage"
  done
done
echo "$(date -Is) OBB_MATRIX_COMPLETE 16/16"
