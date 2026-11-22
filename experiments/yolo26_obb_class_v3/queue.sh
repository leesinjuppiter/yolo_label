#!/usr/bin/env bash
set -Eeuo pipefail
root="${DATASET_ROOT:-$HOME/autodl-tmp/yolo_obb_1122}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code=0
for variant in A600 A779; do
  printf '%s %s\n' "$(date -Is)" "$variant"
  name="${variant,,}_yolo26n_obb_960_b24_val_v3"
  bash "$here/run.sh" "$variant" > "$root/logs/$name.console.log" 2>&1 || code=$?
done
printf '%s\n' "$code" > "$root/logs/val_v3_queue.exit"
exit "$code"
