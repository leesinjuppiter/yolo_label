"""Keep a live portable summary without exposing test sample IDs."""
import csv
import json
from pathlib import Path
from statistics import median
import sys


def main():
    stage = Path(sys.argv[1])
    rows = []
    for dataset in ("A600", "B600", "A1189", "B1189"):
        for scale in "smlx":
            name = f"{dataset.lower()}_yolo26{scale}_obb_960"
            run = stage / "runs" / name
            test = run / "matrix_eval/test_summary.json"
            if not test.exists():
                continue
            result = json.loads(test.read_text())
            lock = json.loads((test.parent/"threshold_lock.json").read_text())
            perf = [json.loads(x) for x in (run/"epoch_performance.jsonl").read_text().splitlines()]
            with (run/"results.csv").open(newline="") as stream:
                train = list(csv.DictReader(stream, skipinitialspace=True))
            gpu_file = stage.parent / "logs" / f"obb_matrix_{name}.gpu.csv"
            with gpu_file.open(newline="") as stream:
                gpu = [(int(x[1]),int(x[2])) for x in csv.reader(stream)
                       if len(x)==4 and x[1].strip().isdigit() and x[2].strip().isdigit()]
            point = result["test_at_val_thresholds"][1]
            val = lock["operating_points"][1]
            rows.append({
                "dataset": dataset, "scale": scale, "checkpoint": result["checkpoint"],
                "checkpoint_sha256": result["sha256"], "epochs": len(train), "actual_batch": perf[-1]["batch"],
                "train_minutes": round(float(train[-1]["time"])/60, 3),
                "train_images_per_second": round(median(x["images_per_second"] for x in perf),3),
                "gpu_median_percent": median(x[0] for x in gpu), "gpu_peak_mib": max(x[1] for x in gpu),
                "val_recalled_of_113": val["foreign_detected"], "val_false_alarm_frames_of_191": val["negative_alarms"],
                "locked_confidence": point["confidence"], "test_recalled_of_256": point["tp"],
                "test_recall": point["instance_recall"], "test_precision": point["instance_precision"],
                "test_false_alarm_frames_of_476": point["negative_false_alarms"],
                "test_false_alarm_rate": point["negative_false_alarm_rate"],
                "test_ap50": result["ap"]["ap50"], "test_ap50_95": result["ap"]["ap50_95"],
                "test_p50_ms": result["timing_ms"]["p50"], "test_p95_ms": result["timing_ms"]["p95"],
                "test_frames_at_max_det": result["frames_at_max_det"],
            })
    for name in ("summary.csv","summary.json"):
        path=stage / name
        temp=path.with_suffix(path.suffix+".part")
        if name.endswith(".json"):
            temp.write_text(json.dumps(rows, indent=2)+"\n")
        else:
            with temp.open("w",newline="") as f:
                writer=csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        temp.replace(path)
    print(f"MATRIX_PROGRESS {len(rows)}/16",flush=True)


if __name__ == "__main__":
    main()
