"""Evaluate class-aware OBB predictions on the fixed 300-image validation split."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

CONFIDENCE_FLOOR = 0.001
IOU_THRESHOLD = 0.5


def polygon_iou(left, right):
    if not cv2.isContourConvex(left):
        left = cv2.convexHull(left)
    if not cv2.isContourConvex(right):
        right = cv2.convexHull(right)
    intersection, _ = cv2.intersectConvexConvex(left, right)
    union = cv2.contourArea(left) + cv2.contourArea(right) - intersection
    return min(1.0, intersection / union) if union > 0 else 0.0


def for_class(boxes, class_id):
    return [box for box in boxes if box.get("cls", 0) == class_id]


def match_predictions(predictions, truth):
    contours = [np.asarray(points, dtype=np.float32) for points in truth]
    unmatched = set(range(len(contours)))
    for det in predictions:
        if not unmatched:
            break
        polygon = np.asarray(det["points"], dtype=np.float32)
        overlap, index = max((polygon_iou(polygon, contours[i]), i) for i in sorted(unmatched))
        if overlap >= IOU_THRESHOLD:
            unmatched.remove(index)
    return sorted(unmatched)


def foreign_ground_truth(row):
    return [b["points"] if isinstance(b, dict) else b for b in row["ground_truth"]
            if not isinstance(b, dict) or b["cls"] == 0]


def operating_point(rows, budget):
    predictions_by_row = [for_class(row["predictions"], 0) for row in rows]
    negative_peaks = sorted(
        (boxes[0]["score"] for row, boxes in zip(rows, predictions_by_row) if row["negative"] and boxes),
        reverse=True,
    )
    threshold = (math.nextafter(negative_peaks[budget], math.inf)
                 if len(negative_peaks) > budget else CONFIDENCE_FLOOR)
    threshold = max(threshold, CONFIDENCE_FLOOR)
    matched, positive_alarms, positive_matched = 0, 0, 0
    misses, false_alarms = [], []
    for row, boxes in zip(rows, predictions_by_row):
        predictions = [det for det in boxes if det["score"] >= threshold]
        if row["negative"]:
            if predictions:
                false_alarms.append({"id": row["id"], "file": row["file"], "score": predictions[0]["score"]})
            continue
        positive_alarms += bool(predictions)
        truth = foreign_ground_truth(row)
        unmatched = match_predictions(predictions, truth)
        matched += len(truth) - len(unmatched)
        positive_matched += len(truth) > len(unmatched)
        if unmatched:
            misses.append({"id": row["id"], "file": row["file"], "missed_indices": unmatched})
    return {"negative_budget": budget, "confidence": threshold,
            "foreign_detected": matched, "positive_alarms": positive_alarms,
            "positive_matched": positive_matched, "negative_alarms": len(false_alarms),
            "false_alarms": false_alarms, "misses": misses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO
    frames = json.loads((args.dataset_dir / "review/frames.json").read_text())
    val = [frame for frame in frames if frame["split"] == "val"]
    assert (len(val), sum(frame["negative"] for frame in val),
            sum(len(frame["foreign"]) for frame in val)) == (300, 191, 113)
    options = {"imgsz": 960, "device": 0, "conf": CONFIDENCE_FLOOR,
               "iou": 0.7, "batch": 1, "max_det": args.max_det, "agnostic_nms": False, "verbose": False, "end2end": False}
    model = YOLO(str(args.weights), task="obb")
    first = cv2.imread(str(args.dataset_dir / "images/val" / val[0]["name"]))
    for _ in range(10):
        model.predict(first, **options)
    rows = []
    for frame in val:
        photo_path = args.dataset_dir / "images/val" / frame["name"]
        photo = cv2.imread(str(photo_path))
        if photo is None:
            raise ValueError(f"Could not decode {photo_path}")
        label = args.dataset_dir / "labels/val" / f"{photo_path.stem}.txt"
        truth = []
        for line in label.read_text().splitlines():
            cls, *coords = line.split()
            coords = np.asarray(coords, dtype=np.float32).reshape(4, 2)
            truth.append({"cls": int(cls), "points": (coords * [photo.shape[1], photo.shape[0]]).tolist()})
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = model.predict(photo, **options)[0]
        detections = sorted(
            ({"cls": int(cls), "score": float(score), "points": points} for cls, score, points in
             zip(result.obb.cls.tolist(), result.obb.conf.tolist(), result.obb.xyxyxyxy.tolist())),
            key=lambda det: det["score"], reverse=True,
        )
        torch.cuda.synchronize()
        elapsed_ms = 1000 * (time.perf_counter() - start)
        rows.append({"id": frame["id"], "file": frame["name"], "negative": frame["negative"],
                     "ground_truth": truth, "predictions": detections, "milliseconds": elapsed_ms})
    times = sorted(row["milliseconds"] for row in rows)
    summary = {
        "weights": str(args.weights), "predict_settings": options,
        "polygon_iou_threshold": IOU_THRESHOLD, "foreign_total": 113,
        "positive_total": 109, "negative_total": 191,
        "frames_at_max_det": sum(len(r["predictions"]) >= args.max_det for r in rows),
        "timing_ms": {"mean": statistics.mean(times), "p50": statistics.median(times),
                      "p95": times[math.ceil(0.95 * len(times)) - 1]},
        "operating_points": [operating_point(rows, budget) for budget in (0, 1, 2)],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "val_predictions.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "val_operating_points.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
