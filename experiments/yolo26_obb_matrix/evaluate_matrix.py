"""Calibrate on val_v3, freeze the model and thresholds, then evaluate T724."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
import time

import cv2
import numpy as np

from evaluate import (CONFIDENCE_FLOOR, foreign_ground_truth, match_predictions,
                      operating_point, polygon_iou)


def checksum(path):
    with path.open("rb") as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def evaluate_checkpoint(weight, root, split):
    import torch
    from ultralytics import YOLO

    records = json.loads((root / "review/frames.json").read_text())
    frames = [r for r in records if r["split"] == split]
    expected = (300, 191, 113) if split == "val" else (724, 476, 256)
    assert (len(frames), sum(r["negative"] for r in frames), sum(len(r["foreign"]) for r in frames)) == expected
    options = {"imgsz": 960, "device": 0, "conf": CONFIDENCE_FLOOR, "iou": .70, "max_det": 300,
               "batch": 1, "end2end": False, "agnostic_nms": False, "classes": [0], "verbose": False}
    model = YOLO(str(weight), task="obb")

    def read(frame):
        p = root / f"images/{split}" / frame["name"]
        im = cv2.imread(str(p))
        if im is None:
            raise ValueError(f"Cannot read image: {p}")
        return im

    first = read(frames[0])
    for _ in range(10):
        model.predict(first, **options)
    rows = []
    for frame in frames:
        image = read(frame)
        label = root / f"labels/{split}" / f"{Path(frame['name']).stem}.txt"
        truth = []
        for line in label.read_text().splitlines():
            cls, *coords = line.split()
            if cls == "0":
                truth.append((np.asarray(coords, dtype=np.float32).reshape(4,2) *
                              [image.shape[1], image.shape[0]]).tolist())
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = model.predict(image, **options)[0].obb
        scores = result.conf.tolist()
        polygons = result.xyxyxyxy.tolist()
        torch.cuda.synchronize()
        elapsed = 1000*(time.perf_counter() - start)
        detections = [{"score": score, "points": polygon} for score, polygon in zip(scores, polygons)]
        detections.sort(key=lambda box: box["score"], reverse=True)
        rows.append({"id": frame["id"], "file": frame["name"], "negative": frame["negative"],
                     "ground_truth": truth, "predictions": detections, "milliseconds": elapsed})
    return rows, options


def report_at_threshold(rows, threshold):
    tp = fp = fn = negative = positive = matched_images = 0
    missed, false_alarms = [], []
    for row in rows:
        predictions = [d for d in row["predictions"] if d["score"] >= threshold]
        truth = foreign_ground_truth(row)
        unmatched = match_predictions(predictions, truth)
        correct = len(truth)-len(unmatched)
        tp += correct
        fn += len(unmatched)
        fp += len(predictions)-correct
        matched_images += bool(correct)
        if row["negative"]:
            negative += bool(predictions)
            if predictions:
                false_alarms.append({"id": row["id"], "file": row["file"], "score": predictions[0]["score"]})
        else:
            positive += bool(predictions)
            if unmatched:
                missed.append({"id": row["id"], "file": row["file"], "missed_indices": unmatched})
    return {"confidence": threshold, "tp": tp, "fp": fp, "fn": fn,
            "instance_recall": tp/(tp+fn), "instance_precision": tp/(tp+fp) if tp+fp else 0,
            "negative_false_alarms": negative, "negative_total": 476, "negative_false_alarm_rate": negative/476,
            "positive_alarms": positive, "positive_matched": matched_images, "positive_total": 248,
            "misses": missed, "false_alarms": false_alarms}


def average_precision(rows):
    """Compute 101-point interpolated foreign AP from rotated polygon matches."""
    total = sum(len(r["ground_truth"]) for r in rows)
    overlap = []
    for i, row in enumerate(rows):
        truth = [np.asarray(p, dtype=np.float32) for p in row["ground_truth"]]
        for d in row["predictions"]:
            pred = np.asarray(d["points"], dtype=np.float32)
            ranked = sorted(((polygon_iou(pred, p), j) for j,p in enumerate(truth)), reverse=True)
            overlap.append((d["score"], i, ranked))
    overlap.sort(key=lambda item: item[0], reverse=True)
    ap = {}
    for index in range(50,100,5):
        iou = index/100
        matched = set()
        true_positives = []
        for _, image, ranked in overlap:
            match = next((j for value,j in ranked if value >= iou and (image,j) not in matched), None)
            if match is not None:
                matched.add((image,match))
            true_positives.append(match is not None)
        count = np.cumsum(true_positives)
        precision = count/np.arange(1,len(count)+1)
        envelope = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
        samples = np.searchsorted(count/total, np.linspace(0,1,101), side="left")
        ap[str(index)] = float(np.mean(np.append(envelope, 0)[samples]))
    return {"ap50": ap["50"], "ap50_95": sum(ap.values())/len(ap), "ap_by_iou": ap}


def write(path, data):
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(data, indent=2)+"\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("dataset", type=Path)
    p.add_argument("test", type=Path)
    args = p.parse_args()
    weights = args.run / "weights"
    out = args.run / "matrix_eval"
    out.mkdir(exist_ok=True)
    fixed = out / "threshold_lock.json"
    if not fixed.exists():
        candidates = [weights/"best.pt", weights/"last.pt", *sorted(weights.glob("epoch*.pt"))]
        results = []
        previous = {}
        for checkpoint in candidates:
            digest = checksum(checkpoint)
            if digest not in previous:
                rows, _ = evaluate_checkpoint(checkpoint, args.dataset, "val")
                previous[digest] = [operating_point(rows,b) for b in (0,1,2)]
            results.append({"checkpoint": checkpoint.name, "sha256": digest, "operating_points": previous[digest]})
            print("CALIBRATED", checkpoint.name, [x["foreign_detected"] for x in previous[digest]], flush=True)
        chosen = max(results, key=lambda row: tuple(row["operating_points"][i]["foreign_detected"] for i in (1,0,2)))
        write(out/"val_checkpoint_comparison.json", results)
        write(fixed, chosen)
    selected = json.loads(fixed.read_text())
    checkpoint = weights / selected["checkpoint"]
    assert checksum(checkpoint) == selected["sha256"]
    rows, options = evaluate_checkpoint(checkpoint, args.test, "test")
    times = sorted(r["milliseconds"] for r in rows)
    report = {"checkpoint": checkpoint.name, "sha256": selected["sha256"], "predict_settings": options,
              "ap": average_precision(rows), "frames_at_max_det": sum(len(r["predictions"])>=options["max_det"] for r in rows),
              "timing_ms": {"p50": median(times), "p95": times[math.ceil(.95*len(times))-1]},
              "test_at_val_thresholds": [
                  {"validation_negative_budget": point["negative_budget"], **report_at_threshold(rows, point["confidence"])}
                  for point in selected["operating_points"]]}
    write(out/"test_predictions.json", rows)
    write(out/"test_summary.json", report)
    main_point = report["test_at_val_thresholds"][1]
    print("T724_TEST_COMPLETE", checkpoint.name, f"{main_point['tp']}/256",
          f"{main_point['negative_false_alarms']}/476", report["ap"]["ap50"], flush=True)


if __name__ == "__main__":
    main()
