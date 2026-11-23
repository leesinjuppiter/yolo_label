"""Compare paired foreign-object outcomes and preview material recall."""
import argparse
from collections import Counter
import json
from pathlib import Path

from dataset import in_region
from evaluate import for_class, match_predictions, operating_point


def material_metrics(rows, foreign_threshold, material_threshold, use_predictions):
    counts = Counter(tp=0, fp=0, fn=0, retained=0)
    for row in rows:
        truth = for_class(row["ground_truth"], 1)
        foreigners = (for_class(row["predictions"], 0) if use_predictions
                      else for_class(row["ground_truth"], 0))
        foreigners = [b for b in foreigners if b.get("score", 1.0) >= foreign_threshold]
        predictions = [b for b in for_class(row["predictions"], 1) if b["score"] >= material_threshold
                       and in_region(b, foreigners, 1, 1, 0.5)]
        missed = len(match_predictions(predictions, [b["points"] for b in truth]))
        tp = len(truth) - missed
        counts.update(tp=tp, fp=len(predictions)-tp, fn=missed, retained=len(predictions))
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    return dict(counts, precision=tp/(tp+fp) if tp+fp else 0, recall=tp/(tp+fn) if tp+fn else 0)


def misses(point):
    return {(r["id"], i) for r in point["misses"] for i in r["missed_indices"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--material-threshold", type=float, default=.25)
    args = parser.parse_args()
    rows = json.loads((args.b / "val_predictions.json").read_text())
    b_summary = json.loads((args.b / "val_operating_points.json").read_text())
    a_summary = json.loads(args.a.read_text())
    a_point, b_point = (data["operating_points"][1] for data in (a_summary, b_summary))
    a_misses, b_misses = misses(a_point), misses(b_point)
    frames = [r for r in json.loads((args.dataset / "review/frames.json").read_text()) if r["split"] == "val"]
    index = {r["id"]: r for r in frames}
    assert len(rows) == len(index) == 300 and {r["id"] for r in rows} == index.keys()
    subtype = Counter(box["subtype"] for r in frames for box in r["foreign"])
    details = {}
    for key, targets in (("recovered_by_b", a_misses-b_misses), ("new_misses_in_b", b_misses-a_misses),
                         ("missed_by_both", a_misses & b_misses)):
        details[key] = [{"id": image_id, "file": index[image_id]["name"], "foreign_index": i,
                         "subtype": index[image_id]["foreign"][i]["subtype"]} for image_id, i in sorted(targets)]
    pending = sum(not r["negative"] and not r["material_reviewed"] for r in frames)
    material = None
    if args.material_threshold is not None:
        material = {"confidence": args.material_threshold, "validation_positives_pending_review": pending,
                    "oracle_neighborhood": material_metrics(rows, b_point["confidence"], args.material_threshold, False),
                    "predicted_neighborhood": material_metrics(rows, b_point["confidence"], args.material_threshold, True)}
    output = {"a": str(args.a), "b_operating_points": b_summary["operating_points"],
              "a_fp1": a_point, "paired": details, "subtypes": dict(subtype),
              "timing_raw_inference_ms": b_summary["timing_ms"], "frames_at_max_det": b_summary["frames_at_max_det"],
              "material": material}
    output_path = args.output or args.b / "paired_diagnostics.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
