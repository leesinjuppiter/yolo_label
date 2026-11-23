"""Replay frozen A1189 outputs and catch B category/gate regressions."""
from pathlib import Path
import json
import sys

from dataset import in_region
from diagnostics import material_metrics
from evaluate import operating_point


def checks():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    material = [[13, 0], [19, 0], [19, 5], [13, 5]]
    b = {"cls": 1, "score": .99, "points": square}
    rows = [{"id": "negative", "file": "negative.jpg", "negative": True, "ground_truth": [], "predictions": [b]},
            {"id": "positive", "file": "positive.jpg", "negative": False,
             "ground_truth": [{"cls": 0, "points": square}], "predictions": [b]}]
    point = operating_point(rows, 1)
    assert point["negative_alarms"] == point["foreign_detected"] == 0
    foreign = {"cls": 0, "score": .9, "points": square}
    m = {"cls": 1, "score": .9, "points": material}
    assert in_region(m, [foreign], 1, 1, .5)
    row = {"ground_truth": [foreign, m], "predictions": [foreign, dict(foreign), m]}
    assert material_metrics([row], .5, .25, True)["tp"] == 1
    assert material_metrics([dict(row, predictions=[m])], .5, .25, True)["retained"] == 0


def main():
    checks()
    base = Path(sys.argv[1])
    for variant, expected in (("best", (85, 85, 87)), ("epoch50", (73, 87, 90))):
        rows = json.loads((base / variant / "val_predictions.json").read_text())
        history = json.loads((base / variant / "val_operating_points.json").read_text())
        actual = [operating_point(rows, i) for i in range(3)]
        assert tuple(p["foreign_detected"] for p in actual) == expected
        for point, earlier in zip(actual, history["operating_points"]):
            assert all(point[k] == earlier[k] for k in (
                "confidence", "foreign_detected", "negative_alarms", "positive_alarms", "misses"))
        print(f"A1189 {variant} verified: {expected}", flush=True)


if __name__ == "__main__":
    main()
