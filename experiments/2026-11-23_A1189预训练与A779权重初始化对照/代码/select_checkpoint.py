"""Evaluate standard and periodic checkpoints at equal negative-frame false alarm budgets."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

from train import sha256


def main():
    run, dataset = (Path(arg) for arg in sys.argv[1:])
    evaluator = Path(__file__).with_name("evaluate.py")
    weights = run / "weights"
    checkpoints = [weights / "best.pt", weights / "last.pt", *sorted(weights.glob("epoch*.pt"))]
    seen = {}
    rows = []
    root = run / "checkpoint_eval"
    root.mkdir(exist_ok=True)
    for checkpoint in checkpoints:
        digest = sha256(checkpoint)
        if digest not in seen:
            output = root / checkpoint.stem
            metrics = output / "val_operating_points.json"
            if not metrics.exists():
                subprocess.run([sys.executable, "-B", str(evaluator), "--weights", str(checkpoint),
                                "--dataset-dir", str(dataset), "--output-dir", str(output)], check=True)
            points = json.loads(metrics.read_text())["operating_points"]
            seen[digest] = (checkpoint.stem, [{key: v for key, v in p.items() if key != "misses"} for p in points])
        result, points = seen[digest]
        rows.append({"checkpoint": checkpoint.name, "sha256": digest,
                     "evaluation_dir": result, "operating_points": points})
        print(checkpoint.name, [p["foreign_detected"] for p in points], flush=True)
    # Prefer the validation FP=1 target, then FP=0 and FP=2. Stable ordering favors best.pt on a full tie.
    selected = max(rows, key=lambda row: tuple(row["operating_points"][i]["foreign_detected"] for i in (1, 0, 2)))
    shutil.copyfile(weights / selected["checkpoint"], weights / "best_recall_fp1.pt")
    (root / "summary.json").write_text(json.dumps({"selected": selected, "checkpoints": rows}, indent=2) + "\n")
    print("SELECTED", selected["checkpoint"], flush=True)


if __name__ == "__main__":
    main()
