import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from train import prepare_dataset, sha256

A = "4911_A1189_foreign_YOLO_OBB_val_v3.zip"
B = "4911_B1189_foreign_material_0p5L_YOLO_OBB_val_v3_pending_review.zip"
A_SHA = "4c05e91c2f27289ebed11d0829f2d46c1d331a14c6040c7f1a1bb4f5ec96a075"
B_SHA = "b0ed0f8bdcac5e54e659c611d19de9680f5495cae92e775a7271a90f0a7b2ddc"
CHECKPOINTS = {
    "official": "6f51c78197aacda4a33be77294065a9001675fb893f56227a179731b53dbd2b0",
    "a779_best": "59469be2d851d602695b669f2761b41d75d34367371f7c526c2aea8db80b9728",
}
COUNTS = {
    "train": {"images": 1189, "negative": 694, "positive": 495, "foreign": 520},
    "val": {"images": 300, "negative": 191, "positive": 109, "foreign": 113},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    for archive, digest in ((root / A, A_SHA), (root / B, B_SHA)):
        assert sha256(archive) == digest, f"Dataset ZIP checksum mismatch: {archive.name}"
        with ZipFile(archive) as data:
            assert data.testzip() is None, archive.name
    with ZipFile(root / A) as data:
        summary = json.loads(data.read("review/summary.json"))
        assert summary["validation_revision"] == "val_v3"
        assert summary["train_sources"] == {"4911_base": 600, "4911_additional": 589}
        fixed = root / "class_v3/A779"
        val = [row for row in json.loads(data.read("review/frames.json")) if row["split"] == "val"]
        earlier = [row for row in json.loads((fixed / "review/frames.json").read_text()) if row["split"] == "val"]
        assert {(r["id"], r["name"]) for r in val} == {(r["id"], r["name"]) for r in earlier}
        for r in val:
            paths = (f"images/val/{r['name']}", f"labels/val/{Path(r['name']).stem}.txt")
            assert all(data.read(path) == (fixed / path).read_bytes() for path in paths), r["id"]
    official = root.parent / "weights/yolo26n-obb.pt"
    a779 = root / "runs/a779_yolo26n_obb_960_b24_val_v3/weights/best.pt"
    for tag, path in (("official", official), ("a779_best", a779)):
        assert sha256(path) == CHECKPOINTS[tag], tag
    stage = root / "a1189_initialization_20261123"
    spec = {"sha256": A_SHA, "counts": COUNTS}
    data = prepare_dataset(root / A, stage / "A1189", spec)
    provenance = {"dataset_sha256": A_SHA, "dataset_b_sha256": B_SHA, "counts": COUNTS,
                  "validation_revision": "val_v3", "dataset": str(data),
                  "weights": {"official": str(official), "a779_best": str(a779)},
                  "weights_sha256": CHECKPOINTS}
    (stage / "inputs.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Verified A/B archives, both checkpoints, and exact val_v3 image and label equality", flush=True)


if __name__ == "__main__":
    main()
