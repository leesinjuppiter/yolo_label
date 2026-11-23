"""Verify the frozen A/B training references and extract the independent test split."""
from collections import Counter
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

A600 = "class_v3/4911_A600_foreign_YOLO_OBB_val_v3.zip"
A1189 = "4911_A1189_foreign_YOLO_OBB_val_v3.zip"
B1189 = "4911_B1189_foreign_material_0p5L_YOLO_OBB_val_v3_pending_review.zip"
T724 = "4911_T724_independent_test_foreign_YOLO_OBB.zip"
HASHES = {
    A600: "c5fa3a9fe5a6ef997ed03677ec24c42259a427a9a353dc31f53785fff23e2a0f",
    A1189: "4c05e91c2f27289ebed11d0829f2d46c1d331a14c6040c7f1a1bb4f5ec96a075",
    B1189: "b0ed0f8bdcac5e54e659c611d19de9680f5495cae92e775a7271a90f0a7b2ddc",
    T724: "97e109a63890c573438cd8a1f645f7cec72c25f2648d68fdf0aa4bacfde3518b",
}
DATASETS = {
    "A600": ("class_v3/A600", A600, 600, 269, 0),
    "B600": ("b_series_20261123/B600", B1189, 600, 269, 4263),
    "A1189": ("a1189_initialization_20261123/A1189", A1189, 1189, 520, 0),
    "B1189": ("b_series_20261123/B1189", B1189, 1189, 520, 8330),
}


def checksum(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def members(row):
    split, name = row["split"], row["name"]
    assert split in {"train", "val", "test"} and re.fullmatch(r"[\w-]+\.jpe?g", name, re.ASCII)
    return f"images/{split}/{name}", f"labels/{split}/{Path(name).stem}.txt"


def rows_from(archive):
    rows = json.loads(archive.read("review/frames.json"))
    assert len(rows) == len({row["id"] for row in rows})
    return rows


def inspect_labels(raw, row):
    counts = Counter()
    for line in raw.splitlines():
        cls, *values = line.split()
        assert cls in (b"0", b"1") and len(values) == 8 and all(0 <= float(v) <= 1 for v in values)
        counts[int(cls)] += 1
    assert counts[0] == len(row["foreign"]) and bool(counts[0]) != row["negative"]
    assert not row["negative"] or not counts[1]
    return counts


def verify_train(root, name, specification):
    location, source_file, total, foreign, material = specification
    dataset = root / location
    config = json.loads((dataset / "data.yaml").read_text())
    classes = ["foreign", "surrounding_material"] if name.startswith("B") else ["foreign"]
    assert config == {"path": str(dataset), "names": classes, "train": "images/train", "val": "images/val"}
    rows = json.loads((dataset / "review/frames.json").read_text())
    counts = Counter()
    ids = {"train": set(), "val": set()}
    subjects = set()
    with ZipFile(root / source_file) as source:
        expected = {r["id"]: r for r in rows_from(source)}
        for row in rows:
            raw_meta = expected[row["id"]]
            assert all(row[key] == raw_meta[key] for key in ("name", "split", "foreign", "negative", "capture_group"))
            image, label = members(row)
            raw_image, raw_label = source.read(image), source.read(label)
            assert raw_image == (dataset / image).read_bytes()
            assert raw_label == (dataset / label).read_bytes()
            ty = row["split"]
            counts[ty] += 1
            for k, count in inspect_labels(raw_label, row).items():
                counts[ty, k] += count
            ids[ty].add(row["id"])
            subjects.add((row["capture_group"], row["name"], sha256(raw_image).hexdigest()))
    assert counts["train"] == total and counts["train",0] == foreign and counts["train",1] == material
    assert counts["val"] == 300 and counts["val",0] == 113 and counts["val",1] == (1685 if name.startswith("B") else 0)
    assert len(ids["train"]) == total and len(ids["val"]) == 300
    print("DATA_VERIFIED", name, total, foreign, material, flush=True)
    return {"path": str(dataset), "source_sha256": HASHES[source_file], "train_images": total,
            "train_foreign": foreign, "train_material": material}, ids, subjects


def verify_test(root, target, subjects):
    other_groups, other_names, other_hashes = map(set, zip(*subjects))
    with ZipFile(root / T724) as source:
        assert source.testzip() is None
        rows = rows_from(source)
        assert len(rows) == 724 and all(r["split"] == "test" and r["foreign_reviewed"] for r in rows)
        counts = Counter()
        seen = set()
        for row in rows:
            image, label = members(row)
            raw_image, raw_label = source.read(image), source.read(label)
            image_hash = sha256(raw_image).hexdigest()
            assert row["capture_group"] not in other_groups and row["name"] not in other_names
            assert image_hash == row["image_sha256"] and image_hash not in other_hashes | seen
            seen.add(image_hash)
            box_counts = inspect_labels(raw_label, row)
            assert box_counts[1] == 0
            counts.update(foreign=box_counts[0], negative=int(row["negative"]), positive=int(not row["negative"]))
            for relative, payload in ((image, raw_image), (label, raw_label)):
                path = target / relative
                if path.exists():
                    assert path.read_bytes() == payload, path
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temp = path.with_suffix(path.suffix + ".part")
                    temp.write_bytes(payload)
                    temp.replace(path)
        assert counts == {"foreign": 256, "negative": 476, "positive": 248}
        (target / "review").mkdir(exist_ok=True)
        (target / "review/frames.json").write_bytes(source.read("review/frames.json"))
    print("TEST_VERIFIED", dict(counts), flush=True)


def main():
    root, stage = (Path(p).resolve() for p in sys.argv[1:])
    for path, expected in HASHES.items():
        assert checksum(root / path) == expected, path
    specs, splits, subjects = {}, {}, set()
    for name, specification in DATASETS.items():
        specs[name], splits[name], content = verify_train(root, name, specification)
        subjects.update(content)
    assert splits["A600"] == splits["B600"] and splits["A1189"] == splits["B1189"]
    assert splits["A600"]["val"] == splits["A1189"]["val"]
    verify_test(root, stage / "T724", subjects)
    scales = {}
    for scale in ("s", "m", "l", "x"):
        weight = root.parent / "weights" / f"yolo26{scale}-obb.pt"
        assert weight.is_file() and weight.stat().st_size > 1_000_000, weight
        scales[scale] = {"path": str(weight), "sha256": checksum(weight)}
    (stage / "inputs.json").write_text(json.dumps({"datasets": specs, "weights": scales,
                                                   "test_sha256": HASHES[T724]}, indent=2)+"\n")


if __name__ == "__main__":
    main()
