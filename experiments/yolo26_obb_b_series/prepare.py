"""Build paired B600/B1189 from the frozen B1189 ZIP and A reference splits."""
from collections import Counter
import json
import re
import sys
from pathlib import Path
from zipfile import ZipFile

from train import sha256

B_SHA = "b0ed0f8bdcac5e54e659c611d19de9680f5495cae92e775a7271a90f0a7b2ddc"
OFFICIAL_SHA = "6f51c78197aacda4a33be77294065a9001675fb893f56227a179731b53dbd2b0"
CLASSES = ["foreign", "surrounding_material"]
CONFIG = {"train": "images/train", "val": "images/val", "names": CLASSES}


def verify_or_write(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing file differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".part")
        temp.write_bytes(data)
        temp.replace(path)


def build(archive, source, target, variant, initial):
    a_frames = json.loads((source / "review/frames.json").read_text())
    b_frames = {r["id"]: r for r in json.loads(archive.read("review/frames.json"))}
    frames = []
    counts = {"train": Counter(), "val": Counter()}
    groups = {"train": set(), "val": set()}
    for a in a_frames:
        frame = b_frames[a["id"]]
        for key in ("id", "name", "split", "negative", "capture_group"):
            if a[key] != frame[key]:
                raise ValueError(f"{variant}: sample differs on {key}: {a['id']}")
        name, split = a["name"], a["split"]
        if split not in counts or not re.fullmatch(r"[\w-]+\.jpe?g", name, flags=re.ASCII):
            raise ValueError(f"Unexpected dataset entry: {name}")
        image = f"images/{split}/{name}"
        label = f"labels/{split}/{Path(name).stem}.txt"
        raw = archive.read(label)
        by_class = Counter()
        foreign = []
        for line in raw.splitlines(keepends=True):
            cls, *points = line.split()
            if cls not in (b"0", b"1") or len(points) != 8 or any(not 0 <= float(p) <= 1 for p in points):
                raise ValueError(f"Invalid OBB: {label}")
            by_class[int(cls)] += 1
            if cls == b"0":
                foreign.append(line)
        if (b"".join(foreign) != (source / label).read_bytes()
                or by_class[0] != len(frame["foreign"]) or by_class[1] != len(frame["material_ids"])
                or frame["negative"] != (not foreign) or (frame["negative"] and raw.strip())):
            raise ValueError(f"{variant}: incorrect A/B label pairing for {label}")
        image_data = archive.read(image)
        if image_data != (source / image).read_bytes():
            raise ValueError(f"{variant}: A/B images differ: {image}")
        verify_or_write(target / image, image_data)
        verify_or_write(target / label, raw)
        frames.append(frame)
        groups[split].add(frame["capture_group"])
        counts[split].update(images=1, foreign=by_class[0], material=by_class[1],
                             negative=int(frame["negative"]), positive=int(not frame["negative"]),
                             material_review_pending_positive=int(not frame["negative"] and not frame["material_reviewed"]))
    assert len(frames) == len({r["id"] for r in frames}) == len({Path(r["name"]).stem for r in frames})
    assert counts["train"]["images"] == int(variant[1:]) and not groups["train"] & groups["val"]
    assert (counts["val"]["images"], counts["val"]["negative"], counts["val"]["foreign"], counts["val"]["material"]) == (300, 191, 113, 1685)
    if variant == "B1189":
        assert set(b_frames) == {r["id"] for r in frames}
        assert counts["train"]["foreign"] == 520 and counts["train"]["material"] == 8330
    verify_or_write(target / "review/frames.json", json.dumps(frames, indent=2).encode())
    dataset = target / "data.yaml"
    verify_or_write(dataset, json.dumps({"path": str(target), **CONFIG}, indent=2).encode())
    spec = {"variant": variant, "validation_revision": "val_v3", "dataset_sha256": B_SHA,
            "dataset_sha256_role": "source_B1189_archive",
            "frames_sha256": sha256(target / "review/frames.json"),
            "counts": counts, "dataset": str(dataset),
            "weights": {"official": str(initial)}, "weights_sha256": {"official": OFFICIAL_SHA}}
    verify_or_write(target / "inputs.json", json.dumps(spec, indent=2).encode())
    print(variant, json.dumps(counts), flush=True)


def main():
    root = Path(sys.argv[1]).resolve()
    path = root / "4911_B1189_foreign_material_0p5L_YOLO_OBB_val_v3_pending_review.zip"
    initial = root.parent / "weights/yolo26n-obb.pt"
    assert sha256(path) == B_SHA and sha256(initial) == OFFICIAL_SHA
    with ZipFile(path) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert json.loads(archive.read("data.yaml")) == CONFIG
        for variant, source in (("B600", root / "class_v3/A600"),
                                ("B1189", root / "a1189_initialization_20261123/A1189")):
            build(archive, source, root / "b_series_20261123" / variant, variant, initial)


if __name__ == "__main__":
    main()
