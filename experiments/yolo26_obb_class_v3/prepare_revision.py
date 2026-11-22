"""Create the shared cap-free, board-merged A600/A779 dataset revision."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

SOURCES = {
    "A600": ("val_v2/4911_A600_foreign_YOLO_OBB_val_v2.zip",
             "4b83be6d994efe546f83aff973c1ec400fa7191b49d3e421bac8780a1617bf71"),
    "A779": ("4911_A779_foreign_YOLO_OBB_val_v2.zip",
             "35d69c95ffcc794455aaa8d335d8abe36154ed8cb019218c0279c9bf3dd5a954"),
}
CAP_SAMPLE = "d4911_l0_001498"
VAL_COUNTS = {"images": 300, "positive": 109, "negative": 191, "foreign": 113}


def checksum(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def rewrite(source_path, output, variant):
    counts = {key: Counter() for key in ("train", "val")}
    edits = {"removed_cap": [], "merged_stick_into_board": []}
    with ZipFile(source_path) as source, ZipFile(output, "x", ZIP_DEFLATED) as dest:
        rows = json.loads(source.read("review/frames.json"))
        changed_labels = {}
        for row in rows:
            boxes = row["foreign"]
            name = f"labels/{row['split']}/{Path(row['name']).stem}.txt"
            entries = source.read(name).splitlines(keepends=True)
            assert len(entries) == len(boxes)
            selected = [(ann, label) for ann, label in zip(boxes, entries) if ann["subtype"] != "cap"]
            if len(selected) != len(boxes):
                edits["removed_cap"].append({"id": row["id"], "name": row["name"], "split": row["split"],
                                              "removed": len(boxes) - len(selected)})
                changed_labels[name] = b"".join(label for _, label in selected)
                row["negative"] = not selected
            updated = []
            for ann, _ in selected:
                if ann["subtype"] == "stick":
                    edits["merged_stick_into_board"].append({"id": row["id"], "name": row["name"],
                                                             "split": row["split"], "box": ann["id"]})
                    ann = ann | {"subtype": "board"}
                updated.append(ann)
            row["foreign"] = updated
            count = counts[row["split"]]
            count.update(images=1, foreign=len(updated))
            count["negative" if row["negative"] else "positive"] += 1

        assert counts["val"] == VAL_COUNTS
        assert counts["train"] == {
            "images": int(variant[1:]), "negative": 347,
            "positive": 253 if variant == "A600" else 432,
            "foreign": 269 if variant == "A600" else 626,
        }
        assert [r["id"] for r in edits["removed_cap"]] == [CAP_SAMPLE]
        assert len(edits["merged_stick_into_board"]) == (15 if variant == "A600" else 16)
        revision = {"variant": variant, "validation_revision": "val_v3",
                    "foreign_subtypes": {"removed": ["cap"], "merged": {"stick": "board"}},
                    "counts": counts, "edits": edits}
        changed = changed_labels | {
            "review/frames.json": (json.dumps(rows, ensure_ascii=False, indent=2) + "\n").encode(),
            "review/summary.json": (json.dumps(revision, ensure_ascii=False, indent=2) + "\n").encode(),
        }
        for member in source.infolist():
            name = member.filename
            body = changed[name] if name in changed else source.read(member)
            dest.writestr(name, body, compress_type=member.compress_type)
    with ZipFile(source_path) as source, ZipFile(output) as dest:
        assert dest.testzip() is None
        assert source.namelist() == dest.namelist()
        modified = {name for name in source.namelist()
                    if source.getinfo(name).CRC != dest.getinfo(name).CRC}
        assert modified == set(changed)
        assert dest.read(next(iter(changed_labels))) == b""
    return revision | {"file": output.name, "sha256": checksum(output)}


def prepare(root):
    final = root / "class_v3"
    if final.exists():
        raise FileExistsError(final)
    with TemporaryDirectory(prefix=".class_v3-", dir=root) as scratch:
        stage = Path(scratch) / final.name
        stage.mkdir()
        versions = {}
        for variant, (relative, expected) in SOURCES.items():
            source = root / relative
            if checksum(source) != expected:
                raise ValueError(f"Input archive hash mismatch: {source}")
            filename = f"4911_{variant}_foreign_YOLO_OBB_val_v3.zip"
            revision = rewrite(source, stage / filename, variant)
            versions[variant] = revision | {"source_sha256": expected}
        with ZipFile(stage / versions["A600"]["file"]) as a, ZipFile(stage / versions["A779"]["file"]) as b:
            for item in a.infolist():
                if item.filename.startswith(("images/", "labels/")):
                    assert item.CRC == b.getinfo(item.filename).CRC, item.filename
        (stage / "revision.json").write_text(json.dumps(versions, ensure_ascii=False, indent=2) + "\n")
        stage.rename(final)
    for variant, revision in versions.items():
        print(variant, revision["counts"], revision["sha256"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    prepare(parser.parse_args().root.resolve(strict=True))
