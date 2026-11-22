"""Build paired YOLO OBB datasets from one reviewed annotation snapshot."""

import json
import math
import re
from pathlib import PurePosixPath
from tempfile import TemporaryFile
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from dataset import in_region


SPLITS = ("train", "val", "test")
NEIGHBORHOOD_FACTOR = 0.5
DATASETS = {
    "A_foreign": ["foreign"],
    "B_foreign_material": ["foreign", "surrounding_material"],
}


def label_row(box, class_id):
    points = []
    for point in box["points"]:
        if not all(math.isfinite(value) for value in point):
            raise ValueError(f"标注框 {box['id']} 包含无效坐标")
        # Legacy XML may place corners outside the image. Keep the original
        # geometry for membership checks and normalize boundary points for YOLO.
        points.append(tuple(min(1.0, max(0.0, value)) for value in point))
    area = sum(
        a[0] * b[1] - b[0] * a[1]
        for a, b in zip(points, points[1:] + points[:1])
    )
    if len(set(points)) < 3 or abs(area) <= 1e-14:
        raise ValueError(f"标注框 {box['id']} 裁至图内后面积为零，请修正边界框")
    coords = " ".join(f"{value:.15g}" for point in points for value in point)
    return f"{class_id} {coords}\n"


def dataset_yaml(splits, classes):
    settings = {
        name: f"images/{name}" if name in splits else None
        for name in SPLITS
        if name != "test" or name in splits
    }
    settings["names"] = classes
    # JSON mappings also parse as YAML, without requiring a YAML dependency.
    return json.dumps(settings, indent=2) + "\n"


def write_frame(archive, dataset, frame, filenames):
    name, split = frame["name"], frame["split"]
    if split not in SPLITS or not re.fullmatch(r"[A-Za-z0-9_.-]+\.(?:jpg|jpeg|png)", name, re.IGNORECASE):
        raise ValueError(f"无法导出的样本文件名或分集：{name}")
    stem = PurePosixPath(name).stem
    label_key = (split, stem.casefold())
    if label_key in filenames:
        raise ValueError(f"{split} 中存在重复的标签文件名：{stem}")
    filenames.add(label_key)

    foreign = [box for box in frame["boxes"] if box["kind"] == "foreign"]
    material = []
    if frame["negative"]:
        if foreign:
            raise ValueError(f"{name} 已确认无异物，请先核对仍存在的异物框")
    else:
        if not foreign:
            raise ValueError(f"{name} 已审核为阳性，但没有异物框")
        material = [
            box for box in frame["boxes"]
            if box["kind"] == "material"
            and in_region(box, foreign, frame["width"], frame["height"], NEIGHBORHOOD_FACTOR)
        ]

    try:
        a_rows = "".join(label_row(box, 0) for box in foreign)
        b_rows = a_rows + "".join(label_row(box, 1) for box in material)
    except ValueError as exc:
        raise ValueError(f"{name}：{exc}") from exc
    source = dataset.image_path(frame["id"])
    for directory, rows in zip(DATASETS, (a_rows, b_rows)):
        archive.write(source, f"{directory}/images/{split}/{name}", compress_type=ZIP_STORED)
        archive.writestr(f"{directory}/labels/{split}/{stem}.txt", rows)

    metadata = {
        key: frame[key] for key in (
            "id", "name", "bucket", "pilot", "candidate", "split", "negative",
            "foreign_reviewed", "material_reviewed", "verdict", "note", "version",
        )
    }
    metadata["foreign"] = [{"id": box["id"], "subtype": box["subtype"]} for box in foreign]
    metadata["material_ids"] = [box["id"] for box in material]
    return metadata


def build_export(dataset, frames, temp_dir=None):
    stream = TemporaryFile(dir=temp_dir)
    samples, split_counts, filenames = [], {}, set()
    try:
        with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
            for frame in frames:
                if frame["foreign_reviewed"] and (frame["negative"] or frame["material_reviewed"]):
                    samples.append(write_frame(archive, dataset, frame, filenames))
                    split = frame["split"]
                    split_counts[split] = split_counts.get(split, 0) + 1
            if not samples:
                raise ValueError("暂无可配对导出的样本，请先完成异物和邻域审核或确认无异物")

            for directory, classes in DATASETS.items():
                archive.writestr(f"{directory}/data.yaml", dataset_yaml(split_counts, classes))
            summary = {
                "neighborhood_factor": NEIGHBORHOOD_FACTOR,
                "distance_rule": "material_obb_to_foreign_obb_pixels",
                "split_sample_counts": split_counts,
                "has_train_and_val": all(split in split_counts for split in ("train", "val")),
                "classes": DATASETS,
            }
            archive.writestr("review/summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            archive.writestr("review/frames.json", json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
        stream.seek(0)
        return stream
    except Exception:
        stream.close()
        raise
