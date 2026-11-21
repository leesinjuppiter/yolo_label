"""Read the 4911 review manifest and the original rotated XML annotations."""

import csv
import math
from pathlib import Path, PurePosixPath
import re
from xml.etree import ElementTree


BUCKET_SPLITS = {
    "01_train_0600_base": "train",
    "02_train_add_0600_to_1200": "train",
    "03_train_add_0700_to_1900": "train",
    "04_val_0300": "val",
    "05_test_0727": "test",
}
FOREIGN_SUBTYPES = frozenset(("rockbolt", "mesh", "stick", "board", "cap", "bottle"))
PILOT_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*\[原图\]\(candidate_(?:positive|negative)/images/([^/()]+)\)",
    flags=re.MULTILINE,
)


class Dataset:
    """Index exported samples by stable ID, including their pilot membership."""

    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.records = {}
        self._images = {}
        self._annotations = {}
        index = self._path("00_pilot_100/首批试标入口.md")
        pilot_order = {}
        if index.is_file():
            pilot_order = {name: int(order) for order, name in PILOT_ROW.findall(index.read_text(encoding="utf-8-sig"))}

        source = self._path("sample_manifest.csv")
        with source.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                image = (row.get("relative_image") or "").strip()
                if not image:
                    continue
                sample_id = row.get("sample_id", "")
                bucket = row.get("bucket", "")
                candidate = row.get("candidate_status", "")
                pilot = row.get("pilot", "").lower()
                relative = PurePosixPath(image)
                if (
                    not re.fullmatch(r"[a-zA-Z0-9_-]+", sample_id)
                    or sample_id in self.records
                    or bucket not in BUCKET_SPLITS
                    or candidate not in {"positive", "negative"}
                    or pilot not in {"true", "false"}
                    or relative.parts[:3] != (bucket, f"candidate_{candidate}", "images")
                    or len(relative.parts) != 4
                    or not re.fullmatch(r"[a-zA-Z0-9_-]+\.jpe?g", relative.name, flags=re.IGNORECASE)
                    or (pilot == "true" and bucket != "01_train_0600_base")
                ):
                    raise ValueError(f"Invalid dataset manifest row for {sample_id!r}")
                self.records[sample_id] = {
                    "id": sample_id,
                    "name": relative.name,
                    "bucket": bucket,
                    "pilot": pilot == "true",
                    "pilot_order": pilot_order.get(relative.name) if pilot == "true" else None,
                    "candidate": candidate,
                    "split": BUCKET_SPLITS[bucket],
                    "capture_group": row.get("capture_group", ""),
                    "original_split": row.get("original_split", ""),
                }
                self._images[sample_id] = relative.as_posix()
                self._annotations[sample_id] = (
                    relative.parent.parent / "legacy_xml" / relative.with_suffix(".xml").name
                ).as_posix()

    def _path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Dataset file must remain inside its configured directory")
        return path

    def image_path(self, sample_id):
        return self._path(self._images[sample_id])

    def frame(self, sample_id):
        path = self._path(self._annotations[sample_id])
        root = ElementTree.parse(path).getroot()
        try:
            width = int(root.findtext("size/width", "0"))
            height = int(root.findtext("size/height", "0"))
            if width <= 0 or height <= 0:
                raise ValueError("Invalid image dimensions")

            boxes = []
            unknown = []
            for index, obj in enumerate(root.findall("object"), start=1):
                subtype = obj.findtext("name", "").strip()
                if subtype == "coal":
                    kind = "material"
                elif subtype in FOREIGN_SUBTYPES:
                    kind = "foreign"
                else:
                    unknown.append(subtype)
                    continue
                cx, cy, box_width, box_height, angle = (
                    float(obj.findtext(f"robndbox/{field}", "nan"))
                    for field in ("cx", "cy", "w", "h", "angle")
                )
                if not all(map(math.isfinite, (cx, cy, box_width, box_height, angle))) or min(box_width, box_height) <= 0:
                    raise ValueError(f"Invalid rotation box at object {index}")
                cos_angle, sin_angle = math.cos(angle), math.sin(angle)
                half_width, half_height = box_width / 2, box_height / 2
                local = (
                    (-half_width, -half_height),
                    (half_width, -half_height),
                    (half_width, half_height),
                    (-half_width, half_height),
                )
                # The editor and neighborhood geometry need unclipped corners, including at image borders.
                points = [
                    [(cx + u*cos_angle - v*sin_angle) / width, (cy + u*sin_angle + v*cos_angle) / height]
                    for u, v in local
                ]
                boxes.append({"id": f"legacy-{index}", "kind": kind, "subtype": subtype, "points": points})
        except ValueError as error:
            raise ValueError(f"{path.name}: {error}") from error
        return {
            **self.records[sample_id],
            "width": width,
            "height": height,
            "boxes": boxes,
            "legacy_unknown": unknown,
        }


def in_region(material, foreigners, width, height, factor):
    """Check material center distance to any foreign OBB in original pixels."""
    material_points = material["points"]
    center_x = sum(point[0] for point in material_points) * width / 4
    center_y = sum(point[1] for point in material_points) * height / 4
    for foreign in foreigners:
        p0, p1, _, p3 = foreign["points"]
        x = (p1[0] - p0[0]) * width
        y = (p1[1] - p0[1]) * height
        z = (p3[0] - p0[0]) * width
        t = (p3[1] - p0[1]) * height
        box_width, box_height = math.hypot(x, y), math.hypot(z, t)
        if not box_width or not box_height:
            continue
        offset_x = center_x - p0[0] * width
        offset_y = center_y - p0[1] * height
        u = (offset_x*x + offset_y*y) / box_width
        v = (offset_x*z + offset_y*t) / box_height
        distance = math.hypot(max(0, -u, u-box_width), max(0, -v, v-box_height))
        if distance <= factor * max(box_width, box_height) + 1e-9:
            return True
    return False
