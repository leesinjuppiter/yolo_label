"""Persist review state independently of the original images and labels."""

from contextlib import closing
import json
import math
import sqlite3


VERDICTS = ("half", "full", "neither", "undecided")
FLAGS = ("foreign_reviewed", "material_reviewed", "negative")
DEFAULT = dict.fromkeys(FLAGS, False) | {"verdict": "", "note": "", "version": 0}

BATCHES = {
    "pilot": "首批试标 · 100 张",
    "01_train_0600_base": "训练 · 首轮",
    "02_train_add_0600_to_1200": "训练 · 扩至 1,200 张",
    "03_train_add_0700_to_1900": "训练 · 扩至 1,900 张",
    "04_val_0300": "验证 · 300 张",
    "05_test_0727": "独立测试 · 727 张",
}


class Conflict(ValueError):
    pass


def review_fields(state):
    return {key: state[key] for key in DEFAULT}


def is_exportable(state):
    return state["foreign_reviewed"] and (state["negative"] or state["material_reviewed"])


def validate_boxes(boxes, current):
    if not isinstance(boxes, list) or len(boxes) > 1500:
        raise ValueError("标注框必须是列表，且每张图最多 1,500 个。")
    seen = set()
    previous = {box["id"]: box["points"] for box in current}
    cleaned = []
    for box in boxes:
        if not isinstance(box, dict):
            raise ValueError("标注框格式错误。")
        box_id, kind, subtype, points = (box.get(key) for key in ("id", "kind", "subtype", "points"))
        if not isinstance(box_id, str) or not 0 < len(box_id) <= 100 or box_id in seen:
            raise ValueError("标注框 ID 缺失或重复。")
        if kind not in ("foreign", "material") or not isinstance(subtype, str) or len(subtype) > 80:
            raise ValueError("标注框类别或子类格式错误。")
        if not isinstance(points, list) or len(points) != 4 or any(
            not isinstance(p, list) or len(p) != 2 or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in p
            ) for p in points
        ):
            raise ValueError("每个标注框须有四个有效角点。")
        if points != previous.get(box_id) and any(not 0 <= v <= 1 for p in points for v in p):
            raise ValueError("新建或修改的框须在图像边界内。")
        cross = []
        for i in range(4):
            a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
            cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        if not (all(v > 0 for v in cross) or all(v < 0 for v in cross)):
            raise ValueError("标注框的角点须沿边界排列且围成有效区域。")
        seen.add(box_id)
        cleaned.append({"id": box_id, "kind": kind, "subtype": subtype, "points": points})
    return cleaned


class Store:
    def __init__(self, dataset, directory):
        self.dataset = dataset
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "reviews.sqlite3"
        with closing(self.connect()) as db, db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS reviews (
                    sample_id TEXT PRIMARY KEY, state TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS history (
                    sample_id TEXT NOT NULL, version INTEGER NOT NULL,
                    saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, state TEXT NOT NULL,
                    PRIMARY KEY(sample_id, version)
                );
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def catalog(self, batch):
        if batch not in BATCHES:
            raise ValueError("未知的标注批次。")
        with closing(self.connect()) as db:
            states = {sample: json.loads(state) for sample, state in db.execute("SELECT sample_id, state FROM reviews")}
        frames = []
        summary = dict.fromkeys(VERDICTS, 0)
        for sample, record in self.dataset.records.items():
            if (batch == "pilot" and record["pilot"]) or record["bucket"] == batch:
                frame = record | review_fields(states.get(sample, DEFAULT))
                frames.append(frame)
                if frame["verdict"]:
                    summary[frame["verdict"]] += 1
        if batch == "pilot":
            frames.sort(key=lambda frame: (frame["pilot_order"] is None, frame["pilot_order"], frame["id"]))
        batches = BATCHES.copy()
        first_round = "01_train_0600_base"
        first_round_count = sum(record["bucket"] == first_round for record in self.dataset.records.values())
        batches[first_round] = f"{batches[first_round]} {first_round_count} 张"
        return {
            "frames": frames,
            "summary": summary,
            "exportable": sum(sample in self.dataset.records and is_exportable(state) for sample, state in states.items()),
            "batches": [{"id": k, "name": v} for k, v in batches.items()],
        }

    def frame(self, sample):
        original = self.dataset.frame(sample)
        with closing(self.connect()) as db:
            row = db.execute("SELECT state FROM reviews WHERE sample_id=?", (sample,)).fetchone()
        return original | (json.loads(row[0]) if row else DEFAULT)

    def save(self, sample, payload):
        expected_keys = {"version", "boxes", "verdict", "note", *FLAGS}
        if not isinstance(payload, dict) or payload.keys() != expected_keys:
            raise ValueError("保存字段缺失或无效。")
        if type(payload["version"]) is not int or any(type(payload[key]) is not bool for key in FLAGS):
            raise ValueError("保存版本和复核状态格式错误。")
        if payload["verdict"] not in ("", *VERDICTS) or not isinstance(payload["note"], str) or len(payload["note"]) > 2000:
            raise ValueError("邻域判断或备注格式错误，备注最多 2,000 个字符。")
        original = self.dataset.frame(sample)
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state FROM reviews WHERE sample_id=?", (sample,)).fetchone()
            current = original | (json.loads(row[0]) if row else DEFAULT)
            if current["version"] != payload["version"]:
                raise Conflict("此帧已有更新。请刷新后核对标注，再重新提交。")
            boxes = validate_boxes(payload["boxes"], current["boxes"])
            state = payload | {"version": current["version"] + 1, "boxes": boxes}
            has_foreign = any(box["kind"] == "foreign" for box in boxes)
            if state["negative"] and (has_foreign or not state["foreign_reviewed"] or state["material_reviewed"] or state["verdict"]):
                raise ValueError("确认阴性须完成整图复核、清除异物框，并清空邻域判断。")
            if state["foreign_reviewed"] and not (has_foreign or state["negative"]):
                raise ValueError("请保留已确认的异物框，或将整帧确认为阴性。")
            if state["material_reviewed"] and not (has_foreign and state["foreign_reviewed"]):
                raise ValueError("请先完成整图异物复核，再确认邻域物料。")
            blob = json.dumps(state, ensure_ascii=False, allow_nan=False)
            db.execute("INSERT INTO reviews VALUES (?, ?) ON CONFLICT(sample_id) DO UPDATE SET state=excluded.state", (sample, blob))
            db.execute("INSERT INTO history (sample_id, version, state) VALUES (?, ?, ?)", (sample, state["version"], blob))
        return original | state

    def export_frames(self):
        with closing(self.connect()) as db:
            reviewed = list(db.execute("SELECT sample_id, state FROM reviews ORDER BY sample_id"))
        for sample, raw in reviewed:
            state = json.loads(raw)
            if sample in self.dataset.records and is_exportable(state):
                yield self.dataset.frame(sample) | state
