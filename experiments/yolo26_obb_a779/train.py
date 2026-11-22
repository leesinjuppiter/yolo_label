"""Train the 4911 A779 YOLO26n-OBB baseline from official pretrained weights."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from zipfile import ZipFile

DATA_SHA256 = "35d69c95ffcc794455aaa8d335d8abe36154ed8cb019218c0279c9bf3dd5a954"
WEIGHTS_SHA256 = "6f51c78197aacda4a33be77294065a9001675fb893f56227a179731b53dbd2b0"
SPLITS = {
    "train": {"images": 779, "positive": 433, "negative": 346, "foreign": 627},
    "val": {"images": 300, "positive": 109, "negative": 191, "foreign": 113},
}


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def prepare_dataset(archive_path, dataset_root):
    if sha256(archive_path) != DATA_SHA256:
        raise ValueError(f"A779 ZIP checksum mismatch: {archive_path}")
    with ZipFile(archive_path) as archive:
        config = json.loads(archive.read("data.yaml"))
        if config != {"train": "images/train", "val": "images/val", "names": ["foreign"]}:
            raise ValueError(f"Unexpected A779 classes or splits: {config}")
        frames = json.loads(archive.read("review/frames.json"))
        counts = {split: Counter() for split in SPLITS}
        groups = {split: set() for split in SPLITS}
        names = set()
        stems = set()
        for frame in frames:
            split, name = frame["split"], frame["name"]
            if split not in SPLITS or not re.fullmatch(r"[\w-]+\.jpe?g", name, flags=re.ASCII):
                raise ValueError(f"Invalid dataset entry: {split}/{name}")
            if name in names or Path(name).stem in stems:
                raise ValueError(f"Duplicate image or label: {name}")
            names.add(name)
            stems.add(Path(name).stem)
            groups[split].add(frame["capture_group"])
            label = f"labels/{split}/{Path(name).stem}.txt"
            rows = archive.read(label).decode().splitlines()
            if bool(rows) == frame["negative"] or len(rows) != len(frame["foreign"]):
                raise ValueError(f"Unexpected foreign annotations: {label}")
            for row in rows:
                cls, *points = row.split()
                if cls != "0" or len(points) != 8 or any(not 0 <= float(v) <= 1 for v in points):
                    raise ValueError(f"Invalid OBB label: {label}")
            counts[split].update(images=1, foreign=len(rows))
            counts[split]["positive" if rows else "negative"] += 1
        if counts != SPLITS or groups["train"] & groups["val"]:
            raise ValueError(f"Unexpected split counts or shared capture groups: {counts}")

        manifest = dataset_root / "review/summary.json"
        if not manifest.exists():
            staging = dataset_root.with_name(dataset_root.name + ".extracting")
            staging.mkdir(parents=True, exist_ok=False)
            for frame in frames:
                split, name = frame["split"], frame["name"]
                for relative in (f"images/{split}/{name}", f"labels/{split}/{Path(name).stem}.txt"):
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(relative) as source, target.open("xb") as dest:
                        shutil.copyfileobj(source, dest)
            for relative in ("review/frames.json", "review/summary.json"):
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(relative))
            print(f"Extracted verified A779 to {dataset_root}", flush=True)
            (staging / "data.yaml").write_text(json.dumps({"path": str(dataset_root), **config}, indent=2) + "\n")
            staging.rename(dataset_root)
        else:
            for frame in frames:
                split, name = frame["split"], frame["name"]
                for relative in (f"images/{split}/{name}", f"labels/{split}/{Path(name).stem}.txt"):
                    if (dataset_root / relative).read_bytes() != archive.read(relative):
                        raise ValueError(f"Extracted A779 differs from ZIP: {relative}")
        print(f"Verified {SPLITS}", flush=True)
    return dataset_root / "data.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-zip", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    weight = args.weights.resolve(strict=True)
    if sha256(weight) != WEIGHTS_SHA256:
        raise ValueError("Expected official Ultralytics YOLO26n-OBB pretrained weights")
    data = prepare_dataset(args.data_zip, args.dataset_dir.resolve())
    project = args.project.resolve()
    runs = project / args.name
    if runs.exists():
        raise FileExistsError(runs)

    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for A779 training")
    settings = {
        "data": str(data), "imgsz": 960, "epochs": args.epochs, "batch": args.batch,
        "device": 0, "workers": args.workers, "cache": "ram", "amp": True,
        "seed": 26, "deterministic": True, "patience": 20,
        "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01, "weight_decay": 0.0005,
        "cos_lr": True, "mosaic": 0.5, "close_mosaic": 10, "mixup": 0.0,
        "copy_paste": 0.0, "degrees": 0.0, "translate": 0.05, "scale": 0.3,
        "fliplr": 0.5, "flipud": 0.0, "plots": True,
        "project": str(project), "name": args.name,
    }

    def write_run_info(trainer):
        info = {
            "data_sha256": DATA_SHA256, "weights_sha256": WEIGHTS_SHA256,
            "weights": str(weight), "torch": torch.__version__, "cuda": torch.version.cuda,
            "ultralytics": ultralytics.__version__, "gpu": torch.cuda.get_device_name(0),
            "amp_enabled": bool(trainer.amp), "config": settings,
        }
        (Path(trainer.save_dir) / "a779_run.json").write_text(json.dumps(info, indent=2) + "\n")

    def start_epoch(trainer):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        trainer.a779_epoch_start = time.perf_counter()

    def finish_epoch(trainer):
        torch.cuda.synchronize()
        seconds = time.perf_counter() - trainer.a779_epoch_start
        perf = {"epoch": trainer.epoch + 1, "seconds": seconds,
                "images_per_second": SPLITS["train"]["images"] / seconds,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
        with (Path(trainer.save_dir) / "epoch_performance.jsonl").open("a") as file:
            file.write(json.dumps(perf) + "\n")
    model = YOLO(str(weight), task="obb")
    model.add_callback("on_pretrain_routine_end", write_run_info)
    model.add_callback("on_train_epoch_start", start_epoch)
    model.add_callback("on_train_epoch_end", finish_epoch)
    model.train(**settings)
    best = runs / "weights/best.pt"
    if not best.is_file() or not (runs / "results.csv").is_file():
        raise RuntimeError(f"Training results missing: {runs}")
    print(f"A779_FINISHED {best} sha256={sha256(best)}", flush=True)


if __name__ == "__main__":
    main()
