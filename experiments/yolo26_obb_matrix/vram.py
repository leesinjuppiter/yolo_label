"""Measure a dense, real-image 960-pixel training and validation workload."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

TARGET_MIB = 10_752
START_BATCH = {"s": 11, "m": 4, "l": 4, "x": 2}


class ProbeFinished(Exception):
    pass


class MemoryLimit(Exception):
    pass


def stress_dataset(source, root):
    config = json.loads((source / "data.yaml").read_text())
    for split, count in (("train", 144), ("val", 96)):
        directory = source / f"images/{split}"
        ranked = sorted(directory.iterdir(), key=lambda p: (
            -(source / f"labels/{split}/{p.stem}.txt").read_bytes().count(b"\n"), p.name))
        for image in ranked[:count]:
            for relative in (f"images/{split}/{image.name}", f"labels/{split}/{image.stem}.txt"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    os.link(source / relative, path)
    config["path"] = str(root)
    path = root / "data.yaml"
    path.write_text(json.dumps(config))
    return path


def run_probe(settings_path, output):
    import torch
    from ultralytics import YOLO

    spec = json.loads(settings_path.read_text())
    settings = spec["settings"]
    observations = {"train_batches": 0, "val_batches": 0, "peak_mib": 0, "actual_batch": settings["batch"]}

    def sample():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        external = total - free - torch.cuda.memory_reserved()
        used = max(total - free, external + torch.cuda.max_memory_reserved())
        observations["peak_mib"] = max(observations["peak_mib"], round(used / 1024**2))
        if observations["peak_mib"] > TARGET_MIB:
            raise MemoryLimit()

    def train_batch(trainer):
        observations["train_target"] = min(12, len(trainer.train_loader))
        observations["actual_batch"] = trainer.batch_size
        observations["train_batches"] += 1
        sample()
        if observations["train_batches"] >= observations["train_target"]:
            trainer.stop = True

    def val_batch(validator):
        observations["val_target"] = min(6, len(validator.dataloader))
        observations["val_batches"] += 1
        sample()
        if observations["val_batches"] >= observations["val_target"]:
            raise ProbeFinished()

    model = YOLO(spec["weights"], task="obb")
    model.add_callback("on_train_batch_end", train_batch)
    model.add_callback("on_val_batch_end", val_batch)
    # Fail the probe explicitly if Ultralytics internally lowers the requested batch.
    try:
        model.train(**settings)
    except ProbeFinished:
        pass
    except MemoryLimit:
        observations["limited"] = True
    except RuntimeError as error:
        message = str(error).lower()
        if "out of memory" not in message or "cuda" not in message:
            raise
        observations["oom"] = True
    output.write_text(json.dumps(observations))


def select_batch(inputs_path, dataset, scale, weight, settings):
    work = inputs_path.parent / "calibration" / "vram_20261124" / scale
    work.mkdir(parents=True, exist_ok=True)
    report = work / "selection.json"
    if report.exists():
        return json.loads(report.read_text())
    sources = json.loads(inputs_path.read_text())["datasets"]
    data = stress_dataset(Path(sources["B1189"]["path"]), work / "dense_b1189")
    trials = []
    best = None
    upper = 33
    batch = START_BATCH[scale]
    probe_python = Path(__file__).resolve()
    for index in range(6):
        slot = work / f"trial{index + 1}_batch{batch}"
        slot.mkdir(exist_ok=True)
        candidate = copy.copy(settings)
        candidate.update(data=str(data), batch=batch, epochs=1, warmup_epochs=3,
                         save=False, save_period=-1, plots=False, cache=False, workers=4,
                         close_mosaic=0, project=str(slot), name="probe", exist_ok=True)
        options, result = slot / "settings.json", slot / "measurement.json"
        options.write_text(json.dumps({"weights": weight, "settings": candidate}))
        with (slot / "console.log").open("w") as log:
            subprocess.run([sys.executable, "-B", str(probe_python), str(options), str(result)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        measured = json.loads(result.read_text())
        safe = (measured["actual_batch"] == batch
                and 0 < measured.get("train_target", 0) <= measured["train_batches"]
                and 0 < measured.get("val_target", 0) <= measured["val_batches"]
                and measured["peak_mib"] <= TARGET_MIB
                and not measured.get("limited") and not measured.get("oom"))
        trials.append({"batch": batch, "safe": safe, **measured})
        print(f"VRAM_PROBE {dataset} {scale} {trials[-1]}", flush=True)
        if safe:
            best = batch
            peak = measured["peak_mib"]
            per_image = max(256, peak / (2 * batch))
            if batch + 1 >= upper or peak + per_image > TARGET_MIB:
                break
            batch = min(upper - 1, batch + min(4, max(1, int((TARGET_MIB - peak) / per_image))))
        else:
            upper = batch
            lower = best or 0
            if batch <= lower + 1:
                break
            batch = max(lower + 1, (lower + batch) // 2)
    if best is None:
        raise RuntimeError(f"No verified batch below VRAM ceiling for {scale}. See {work}")
    selection = {"batch": best, "policy": "dense_B1189_train_and_val_20261124",
                 "target_mib": TARGET_MIB, "trials": trials}
    temp = report.with_suffix(".part")
    temp.write_text(json.dumps(selection, indent=2) + "\n")
    temp.replace(report)
    print(f"VRAM_BATCH_SELECTED {dataset} {scale} batch={best}", flush=True)
    return selection


if __name__ == "__main__":
    run_probe(*(Path(arg) for arg in sys.argv[1:]))
