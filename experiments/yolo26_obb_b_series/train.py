"""Train a verified two-class OBB dataset from the official pretrained model."""
import argparse
import hashlib
import json
from pathlib import Path
import time

def sha256(path):
    with path.open("rb") as file:
        digest = hashlib.sha256()
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True, choices=("official",))
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save-period", type=int, default=10)
    args = parser.parse_args()

    spec = json.loads(args.inputs.read_text())
    weight = Path(spec["weights"][args.init]).resolve(strict=True)
    expected_sha = spec["weights_sha256"][args.init]
    if sha256(weight) != expected_sha:
        raise ValueError(f"Unexpected initial checkpoint: {args.init}")
    data = Path(spec["dataset"]).resolve(strict=True)
    project = args.project.resolve()
    runs = project / args.name
    if runs.exists():
        raise FileExistsError(runs)

    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for OBB training")
    settings = {
        "data": str(data), "imgsz": 960, "epochs": args.epochs, "batch": args.batch,
        "device": 0, "workers": args.workers, "cache": "ram", "amp": True,
        "seed": 26, "deterministic": True, "patience": 20, "save_period": args.save_period,
        "resume": False,
        "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01, "weight_decay": 0.0005,
        "cos_lr": True, "mosaic": 0.5, "close_mosaic": 10, "mixup": 0.0,
        "copy_paste": 0.0, "degrees": 0.0, "translate": 0.05, "scale": 0.3,
        "fliplr": 0.5, "flipud": 0.0, "plots": True,
        "project": str(project), "name": args.name,
    }

    def write_run_info(trainer):
        info = {
            "variant": spec["variant"], "initialization": args.init, "validation_revision": spec["validation_revision"],
            "source_zip_sha256": spec["dataset_sha256"], "frames_sha256": spec["frames_sha256"], "weights_sha256": expected_sha,
            "weights": str(weight), "torch": torch.__version__, "cuda": torch.version.cuda,
            "ultralytics": ultralytics.__version__, "gpu": torch.cuda.get_device_name(0),
            "amp_enabled": bool(trainer.amp), "config": settings,
        }
        (Path(trainer.save_dir) / "run_config.json").write_text(json.dumps(info, indent=2) + "\n")

    def start_epoch(trainer):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        trainer.epoch_start = time.perf_counter()

    def finish_epoch(trainer):
        torch.cuda.synchronize()
        seconds = time.perf_counter() - trainer.epoch_start
        perf = {"epoch": trainer.epoch + 1, "seconds": seconds,
                "images_per_second": spec["counts"]["train"]["images"] / seconds,
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
    print(f"TRAINING_FINISHED {best} sha256={sha256(best)}", flush=True)


if __name__ == "__main__":
    main()
