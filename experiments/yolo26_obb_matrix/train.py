"""Train one matrix cell with the previous 4911 OBB training recipe."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import time

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.torch_utils import strip_optimizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path)
    p.add_argument("dataset", choices=("A600", "B600", "A1189", "B1189"))
    p.add_argument("scale", choices=("s", "m", "l", "x"))
    p.add_argument("run", type=Path)
    p.add_argument("--batch", type=float, default=.82)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    inputs = json.loads(args.inputs.read_text())
    dataset = inputs["datasets"][args.dataset]
    weight = inputs["weights"][args.scale]
    with Path(weight["path"]).open("rb") as stream:
        assert file_digest(stream, "sha256").hexdigest() == weight["sha256"]
    settings = {
        "data": str(Path(dataset["path"]) / "data.yaml"), "imgsz": 960,
        "epochs": 100, "batch": int(args.batch) if args.batch >= 1 else args.batch,
        "device": 0, "workers": args.workers, "cache": "ram", "amp": True,
        "seed": 26, "deterministic": True, "patience": 20, "save_period": 20,
        "optimizer": "AdamW", "lr0": .001, "lrf": .01, "weight_decay": .0005,
        "cos_lr": True, "mosaic": .5, "close_mosaic": 10, "mixup": 0.0,
        "copy_paste": 0.0, "degrees": 0.0, "translate": .05, "scale": .3,
        "fliplr": .5, "flipud": 0.0, "plots": False,
        "project": str(args.run.parent), "name": args.run.name, "exist_ok": False,
    }
    last = args.run / "weights/last.pt"
    if last.exists():
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        if checkpoint.get("epoch", -1) < 0 or checkpoint.get("optimizer") is None:
            assert (args.run / "weights/best.pt").is_file() and (args.run / "results.csv").is_file()
            print("Completed training checkpoint already present", flush=True)
            return
        del checkpoint
        model = YOLO(str(last), task="obb")
        resume = True
    else:
        assert not args.run.exists(), f"Incomplete run with no resumable checkpoint: {args.run}"
        model = YOLO(weight["path"], task="obb")
        assert model.task == "obb"
        resume = False

    def metadata(trainer):
        path = Path(trainer.save_dir) / "matrix_run.json"
        if not path.exists():
            path.write_text(json.dumps({"dataset": args.dataset, "scale": args.scale,
                "data": dataset, "pretrained": weight, "test_sha256": inputs["test_sha256"],
                "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                "gpu": torch.cuda.get_device_name(0), "config": settings, "amp": bool(trainer.amp)}, indent=2)+"\n")

    def epoch_start(trainer):
        trainer.matrix_start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    def epoch_end(trainer):
        torch.cuda.synchronize()
        seconds = time.perf_counter() - trainer.matrix_start_time
        row = {"epoch": trainer.epoch + 1, "batch": trainer.batch_size, "seconds": seconds,
               "images_per_second": dataset["train_images"]/seconds,
               "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2}
        with (Path(trainer.save_dir) / "epoch_performance.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")

    def slim_periodic(trainer):
        path = Path(trainer.wdir) / f"epoch{trainer.epoch}.pt"
        if path.is_file():
            tmp = path.with_suffix(".tmp")
            strip_optimizer(path, tmp)
            tmp.replace(path)

    model.add_callback("on_pretrain_routine_end", metadata)
    model.add_callback("on_train_epoch_start", epoch_start)
    model.add_callback("on_train_epoch_end", epoch_end)
    model.add_callback("on_model_save", slim_periodic)
    if resume:
        model.train(resume=True)
    else:
        model.train(**settings)
    assert (args.run / "weights/best.pt").is_file() and last.is_file()
    print("TRAINING_FINISHED", args.run, flush=True)


if __name__ == "__main__":
    main()
