# 4911 A600 异物检测基线

使用官方 YOLO26n-OBB 通用预训练权重训练单类 `foreign`。A600 ZIP 包含训练 600 张和固定验证 300 张。脚本校验 ZIP 与权重的 SHA-256，核对标签、正负样本数和采集组，再从 ZIP 准备数据。B600 和独立测试集留给后续实验。

## 云端运行

运行环境为 Python 3.12、Ultralytics 8.4.115、PyTorch 2.8.0+cu128。默认在 `~/autodl-tmp/yolo_obb_1122` 读取 A600 ZIP，标准权重位于 `~/autodl-tmp/weights/yolo26n-obb.pt`。该权重可从 [Ultralytics 官方发布](https://github.com/ultralytics/assets/releases/tag/v8.4.0) 获取。

```bash
bash run.sh probe
bash run.sh train <实测选定的批大小>
```

测速分别用 batch 24 和 40 从预训练权重跑 2 轮，比较每轮训练吞吐、GPU 利用率和显存，并确认验证及权重写入成功。正式 A600 从同一通用权重独立开始。输入为 960 像素，训练预算 100 轮，早停耐心为 20 轮，随机种子为 26。启用 AMP 检查、RAM 图像缓存与 8 个数据加载 worker。实际参数与 AMP 状态见运行目录中的 `args.yaml` 和 `a600_run.json`。后续 B600 对照应复用同样的输入和训练设置。

```bash
python evaluate.py --weights /path/to/run/weights/best.pt \
  --dataset-dir ~/autodl-tmp/yolo_obb_1122/A600 \
  --output-dir /path/to/run/val_eval
```

`results.csv` 记录 Ultralytics 每轮指标，`epoch_performance.jsonl` 记录训练吞吐和峰值预留显存，`logs/` 保留 GPU 采样。验证脚本按原图上的多边形 IoU ≥0.50，一对一匹配异物框，报告最多 0、1、2 张确认阴性误报时的异物实例召回、阳性帧放行数和对应漏检。推理计时以已解码图像为输入，预热 10 次后记录 batch=1 的平均、P50、P95 毫秒。验证预测和置信度保存在 `val_predictions.json`。
