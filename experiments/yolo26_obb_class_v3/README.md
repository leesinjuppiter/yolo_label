# A600 和 A779 异物类别修订对照

类别规则移除 cap，木条并入板状物。训练输出仍为单类 `foreign`。A600 有 600 张训练图，A779 在同一批 4911 图上增加 179 张 LSW。两组使用同一份 `val_v3`，共 300 张、113 个真实异物、191 张确认阴性。v3 在 v2 的固定图和异物几何基础上更新子类。

`prepare_revision.py` 校验两份 v2 ZIP 的 SHA-256，然后在服务器 `class_v3/` 生成新版。训练中仅有一张旧 cap-only 图改为空标签，其他训练框几何一致。数据分集、改动明细与哈希记录在 `class_v3/revision.json`。

```bash
python prepare_revision.py ~/autodl-tmp/yolo_obb_1122
bash run.sh A600
bash run.sh A779
```

两组使用官方 YOLO26n-OBB 通用预训练权重，输入 960、batch 24、seed 26，AMP 和 RAM 图像缓存开启，8 个加载 worker，训练预算 100 轮，早停耐心 20 轮。模型选择与重评都使用新版验证。训练完成后自动用最佳权重跑一对多 + NMS，按旋转框多边形 IoU ≥ 0.50 一对一匹配，并记录允许 0、1、2 张阴性误报时的检出数、漏检样本和逐图预测。
