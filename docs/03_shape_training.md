# 3. 七类形状训练

先完成数据检查。所有命令在仓库根目录执行，输出目录必须是新的。

## 配置预检查与补训

```powershell
.\.venv\Scripts\python.exe training/train_shape.py --data data/prepared/v2/data.yaml --run runs/v2 --dry-run
.\.venv\Scripts\python.exe training/train_shape.py --data data/prepared/v2/data.yaml --run runs/v2 --device 0
```

默认从models/current/shape_yolo11n.pt继续补训，CPU使用`--device cpu`。也可用`--weights`明确指定可信的其他七类YOLO11n权重。路径均可含空格，给参数加引号即可。

默认配方见configs/shape_train.json：65轮、batch32、320输入、AdamW、初始学习率0.00025、patience18、固定seed2026101503、AMP关闭、workers0、mosaic0.20。显存不足用`--batch 8`，快速检查可用`--epochs 1`；这些会改变训练结果，不能当作原版完整复现。

从随机初始化训练：

```powershell
.\.venv\Scripts\python.exe training/train_shape.py --data data/prepared/v2/data.yaml --run runs/scratch --scratch --device 0
```

随机初始化使用固定YOLO11n结构，不下载其他权重。小规模实拍数据通常更适合基于已有权重补训；是否采用新模型由验证集决定。

输出runs/v2/shape_training/fit/weights/best.pt、last.pt和训练计划/日志。best.pt是Ultralytics验证指标最优，不一定是实际分拣阈值下最优。

## 按部署阈值选模型

```powershell
.\.venv\Scripts\python.exe training/select_shape.py --data data/prepared/v2/data.yaml --run runs/v2
```

脚本导出best/last静态ONNX，并与仓库当前ONNX比较，只使用val。默认在conf0.55和0.70检查精确率不比基线降低超过0.5个百分点、整体F1不降低超过2.5个百分点，再按平均F1择优。结果在shape_selection.json，正式权重在selected_onnx。

该便携选模规则是整理后的通用入口；历史20261015还分别检查过旧批次表现，见结果文档。实际补训务必继续核对旧七类数据和新数据的逐类变化，不能只看一个总分。

需要单独导出可信PT时：

```powershell
.\.venv\Scripts\python.exe training/export_onnx.py --weights runs/v2/shape_training/fit/weights/best.pt --output runs/export_v2
```

它导出320×320、batch1、opset12、FP32、固定形状、不包含NMS的ONNX，并检查架构是YOLO11n。正式部署优先使用验证后选中的权重。
