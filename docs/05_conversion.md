# 5. 模型转换与独立验证

训练环境导出的ONNX使用RGB浮点输入/255。K230 kmodel使用RGB planar uint8，由模型内部除255，调用方不能再次归一化。形状模型输入[1,3,320,320]，输出[1,11,2100]：xywh+七类分数，没有objectness或NMS。预处理等比缩放、居中补边114，后处理还原到实际相机像素。

姿态输入[1,3,96,96]，输出[1,6]。校准图应是从train标注框裁剪并缩放后的RGB物块图。

## 训练集校准图

```powershell
.\.venv\Scripts\python.exe training/make_calibration.py --dataset data/prepared/v2 --kind shape --count 112 --output runs/v2/calibration_shape
.\.venv\Scripts\python.exe training/make_calibration.py --dataset data/prepared/v2 --kind pose --count 482 --output runs/v2/calibration_pose
```

只从train选择；数量不足时使用现有train图，至少32张。482是历史姿态训练数的示例，不是所有人必须凑的固定数。确保覆盖所有类别、颜色、大小和状态，别用val/test作校准。

## 在nncase环境转换

```powershell
.\.venv-convert\Scripts\python.exe training/compile_kmodel.py --kind shape --model runs/v2/selected_onnx/shape_yolo11n.onnx --calibration runs/v2/calibration_shape --output runs/v2/kmodel_shape
.\.venv-convert\Scripts\python.exe training/compile_kmodel.py --kind pose --model runs/v2/selected_pose/state.onnx --calibration runs/v2/calibration_pose --output runs/v2/kmodel_pose
```

使用已验证的nncase2.11.0配方：预处理启用、输入uint8、NoClip、激活uint8、权重int16，先检查ONNX常量折叠前后等价。输出shape.kmodel或state.kmodel和conversion.json。不能将2.11模型用于不匹配的板端运行时。

## 独立测试

模型、阈值和融合参数确定后，再打开test：

```powershell
.\.venv\Scripts\python.exe training/evaluate.py --model runs/v2/selected_onnx/shape_yolo11n.onnx --data data/prepared/v2/data.yaml --split test --output runs/v2/test_onnx.json --pose-model runs/v2/selected_pose/state.onnx --state-json runs/v2/state_model.json --pose-manifest data/prepared/v2/pose_manifest.json
```

输出conf0.50/0.55/0.70下TP、FP、FN、precision、recall、F1及逐类统计。姿态报告包含形状漏检、姿态未知、姿态误判和联合正确，不使用真值框代替预测框。

用转换环境的官方模拟器检查kmodel：先在该环境额外安装`opencv-python==4.11.0.86`与`PyYAML==6.0.2`，再执行：

```powershell
.\.venv-convert\Scripts\python.exe training/evaluate.py --model runs/v2/kmodel_shape/shape.kmodel --data data/prepared/v2/data.yaml --split val --output runs/v2/val_kmodel.json --pose-model runs/v2/kmodel_pose/state.kmodel --state-json runs/v2/state_model.json --pose-manifest data/prepared/v2/pose_manifest.json
```

模拟器全量运行会较慢。可另准备代表性验证子集，覆盖七类、混摆、空盘和状态，不按模型结果挑选图片。与ONNX同子集对比，人工评估量化是否引入明显退步。电脑OpenCV与板端AI2D预处理并未宣称位级一致，最后仍需实体K230检查。

## 生成自己的更新包

在独立测试结果确认可接受后：

```powershell
.\.venv\Scripts\python.exe training/build_update.py --shape runs/v2/kmodel_shape/shape.kmodel --state runs/v2/kmodel_pose/state.kmodel --state-json runs/v2/state_model.json --test-report runs/v2/test_onnx.json --version vision-custom-v2 --output runs/update_v2
```

输出目录及同名ZIP均须不存在。打包时计算文件校验值，并把board_verified重置为false，避免把旧版实机通过记录套给新权重。脚本不会自动写板或删除SD文件。
