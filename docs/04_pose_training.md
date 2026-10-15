# 4. 三类物块的状态训练

状态判断独立于七类检测：先检测物块，再对三棱柱、五棱柱、圆柱提取裁剪图和几何前景特征。姿态CNN输入96×96，输出三对logit；几何RBF分类器和CNN按验证集选择的权重融合，证据不足输出UNKNOWN。预览还有连续两帧一致确认。

形状ID2/3/5对应五棱柱/三棱柱/圆柱；姿态标签-1=UPRIGHT、+1=LYING。不能改变这套顺序。

## 需要什么数据

数据整理生成的pose_manifest.json只收录description明确填写UPRIGHT/LYING的框。每个形状在train、val、test中都必须同时包含两个状态。三个集合按整张图隔离；混摆里的不同框不能把同一图片分进不同集合。

单件与混摆均可提供明确状态，疑难图可以只标形状。不要按“文件夹叫三棱柱1”猜状态。

## 训练CNN

```powershell
.\.venv\Scripts\python.exe training/train_pose.py --manifest data/prepared/v2/pose_manifest.json --run runs/v2
```

默认使用仓库当前形状网络的前5层作为结构，并载入当前pose_classifier.pt补训。90轮上限，验证早停，torch/NumPy随机种子2026101504，stem与head采用不同学习率。`--epochs`可用于短流程检查；`--scratch`表示姿态参数随机初始化，但形状stem结构仍来自指定的形状模型。

若显式换基线，可使用`--shape-weights`和`--init`。仅加载可信的PT文件。

## 几何特征与融合选择

```powershell
.\.venv\Scripts\python.exe training/prepare_pose_features.py --run runs/v2
.\.venv\Scripts\python.exe training/select_pose.py --run runs/v2 --version pose-custom-v2
```

特征提取与板端state_sample.py使用同一个实现。只对train做亮度、缩放、框抖动增强；几何gamma/regularization、CNN新旧版本及融合权重/拒识间隔只按val选择。历史数据来自--base时标为prior_pose，选模会检查保留它的验证表现。

正式输出为runs/v2/state_model.json及runs/v2/selected_pose/state.onnx、pose_classifier.pt。选中旧CNN也正常，不要强制把所有训练产生的文件都当新版。state_model.json和选中的CNN需要配套转换、部署。

最终测试应通过检测器预测框再判断姿态，不能仅报告人工真值框裁剪后的分类准确率。联合测试命令见下一章。
