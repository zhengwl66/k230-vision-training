# 2. 拍照、标注与数据整理

## 拍哪些照片

使用最终分拣时的相机、镜头、安装高度和角度。先固定托盘位置与照明。每个形状拍不同位置、旋转角度、距离和遮挡情况，覆盖中间及四角；补拍容易漏检或混淆的情况。

三棱柱、五棱柱、圆柱都要拍立放和横躺。每种状态分几次采集、换位转动，最好有独立日期或光照的测试批次。蓝色/浅蓝/深蓝/紫色要交叉摆放，避免某个形状永远对应一种颜色。混摆包含相同颜色不同形状、不同颜色相同形状，以及有间隙/轻微接触的情况。

手离开画面后拍摄；保存未画框、未缩放的原始JPEG。空盘、阴影、螺丝和托盘边缘可作为负样本。不要把识别预览截图当训练原图。

在CanMV运行`deployment/capture_photos.py`前编辑GROUP、COUNT、INTERVAL_MS。它使用板端原uvc_camera模块，把原始MJPEG保存到`/sdcard/train_photos`的新批次目录。拍完下载照片到电脑，停止采集再运行识别。分拣相机若采用不同采集档位，先用对应原相机模块采集，不能擅自改摄像头设置。

## 标注

可安装LabelMe（建议放单独环境，或使用已有标注工具）。矩形紧贴完整物块外轮廓，留少量边缘；不要把阴影一起框入。同张图每件物块各标一个框。类别严格用README七类英文标签或对应中文，不能按混摆文件夹名称猜类别。

对三种需要状态的物块，在每个矩形的`description`中填`UPRIGHT`或`LYING`。立放指物块以底面支撑站立，横躺指侧面支撑，不是图像里的长边朝向。看不清状态时description留空，仅用于形状训练；不能靠颜色或物块形状自动编造姿态标签。

确认空盘图才保存`shapes: []`的JSON。混摆尚未标注、导出TXT为空，不等于空盘。

按采集批次组织，每张图片旁边有同名JSON：

```text
data/raw/
  session_a_triangular_upright/
    000001.jpg
    000001.json
  session_b_mixed/
    000001.jpg
    000001.json
```

脚本以JPEG和同名JSON配对，读取JSON内类别；不会依赖JSON里可能已失效的原电脑imagePath。文件名可以重复出现在不同批次中。仅支持两个点的rectangle，其他形状需先转换并复核。

## 生成训练集

```powershell
.\.venv\Scripts\python.exe training/prepare_dataset.py --source data/raw --output data/prepared/v1 --allow-empty
.\.venv\Scripts\python.exe training/audit_dataset.py --data data/prepared/v1/data.yaml
```

每个批次至少15张唯一图片。默认按自然文件名顺序、连续5张成组，最后两个完整组分别为验证与测试，剩余用于训练。完全相同图片按SHA256去重，有冲突的标注报错；坏图、越界框、未知类别报错；无JSON照片记录并排除，不会当负样本。

生成images、labels、data.yaml、manifest.json、pose_manifest.json和import_report.json。manifest使用可迁移的相对路径。背景图有空TXT。

按批次分组只能减少相邻帧混入，无法证明跨场景泛化。用于最终验收时，另保留整个新采集批次作为测试集，并人工核对划分。不要在看过测试结果后反复调参，再宣称它是独立测试。

## 补充到历史数据

`--base`接收本脚本此前生成的数据目录，原划分原样保留：

```powershell
.\.venv\Scripts\python.exe training/prepare_dataset.py --source data/raw_new --base data/prepared/v1 --output data/prepared/v2 --allow-empty
.\.venv\Scripts\python.exe training/audit_dataset.py --data data/prepared/v2/data.yaml
```

重复图片保留原划分，不再次加入。冲突框需先人工确认，不会静默覆盖旧标注。训练/验证应有全部七类；只拍了四类时需要与此前七类数据合并，以减少旧类遗忘。

如果已有标准YOLO导出，可直接按images/train、labels/train等结构组织并使用audit_dataset/train_shape；data.yaml的names必须和configs/classes.json一致。姿态训练还需要明确的逐物块状态标注，普通YOLO TXT不包含状态信息。
