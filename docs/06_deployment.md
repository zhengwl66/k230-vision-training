# 6. K230预览、分拣接入与更新

## 当前模型怎么复制

连接K230调试口与摄像头，VS Code使用CanMV插件。停止脚本，先把板上当前视觉目录下载到电脑备份，再把仓库`deployment/board_files/sdcard/vision_update_20261013`整个文件夹复制到板端`/sdcard/`，覆盖同名目录。

四个文件必须一起更新：shape.kmodel、state.kmodel、state_model.json、inference_config.json。其余Python、类别表一并保留，不能只复制一个孤立权重。

固定目录名20261013用于兼容原分拣代码；实际模型版本由配置、校验值和启动日志确定。目前应为vision-supplement-20261015-v1，姿态pose-supplement-20261015-v1。

## 预览

若原相机模块缺失，再把deployment/board_files/sdcard/uvc_camera.py、recognize_camera.py复制到SD根目录。uvc_camera.py为原640×480 MJPEG 30FPS配置；相机不同的板先用自己的原模块，不能直接覆盖不同配置。

运行`/sdcard/recognize_camera.py`或模型目录内recognize_camera.py。终端应出现：

```text
VISION_MODEL_VERSION vision-supplement-20261015-v1
VISION_MODEL_PATH /sdcard/vision_update_20261013/shape.kmodel
NEW_CAMERA_MODEL_READY
```

CanMV预览显示中文形状、颜色、姿态和框。若要原分拣1280×960档位的视觉预览，运行模型目录recognize_sorting_camera.py；它依赖sorting_camera.py。两种预览均不启动机械臂。

其他相机可修改预览入口的CAMERA_MODULE_DIR和CAMERA_MODULE，接口需要camera_init、get_frame_rgb888、camera_deinit。更改相机后重新采图/验证，不复用旧角度的准确率结论。

## 队友的fenjian怎么调用新版

保留对方现有机械控制和标定，检查jiance或视觉入口：

```python
KMODEL_PATH = '/sdcard/vision_update_20261013/shape.kmodel'
```

姿态模块：

```python
MODEL_PATH = '/sdcard/vision_update_20261013/state_model.json'
```

state_model.json中的image_model应为state.kmodel，并位于同目录。检查sys.path上同名state_sample.py或shape_k230.py的优先级，确认实际导入的是预期文件。当前用户的k230_sort/state_sample.py已指向正式模型目录；用户现有jiance/fenjian未上传到本仓库，避免把机械参数误当通用配置。

同名旧模型直接备份后覆盖即可；放新目录但不改调用路径不会生效。复制后停止旧脚本并重新运行，或重新上电，让缓存的KPU对象重新加载文件。

当前用户分拣入口仍为`/sdcard/k230_sort/fenjian.py`，开机入口为`/sdcard/main.py`。jiance.CONF_TH=0.70、fenjian.PLAN_CONF_TH=0.50是用户当前值，仓库不自动改阈值或运动策略。

## 自己的板如何验收

先放空盘检查误检；七类单件分别检查；三棱柱/五棱柱/圆柱各测立放与横躺，再换位置和转角；深蓝/浅蓝/紫色互换位置并测四角；最后检查混摆。记录真实摆放、置信度、漏检、误检与未知，不能只确认“能显示画面”。

预览和真正分拣视觉接口都要测。视觉通过不等于机械抓取成功，需要再由机械程序完成闭环验收。

清理旧模型前下载并核对备份，只删自己项目明确不再引用的模型目录；厂商examples、固件、相机配置和机械标定不属于旧视觉权重。
