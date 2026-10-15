# 颜色识别算法来源与许可

本模块独立实现了白托盘参考校正、彩色前景连通区域采样、逐像素亮度归一化、Lab 转换、CIEDE2000 色差及颜色参考样本匹配。K230 上只使用 `math`、列表和现有 RGB888 图像接口，无需安装 OpenCV。

参考来源：

- [OpenCV GrayworldWB 实现](https://github.com/opencv/opencv_contrib/blob/4.x/modules/xphoto/src/grayworld_white_balance.cpp)：参考逐通道增益与饱和度筛选的思路。本项目使用白托盘的中性像素作为参考，没有复制或运行该 C++ 文件，也没有采用对整幅蓝色物块画面直接求灰世界均值的做法。
- [Greg Fiumara 的 CIEDE2000 实现与测试](https://github.com/gfiumara/CIEDE2000)：用于核对 CIEDE2000 公式和 34 组 Sharma 标准测试向量。实现为本项目独立编写的 MicroPython 版本；测试向量提取自其测试文件。MIT 许可全文附于 `licenses/CIEDE2000_LICENSE.txt`。
- [OpenCV GrayworldWB 文档](https://docs.opencv.org/4.13.0/d7/d71/classcv_1_1xphoto_1_1GrayworldWB.html)：饱和度阈值和白平衡接口说明。

上游文本仅用于阅读与测试数据核对，没有执行下载来的源代码。下载地址和文件 SHA256 记录在 `research_sources.json`。

本项目的颜色参考值来自本机训练样本；推理时不会按物块形状或所在位置推断颜色。不同深浅的蓝色共同输出 `BLUE` / “蓝色”。`color_support` 是颜色分界的相对支持度，不是经过校准的正确概率。
