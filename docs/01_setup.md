# 1. 环境搭建

以下命令从仓库根目录执行。可把仓库放在D/E盘任意路径；脚本不要求原电脑目录存在。

## 下载的软件

| 软件 | 官方下载或安装方式 | 用途 |
|---|---|---|
| Python 3.13 x64 | https://www.python.org/downloads/windows/ | 训练与电脑验证 |
| Python 3.12 x64 | 同上；可与3.13共存 | nncase独立转换环境 |
| VS Code | https://code.visualstudio.com/ | 编辑、连接开发板 |
| CanMV插件 | VS Code扩展市场搜索CanMV，选择Kendryte发布者 | 上传文件和运行板端代码 |
| Git | https://git-scm.com/downloads | 克隆/更新仓库；也可网页Download ZIP |
| nncase KPU 2.11.0 | https://github.com/kendryte/nncase/releases/tag/v2.11.0 | 下载Windows x64的nncase_kpu wheel |
| .NET Runtime 7 x64 | https://dotnet.microsoft.com/en-us/download/dotnet/7.0 | 转换环境运行时，选择Runtime的ZIP |

只复制已有kmodel运行K230时，不需要电脑安装PyTorch、nncase或.NET。

## 训练环境

安装Python时包含Python Launcher，便于使用`py -3.13`。没有Launcher时将命令替换为已安装Python的完整路径。

```powershell
py -3.13 -m venv .venv
```

NVIDIA显卡选本项目已验证的CUDA 12.8构建：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

仅CPU则替换为：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
```

两者选一种，随后执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/train.txt
.\.venv\Scripts\python.exe -c "import torch,ultralytics; print(torch.__version__,ultralytics.__version__); print('CUDA',torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

固定Ultralytics 8.3.235、torch 2.8.0、torchvision 0.23.0、numpy 2.2.6等版本。电脑训练不导入板端nncase_runtime或ulab，K230运行时也不安装PyTorch。

## 独立转换环境

不能将nncase需要的numpy1.26与训练环境numpy2.2混装：

```powershell
py -3.12 -m venv .venv-convert
.\.venv-convert\Scripts\python.exe -m pip install -r requirements/convert.txt
# 先从上表官方发布页下载wheel到downloads目录，再执行
.\.venv-convert\Scripts\python.exe -m pip install .\downloads\nncase_kpu-2.11.0-py2.py3-none-win_amd64.whl
```

把.NET Runtime7 ZIP完整解压到仓库`.runtime/dotnet7`，里面应直接看到`dotnet.exe`和`shared`。也支持在本次终端设置`$env:DOTNET_ROOT='D:\实际.NET目录'`。转换脚本自动设置KPU插件路径。

```powershell
.\.venv-convert\Scripts\python.exe -c "import _nncase; print(_nncase.__version__)"
```

应输出2.11.0。目标板固件中的`nncase_runtime.version()`也必须匹配。其他固件请重新转换，不要仅修改配置里的版本字符串。

本机原训练环境和显卡信息是历史验证依据，不是每台电脑必须拥有同型号硬件。CPU可训练但更慢；不同设备训练结果不保证逐位相同。
