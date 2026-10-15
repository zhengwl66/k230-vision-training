"""Shared local-only helpers. Argument parsing and dry runs need no ML packages."""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys

DEFAULT_CLASSES_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'classes.json'
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def fail(message):
    raise ValueError(message)


def entrypoint(main):
    try:
        main()
    except KeyboardInterrupt:
        print("\n已停止。已经写入的训练文件保留；不会自动重启训练。", file=sys.stderr)
        sys.exit(130)
    except (ValueError, FileNotFoundError, ImportError, RuntimeError, OSError) as exc:
        print("\n操作未完成：" + str(exc), file=sys.stderr)
        sys.exit(2)


def file_path(value, suffix=None):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        fail("找不到文件：" + str(path))
    if suffix and path.suffix.lower() != suffix:
        fail("文件后缀必须是 " + suffix + "：" + str(path))
    return path


def new_output(value):
    path = Path(value).expanduser().resolve()
    if path.exists():
        fail("输出目录已经存在，为避免覆盖，请更换 --output：" + str(path))
    return path


def validate_names(names):
    if not isinstance(names,list) or not names or any(not isinstance(n,str) or not n.strip() for n in names):
        fail('类别必须是非空名称列表。')
    if len(set(names)) != len(names):
        fail('类别名称不得重复。')
    return names


def read_classes(value):
    return validate_names(json.loads(file_path(value).read_text(encoding='utf-8-sig')))


def read_data(value, needed=("train", "val")):
    """Accept local YOLO directories only; never forward a download directive."""
    source = file_path(value)
    raw = source.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError:
            fail("此 data.yaml 不是本包生成的 JSON 兼容格式；请安装 PyYAML 后读取普通 YAML。")
        data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        fail("data.yaml 应为包含 path、train、val、test、names 的字典。")
    if "download" in data:
        fail("本工具只使用本地数据，data.yaml 不得包含 download 指令。")
    names = data.get("names")
    if isinstance(names, dict):
        try:
            indexed = {int(k): v for k, v in names.items()}
            names = [indexed[i] for i in range(len(indexed))]
        except (ValueError, TypeError, KeyError):
            fail("names 字典的类别编号必须连续，从0开始。")
    names=validate_names(names)
    if data.get("nc", len(names)) != len(names):
        fail("nc 与 names 长度不一致。")
    raw_root = data.get("path", ".")
    if not isinstance(raw_root, str) or "://" in raw_root:
        fail("path 必须为本地数据目录。")
    root = Path(raw_root).expanduser()
    root = (root if root.is_absolute() else source.parent / root).resolve()
    metadata=root/'classes.json'
    if metadata.is_file() and read_classes(metadata)!=names:
        fail('data.yaml 与 classes.json 类别顺序不同，请重新生成一致的数据。')
    resolved = {"path": str(root), "names": names, "nc": len(names)}
    counts = {}
    for split in needed:
        part = data.get(split)
        if not isinstance(part, str) or "://" in part:
            fail(split + " 必须指向本地图片目录。")
        folder = Path(part)
        folder = (folder if folder.is_absolute() else root / folder).resolve()
        if not folder.is_dir():
            fail("找不到 " + split + " 图片目录：" + str(folder))
        pictures = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
        if not pictures:
            fail(split + " 没有图片。")
        # Match Ultralytics' images/... -> labels/... convention.
        per_class = [0] * len(names)
        for picture in pictures:
            components = list(picture.parts)
            positions = [i for i, x in enumerate(components) if x == "images"]
            if not positions:
                fail("数据应放在 images/<split>/ 与 labels/<split>/ 下：" + str(picture))
            components[positions[-1]] = "labels"
            label = Path(*components).with_suffix(".txt")
            if not label.is_file():
                fail("图片缺少标签文件（背景图也需要空 txt）：" + str(label))
            for line_no, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    items = line.split()
                    if len(items) != 5:
                        raise ValueError()
                    cls = int(items[0])
                    x, y, w, h = map(float, items[1:])
                    if not 0 <= cls < len(names):
                        raise ValueError()
                    if not all(math.isfinite(v) for v in (x, y, w, h)):
                        raise ValueError()
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        raise ValueError()
                    if x - w / 2 < -1e-5 or x + w / 2 > 1.00001 or y - h / 2 < -1e-5 or y + h / 2 > 1.00001:
                        raise ValueError()
                except (ValueError, OverflowError):
                    fail("标签格式/范围错误：{} 第{}行".format(label, line_no))
                per_class[cls] += 1
        if split in ("train", "val") and any(n == 0 for n in per_class):
            fail(split + " 存在没有标注的类别：" + ", ".join(names[i] for i, n in enumerate(per_class) if n == 0))
        counts[split] = {"images": len(pictures), "objects_per_class": per_class}
        resolved[split] = str(folder)
    return source, resolved, counts


def local_images(value):
    path = Path(value).expanduser().resolve()
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if path.is_dir():
        images = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if images:
            return images
    fail("--source 需要一个本地图片文件或包含图片的目录；不读取网络、视频或摄像头。")


def check_numbers(args):
    for name in ("epochs", "batch", "imgsz", "patience", "max_det"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            fail("--" + name.replace("_", "-") + " 必须大于0。")
    if getattr(args, "imgsz", 320) != 320:
        fail("本包部署约定固定 imgsz=320；不同输入尺寸需要同步调整K230预处理。")
    for name in ("conf", "iou", "fraction"):
        value = getattr(args, name, None)
        if value is not None and not 0 < value <= 1:
            fail("--" + name + " 必须在 (0,1] 内。")
    device = getattr(args, "device", "auto")
    if device not in ("auto", "cpu") and not device.isdigit():
        fail("--device 使用 auto、cpu 或单张 NVIDIA 显卡编号，例如 0。")


def versions():
    result = {"python": sys.version.split()[0]}
    for package in ("ultralytics", "torch", "torchvision", "numpy", "Pillow", "onnx", "onnxruntime"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    return result


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def show_plan(config):
    config = dict(config)
    config["installed_versions"] = versions()
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print("仅检查配置：没有开始训练、下载权重或创建输出目录。")


def load_engine(output, online=False):
    missing = [name for name in ("torch", "ultralytics") if versions().get(name) == "not installed"]
    if missing:
        fail("缺少训练依赖：" + ", ".join(missing) + "。请先按说明创建Python环境并安装 training/requirements.txt；--dry-run 不需要这些依赖。")
    installed = versions()["ultralytics"]
    if installed != "8.3.235":
        fail("本包固定使用 ultralytics==8.3.235，当前为 " + installed + "。请在独立环境按 requirements.txt 安装，避免接口变化。")
    output.mkdir(parents=True,exist_ok=True)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "false" if online else "true"
    os.environ["YOLO_CONFIG_DIR"] = str(output / "ultralytics_config")
    try:
        import torch
        from ultralytics import YOLO, settings
    except (ImportError, OSError) as exc:
        fail("训练库无法加载（可能缺少依赖或显卡运行库）：" + str(exc))
    settings.update({"sync": False, "wandb": False, "comet": False, "mlflow": False, "clearml": False, "neptune": False, "hub": False})
    return torch, YOLO


def resolve_device(torch, requested):
    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if requested != "cpu" and (not torch.cuda.is_available() or int(requested) >= torch.cuda.device_count()):
        fail("指定的CUDA显卡不可用。可用 --device cpu，或安装匹配的CUDA版PyTorch。")
    return requested


def assert_yolo11n(model, expected_names=None):
    if model.task != "detect":
        fail("只支持目标检测权重，不能使用分类/分割模型。")
    yaml_info = getattr(model.model, "yaml", {})
    yaml_file = str(yaml_info.get("yaml_file", "")).lower()
    scale = yaml_info.get("scale", "")
    if "yolo11" not in yaml_file or (scale and scale != "n") or (not scale and "yolo11n" not in yaml_file):
        fail("权重架构不是可确认的YOLO11n；本包不会自动换用其他YOLO版本。")
    if expected_names is not None:
        names = model.names
        names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
        if names != expected_names:
            fail("权重中的类别名称/顺序与本次类别文件不一致；请使用匹配的 best.pt 和 classes.json。")


def clean_metrics(metrics):
    results = {}
    for key, value in metrics.results_dict.items():
        value = float(value)
        results[key] = value if math.isfinite(value) else None
    return results
