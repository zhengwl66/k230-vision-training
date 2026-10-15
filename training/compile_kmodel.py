"""Compile the seven-class ONNX with nncase 2.11.0 and training-only calibration.

Reference: Kendryte's official YOLO11 to_kmodel.py and nncase v2 Python API.
Unlike the reference image generator, calibration preserves centered letterbox.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def setup_runtime():
    dotnet = ROOT / '.runtime/dotnet7'
    if not dotnet.is_dir():
        dotnet = Path(os.environ.get('DOTNET_ROOT', 'C:/Program Files/dotnet'))
    if not dotnet.is_dir():
        raise RuntimeError('Install .NET7 x64 runtime or extract it to .runtime/dotnet7')
    site = Path(sys.prefix) / "Lib/site-packages"
    os.environ["DOTNET_ROOT"] = str(dotnet)
    os.environ["NNCASE_PLUGIN_PATH"] = str(site / "nncase/modules/kpu")
    os.environ["PATH"] = str(dotnet) + os.pathsep + str(site) + os.pathsep + os.environ.get("PATH", "")
    os.environ["DOTNET_PROCESSOR_COUNT"] = "12"


def image_tensor(path, side=320):
    import numpy as np
    from PIL import Image
    with Image.open(path) as original:
        image = original.convert("RGB")
        scale = min(side / image.width, side / image.height)
        width, height = round(image.width * scale), round(image.height * scale)
        resized = image.resize((width, height), Image.Resampling.BILINEAR)
        padded = Image.new("RGB", (side, side), (114, 114, 114))
        padded.paste(resized, ((side - width) // 2, (side - height) // 2))
    return np.ascontiguousarray(np.asarray(padded).transpose(2, 0, 1)[None], dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models/shape.onnx")
    parser.add_argument("--calibration", type=Path, default=ROOT / "calibration/images")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/kmodel_noclip")
    parser.add_argument("--calibrate-method", choices=("NoClip", "Kld"), default="NoClip")
    parser.add_argument('--kind',choices=('shape','pose'),default='shape')
    args = parser.parse_args()
    side=320 if args.kind=='shape' else 96
    args.output.mkdir(parents=True, exist_ok=True)
    setup_runtime()
    import numpy as np
    import onnx
    import onnxruntime as ort
    import nncase
    import _nncase
    assert _nncase.__version__ == "2.11.0", _nncase.__version__
    start = time.monotonic()
    graph = onnx.load(str(args.model))
    onnx.checker.check_model(graph)
    optimized = args.output / "constant_folded.onnx"
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(optimized)
    options.intra_op_num_threads = 4
    reference = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(str(optimized), providers=["CPUExecutionProvider"])
    calib_paths = sorted(p for p in args.calibration.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))
    assert len(calib_paths) >= 32, "At least 32 train-only calibration images required"
    # Verify constant folding before importing the optimized graph into nncase.
    for path in calib_paths[::16]:
        x = image_tensor(path, side).astype(np.float32) / 255.0
        old = reference.run(None, {reference.get_inputs()[0].name: x})[0]
        new = candidate.run(None, {candidate.get_inputs()[0].name: x})[0]
        assert np.allclose(old, new, atol=1e-4, rtol=1e-4), str(path)
    print("ONNX_CONSTANT_FOLD_CHECK_OK", flush=True)
    co = nncase.CompileOptions()
    co.target = "k230"
    co.dump_dir = str(args.output / "compiler_dump")
    co.dump_ir = False
    co.dump_asm = False
    co.preprocess = True
    co.input_shape = [1, 3, side, side]
    co.input_type = "uint8"
    co.input_range = [0, 1]
    co.mean = [0, 0, 0]
    co.std = [1, 1, 1]
    co.input_layout = "NCHW"
    co.swapRB = False
    compiler = nncase.Compiler(co)
    print("COMPILER_CREATED", flush=True)
    compiler.import_onnx(optimized.read_bytes(), nncase.ImportOptions())
    print("ONNX_IMPORTED", flush=True)
    ptq = nncase.PTQTensorOptions()
    ptq.samples_count = len(calib_paths)
    ptq.calibrate_method = args.calibrate_method
    ptq.quant_type = "uint8"
    ptq.w_quant_type = "int16"
    # The installed 2.11 wrapper expects outer dimension = samples, inner = inputs.
    ptq.set_tensor_data([[image_tensor(path, side)] for path in calib_paths])
    compiler.use_ptq(ptq)
    print("COMPILING", len(calib_paths), "TRAIN_IMAGES", flush=True)
    compiler.compile()
    data = compiler.gencode_tobytes()
    destination = args.output / ("shape.kmodel" if args.kind=="shape" else "state.kmodel")
    destination.write_bytes(data)
    result = {
        "compiler": _nncase.__version__, "target": "k230", "onnx": str(args.model),
        "onnx_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "kmodel": str(destination), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "calibrate_method": args.calibrate_method, "quant_type": "uint8", "w_quant_type": "int16",
        "calibration_images": len(calib_paths), "calibration_from": "training split only",
        "input_shape": [1, 3, side, side], "input_dtype": "uint8", "input_color": "RGB planar",
        "external_preprocess": "centered letterbox320 pad114" if args.kind=="shape" else "RGB96x96 object crop",
        "normalization": "inside kmodel: uint8 / 255; caller must not divide again",
        "output_shape": [1, 11, 2100] if args.kind=="shape" else [1,6], "output": "xywh + 7 scores; no NMS" if args.kind=="shape" else "6 paired pose logits",
        "elapsed_seconds": time.monotonic() - start, "board_inference_verified": False,
        "versions": {name: importlib.metadata.version(name) for name in ("nncase", "nncase-kpu", "numpy", "onnx", "onnxruntime", "Pillow")},
    }
    (args.output / "conversion.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
