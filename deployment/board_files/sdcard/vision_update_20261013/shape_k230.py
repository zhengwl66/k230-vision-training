"""Seven-class K230 inference module. Importing this module starts no peripherals.

Input: RGB888 planar uint8, or an image with packed RGB888 pixels.
Model: nncase 2.11.0 kmodel, [1,3,320,320] uint8, internal /255.
Output: target dictionaries in original camera pixel coordinates.
"""
from box_adapter import centered_letterbox_geometry, detections_to_targets

CLASSES = ("hollow_cube", "tetrahedron", "pentagonal_prism", "triangular_prism",
           "cone", "cylinder", "hemisphere")


def _iou(a, b):
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def decode_raw(raw, confidence=0.5, nms_iou=0.45, max_detections=30):
    """Decode [1,11,2100], class-wise NMS, retaining padded-model xyxy pixels."""
    if tuple(raw.shape) != (1, 11, 2100):
        raise ValueError("Expected raw output shape [1,11,2100]")
    if not 0 <= confidence <= 1 or not 0 <= nms_iou <= 1:
        raise ValueError("Thresholds must be between 0 and 1")
    candidates = []
    for index in range(2100):
        cid, score = 0, float(raw[0, 4, index])
        for candidate in range(1, 7):
            value = float(raw[0, 4 + candidate, index])
            if value > score:
                cid, score = candidate, value
        if score < confidence:
            continue
        x, y = float(raw[0, 0, index]), float(raw[0, 1, index])
        width, height = float(raw[0, 2, index]), float(raw[0, 3, index])
        if width <= 0 or height <= 0:
            continue
        candidates.append({"xyxy": (x - width / 2, y - height / 2, x + width / 2, y + height / 2),
                           "confidence": score, "class_id": cid})
    candidates.sort(key=lambda row: -row["confidence"])
    kept = []
    for row in candidates:
        suppressed = False
        for other in kept:
            if row["class_id"] == other["class_id"] and _iou(row["xyxy"], other["xyxy"]) > nms_iou:
                suppressed = True
                break
        if not suppressed:
            kept.append(row)
            if len(kept) >= max_detections:
                break
    return kept


class ShapeDetector:
    def __init__(self, model_path, width=640, height=480, confidence=0.5, nms_iou=0.45):
        import nncase_runtime as nn
        import ulab.numpy as np
        self.nn, self.np = nn, np
        self.confidence, self.nms_iou = confidence, nms_iou
        self.geometry = centered_letterbox_geometry(width, height, 320, 320)
        self.width, self.height = width, height
        self.kpu = nn.kpu()
        self.kpu.load_kmodel(model_path)
        self.ai2d = nn.ai2d()
        self.ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
        g = self.geometry
        self.ai2d.set_pad_param(True, [0, 0, 0, 0, g["pad_top"], g["pad_bottom"], g["pad_left"], g["pad_right"]], 0, [114, 114, 114])
        self.ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        self.builder = self.ai2d.build([1, 3, height, width], [1, 3, 320, 320])
        self.input_array = np.zeros((1, 3, 320, 320), dtype=np.uint8)
        self.input_tensor = nn.from_numpy(self.input_array)

    def detect_rgb888p(self, pixels):
        """pixels is a uint8 RGB planar ndarray [3,H,W] or [1,3,H,W]."""
        if self.kpu is None:
            raise RuntimeError("Detector is closed")
        shape = tuple(pixels.shape)
        if shape == (3, self.height, self.width):
            pixels = pixels.reshape((1, 3, self.height, self.width))
        elif shape != (1, 3, self.height, self.width):
            raise ValueError("Frame size/layout differs from detector configuration")
        input_tensor = self.nn.from_numpy(pixels)
        self.builder.run(input_tensor, self.input_tensor)
        self.kpu.set_input_tensor(0, self.input_tensor)
        self.kpu.run()
        output_tensor = self.kpu.get_output_tensor(0)
        raw = output_tensor.to_numpy()
        rows = decode_raw(raw, self.confidence, self.nms_iou)
        targets = detections_to_targets(rows, CLASSES, self.geometry, self.confidence)
        del raw, output_tensor, input_tensor
        return targets

    def detect_rgb888(self, frame):
        """frame is an image.Image containing packed RGB888 pixels."""
        hwc = frame.to_numpy_ref()
        if tuple(hwc.shape) != (self.height, self.width, 3):
            raise ValueError("Expected packed RGB888 image with configured width/height")
        pixels = hwc.reshape((self.height * self.width, 3)).transpose().copy()
        return self.detect_rgb888p(pixels.reshape((3, self.height, self.width)))

    def close(self):
        """Release this detector; shared runtime pool remains available to other AI."""
        import gc
        self.kpu = self.builder = self.ai2d = self.input_tensor = self.input_array = None
        gc.collect()
