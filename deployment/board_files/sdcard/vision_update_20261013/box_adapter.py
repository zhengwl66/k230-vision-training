"""Map decoded, NMS-filtered model-pixel xyxy boxes to project target dicts.

Pure Python: no camera, KPU, HDMI, UART, NumPy or Ultralytics imports.
Input row contract: {"xyxy": (x1, y1, x2, y2), "confidence": 0.9,
                     "class_id": 0}
Coordinates are continuous edge coordinates in the padded MODEL input image.
They are not normalized, xywh, raw YOLO tensors, or already-restored boxes.
"""

import math


def _finite_number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if value != value or value == float("inf") or value == -float("inf"):
        return None
    return value


def _whole_number(value, name, minimum):
    number = _finite_number(value)
    if number is None or number < minimum or int(number) != number:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return int(number)


def letterbox_geometry(source_width, source_height, input_width, input_height,
                       resized_width, resized_height,
                       pad_left, pad_top, pad_right, pad_bottom):
    """Record the ACTUAL integer resize/pad operation used by preprocessing.

    Both axes use their real rounded resize ratio. Do not replace them with
    a single ideal min(input/source) scale after dimensions have been rounded.
    This describes resize + padding only: no crop, rotation, flip, or ROI.
    """
    values = {
        "source_width": source_width, "source_height": source_height,
        "input_width": input_width, "input_height": input_height,
        "resized_width": resized_width, "resized_height": resized_height,
        "pad_left": pad_left, "pad_top": pad_top,
        "pad_right": pad_right, "pad_bottom": pad_bottom,
    }
    for key in values:
        values[key] = _whole_number(values[key], key, 0 if key.startswith("pad_") else 1)
    if values["resized_width"] + values["pad_left"] + values["pad_right"] != values["input_width"]:
        raise ValueError("resized_width + left + right must equal input_width")
    if values["resized_height"] + values["pad_top"] + values["pad_bottom"] != values["input_height"]:
        raise ValueError("resized_height + top + bottom must equal input_height")
    values["scale_x"] = values["resized_width"] / float(values["source_width"])
    values["scale_y"] = values["resized_height"] / float(values["source_height"])
    return values


def centered_letterbox_geometry(source_width=640, source_height=480,
                                input_width=320, input_height=320,
                                allow_upscale=True):
    """Plan a centered letterbox; caller must apply exactly the returned sizes.

    Positive half-pixel ties round up; odd padding puts the extra pixel on
    the right/bottom. This is this helper's contract, NOT a claim about every
    Ultralytics or Ai2d version. For an existing preprocessor, record its actual
    dimensions with letterbox_geometry instead of assuming this convention.
    """
    sw = _whole_number(source_width, "source_width", 1)
    sh = _whole_number(source_height, "source_height", 1)
    iw = _whole_number(input_width, "input_width", 1)
    ih = _whole_number(input_height, "input_height", 1)
    ratio = min(iw / float(sw), ih / float(sh))
    if not allow_upscale:
        ratio = min(ratio, 1.0)
    rw = min(iw, max(1, int(math.floor(sw * ratio + 0.5))))
    rh = min(ih, max(1, int(math.floor(sh * ratio + 0.5))))
    left = (iw - rw) // 2
    top = (ih - rh) // 2
    return letterbox_geometry(sw, sh, iw, ih, rw, rh,
                              left, top, iw - rw - left, ih - rh - top)


def _validated_geometry(geometry):
    try:
        return letterbox_geometry(
            geometry["source_width"], geometry["source_height"],
            geometry["input_width"], geometry["input_height"],
            geometry["resized_width"], geometry["resized_height"],
            geometry["pad_left"], geometry["pad_top"],
            geometry["pad_right"], geometry["pad_bottom"])
    except (KeyError, TypeError):
        raise ValueError("geometry must come from a letterbox geometry helper")


def detections_to_targets(detections, class_names, geometry,
                          confidence_threshold=0.5):
    """Return target dictionaries; discard invalid individual detection rows.

    class_names: a list/tuple in model class order, or {integer_id: name}.
    Boxes are clipped to the source frame. Integer rectangles use floor for
    left/top and ceil for right/bottom; x+w/y+h are exclusive image edges.
    Bad geometry/threshold is a configuration error and raises ValueError.
    This function does NOT decode model output, run NMS, infer color, deduplicate
    across frames, or establish whether a robot grasp/sort succeeded.
    """
    g = _validated_geometry(geometry)
    threshold = _finite_number(confidence_threshold)
    if threshold is None or not 0 <= threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    targets = []
    for row in detections:
        if not isinstance(row, dict):
            continue
        try:
            xyxy = row["xyxy"]
            if len(xyxy) != 4:
                continue
            coords = [_finite_number(v) for v in xyxy]
            if None in coords:
                continue
            x1, y1, x2, y2 = coords
            confidence = _finite_number(row["confidence"])
            class_value = _finite_number(row["class_id"])
            if confidence is None or not threshold <= confidence <= 1:
                continue
            if class_value is None or class_value < 0 or int(class_value) != class_value:
                continue
            class_id = int(class_value)
            shape_name = class_names[class_id]
            if not isinstance(shape_name, str) or not shape_name:
                continue
        except (KeyError, IndexError, TypeError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        x1 = max(0.0, min(float(g["source_width"]), (x1 - g["pad_left"]) / g["scale_x"]))
        y1 = max(0.0, min(float(g["source_height"]), (y1 - g["pad_top"]) / g["scale_y"]))
        x2 = max(0.0, min(float(g["source_width"]), (x2 - g["pad_left"]) / g["scale_x"]))
        y2 = max(0.0, min(float(g["source_height"]), (y2 - g["pad_top"]) / g["scale_y"]))
        if x2 <= x1 or y2 <= y1:
            continue
        x, y = int(math.floor(x1)), int(math.floor(y1))
        right, bottom = int(math.ceil(x2)), int(math.ceil(y2))
        width, height = right - x, bottom - y
        targets.append({
            "x": x, "y": y, "w": width, "h": height,
            "cx": x + width // 2, "cy": y + height // 2,
            "area": width * height,
            "shape_name": shape_name, "confidence": confidence,
            "class_id": class_id, "color_hint": "UNKNOWN",
        })
    return targets


def largest_target(targets):
    """Preserve the existing largest-bounding-box target selection policy."""
    best = None
    for target in targets:
        if best is None or target["area"] > best["area"]:
            best = target
    return best
