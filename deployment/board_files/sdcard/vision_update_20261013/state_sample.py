"""Supervised three-shape pose classifier, identical features on PC and K230.

The kernel model is fitted from labelled real images; no fixed aspect-ratio
rule decides the state. Model data are shared by preview and sorting programs.
"""
import math
import json

STATE_ALGORITHM_VERSION = 'pose-image-geometry-fusion-20261013-v1'
MODEL_PATH = '/sdcard/vision_update_20261013/state_model.json'
STATEFUL_CLASS_IDS = (2, 3, 5)
_model = None
_model_error = None
_tracks = []
_last_ms = None
_image_engine = None


def fuse_evidence(geometry_score, image_score, geometry_margin, image_margin):
    geometry_ok=abs(geometry_score)>=geometry_margin
    image_ok=abs(image_score)>=image_margin
    if geometry_ok and image_ok and (geometry_score>0)!=(image_score>0):
        return 'UNKNOWN',0.0,'evidence_conflict'
    if geometry_ok:
        score=geometry_score
        reason='geometry_confirmed_by_image' if image_ok else 'strong_geometry_evidence'
    elif image_ok:
        score=image_score;reason='strong_image_evidence'
    else:return 'UNKNOWN',(geometry_score+image_score)*.5,'ambiguous_score'
    return ('LYING' if score>0 else 'UPRIGHT'),score,reason


class _PoseImageEngine:
    def __init__(self, model_path):
        import nncase_runtime as nn
        import ulab.numpy as np
        self.nn = nn; self.np = np
        self.kpu = nn.kpu(); self.kpu.load_kmodel(model_path)
        self.ai2d = nn.ai2d()
        self.ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
        self.ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        self.output = np.zeros((1,3,96,96), dtype=np.uint8)
        self.output_tensor = nn.from_numpy(self.output)
    def predict(self, pixels, target, pair_index):
        height,width,channels = pixels.shape
        x0=max(0,int(target['x']));y0=max(0,int(target['y']))
        x1=min(width,int(target['x']+target['w']));y1=min(height,int(target['y']+target['h']))
        if x1-x0<8 or y1-y0<8:raise ValueError('Pose crop too small')
        planar=pixels.reshape((height*width,3)).transpose().copy().reshape((1,3,height,width))
        tensor=self.nn.from_numpy(planar)
        self.ai2d.set_crop_param(True,x0,y0,x1-x0,y1-y0)
        builder=self.ai2d.build([1,3,height,width],[1,3,96,96])
        builder.run(tensor,self.output_tensor)
        self.kpu.set_input_tensor(0,self.output_tensor);self.kpu.run()
        output_tensor=self.kpu.get_output_tensor(0);raw=output_tensor.to_numpy()
        if tuple(raw.shape)!=(1,6):raise ValueError('Pose model output must be [1,6]')
        delta=float(raw[0,pair_index*2+1])-float(raw[0,pair_index*2])
        value=2.0/(1.0+math.exp(max(-30.0,min(30.0,-delta))))-1.0
        del raw,output_tensor,tensor,planar,builder
        return value
    def close(self):
        self.kpu=self.ai2d=self.output=self.output_tensor=None


def extract_features(pixels, target):
    height, width, channels = pixels.shape
    if channels != 3:
        raise ValueError('State input must be RGB888')
    x0 = max(0, int(target['x'])); y0 = max(0, int(target['y']))
    x1 = min(width, int(target['x'] + target['w']))
    y1 = min(height, int(target['y'] + target['h']))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, {'reason': 'small_box'}
    side = 32
    mask = [0] * (side * side)
    values = [0.0] * (side * side)
    for row in range(side):
        y = y0 + (2 * row + 1) * (y1 - y0) // (2 * side)
        for col in range(side):
            x = x0 + (2 * col + 1) * (x1 - x0) // (2 * side)
            r = int(pixels[y, x, 0]); g = int(pixels[y, x, 1]); b = int(pixels[y, x, 2])
            hi = max(r, g, b); lo = min(r, g, b)
            i = row * side + col
            if hi >= 45 and hi - lo >= hi * .23:
                mask[i] = 1
                values[i] = hi

    # The largest connected component discards neighbouring objects and isolated
    # background pixels. Four-connected neighbourhood is deterministic on both
    # Python implementations.
    seen = [0] * len(mask); best = []
    for index in range(len(mask)):
        if not mask[index] or seen[index]:
            continue
        queue = [index]; seen[index] = 1; pos = 0
        while pos < len(queue):
            i = queue[pos]; pos += 1; row = i // side; col = i % side
            neighbours = []
            if row: neighbours.append(i - side)
            if row < side - 1: neighbours.append(i + side)
            if col: neighbours.append(i - 1)
            if col < side - 1: neighbours.append(i + 1)
            for j in neighbours:
                if mask[j] and not seen[j]:
                    seen[j] = 1; queue.append(j)
        if len(queue) > len(best): best = queue
    if len(best) < 48:
        return None, {'reason': 'insufficient_foreground', 'points': len(best)}
    left = min(i % side for i in best); right = max(i % side for i in best) + 1
    top = min(i // side for i in best); bottom = max(i // side for i in best) + 1
    bw = right - left; bh = bottom - top
    if bw < 5 or bh < 5:
        return None, {'reason': 'thin_foreground'}
    selected = [0] * len(mask)
    for i in best: selected[i] = 1
    level = sum(values[i] for i in best) / len(best)
    counts = [0] * 64; total = [0] * 64; luminance = [0.0] * 64
    sx = sy = sxx = syy = sxy = 0.0
    for row in range(top, bottom):
        gy = min(7, (row - top) * 8 // bh)
        for col in range(left, right):
            gx = min(7, (col - left) * 8 // bw); cell = gy * 8 + gx
            total[cell] += 1; i = row * side + col
            if selected[i]:
                counts[cell] += 1
                luminance[cell] += max(-.8, min(.8, values[i] / level - 1.0))
                xn = (col - left + .5) / bw; yn = (row - top + .5) / bh
                sx += xn; sy += yn; sxx += xn * xn; syy += yn * yn; sxy += xn * yn
    count = len(best); mx = sx / count; my = sy / count
    vx = max(1e-8, sxx / count - mx * mx)
    vy = max(1e-8, syy / count - my * my); cov = sxy / count - mx * my
    aspect = bw * (x1 - x0) / float(bh * (y1 - y0))
    vector = [counts[i] / float(max(1, total[i])) for i in range(64)]
    vector += [luminance[i] / max(1, counts[i]) * 2.0 for i in range(64)]
    vector += [math.log(max(.1, min(10.0, aspect))) * 2.0,
               count / float(bw * bh) * 2.0, mx * 2.0, my * 2.0,
               vx * 12.0, vy * 12.0, cov * 12.0]
    diagnostic = {'points': count, 'fill': count / float(bw * bh),
                  'aspect': aspect, 'reason': 'ok'}
    return vector, diagnostic


def set_model(model):
    """Inject fitted model on PC, or load one from disk on K230."""
    global _model, _model_error
    if model.get('algorithm') != STATE_ALGORITHM_VERSION:
        raise ValueError('State model feature version mismatch')
    if model.get('feature_dim') != 135:
        raise ValueError('State feature dimension mismatch')
    _model = model; _model_error = None


def _load_model():
    global _model_error
    if _model is None and _model_error is None:
        try:
            with open(MODEL_PATH, 'r') as handle: set_model(json.load(handle))
        except Exception as exc:
            _model_error = str(exc)
    return _model


def classify_features(features, class_id):
    model = _load_model()
    if model is None:
        return 'UNKNOWN', 0.0, 0.0, 'model_unavailable'
    profile = model['profiles'].get(str(class_id))
    if not profile or not profile.get('enabled'):
        return 'UNKNOWN', 0.0, 0.0, 'profile_not_validated'
    gamma = profile['gamma']; score = 0.0; nearest = 1e30
    for support, coefficient in zip(profile['support_vectors'], profile['coefficients']):
        distance = 0.0
        for j in range(len(features)):
            delta = features[j] - support[j]
            distance += delta * delta
        if distance < nearest: nearest = distance
        score += coefficient * math.exp(-gamma * distance)
    if nearest > profile['max_distance']:
        return 'UNKNOWN', score, nearest, 'out_of_distribution'
    if abs(score) < profile['margin']:
        return 'UNKNOWN', score, nearest, 'ambiguous_score'
    return ('LYING' if score > 0 else 'UPRIGHT'), score, nearest, 'supervised_model'


def attach_state(pixels, target, context=None):
    global _image_engine
    class_id = int(target.get('class_id', -1))
    target['state_algorithm_version'] = STATE_ALGORITHM_VERSION
    if class_id not in STATEFUL_CLASS_IDS:
        target.update(state_hint='NOT_APPLICABLE', state_name_zh='不适用',
                      state_reason='shape_not_stateful', state_calibrated=False,
                      state_score=0.0, state_confidence=0.0)
        return target
    features, diagnostic = extract_features(pixels, target)
    target['state_features'] = diagnostic
    model = _load_model()
    target['state_profile_version'] = model['version'] if model else 'unavailable'
    target['state_calibrated'] = bool(model and model['profiles'].get(str(class_id), {}).get('enabled'))
    if features is None:
        hint, score, distance, reason = 'UNKNOWN', 0.0, 0.0, diagnostic['reason']
    else:
        hint, score, distance, reason = classify_features(features, class_id)
        if model and reason!='out_of_distribution':
            fusion=model['fusion'][str(class_id)];alpha=float(fusion['geometry_weight'])
            target['state_geometry_score']=score
            if alpha<1.0:
                if _image_engine is None:
                    _image_engine=_PoseImageEngine(MODEL_PATH.rsplit('/',1)[0]+'/'+model['image_model'])
                image_score=_image_engine.predict(pixels,target,(0 if class_id==2 else 1 if class_id==3 else 2))
                target['state_image_score']=image_score
                score=alpha*score+(1.0-alpha)*image_score
            if fusion.get('mode')=='confidence_gate':
                hint,score,reason=fuse_evidence(target['state_geometry_score'],target['state_image_score'],
                                              float(fusion['geometry_margin']),float(fusion['image_margin']))
            else:
                margin=float(fusion['margin'])
                hint='UNKNOWN' if abs(score)<margin else 'LYING' if score>0 else 'UPRIGHT'
                reason='ambiguous_score' if hint=='UNKNOWN' else 'supervised_image_geometry_fusion'
    target['state_hint'] = hint
    target['state_name_zh'] = {'UPRIGHT':'竖直立放', 'LYING':'横躺', 'UNKNOWN':'姿态不确定'}[hint]
    target['state_reason'] = reason; target['state_score'] = score
    # Decision strength is not a calibrated probability.
    target['state_confidence'] = min(1.0, abs(score)) if hint != 'UNKNOWN' else 0.0
    target['state_distance'] = distance
    return target


def reset_state_history():
    global _tracks, _last_ms, _image_engine
    _tracks = []; _last_ms = None
    if _image_engine is not None:_image_engine.close();_image_engine=None


def stabilize_states(targets):
    """Require two consecutive agreeing readings; do not retain stale poses."""
    global _tracks, _last_ms
    import time
    now = time.ticks_ms() if hasattr(time, 'ticks_ms') else int(time.monotonic()*1000)
    if _last_ms is not None:
        elapsed = time.ticks_diff(now, _last_ms) if hasattr(time, 'ticks_diff') else now - _last_ms
        if elapsed > 1800 or elapsed < 0: _tracks = []
    _last_ms = now
    following = []; used = []
    for target in targets:
        raw = target.get('state_hint', 'UNKNOWN'); target['state_hint_raw'] = raw
        if raw == 'NOT_APPLICABLE': continue
        candidate = None; best = .3
        for index, track in enumerate(_tracks):
            box = track['box']
            if index in used or box['class_id'] != target['class_id']: continue
            x0 = max(target['x'], box['x']); y0 = max(target['y'], box['y'])
            x1 = min(target['x']+target['w'], box['x']+box['w'])
            y1 = min(target['y']+target['h'], box['y']+box['h'])
            intersection = max(0, x1-x0)*max(0, y1-y0)
            overlap = intersection / float(max(1, target['w']*target['h'] + box['w']*box['h'] - intersection))
            if overlap > best: best = overlap; candidate = index
        previous = None
        if candidate is not None:
            used.append(candidate); previous = _tracks[candidate]['raw']
        consistent = raw in ('UPRIGHT', 'LYING') and raw == previous
        target['state_temporal_consistent'] = consistent
        if raw in ('UPRIGHT', 'LYING') and not consistent:
            target['state_hint'] = 'UNKNOWN'; target['state_name_zh'] = '姿态确认中'
            target['state_reason'] = 'awaiting_second_frame'; target['state_confidence'] = 0.0
        following.append({'box':{key:target[key] for key in ('x','y','w','h','class_id')}, 'raw':raw})
    _tracks = following
    return targets
