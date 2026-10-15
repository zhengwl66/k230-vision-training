"""K230 callable interface: current working camera frame -> shape/color targets."""
import json
from shape_k230 import ShapeDetector
from color_sample import attach_color,color_context,stabilize_colors,reset_color_history
from state_sample import attach_state,stabilize_states,reset_state_history
NAMES_ZH=('镂空正方体','正四面体','五棱柱','三棱柱','圆锥','圆柱','半球')

class VisionModule:
    def __init__(self,model_path,confidence=.5,nms_iou=.45):
        self.model_path=model_path;self.confidence=confidence;self.nms_iou=nms_iou
        self.detector=None;self.size=None
    def detect(self,frame):
        """frame must be an unannotated packed RGB888 image.Image."""
        size=(frame.width(),frame.height())
        if self.detector is None or size!=self.size:
            self.close();self.size=size
            self.detector=ShapeDetector(self.model_path,size[0],size[1],self.confidence,self.nms_iou)
        targets=self.detector.detect_rgb888(frame);pixels=frame.to_numpy_ref()
        context=color_context(pixels)
        for t in targets:
            attach_color(pixels,t,context)
            try:attach_state(pixels,t,context)
            except Exception as exc:
                t['state_hint']='UNKNOWN';t['state_name_zh']='姿态不确定'
                t['state_reason']='state_module_error';t['state_calibrated']=False
            t['shape_name_zh']=NAMES_ZH[t['class_id']]
        del pixels
        stabilize_colors(targets);stabilize_states(targets)
        return targets
    def close(self):
        if self.detector is not None:self.detector.close()
        self.detector=None;self.size=None
        reset_color_history();reset_state_history()

def draw_targets(preview,targets):
    for t in targets:
        preview.draw_rectangle(t['x'],t['y'],t['w'],t['h'],color=(0,255,0),thickness=2)
        label=t.get('color_name_zh','')+NAMES_ZH[t['class_id']]
        if t.get('state_hint')!='NOT_APPLICABLE':label+=' '+t.get('state_name_zh','姿态不确定')
        preview.draw_string_advanced(t['x'],max(0,t['y']-24),20,label+' %.2f'%t['confidence'],color=(255,255,0))
    return preview

