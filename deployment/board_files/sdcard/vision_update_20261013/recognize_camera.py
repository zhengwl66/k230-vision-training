"""CanMV entry using the existing working camera module without changing its mode."""
import sys,os,time,gc,json
MODEL_DIR='/sdcard/vision_update_20261013'
# Point these two values at the camera module used by your successful capture script.
CAMERA_MODULE_DIR='/sdcard'
CAMERA_MODULE='uvc_camera'
MAX_FRAMES=0
sys.path.insert(0,CAMERA_MODULE_DIR)
sys.path.insert(0,MODEL_DIR)
for _name in ('vision_updated','shape_k230','box_adapter','color_sample','state_sample'):
    if _name in sys.modules:del sys.modules[_name]
from vision_updated import VisionModule,draw_targets

def main():
    camera=__import__(CAMERA_MODULE);vision=None;started=False;count=0
    try:
        config=json.load(open(MODEL_DIR+'/inference_config.json'))
        vision=VisionModule(MODEL_DIR+'/shape.kmodel',config['confidence'],config['nms_iou'])
        started=True;camera.camera_init();warmup=time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(),warmup)<1500:
            os.exitpoint();initial=camera.get_frame_rgb888()
            if initial is not None:del initial
            time.sleep_ms(5)
        last=time.ticks_ms()
        print('VISION_MODEL_VERSION',config['version'])
        print('VISION_MODEL_PATH',MODEL_DIR+'/shape.kmodel')
        print('NEW_CAMERA_MODEL_READY')
        while not MAX_FRAMES or count<MAX_FRAMES:
            os.exitpoint();frame=camera.get_frame_rgb888()
            if frame is None:
                if time.ticks_diff(time.ticks_ms(),last)>10000:raise RuntimeError('Camera timeout')
                time.sleep_ms(10);continue
            last=time.ticks_ms();targets=vision.detect(frame)
            preview=draw_targets(frame.to_rgb565(),targets);preview.compress_for_ide()
            if count%5==0:print('VISION_RESULT',json.dumps({'frame':count,'width':frame.width(),'height':frame.height(),'targets':targets}))
            count+=1;del preview,frame;gc.collect()
    finally:
        if vision is not None:vision.close()
        if started:camera.camera_deinit()
        print('VISION_STOPPED',count)

if __name__=='__main__':main()
