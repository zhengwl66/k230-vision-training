"""Run with CanMV on the K230; save unannotated full camera frames for labeling."""
import sys
import os
import gc
import time

if "/sdcard" not in sys.path:
    sys.path.insert(0, "/sdcard")
from uvc_camera import camera_init, camera_deinit
from media.uvc import UVC
import image

# Change these settings for each collection batch.
GROUP = "cone"
COUNT = 30
INTERVAL_MS = 3000
ROOT = "/sdcard/train_photos"

GROUPS = ("hollow_cube", "tetrahedron", "pentagonal_prism", "triangular_prism",
          "cone", "cylinder", "hemisphere", "background", "mixed")
if GROUP not in GROUPS or COUNT <= 0 or INTERVAL_MS < 200:
    raise ValueError("Check GROUP, COUNT and INTERVAL_MS")


def new_batch():
    try:
        os.mkdir(ROOT)
    except OSError as error:
        if error.args[0] != 17:  # EEXIST
            raise
    for number in range(1, 10000):
        folder = "%s/%s_%04d" % (ROOT, GROUP, number)
        try:
            os.mkdir(folder)
            return folder
        except OSError as error:
            if error.args[0] != 17:
                raise
    raise RuntimeError("Batch names exhausted")


folder = new_batch()
saved = 0
try:
    camera_init()
    print("照片保存目录：", folder)
    print("请改变物体的位置、朝向或组合；每%d毫秒保存一张" % INTERVAL_MS)
    last_saved = time.ticks_ms()
    last_received = last_saved
    while saved < COUNT:
        os.exitpoint()
        raw = UVC.snapshot()
        now = time.ticks_ms()
        if raw is None:
            if time.ticks_diff(now, last_received) > 10000:
                raise RuntimeError("摄像头连续10秒没有返回画面")
            time.sleep_ms(10)
            continue
        last_received = now

        if time.ticks_diff(now, last_saved) >= INTERVAL_MS:
            if raw.format() != image.JPEG:
                raise RuntimeError("采集要求MJPEG/cvt=False，请检查uvc_camera.py")
            filename = "%s/%06d.jpg" % (folder, saved + 1)
            # Save the camera JPEG before preview conversion or any drawing.
            raw.save(filename)
            if os.stat(filename)[6] <= 0:
                raise RuntimeError("照片写入为空，请检查TF卡")
            saved += 1
            last_saved = time.ticks_ms()
            print("已保存 %d/%d：%s" % (saved, COUNT, filename))

        preview = raw.to_rgb565()
        preview.compress_for_ide()
        del raw, preview
        gc.collect()
    print("采集完成，共%d张。目录：%s" % (saved, folder))
finally:
    camera_deinit()
    print("摄像头已释放，本批已保存%d张" % saved)
