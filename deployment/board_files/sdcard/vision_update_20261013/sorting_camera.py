import time,gc
from media.uvc import UVC
from media.media import MediaManager
SHOW_CAMERA=False
VERBOSE=False
_cam_ok=False
def _standby_take(name):return False
FRAME_DRAIN=8
CAMERA_MODE_W=1280
CAMERA_MODE_H=960
CAM_TRIES=5
def cam_start():
    """起摄像头。带重试（见上面 CAM_TRIES 的说明）。

       ★★ .151 幂等：已经起过就直接返回，不重复初始化 ★★
         两条"已经起过"：
           · 本次运行里起过（_cam_ok = True）          → 直接返回
           · 待机期（main.py）已经枚举 + 自检过（交接单）→ 接过标志、直接返回
         两条都是"打一行说明 + return"，绝不重复
         MediaManager.init() / UVC.probe() / UVC.start()
         —— 重复起就是自己和自己抢 USB 摄像头 ✗
       ★ cam_stop() 会把 _cam_ok 清掉：所以 cam_selftest() 里
         "停掉重起一次（等于软件拔插）"那条路照旧真的重起 ✓
    """
    global _cam_ok

    if _cam_ok:
        print("摄像头已经在跑（.151 幂等）—— 不重复初始化")
        return

    if _standby_take("cam"):
        _cam_ok = True
        print("摄像头复用：[待机期]已经枚举 + 自检过（.151 幂等）——"
              "不重复初始化，免得和自己抢 USB 摄像头")
        return

    last = "还没试"

    for attempt in range(1, CAM_TRIES + 1):

        plugin = False

        # ---- 1) MediaManager ----
        try:
            MediaManager.init()
        except Exception as e:
            last = "MediaManager.init 失败：%s" % e
            plugin = False
        else:
            # ---- 2) 探测设备 ----
            try:
                plugin, dev = UVC.probe()
            except Exception as e:
                plugin = False
                last = "UVC.probe 异常：%s" % e
            else:
                if not plugin:
                    last = "没找到 USB 摄像头"

        # ---- 3) 选档 + 启动 ----
        if plugin:
            try:
                mode = UVC.video_mode(CAMERA_MODE_W, CAMERA_MODE_H,
                                      UVC.FOURCC_MJPEG, 30)
                ok, mode = UVC.select_video_mode(mode)

                if not ok:
                    last = ("摄像头不支持 %dx%d MJPEG"
                            "（★ 上一次运行没把它放开时也是报这句 —— "
                            "先按断电/拔插处理）"
                            % (CAMERA_MODE_W, CAMERA_MODE_H))
                else:
                    UVC.start(cvt=False)
                    _cam_ok = True          # ★ .151：起来了 → 幂等标志置上
                    if attempt == 1:
                        if VERBOSE:
                            print("摄像头 OK")
                    else:
                        if VERBOSE:
                            print("摄像头 OK（第 %d 次才起来 —— 说明是枚举慢，"
                              "不是硬件坏）" % attempt)
                    return
            except Exception as e:
                last = "UVC.start 失败：%s" % e

        # ---- 4) 失败：清干净，等 1 秒再来 ----
        print("   ⚠️ 摄像头第 %d/%d 次没起来（%s）"
              % (attempt, CAM_TRIES, last))
        cam_stop()
        time.sleep_ms(1000)

    raise RuntimeError(
        "摄像头起不来（试了 %d 次）：%s\n"
        "   → 把板子【断电 30 秒】再上电（软复位解不开 USB 摄像头，"
        "这是老毛病）\n"
        "   → 顺手确认：摄像头插在板子的 USB Host 口、线是数据线"
        "（不是只充电的）\n"
        "   → ★ .6 起相机是在【待机期】先起的：如果这句只在这个版本出现，"
        "把 main.py 的 PRELOAD_CAMERA_IN_STANDBY 改成 False"
        "（相机退回按 KEY 之后才起）" % (CAM_TRIES, last))

def cam_stop():
    global _cam_ok

    # ★ .151：关掉了 → 幂等标志必须清掉，
    #   这样下一次 cam_start() 会【真的】重起（cam_selftest 的"软件拔插"靠它）
    _cam_ok = False

    try:
        UVC.stop()
    except Exception:
        pass
    # ★ 官方例子 UVC.stop() 之后会 sleep 100ms 再 deinit 显示 ——
    #   给 USB 一点时间把接口真正交出来（2026-10-06.34）
    try:
        time.sleep_ms(100)
    except Exception:
        pass
    try:
        MediaManager.deinit()
    except Exception:
        pass

def grab():
    """抓一帧 RGB888。

       ★★ 抓之前先把积压的旧帧倒掉（2026-10-05）★★

       现场症状（用户从电脑预览上看出来的）：
         夹取完成后做【抓完验证】，拿到的画面还是【夹取之前】那一帧 ——
         于是"明明夹走了"被判成"还在托盘里"，白重试一次、
         成功数也记不对。
         电脑上的预览也只在夹爪回到停泊位时刷新一次，内容却是旧的。

       原因：
         机械臂运动那十几秒里，一次都没调 UVC.snapshot()。
         管线里攒了一串旧帧，再调 snapshot() 拿到的是最早那一张。
         —— 验证看到的是"过去"，自然怎么验都不对。

       做法：
         连续 snapshot() 几次，把旧帧扔掉，只用最后一张。
         每次约 2ms，8 次也就 16ms，比一次误判便宜太多。
    """

    raw = None

    for _ in range(FRAME_DRAIN):
        r = UVC.snapshot()
        if r is not None:
            if raw is not None:
                try:
                    raw.__del__()
                except Exception:
                    pass
            raw = r

    if raw is None:
        return None

    img = raw.to_rgb888()

    # 原图用完就放（官方案例也是用完立刻 __del__）
    try:
        raw.__del__()
    except Exception:
        pass

    # 收一次，把上面那些旧帧真正还给显存池
    try:
        gc.collect()
    except Exception:
        pass

    if SHOW_CAMERA:
        try:
            img.compress_for_ide()
        except Exception:
            pass

    return img

def camera_init():cam_start()
def camera_deinit():cam_stop()
def get_frame_rgb888():return grab()
