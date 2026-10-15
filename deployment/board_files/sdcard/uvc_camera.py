# ============================================================
# uvc_camera.py
#
# K230 USB UVC Camera Module
#
# Verified baseline:
#
# USB UVC camera
# 640 x 480
# MJPEG
# 30 FPS
#
# Output:
# RGB888 image
# ============================================================

from media.uvc import *
from media.media import *


# ============================================================
# Camera configuration
# ============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30


# ============================================================
# Internal state
#
# This variable is used only inside this module.
# Other modules do not access it directly.
# ============================================================

_camera_initialized = False


# ============================================================
# Initialize camera
#
# Return:
#
# True
#
# If camera is not found or mode selection fails,
# RuntimeError will be raised.
# ============================================================

def camera_init():

    global _camera_initialized


    # Already initialized
    if _camera_initialized:

        return True


    # --------------------------------------------------------
    # Initialize media system
    # --------------------------------------------------------

    MediaManager.init()


    try:

        # ----------------------------------------------------
        # Find USB UVC camera
        # ----------------------------------------------------

        plugin, dev = UVC.probe()


        if not plugin:

            raise RuntimeError(
                "USB camera not found"
            )


        # ----------------------------------------------------
        # Request:
        #
        # 640 x 480
        # MJPEG
        # 30 FPS
        # ----------------------------------------------------

        mode = UVC.video_mode(
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            UVC.FOURCC_MJPEG,
            CAMERA_FPS
        )


        # ----------------------------------------------------
        # Select video mode
        # ----------------------------------------------------

        success, mode = UVC.select_video_mode(
            mode
        )


        if not success:

            raise RuntimeError(
                "UVC video mode select failed"
            )


        # ----------------------------------------------------
        # Start UVC
        #
        # Keep the verified setting:
        #
        # cvt=False
        # ----------------------------------------------------

        UVC.start(
            cvt=False
        )


        _camera_initialized = True


        return True


    except Exception:

        # ----------------------------------------------------
        # Initialization failed
        #
        # Release MediaManager before passing the error upward.
        # ----------------------------------------------------

        MediaManager.deinit()

        raise


# ============================================================
# Get one RGB888 frame
#
# Return:
#
# RGB888 image
#
# or:
#
# None
# ============================================================

def get_frame_rgb888():

    if not _camera_initialized:

        raise RuntimeError(
            "Camera is not initialized"
        )


    # --------------------------------------------------------
    # Get one UVC frame
    # --------------------------------------------------------

    camera_img = UVC.snapshot()


    if camera_img is None:

        return None


    # --------------------------------------------------------
    # Convert to RGB888
    #
    # This is the standard image format used by later
    # vision modules.
    # --------------------------------------------------------

    img_rgb888 = camera_img.to_rgb888()


    return img_rgb888


# ============================================================
# Deinitialize camera
# ============================================================

def camera_deinit():

    global _camera_initialized


    if not _camera_initialized:

        return


    # --------------------------------------------------------
    # Stop UVC
    # --------------------------------------------------------

    UVC.stop()


    # --------------------------------------------------------
    # Release media resources
    # --------------------------------------------------------

    MediaManager.deinit()


    _camera_initialized = False
