"""
Track perception: find the yellow tape line and nearby blue boundary tape.

Usage from other code:
    from perception import Perception
    p = Perception()
    result = p.process(frame)   # frame is a BGR image from OpenCV
    result.error                # -1 (track far left) .. +1 (track far right)

HSV thresholds are loaded from hsv_config.json (written by hsv_tuner.py) if it
exists, otherwise the defaults below are used.
"""

import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hsv_config.json")

# OpenCV HSV ranges: H 0-179, S 0-255, V 0-255
DEFAULT_CONFIG = {
    "yellow_lo": [20, 80, 80],
    "yellow_hi": [35, 255, 255],
    "blue_lo": [95, 80, 50],
    "blue_hi": [130, 255, 255],
}

PROC_WIDTH, PROC_HEIGHT = 320, 240  # frames are resized to this before processing
ROI_TOP_FRACTION = 0.66             # use only the bottom third of the image
MIN_YELLOW_PIXELS = 200             # fewer than this => track "not found"
BLUE_WARN_FRACTION = 0.05           # blue covering >5% of an ROI half => warning


def load_config(path=CONFIG_PATH):
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def save_config(cfg, path=CONFIG_PATH):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


@dataclass
class PerceptionResult:
    found: bool             # was enough yellow seen in the ROI?
    error: float            # -1 (left) .. +1 (right); last known value if not found
    cx: float | None        # centroid x in processed-image pixels (None if not found)
    yellow_pixels: int
    blue_left: float        # fraction of left half of ROI that is blue
    blue_right: float       # fraction of right half of ROI that is blue
    blue_warning: bool      # True if either side exceeds BLUE_WARN_FRACTION
    # Debug images (processed resolution), useful for visualization
    frame: np.ndarray = field(repr=False, default=None)
    yellow_mask: np.ndarray = field(repr=False, default=None)
    blue_mask: np.ndarray = field(repr=False, default=None)
    roi_top: int = 0


class Perception:
    def __init__(self, config=None):
        self.set_config(config or load_config())
        self.last_error = 0.0

    def set_config(self, cfg):
        self.cfg = cfg
        self.y_lo = np.array(cfg["yellow_lo"], dtype=np.uint8)
        self.y_hi = np.array(cfg["yellow_hi"], dtype=np.uint8)
        self.b_lo = np.array(cfg["blue_lo"], dtype=np.uint8)
        self.b_hi = np.array(cfg["blue_hi"], dtype=np.uint8)

    def process(self, frame_bgr):
        frame = cv2.resize(frame_bgr, (PROC_WIDTH, PROC_HEIGHT))
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        roi_top = int(h * ROI_TOP_FRACTION)

        # Full-frame masks (for display); measurements use only the ROI slice.
        yellow_mask = cv2.inRange(hsv, self.y_lo, self.y_hi)
        blue_mask = cv2.inRange(hsv, self.b_lo, self.b_hi)
        # Remove speckle noise
        kernel = np.ones((3, 3), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        y_roi = yellow_mask[roi_top:, :]
        b_roi = blue_mask[roi_top:, :]

        yellow_pixels = int(np.count_nonzero(y_roi))
        m = cv2.moments(y_roi, binaryImage=True)
        if yellow_pixels >= MIN_YELLOW_PIXELS and m["m00"] > 0:
            cx = m["m10"] / m["m00"]
            error = float(np.clip((cx - w / 2) / (w / 2), -1.0, 1.0))
            found = True
            self.last_error = error
        else:
            cx, found = None, False
            error = self.last_error  # remember which side the track was last seen on

        half = w // 2
        blue_left = np.count_nonzero(b_roi[:, :half]) / b_roi[:, :half].size
        blue_right = np.count_nonzero(b_roi[:, half:]) / b_roi[:, half:].size

        return PerceptionResult(
            found=found,
            error=error,
            cx=cx,
            yellow_pixels=yellow_pixels,
            blue_left=float(blue_left),
            blue_right=float(blue_right),
            blue_warning=bool(max(blue_left, blue_right) > BLUE_WARN_FRACTION),
            frame=frame,
            yellow_mask=yellow_mask,
            blue_mask=blue_mask,
            roi_top=roi_top,
        )


def open_camera(index=0):
    """Open a webcam. On Windows, DirectShow usually starts much faster."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}")
    return cap
