"""Track perception: find the yellow tape line and nearby blue boundary tape.

Ported from the PC "Camera Test" prototype. This module has no ROS
dependency, so the same code runs in the track_test node, in navigation.py
(when motor control is added) and on a PC.

    from team_navigation.perception import Perception
    p = Perception()
    result = p.process(frame)   # frame is a BGR image from OpenCV
    result.error                # -1 (track far left) .. +1 (track far right)
"""
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

# OpenCV HSV ranges: H 0-179, S 0-255, V 0-255
DEFAULT_CONFIG = {
    'yellow_lo': [20, 80, 80],
    'yellow_hi': [35, 255, 255],
    'blue_lo': [95, 80, 50],
    'blue_hi': [130, 255, 255],
}
CONFIG_KEYS = tuple(DEFAULT_CONFIG)
HSV_MAX = (179, 255, 255)

PROC_WIDTH, PROC_HEIGHT = 320, 240  # frames are resized to this before processing
ROI_TOP_FRACTION = 0.66             # use only the bottom third of the image
MIN_YELLOW_PIXELS = 200             # fewer than this => track "not found"
BLUE_WARN_FRACTION = 0.05           # blue covering >5% of an ROI half => warning


def validate_triplet(value):
    """Return value as a list of three in-range HSV ints, or raise ValueError."""
    vals = [int(v) for v in value]
    if len(vals) != 3:
        raise ValueError(f'expected 3 values (H, S, V), got {len(vals)}')
    for v, top in zip(vals, HSV_MAX):
        if not 0 <= v <= top:
            raise ValueError(f'{vals} out of range; limits are H 0-179, S 0-255, V 0-255')
    return vals


def load_config(path):
    """Defaults, overridden by the JSON file at path if it exists."""
    cfg = {k: list(v) for k, v in DEFAULT_CONFIG.items()}
    path = os.path.expanduser(path)
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        cfg.update({k: validate_triplet(saved[k]) for k in CONFIG_KEYS if k in saved})
    return cfg


def save_config(cfg, path):
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump({k: list(cfg[k]) for k in CONFIG_KEYS}, f, indent=2)
    return path


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
        self.set_config(config or DEFAULT_CONFIG)
        self.last_error = 0.0

    def set_config(self, cfg):
        self.cfg = {k: list(cfg[k]) for k in CONFIG_KEYS}
        self.y_lo = np.array(cfg['yellow_lo'], dtype=np.uint8)
        self.y_hi = np.array(cfg['yellow_hi'], dtype=np.uint8)
        self.b_lo = np.array(cfg['blue_lo'], dtype=np.uint8)
        self.b_hi = np.array(cfg['blue_hi'], dtype=np.uint8)

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
        if yellow_pixels >= MIN_YELLOW_PIXELS and m['m00'] > 0:
            cx = m['m10'] / m['m00']
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


# ---------------------------------------------------------------- drawing
SCALE = 2  # upscale the 320x240 processed image for display


def _text(img, text, org, scale, color, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw_overlay(r, left_cmd, right_cmd, fps):
    """Camera view with ROI box, centre line, centroid, error bar and warnings."""
    img = r.frame.copy()
    h, w = img.shape[:2]

    cv2.rectangle(img, (0, r.roi_top), (w - 1, h - 1), (0, 255, 0), 1)
    cv2.line(img, (w // 2, r.roi_top), (w // 2, h), (255, 255, 255), 1)
    if r.found:
        cy = (r.roi_top + h) // 2
        cv2.circle(img, (int(r.cx), cy), 6, (0, 0, 255), -1)
        cv2.line(img, (w // 2, cy), (int(r.cx), cy), (0, 0, 255), 2)

    img = cv2.resize(img, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
    W = w * SCALE

    bar_y, bar_h = 10, 14
    cv2.rectangle(img, (10, bar_y), (W - 10, bar_y + bar_h), (60, 60, 60), -1)
    mid = W // 2
    x = int(mid + r.error * (W / 2 - 10))
    color = (0, 200, 0) if r.found else (0, 140, 255)
    cv2.rectangle(img, (min(mid, x), bar_y), (max(mid, x), bar_y + bar_h), color, -1)
    cv2.line(img, (mid, bar_y - 3), (mid, bar_y + bar_h + 3), (255, 255, 255), 1)

    status = 'TRACK' if r.found else 'LOST (using last error)'
    lines = [
        f'{status}   error={r.error:+.2f}   yellow px={r.yellow_pixels}',
        f'blue L={r.blue_left:.1%}  R={r.blue_right:.1%}',
        f'preview L={left_cmd:+.2f}  R={right_cmd:+.2f}   {fps:.0f} fps',
    ]
    for i, text in enumerate(lines):
        _text(img, text, (10, 50 + i * 24), 0.6, (255, 255, 255))

    if r.blue_warning:
        side = 'LEFT' if r.blue_left > r.blue_right else 'RIGHT'
        _text(img, f'BLUE WARNING ({side})', (10, h * SCALE - 15), 0.9, (255, 80, 0), 2)
    return img


def draw_masks(r):
    """What the robot "sees": yellow mask in yellow, blue mask in blue."""
    h, w = r.yellow_mask.shape
    vis = np.zeros((h, w, 3), np.uint8)
    vis[r.yellow_mask > 0] = (0, 255, 255)
    vis[r.blue_mask > 0] = (255, 80, 0)
    vis[: r.roi_top] //= 3  # dim everything above the ROI; it is ignored
    cv2.line(vis, (0, r.roi_top), (w, r.roi_top), (0, 255, 0), 1)
    return cv2.resize(vis, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)


def draw_debug(r, left_cmd, right_cmd, fps):
    return np.hstack([draw_overlay(r, left_cmd, right_cmd, fps), draw_masks(r)])
