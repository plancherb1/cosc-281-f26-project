"""Receive sensors and request motion through the supplied safety node.

The supplied hooks return no action. Implement perception and control here
or distribute those responsibilities across additional team nodes.
"""
# Course attribution: this file began as the navigation starter prepared by the
# A2R Lab for COSC 81/281, Fall 2026, released under the MIT license (see
# robot/NOTICE.md and LICENSE).
#
# Team The Perceptrons: drive-basic wiring. Each hook hands off to a teammate's
# part where one exists, so each part can be swapped or improved on its own:
#   process_image   -> perception.py  (Tanner)  yellow road + blue boundary
#   update_estimate -> estimation.py  (Michael) smoothing, stale/glitch checks
#   choose_action   -> basic planning (placeholder; replace with planning's version)
#   compute_motion  -> basic P control (placeholder; replace with control's version)
#
# Sign conventions:
#   perception error / estimate offset: -1 = road far LEFT, +1 = road far RIGHT
#   yaw rate: positive = turn LEFT (ROS convention)
#   so road on the right (offset > 0) -> right turn (yaw < 0): yaw = -gain * offset

import csv
import math
import os
from dataclasses import dataclass, replace
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
from course_robot_interfaces.msg import MotionRequest

from team_navigation.perception import Perception, load_config
from team_navigation.estimation import RoadEstimator


# Every tunable number, with defaults. config/navigation.yaml overrides these,
# and `ros2 param set "/$ROBOT_NAMESPACE/navigation" NAME VALUE` changes them
# live (they're re-read every tick). Floats must keep a decimal point.
PARAMETER_DEFAULTS = {
    # --- control ---
    'requested_forward_speed_m_s': 0.05,   # starter's name, kept
    'steering_gain_rad_s': 0.8,            # rad/s of turn per unit of offset (offset is -1..+1)
    'max_yaw_rate_rad_s': 0.8,
    # --- planning ---
    'blue_slowdown_factor': 1.0,   # multiply speed by this while blue boundary is close (1.0 = off)
    'branch_choice': 'all',        # 'all' = center of all yellow (perception default);
                                   # 'left' / 'right' / 'largest' = follow only that yellow blob (for forks)
    'min_branch_pixels': 60,       # blobs smaller than this (in perception's 320x240 image) are ignored
    # --- perception colors ---
    'hsv_config_path': '~/hsv_config.json',   # same file track_test's sliders save to
    # --- estimation (passed to RoadEstimator) ---
    'smoothing_alpha': 0.2,
    'max_image_age_s': 0.25,
    'road_lost_timeout_s': 0.5,
    'max_jump': 0.3,
    'outlier_accept_count': 3,
    # --- logging ---
    'log_period_s': 0.25,
    'csv_logging': True,
    'log_dir': '~/nav_runs',
}


@dataclass
class Observation:
    """process_image's output: perception's result plus the image capture time."""
    result: object        # perception.PerceptionResult
    stamp_s: float


@dataclass
class Action:
    """choose_action's output."""
    mode: str             # 'follow' or 'stop'
    reason: str
    offset: float = 0.0
    speed_scale: float = 1.0


class Navigation(Node):
    def __init__(self):
        super().__init__('navigation')
        for name, default in PARAMETER_DEFAULTS.items():
            self.declare_parameter(name, default)
        self.read_parameters()

        self.bridge = CvBridge()
        self.image = self.image_stamp = self.imu = None
        self.previous_error = None
        self.last_observation = None

        # Perception colors: defaults, overridden by the JSON file track_test saves.
        cfg_path = self.cfg['hsv_config_path']
        self.perception = Perception(load_config(cfg_path))
        self.get_logger().info(f'Perception HSV config: {self.perception.cfg} '
                               f'(from {cfg_path} if it exists, else defaults)')
        self.estimator = RoadEstimator()
        self.apply_estimator_params()

        self.last_print_s = -math.inf
        self.last_flush_s = -math.inf
        self.csv_file = self.csv_writer = None
        self.open_log()

        self.create_subscription(Image, 'camera/image_raw', self.image_cb, qos_profile_sensor_data)
        self.create_subscription(Imu, 'imu/data', self.imu_cb, qos_profile_sensor_data)
        self.publisher = self.create_publisher(
            MotionRequest, 'navigation/request',
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_timer(.05, self.tick)

    def image_cb(self, msg):
        try:
            # OpenCV uses BGR channel order. The image is an H x W x 3 array.
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_stamp = msg.header.stamp
        except Exception as exc:
            self.image = self.image_stamp = None
            self.get_logger().error(f'Image conversion failed: {exc}')

    def imu_cb(self, msg):
        self.imu = msg

    # ------------------------------------------------------------ helpers
    def read_parameters(self):
        self.cfg = {name: self.get_parameter(name).value for name in PARAMETER_DEFAULTS}

    def apply_estimator_params(self):
        for name in ('smoothing_alpha', 'max_image_age_s', 'road_lost_timeout_s',
                     'max_jump', 'outlier_accept_count'):
            setattr(self.estimator, name, self.cfg[name])

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------ hook 1: perception
    def process_image(self, image):
        """Tanner's perception on the newest frame; None if no image yet."""
        if image is None or self.image_stamp is None:
            return None
        stamp_s = self.image_stamp.sec + self.image_stamp.nanosec * 1e-9
        result = self.perception.process(image)
        if self.cfg['branch_choice'] != 'all' and result.found:
            result = self.pick_branch(result)
        return Observation(result, stamp_s)

    def pick_branch(self, r):
        """At a fork, follow ONE yellow blob instead of the center of all yellow.

        Perception averages every yellow pixel in its region, so with two
        branches in view it would aim at the gap between them. Here we split
        the yellow mask into separate blobs (connected components) and keep
        only the leftmost / rightmost / largest one.
        """
        roi = r.yellow_mask[r.roi_top:, :]
        count, _, stats, centroids = cv2.connectedComponentsWithStats(roi)
        # label 0 is the background; keep blobs above the size floor
        blobs = [(stats[i, cv2.CC_STAT_AREA], centroids[i][0])
                 for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= self.cfg['min_branch_pixels']]
        if len(blobs) < 2:
            return r                                  # no fork in view: keep perception's answer
        choice = self.cfg['branch_choice']
        if choice == 'left':
            _, cx = min(blobs, key=lambda b: b[1])
        elif choice == 'right':
            _, cx = max(blobs, key=lambda b: b[1])
        else:                                         # 'largest'
            _, cx = max(blobs, key=lambda b: b[0])
        w = roi.shape[1]
        error = float(np.clip((cx - w / 2) / (w / 2), -1.0, 1.0))
        return replace(r, error=error, cx=float(cx))

    # ------------------------------------------------------------ hook 2: estimation
    def update_estimate(self, observation, imu):
        """Michael's RoadEstimator (IMU unused)."""
        now_s = self.now_s()
        if observation is None:
            return self.estimator.update(None, False, None, now_s)
        r = observation.result
        return self.estimator.update(r.error, r.found, observation.stamp_s, now_s)

    # ------------------------------------------------------------ hook 3: planning (basic)
    def choose_action(self, estimate):
        """Follow while the estimate is valid; otherwise stop. Slow down near blue."""
        if estimate is None or not estimate.valid:
            status = estimate.status if estimate is not None else 'no_estimate'
            return Action('stop', status)
        scale = 1.0
        if self.last_observation is not None and self.last_observation.result.blue_warning:
            scale = self.cfg['blue_slowdown_factor']
        return Action('follow', estimate.status, estimate.offset, scale)

    # ------------------------------------------------------------ hook 4: control (basic)
    def compute_motion(self, action):
        """Return (forward m/s, left-turn rad/s); no action means zero motion.

        Implement your controller here. A positive yaw request turns left.
        For student-determined separation b (metres), safety converts to treads
        v - b*w/2 and v + b*w/2. Y selects the global tread-speed limit:
        Low 0.15, Medium 0.60, or High 1.40 m/s. If either exceeds that limit,
        it scales both together, preserving their ratio without disarming.
        Invalid or stale requests still stop and disarm. See the README.

        Basic proportional control: constant forward speed, turn rate
        proportional to the offset. Offset +0.5 with gain 0.8 -> yaw -0.40 (right).
        """
        if action is None or action.mode != 'follow':
            return 0., 0.
        forward = self.cfg['requested_forward_speed_m_s'] * action.speed_scale
        yaw = -self.cfg['steering_gain_rad_s'] * action.offset
        limit = self.cfg['max_yaw_rate_rad_s']
        return forward, max(-limit, min(limit, yaw))

    # ------------------------------------------------------------ main loop
    def tick(self):
        forward, yaw = 0., 0.
        observation = estimate = action = None
        self.last_observation = None
        error_text = ''
        try:
            self.read_parameters()
            self.apply_estimator_params()
            observation = self.process_image(self.image)
            self.last_observation = observation
            estimate = self.update_estimate(observation, self.imu)
            action = self.choose_action(estimate)
            forward, yaw = self.compute_motion(action)
            if not all(math.isfinite(v) for v in (forward, yaw)):
                raise ValueError('controller returned a nonfinite command')
            self.previous_error = None
        except Exception as exc:
            forward, yaw = 0., 0.
            error = str(exc)
            error_text = error
            if error != self.previous_error:
                self.get_logger().error(f'Navigation failed; requesting zero motion: {error}')
                self.previous_error = error
        msg = MotionRequest()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.forward_speed_m_s, msg.yaw_rate_rad_s = float(forward), float(yaw)
        self.publisher.publish(msg)
        self.record(observation, estimate, action, forward, yaw, error_text)

    # ------------------------------------------------------------ logging
    COLUMNS = ['ros_time_s', 'image_age_s', 'found', 'raw_error', 'yellow_pixels',
               'blue_left', 'blue_right', 'estimate', 'valid', 'est_status',
               'action', 'forward_m_s', 'yaw_rad_s', 'error']

    def open_log(self):
        """One CSV per launch in ~/nav_runs/run_<date>_<time>/ (outside the repo)."""
        if not self.cfg['csv_logging']:
            return
        try:
            run_dir = os.path.join(os.path.expanduser(self.cfg['log_dir']),
                                   datetime.now().strftime('run_%Y%m%d_%H%M%S'))
            os.makedirs(run_dir, exist_ok=True)
            path = os.path.join(run_dir, 'nav_log.csv')
            self.csv_file = open(path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(self.COLUMNS)
            self.get_logger().info(f'CSV log: {path}')
        except Exception as exc:
            self.csv_file = self.csv_writer = None
            self.get_logger().error(f'CSV logging off: {exc}')

    def record(self, observation, estimate, action, forward, yaw, error_text):
        try:
            now_s = self.now_s()
            r = observation.result if observation else None
            age = estimate.image_age_s if estimate else math.nan
            raw = r.error if (r and r.found) else math.nan
            est = estimate.offset if (estimate and estimate.offset is not None) else math.nan
            row = [f'{now_s:.4f}', f'{age:.4f}', int(bool(r and r.found)), f'{raw:+.4f}',
                   r.yellow_pixels if r else 0,
                   f'{r.blue_left:.3f}' if r else '', f'{r.blue_right:.3f}' if r else '',
                   f'{est:+.4f}', int(bool(estimate and estimate.valid)),
                   estimate.status if estimate else '',
                   action.mode if action else 'stop', f'{forward:.4f}', f'{yaw:+.4f}', error_text]
            if self.csv_writer is not None:
                self.csv_writer.writerow(row)
                if now_s - self.last_flush_s >= 1.0:
                    self.csv_file.flush()
                    self.last_flush_s = now_s
            if now_s - self.last_print_s >= self.cfg['log_period_s']:
                self.last_print_s = now_s
                blue = ' BLUE' if (r and r.blue_warning) else ''
                self.get_logger().info(
                    f'{row[10]:6s} | raw {raw:+.3f} | est {est:+.3f} valid {row[8]} '
                    f'({row[9]}) | img age {age:.3f}s | v {forward:.3f} w {yaw:+.3f}{blue}')
        except Exception as exc:
            self.get_logger().warning(f'Logging failed: {exc}', throttle_duration_sec=5.0)

    def close_log(self):
        if self.csv_file is not None:
            try:
                self.csv_file.close()
            except Exception:
                pass
            self.csv_file = self.csv_writer = None


def main():
    rclpy.init()
    node = Navigation()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close_log()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
