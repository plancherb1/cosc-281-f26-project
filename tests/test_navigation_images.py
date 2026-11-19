import pytest
import cv2
import types
from pathlib import Path

# Tests for specific images, written by claude

DATA = Path(__file__).parent / 'data'


def _obs(navigation, name):
    raw_image = cv2.imread(str(DATA / name))
    now = navigation.get_clock().now()
    navigation.image_stamp = types.SimpleNamespace(
        sec=now.nanoseconds // 10**9,
        nanosec=now.nanoseconds % 10**9
    )
    return navigation.process_image(raw_image)


# ---------------------------------------------------------------- path_straight.jpg

@pytest.fixture
def obs_straight(navigation):
    return _obs(navigation, 'path_straight.jpg')

@pytest.fixture
def est_straight(navigation, obs_straight):
    return navigation.update_estimate(obs_straight, imu=navigation.imu)

@pytest.fixture
def action_straight(navigation, est_straight):
    return navigation.choose_action(est_straight)

@pytest.fixture
def motion_straight(navigation, action_straight):
    return navigation.compute_motion(action_straight)


def test_straight_found(obs_straight):
    assert obs_straight.result.found

def test_straight_error_near_zero(obs_straight):
    assert abs(obs_straight.result.error) < 0.05

def test_straight_follows(action_straight):
    assert action_straight.mode == 'follow'

def test_straight_yaw_near_zero(motion_straight):
    forward, yaw = motion_straight
    assert forward == pytest.approx(0.05)
    assert abs(yaw) < 0.05


# ---------------------------------------------------------------- path_curve_right.jpg
# Road centroid is to the right of centre: error > 0, robot turns right (yaw < 0).

@pytest.fixture
def obs_curve_right(navigation):
    return _obs(navigation, 'path_curve_right.jpg')

@pytest.fixture
def est_curve_right(navigation, obs_curve_right):
    return navigation.update_estimate(obs_curve_right, imu=navigation.imu)

@pytest.fixture
def action_curve_right(navigation, est_curve_right):
    return navigation.choose_action(est_curve_right)

@pytest.fixture
def motion_curve_right(navigation, action_curve_right):
    return navigation.compute_motion(action_curve_right)


def test_curve_right_found(obs_curve_right):
    assert obs_curve_right.result.found

def test_curve_right_positive_error(obs_curve_right):
    assert obs_curve_right.result.error > 0

def test_curve_right_more_blue_left(obs_curve_right):
    r = obs_curve_right.result
    assert r.blue_left > r.blue_right

def test_curve_right_follows(action_curve_right):
    assert action_curve_right.mode == 'follow'

def test_curve_right_turns_right(motion_curve_right):
    _, yaw = motion_curve_right
    assert yaw < 0   # negative yaw = turn right (ROS convention)


# ---------------------------------------------------------------- path_curve_left.jpg
# Road centroid is to the left of centre: error < 0, robot turns left (yaw > 0).

@pytest.fixture
def obs_curve_left(navigation):
    return _obs(navigation, 'path_curve_left.jpg')

@pytest.fixture
def est_curve_left(navigation, obs_curve_left):
    return navigation.update_estimate(obs_curve_left, imu=navigation.imu)

@pytest.fixture
def action_curve_left(navigation, est_curve_left):
    return navigation.choose_action(est_curve_left)

@pytest.fixture
def motion_curve_left(navigation, action_curve_left):
    return navigation.compute_motion(action_curve_left)


def test_curve_left_found(obs_curve_left):
    assert obs_curve_left.result.found

def test_curve_left_negative_error(obs_curve_left):
    assert obs_curve_left.result.error < 0

def test_curve_left_more_blue_right(obs_curve_left):
    r = obs_curve_left.result
    assert r.blue_right > r.blue_left

def test_curve_left_follows(action_curve_left):
    assert action_curve_left.mode == 'follow'

def test_curve_left_turns_left(motion_curve_left):
    _, yaw = motion_curve_left
    assert yaw > 0   # positive yaw = turn left (ROS convention)


# ---------------------------------------------------------------- path_dead_end.jpg
# Blue boundary is symmetric on both sides; road is still found.

@pytest.fixture
def obs_dead_end(navigation):
    return _obs(navigation, 'path_dead_end.jpg')

@pytest.fixture
def est_dead_end(navigation, obs_dead_end):
    return navigation.update_estimate(obs_dead_end, imu=navigation.imu)

@pytest.fixture
def action_dead_end(navigation, est_dead_end):
    return navigation.choose_action(est_dead_end)


def test_dead_end_found(obs_dead_end):
    assert obs_dead_end.result.found

def test_dead_end_balanced_blue(obs_dead_end):
    r = obs_dead_end.result
    assert abs(r.blue_left - r.blue_right) < 0.01

def test_dead_end_follows(action_dead_end):
    assert action_dead_end.mode == 'follow'


# ---------------------------------------------------------------- path_wide_open.jpg
# More yellow visible than any other test image.

@pytest.fixture
def obs_wide_open(navigation):
    return _obs(navigation, 'path_wide_open.jpg')

@pytest.fixture
def est_wide_open(navigation, obs_wide_open):
    return navigation.update_estimate(obs_wide_open, imu=navigation.imu)

@pytest.fixture
def action_wide_open(navigation, est_wide_open):
    return navigation.choose_action(est_wide_open)

@pytest.fixture
def motion_wide_open(navigation, action_wide_open):
    return navigation.compute_motion(action_wide_open)


def test_wide_open_found(obs_wide_open):
    assert obs_wide_open.result.found

def test_wide_open_most_yellow(obs_wide_open):
    assert obs_wide_open.result.yellow_pixels > 20000

def test_wide_open_follows(action_wide_open):
    assert action_wide_open.mode == 'follow'

def test_wide_open_forward_speed(motion_wide_open):
    forward, _ = motion_wide_open
    assert forward == pytest.approx(0.05)


# ---------------------------------------------------------------- frame.jpg
# Off-centre frame: large negative error, heavy blue on the right, large left turn.

@pytest.fixture
def obs_frame(navigation):
    return _obs(navigation, 'frame.jpg')

@pytest.fixture
def est_frame(navigation, obs_frame):
    return navigation.update_estimate(obs_frame, imu=navigation.imu)

@pytest.fixture
def action_frame(navigation, est_frame):
    return navigation.choose_action(est_frame)

@pytest.fixture
def motion_frame(navigation, action_frame):
    return navigation.compute_motion(action_frame)


def test_frame_found(obs_frame):
    assert obs_frame.result.found

def test_frame_large_negative_error(obs_frame):
    assert obs_frame.result.error < -0.4

def test_frame_heavy_right_blue(obs_frame):
    assert obs_frame.result.blue_right > 0.9

def test_frame_follows(action_frame):
    assert action_frame.mode == 'follow'

def test_frame_turns_left(motion_frame):
    _, yaw = motion_frame
    assert yaw > 0   # large leftward correction
