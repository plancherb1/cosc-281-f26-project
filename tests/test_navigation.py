import pytest
import glob
import cv2
import types

from pathlib import Path

@pytest.fixture(params=list((Path(__file__).parent / 'data').glob('*.jpg')))
def image(request):
   return request.param

@pytest.fixture
def observation(navigation, image):
    raw_image = cv2.imread(str(image))
    now = navigation.get_clock().now()
    navigation.image_stamp = types.SimpleNamespace(sec=now.nanoseconds // 10**9, nanosec=now.nanoseconds % 10**9)
    return navigation.process_image(raw_image)

@pytest.fixture
def estimate(navigation, observation):
    return navigation.update_estimate(observation, imu=navigation.imu)

@pytest.fixture
def action(navigation, estimate):
   return navigation.choose_action(estimate)

@pytest.fixture
def motion(navigation, action):
   return navigation.compute_motion(action)

def test_process_image_returns_none_on_missing_image(navigation):
    assert navigation.process_image(None) is None

def test_process_image(observation):
    # Test that given an image, an observation is found with a ~somewhat low error
    assert observation is not None
    assert observation.result.found is True
    assert abs(observation.result.error) < 0.5

def test_update_estimate(estimate):
    # Test that the code is capable of creating valid estimates
    assert estimate is not None
    assert estimate.valid is True

def test_choose_action(action):
    # Test that an action exists, and is not "stop" (test cases shouldn't cause stoppages yet)
    assert action is not None
    assert action.mode != 'stop'

def test_ignore_stale_action(navigation, image):
    raw_image = cv2.imread(str(image))
    navigation.image_stamp = types.SimpleNamespace(sec=0, nanosec=0)
    obs = navigation.process_image(raw_image)
    est = navigation.update_estimate(obs, imu=navigation.imu)
    assert est is not None
    act = navigation.choose_action(est)
    assert act.mode == 'stop'
    assert act.reason == 'stale_image'

def test_compute_motion(motion):
    assert motion is not None
    # Not an empty tuple either
    assert motion[0] != 0 and motion[1] != 0

