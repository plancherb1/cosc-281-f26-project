"""Road estimator (Team The Perceptrons, estimation: Michael Dong).

Takes the road position that perception finds in each camera frame and turns
it into a steadier value, plus a flag saying whether it is safe to steer on.

Road position (offset): -1 = road far left, 0 = centered, +1 = road far right.
All times are in seconds.
"""
import math
from dataclasses import dataclass


@dataclass
class RoadEstimate:
    offset: float | None   # steadied road position; None until the road is first seen
    valid: bool            # True if it is safe to steer on `offset`
    status: str            # what happened this frame (shown in the logs)
    raw: float | None      # perception's value this frame; None if no road found
    image_age_s: float     # how old the camera frame is
    road_age_s: float      # how long since the road was last seen


class RoadEstimator:
    """Averages new readings with the old estimate and rejects bad readings.

    smoothing_alpha      how much each new reading counts (0.2 = 20% new, 80% old)
    max_image_age_s      camera frames older than this are not used
    road_lost_timeout_s  how long to keep the last estimate when the road disappears
    max_jump             a reading that jumps more than this is ignored...
    outlier_accept_count ...unless it happens this many frames in a row
    """

    def __init__(self, smoothing_alpha=0.2, max_image_age_s=0.25,
                 road_lost_timeout_s=0.5, max_jump=0.3, outlier_accept_count=3):
        self.smoothing_alpha = smoothing_alpha
        self.max_image_age_s = max_image_age_s
        self.road_lost_timeout_s = road_lost_timeout_s
        self.max_jump = max_jump
        self.outlier_accept_count = outlier_accept_count
        self.reset()

    def reset(self):
        self.offset = None
        self.last_road_stamp_s = None
        self.last_used_stamp_s = None
        self.outlier_streak = 0

    def update(self, offset, found, image_stamp_s, now_s):
        """Update the estimate with one frame. image_stamp_s is None if there is no image yet."""
        raw = offset if found else None

        if image_stamp_s is None:
            return self._answer('no_image', raw, math.inf, now_s)

        # Don't use old camera frames.
        image_age_s = now_s - image_stamp_s
        if image_age_s > self.max_image_age_s:
            return self._answer('stale_image', raw, image_age_s, now_s)

        # Skip a frame we already used (the loop can run twice on the same frame).
        if self.last_used_stamp_s is not None and image_stamp_s <= self.last_used_stamp_s:
            return self._answer('repeat_frame', raw, image_age_s, now_s)
        self.last_used_stamp_s = image_stamp_s

        # No road in this frame: keep the last estimate.
        if not found or raw is None or not math.isfinite(raw):
            lost = self._road_age(now_s) > self.road_lost_timeout_s
            return self._answer('lost' if lost else 'holding', None, image_age_s, now_s)

        # First time seeing the road, or seeing it again after losing it: start over.
        if (self.offset is None or
                image_stamp_s - self.last_road_stamp_s > self.road_lost_timeout_s):
            self._accept(raw, image_stamp_s)
            return self._answer('first', raw, image_age_s, now_s)

        # A sudden big jump is probably glare, so ignore it,
        # unless it keeps happening (then the road really moved).
        if abs(raw - self.offset) > self.max_jump:
            self.outlier_streak += 1
            if self.outlier_streak < self.outlier_accept_count:
                return self._answer('outlier_ignored', raw, image_age_s, now_s)
            self._accept(raw, image_stamp_s)
            return self._answer('outlier_accepted', raw, image_age_s, now_s)

        # Normal case: mix the new reading into the old estimate.
        a = self.smoothing_alpha
        self._accept((1.0 - a) * self.offset + a * raw, image_stamp_s)
        return self._answer('smoothed', raw, image_age_s, now_s)

    def _accept(self, offset, image_stamp_s):
        """Save a new estimate and remember when the road was seen."""
        self.offset = offset
        self.outlier_streak = 0
        self.last_road_stamp_s = image_stamp_s

    def _road_age(self, now_s):
        if self.last_road_stamp_s is None:
            return math.inf
        return now_s - self.last_road_stamp_s

    def _answer(self, status, raw, image_age_s, now_s):
        """Build the result. Valid only if the frame is recent and the road was seen recently."""
        road_age_s = self._road_age(now_s)
        valid = (self.offset is not None and
                 image_age_s <= self.max_image_age_s and
                 road_age_s <= self.road_lost_timeout_s)
        return RoadEstimate(self.offset, valid, status, raw, image_age_s, road_age_s)
