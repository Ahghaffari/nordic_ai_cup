"""Where to point the camera next.

New objects only enter at the edge the ground is coming from, and the world model keeps everything
already seen, so after one sweep of the full frame at Level 1 the camera patrols that edge, where
unseen objects appear. Every command is made legal against the request's own constraints first.

Which edge that is is read off the fitted homography rather than assumed: every flight recorded so
far drifts the ground downwards (~68 px per frame at the frame centre, so new ground enters at the
top) and that is the default, but a flight running the other way would put the camera on the edge
objects are leaving by, and each one would be picked up only at the end of its life. The band is
chosen once, from the first confident fit, and then held for the rest of the flight.
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from dtos import (ALLOWED_RESOLUTION_LEVELS, FULL_FRAME_CENTER, MAXIMUM_CENTER_DELTA_PIXELS,
                  DroneFlybyPredictRequestDto, RequestedViewDto)

logger = logging.getLogger(__name__)

Target = Tuple[int, int, int]

INITIAL_SWEEP: List[Target] = [
    (1, 960, 540), (1, 1920, 540), (1, 2880, 540),
    (1, 2880, 1620), (1, 1920, 1620), (1, 960, 1620),
]
# The horizontal bands dwell on the middle position, which the ground crosses twice; a vertical edge
# has only the two Level 1 centres, and 1080 px apart they are within the 1102 px per-frame limit.
# Each band starts at the end of the sweep's own corner where it can, so entering it costs no frame;
# 'right' cannot (1920 px away) and takes next_target's nearest-legal fallback for one frame.
PATROL_BANDS = {
    'top': [(1, 960, 540), (1, 1920, 540), (1, 2880, 540), (1, 1920, 540)],
    'bottom': [(1, 960, 1620), (1, 1920, 1620), (1, 2880, 1620), (1, 1920, 1620)],
    'left': [(1, 960, 540), (1, 960, 1620)],
    'right': [(1, 2880, 1620), (1, 2880, 540)],
}
DEFAULT_BAND = 'top'
PATROL: List[Target] = PATROL_BANDS[DEFAULT_BAND]
GRID: List[Target] = [(1, x, y) for y in (540, 1620) for x in (960, 1920, 2880)]

# Enough fits that the window holds real correspondences, and enough drift that the direction is not noise.
MIN_FITS_BEFORE_SWITCH = 3
MIN_DRIFT_PIXELS = 5.0

# Level 2 is the only level that sends native pixels; Level 1 is halved. DRONE_PATROL trades one
# against the other after the opening sweep. The move limit comes from the level the camera is on,
# and at Level 2 that is 551 px, so a Level 2 patrol has to step in small hops and cannot jump back
# out to a distant Level 1 centre in one frame.
#   l1      Level 1 along the incoming edge
#   l2dip   Level 1 for coverage, dipping to Level 2 at the same centre for a detailed look
PATROL_MODE = os.environ.get('DRONE_PATROL', 'l2dip')


def patrol_for(band: str) -> List[Target]:
    base = PATROL_BANDS[band]
    if PATROL_MODE == 'l2dip':
        # Coming back to Level 1 at the same centre before moving keeps every hop legal: the
        # distance is zero, and only the level changes.
        return [t for level, x, y in base for t in ((1, x, y), (2, x, y), (1, x, y))]
    return base


def incoming_edge(H: Sequence[Sequence[float]]) -> Optional[str]:
    """The edge new ground enters from, or None when the motion is too small to call.

    Measured as the displacement of the frame centre, not H's translation column, so the scale terms
    are accounted for. The ground moving down means unseen ground arrives at the top.
    """
    cx, cy = FULL_FRAME_CENTER
    q = np.asarray(H, float) @ np.array([cx, cy, 1.0])
    dx, dy = q[0] / q[2] - cx, q[1] / q[2] - cy
    if max(abs(dx), abs(dy)) < MIN_DRIFT_PIXELS:
        return None
    if abs(dy) >= abs(dx):
        return 'top' if dy > 0 else 'bottom'
    return 'left' if dx > 0 else 'right'


@dataclass
class CameraPlanner:
    """The schedule is locked to the frame number, not to how many commands were applied, so a
    late response or a skipped frame shifts the camera for one frame instead of for the rest of
    the flight."""

    start_frame: Optional[int] = None
    last_command: Optional[Target] = None
    band: str = DEFAULT_BAND
    band_fixed: bool = False

    def observe_motion(self, H, fits: int) -> None:
        """Fix the patrol band from the flight's own motion, once, before the sweep ends."""
        if self.band_fixed or H is None or fits < MIN_FITS_BEFORE_SWITCH:
            return
        edge = incoming_edge(H)
        if edge is None:
            return
        self.band_fixed = True
        if edge != self.band:
            logger.warning('Ground enters from the %s, not the %s: patrolling the %s band',
                           edge, self.band, edge)
            self.band = edge

    def desired(self, step: int) -> Target:
        if step <= 0:
            return (0, 1920, 1080)
        if step <= len(INITIAL_SWEEP):
            return INITIAL_SWEEP[step - 1]
        patrol = patrol_for(self.band)
        return patrol[(step - len(INITIAL_SWEEP) - 1) % len(patrol)]

    def next_target(self, frame: int, current: Target) -> Optional[Target]:
        """The view in a request can lag one command behind the evaluator's camera, so a command must
        be legal from the reported view and from the last view we asked for, whichever is real."""
        if self.start_frame is None:
            self.start_frame = frame
        target = self.desired(frame - self.start_frame + 1)
        anchors = [current]
        if self.last_command is not None and self.last_command != current:
            anchors.append(self.last_command)
        if _legal_from_all(target, anchors):
            return target

        options = [g for g in GRID + [current] if _legal_from_all(g, anchors)]
        if not options:
            return None
        return min(options, key=lambda g: math.hypot(g[1] - target[1], g[2] - target[2]))



def _legal_from_all(target: Target, anchors: List[Target]) -> bool:
    for level, cx, cy in anchors:
        if target[0] not in ALLOWED_RESOLUTION_LEVELS[level]:
            return False
        if target[0] == 0:
            continue
        if math.hypot(target[1] - cx, target[2] - cy) > MAXIMUM_CENTER_DELTA_PIXELS[level]:
            return False
    return True


def legalise(request: DroneFlybyPredictRequestDto, target: Target) -> Optional[RequestedViewDto]:
    constraints = request.camera_constraints
    view = request.view
    level, tx, ty = target

    if level not in constraints.allowed_resolution_levels:
        level = min(constraints.allowed_resolution_levels, key=lambda l: abs(l - level))
    bounds = constraints.bounds_for_level(level)
    if bounds is None:
        return None
    if level == 0:
        return RequestedViewDto(resolution_level=0, center_x=1920, center_y=1080)

    tx = min(max(tx, bounds.minimum_center_x), bounds.maximum_center_x)
    ty = min(max(ty, bounds.minimum_center_y), bounds.maximum_center_y)
    dx, dy = tx - view.center_x, ty - view.center_y
    distance = math.hypot(dx, dy)
    limit = constraints.maximum_center_delta - 1.0
    if distance > limit:
        tx = view.center_x + dx * limit / distance
        ty = view.center_y + dy * limit / distance
    cx = int(min(max(math.floor(tx) if dx > 0 else math.ceil(tx), bounds.minimum_center_x), bounds.maximum_center_x))
    cy = int(min(max(math.floor(ty) if dy > 0 else math.ceil(ty), bounds.minimum_center_y), bounds.maximum_center_y))
    return RequestedViewDto(resolution_level=int(level), center_x=cx, center_y=cy)
