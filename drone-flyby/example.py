"""Detector + world model + camera policy behind the /predict endpoint.

Per request:
  1. advance the world model to this frame (propagate every known object with the ground homography)
  2. detect in the transmitted view and fold the detections into the world model
  3. answer with every tracked object in the whole source frame
  4. pick the next camera view and make it legal against the request's constraints

The original baseline lives in example_baseline.py.
"""

import base64
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List

import motion
from camera_policy import CameraPlanner, legalise
import detector as detector_module
from detector import build_detector
from dtos import OBJECT_CLASSES, DroneFlybyPredictionDto, DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from tracker import WorldModel
from utils import clip_bbox_to_frame, decode_view, source_bbox_to_global

logger = logging.getLogger(__name__)

DETECTOR = build_detector()
MAX_ANNOTATIONS = 500
IDLE_BEFORE_SWAP_S = 20.0
_last_request = [time.monotonic()]
MAX_SEQUENCES = 8
_detector_lock = threading.Lock()
RECORD_DIR = os.environ.get('DRONE_RECORD_DIR')
_record_queue: 'queue.Queue' = queue.Queue(maxsize=1000)


def _recorder():
    while True:
        request, response = _record_queue.get()
        try:
            directory = Path(RECORD_DIR) / request.sequence_id.replace('/', '_').replace(':', '_')
            directory.mkdir(parents=True, exist_ok=True)
            stem = f'{request.frame_index:04d}_f{request.frame:06d}'
            (directory / f'{stem}.png').write_bytes(base64.b64decode(request.view.image))
            meta = request.model_dump(exclude={'view': {'image'}})
            meta['response'] = response.model_dump()
            (directory / f'{stem}.json').write_text(json.dumps(meta))
        except Exception:
            logger.exception('Recording failed')


if RECORD_DIR:
    threading.Thread(target=_recorder, daemon=True).start()


def _model_reloader():
    """Swap in a new detector when models/detector.pt is repointed, but only while no attempt is running."""
    global DETECTOR
    loaded = os.path.realpath(detector_module.MODEL_PATH)
    pending, pending_path = None, None
    while True:
        time.sleep(5)
        try:
            target = os.path.realpath(detector_module.MODEL_PATH)
            idle = lambda: time.monotonic() - _last_request[0] > IDLE_BEFORE_SWAP_S
            if target != loaded and target != pending_path and idle():
                pending, pending_path = detector_module.YoloDetector(target), target
            if pending is not None and idle():
                with _detector_lock:
                    DETECTOR = pending
                logger.warning('DETECTOR SWAPPED to %s', pending_path)
                loaded, pending, pending_path = pending_path, None, None
        except Exception:
            logger.exception('Model reload failed')
            time.sleep(30)


if os.environ.get('DRONE_ORACLE') != '1':
    threading.Thread(target=_model_reloader, daemon=True).start()


class SequenceState:
    def __init__(self):
        self.world = WorldModel()
        self.camera = CameraPlanner()
        self.motion = motion.MotionEstimator()
        self.lock = threading.Lock()


_states: Dict[str, SequenceState] = {}
_states_lock = threading.Lock()


def _state(request: DroneFlybyPredictRequestDto) -> SequenceState:
    with _states_lock:
        state = _states.get(request.sequence_id)
        restarted = state is not None and state.world.frame is not None and request.frame < state.world.frame
        if state is None or restarted:
            state = _states[request.sequence_id] = SequenceState()
            while len(_states) > MAX_SEQUENCES:
                _states.pop(next(iter(_states)))
        return state


def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    if request.camera_command_feedback is not None:
        logger.warning('Camera command from frame %s ignored: %s',
                       request.camera_command_feedback.frame, request.camera_command_feedback.reason)

    _last_request[0] = time.monotonic()
    state = _state(request)
    with state.lock:
        annotations: List[DroneFlybyPredictionDto] = []
        image = None
        try:
            image = decode_view(request.view)
            state.world.H = state.motion.current()
            state.world.advance(request.frame)
            with _detector_lock:
                detections = DETECTOR(image, request)
            state.world.update(detections, request.view.source_region_xyxy, request.view.resolution_level)
            annotations = _annotations(state.world, request)
        except Exception:
            logger.exception('Detection failed on frame %s', request.frame)

        requested_view = None
        try:
            view = request.view
            state.camera.observe_motion(state.motion.current(), state.motion.pairs)
            target = state.camera.next_target(request.frame, (view.resolution_level, view.center_x, view.center_y))
            requested_view = legalise(request, target) if target is not None else None
            if requested_view is not None:
                state.camera.last_command = (requested_view.resolution_level, requested_view.center_x, requested_view.center_y)
        except Exception:
            logger.exception('Camera policy failed on frame %s', request.frame)

    response = DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=annotations,
        requested_view=requested_view,
    )
    if image is not None:
        threading.Thread(target=_update_motion, args=(state, request.frame, image, request.view.source_region_xyxy), daemon=True).start()
    if RECORD_DIR:
        try:
            _record_queue.put_nowait((request, response))
        except queue.Full:
            pass
    return response


def _update_motion(state: SequenceState, frame: int, image, region) -> None:
    with state.lock:
        try:
            state.motion.process(frame, image, region)
        except Exception:
            logger.exception('Motion update failed on frame %s', frame)


def _annotations(world: WorldModel, request: DroneFlybyPredictRequestDto) -> List[DroneFlybyPredictionDto]:
    out = []
    for class_id, box, confidence in sorted(world.report(), key=lambda r: -r[2])[:MAX_ANNOTATIONS]:
        bbox = clip_bbox_to_frame(source_bbox_to_global(box, request.original_width, request.original_height))
        if bbox is None:
            continue
        out.append(DroneFlybyPredictionDto(
            object_id=OBJECT_CLASSES[class_id],
            bbox=[round(float(c), 6) for c in bbox],
            confidence=round(confidence, 4),
        ))
    return out
