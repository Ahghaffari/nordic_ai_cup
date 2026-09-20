# Drone flyby — solution

Evaluation score **0.2484**. Validation 0.6954 on the 249-frame sequence.

The gap between those two numbers is the interesting part of this submission and is analysed below.

## Approach

The protocol is asymmetric: the camera shows one 960×540 crop, but every frame is scored against
every object in the full 3840×2160 source frame. At Level 2 that crop is a sixteenth of the ground.
Answering only for what is visible caps the score at whatever fraction the camera happens to cover.

The way out is that the objects are stationary and the drone flies a straight line, so the ground
moves through the frame by an almost exact per-frame homography — 0.7 px median centre error when
fitted on the reference scene. Once an object has been seen it can be reported on every later frame
by propagating its box, whether or not the camera is still looking at it.

```
960x540 crop
  -> homography fitted live from SIFT matches between consecutive views
  -> world.advance(frame)        every known box warped forward by H^steps
  -> YOLO at 960 and 1152, merged per class
  -> world.update(...)           detections matched to tracks, boxes and classes corrected
  -> world.report()              every tracked object, in full-frame coordinates
  -> camera policy               where to point next, made legal against the request's constraints
```

### The world model

A track holds a box, a per-class score vector rather than a single label, hit and miss counts, and
the best resolution level it has been seen at. Detections are matched to tracks on IoU ≥ 0.2 **or**
centre distance below 0.6 × size — the second clause matters because these objects are a few pixels
across, where IoU is unforgiving of a 2 px error.

Merging is weighted by the level the observation came from (0.35 / 0.65 / 0.85 for levels 0/1/2): a
Level 2 look has real pixels behind it and dominates, a Level 0 look only nudges. An unmatched track
earns a miss only when it *should* have been visible — inside the current view and large enough at
this level — so an object is never penalised for the camera looking elsewhere.

Reported confidence is `purity × evidence × staleness`, where staleness is `0.97^(frames since seen)`.
Since mAP ranks predictions by confidence, a long-unseen track should sit below a freshly confirmed
one rather than be deleted: it still earns recall if right, without outranking live detections.

### Camera control

New objects can only enter at the edge the ground is coming from, and the world model keeps
everything already seen. So the camera sweeps the whole frame once at Level 1, then patrols that
edge. Which edge it is is read off the fitted homography rather than assumed, by warping the frame
centre and taking the direction of travel.

Two details were necessary rather than decorative:

**The schedule is locked to the frame number**, not to how many commands were applied. Before that, a
single late response shifted the patrol for the rest of the flight, and runs differed by ±0.05 for
reasons unrelated to the model.

**Every command must be legal from both the reported view and the last command sent**, because a
request's view can lag one command behind the evaluator's camera. Commands that were legal from the
stale view were being refused against the real one. Stress-tested over 67,913 simulated commands with
stale views and skipped frames: none refused.

### Level 2 dipping

Level 2 is the only level that transmits native pixels; Level 1 is halved. The patrol dips to Level 2
at each position for one frame and returns to Level 1 before moving — returning first keeps every hop
legal, since the move limit comes from the level the camera is currently on and at Level 2 that is
551 px.

Measured on the reference scene over 18 runs, matched on skipped frames so the frame clock cannot
flatter either side:

| patrol | mean mAP | sd |
|---|---|---|
| Level 1 only | 0.703 | 0.024 |
| **Level 1 with Level 2 dips** | **0.762** | 0.016 |
| Level 2 throughout | 0.541 | — |

Permutation test p < 0.0001, and 0.819 → 0.842 offline where the run is deterministic. Staying at
Level 2 loses 0.157: a sixteenth of the frame per look starves the world model of anything to
propagate. Zoom pays as a way of sampling, not as somewhere to live.

### Timing

The frame interval is 333 ms and a slow answer costs the frames that pass while the server is busy.
Measured over 1,021 requests: p50 81 ms, p90 136 ms, p99 226 ms, zero over budget. The homography fit
is the dominant cost, so its window is deliberately small (6 pairs, 600 points, 1000 RANSAC
iterations) and it runs in a background thread after the response is sent.

In the graded run this held: **250 frames answered, none skipped, no camera command refused.**

## Why the evaluation score is a third of the validation score

The run itself was faultless. What failed is generalisation, and the per-class predictions show it
exactly:

| class | evaluation | validation |
|---|---|---|
| hangar | 0 | 65 |
| medium_launcher | 0 | 86 |
| mine_roller | 0 | 180 |
| small_tower | 0 | 68 |
| condor | 0 | 27 |
| tank | 17 | 221 |

Five classes were never predicted once and a sixth nearly vanished. The score is COCO mAP
**macro-averaged per class**, so each missed class contributes a flat zero: six near-zero classes is
37% of the score before the remaining ten are considered, and those scored lower too (1,248
detections against 1,935 at comparable confidence).

The cause is a selection error rather than a modelling one. The detector was trained on synthetic
composites built from the 25 reference sprites, and then every subsequent choice — the model soup,
the inference scales, the confidence threshold, the merge IoU — was selected against ground truth
built for the *validation* scene. That ground truth was itself constructed by running three of our own
models over the validation recording and hand-verifying the tracks, so it contains only objects those
models could already find. Optimising against it selects for agreement with them on one specific
scene.

The warning sign was present before the attempt: the local scorer read 0.581 where the graders read
0.518 for the same model. That gap widened on a third scene.

What would fix it is holding out a scene that is never used for selection, or scoring on the
reference scene alone and accepting a noisier signal. Training more, or tuning inference further, was
not the constraint.

## Running it

```bash
pip install -r requirements.txt
./serve.sh start            # serves :9053, or: python api.py
python local_evaluator.py   # scores the supplied reference scene
```

The trained detector ships with this repository at `models/detector.pt` (167 MB, Git LFS) — a weight
average of three checkpoints, md5 `b2e43c8a40a8`. `serve.sh` pins the inference settings that were
submitted.

| setting | value |
|---|---|
| inference sizes | 960 and 1152, merged per class at IoU 0.75 |
| confidence | 0.10 |
| patrol | `l2dip` |
| homography window | 600 points, 6 pairs, 1000 iterations |

`training/` holds the pipeline that produced the detector: sprite extraction, synthetic dataset
construction, training, weight averaging, and the scoring tools used during development.

## Layout

| file | contents |
|---|---|
| `example.py` | the `/predict` entry point and per-sequence state |
| `detector.py` | YOLO wrapper, multi-scale inference, per-class merging |
| `tracker.py` | the frame-global world model |
| `motion.py` | live homography estimation from SIFT correspondences |
| `camera_policy.py` | where to point next, and making it legal |
| `training/` | dataset construction, training, model soup, scoring tools |
| `EXPERIMENTS.md` | what was measured, including what failed |
