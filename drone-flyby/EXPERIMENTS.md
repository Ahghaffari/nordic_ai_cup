# Drone flyby — validation log

Validation = 249-frame sequence on cases.nordicaicup.com. Recordings in /mnt/data/nordicai/drone-flyby/recordings/.

| # | UTC | Detector | Change vs previous best | Score | Skipped |
|---|---|---|---|---|---|
| 1 | 09:50 | oracle (Helsinki GT, invalid) | — | 0.000 | 0 |
| 2 | 10:07 | y26m_synth_v1 epoch 1 | fixed homography; run disturbed by a concurrent local test | 0.376 | — |
| 3 | 10:41 | y26m_synth_v1 epoch 1 | live SIFT homography, per-sequence state, staleness 0.97 | **0.452** | 0 |
| 4 | 10:47 | y26m_synth_v1 epoch 1 | staleness 1.0 (no decay) | 0.423 | 0 |
| 5 | 10:53 | y26m_synth_v1 epoch 1 | staleness 0.93 | 0.412 | 0 |
| 6 | 11:01 | y26m_synth_v1 epoch 1 | repeat of #3 settings (staleness 0.97), motion update after response | 0.403 | 2 |

Finding after #6: all runs see identical images, but the camera route drifted between runs (a late first response or a skipped frame shifted the list-based patrol for the rest of the flight; training validation on GPU0 caused skips in #6). Differences of ±0.05 between #3–#6 are mostly this noise. Fix: patrol schedule locked to frame number.

All runs #2–#6 used models/detector.pt -> y26m_synth_v1_e1.pt (yolo26m, epoch 1, md5 5f77a6b5).

| # | UTC | Detector | Change | Score | Skipped |
|---|---|---|---|---|---|
| 7 | 11:07 | y26m_synth_v1 epoch 1 | frame-locked camera schedule (staleness 0.97) | 0.395 | see recording |
| 8 | 11:46 | y26m_long_v1 epoch 2 (auto-deployed) | same pipeline as #7 | ? | 0 |
| 9 | 11:58 | y26m_long_v1 epoch 2 (auto-deployed) | same pipeline as #7 | ? | 1 |

## Training / deployment
- y26m_long_v1: synth_v1, selection val = val_synth (600) + real Helsinki views (130). Epoch mAP50: 0.877, **0.890**, 0.874. Stopped after epoch 3 to switch data.
- valbg_v2: 3,400 L1 composites/negatives + 893 L2 crops on recorded validation-scene views (candidate objects inpainted at conf>=0.05).
- y26m_long_v2: synth_v1 + valbg_v2, init from y26m_long_v1 epoch 2, 40 epochs, patience 10, no RAM cache (RAM caching both ranks deadlocked DDP).
- training/deploy_best.py repoints models/detector.pt to any epoch beating the deployed selection mAP50 by 0.002; the server hot-swaps only after 20 s without requests.

| # | UTC | Detector | Change | Score | Skipped |
|---|---|---|---|---|---|
| 10 | 12:41 | y26m_long_v2 epoch 1 (select mAP50 0.924) | — | 0.379 | 2, plus 2 refused camera moves |

Finding after #10: a request's view can lag one command behind the evaluator's camera; commands legal from the stale view were refused against the real camera (frames 5, 7). Fix: every command must be legal from both the reported view and the last commanded view (stress test: 67,913 simulated commands with stale views/skips, 0 refused).
| 11 | 12:50 | y26m_long_v2 epoch 1 | camera commands legal from both anchors | ? | 0 |
| 12 | 13:03 | y26m_long_v2 epoch 1 | — | ? | 0 |
| 13 | 13:28 | y26m_long_v2 epoch 2 (select mAP50 0.900) | manual deploy | 0.358 | see recording |
| 14 | 13:31 | y26m_long_v2 epoch 3 (select mAP50 0.888) | manual deploy | 0.323 | 0 |

Finding after #14: validation falls with every epoch on valbg_v2 (0.379 → 0.358 → 0.323) while the synthetic selection set says otherwise; boxes per frame 6.3 → 2.9. Real objects the old model missed were left in the v2 backgrounds and learned as background. Stopped y26m_long_v2.
| 15 | 13:35 | y26m_long_v1 epoch 2 (synth_v1 only, select mAP50 0.890) | manual deploy | 0.403 | |

Finding after #15: a 2nd synthetic epoch is no better on real validation than epoch 1 (0.403 vs 0.395–0.452). The synthetic selection set does not predict the validation score. Next: build a ground truth for the recorded validation flight to score models offline. Live model reverted to y26m_synth_v1_e1.

## Local validation scorer (valgt)
Ground truth for the recorded validation flight: detections of 3 models on all 710 recorded views, linked into tracks with per-step homographies (median 1,906 SIFT inliers), verified visually (40 accepted objects, 9 ignore regions, ~70 confident false-positive tracks rejected). `training/valgt_replay.py` replays run 11's requests offline with any model; `training/valgt_score.py` scores COCO mAP@0.5 like the evaluator. Local vs official over 9 runs: Pearson 0.87, Spearman 0.91 (local runs ~0.05–0.1 higher). Replay reproduces recorded runs exactly.

| Model | local | official |
|---|---|---|
| synth_v1 e1 | 0.449 | 0.395–0.452 |
| long_v1 e1 / e2 / e3 | 0.460 / 0.500 / 0.301 | – / 0.403 / – |
| long_v2 e1 / e2 / e3 | 0.415 / 0.393 / 0.417 | 0.379 / 0.358 / 0.323 |
| y26s e2 / e3 | 0.255 / 0.178 | – |
| soup long_v1 e1+e2 | 0.578 | |
| soup long_v1 e1+e2+e3 | 0.485 | |
| soup long_v1 e1,e2 + long_v2 e1,e2,e3 | 0.507 | |
| **soup synth_v1 e1 + long_v1 e1 + e2** | **0.581** | **0.518** (run 17) |

## Inference image size and multi-scale merging (2026-09-18)

The detector ran at imgsz 960 on a 960x540 crop. Raising only the internal input size, with no retraining, and
merging detections from several sizes per class (detector.py: DRONE_IMGSZ, DRONE_IMGSZ_MULTI, DRONE_MERGE_IOU)
moves the local scorer a long way. Scales fail on different classes: 1280 finds mine layers (0.38 -> 0.93) and
medium hulls (0.00 -> 0.44) that 960 misses, while 960 keeps jet planes and spacecraft that 1280 drops to 0.00.

| detector setting (conf 0.10 throughout) | local mAP50 |
|---|---|
| 960 (deployed) | 0.599 |
| 1280 | 0.624 |
| 960 + 1280 | 0.647 |
| 960 + 1280 + 1536 | 0.655 |
| 960 + 1024 + 1152 + 1280 | 0.674 |
| 960 + 1152 + 1280 | 0.683 |
| **960 + 1152 + 1280, merge IoU 0.75** | **0.685** |

Cost: 213 ms per frame of detector time against 33 ms single-scale, measured on an idle P100. Frames arrive every
333 ms and a frame that is too slow is skipped, so this needs measuring on the GPU that actually serves - the
drone server shares GPU 0 with training, where it will be slower. 960+1280 (0.647) costs about 140 ms and is the
safer setting if headroom is short.

yolo26x was checked too: it runs (93 ms/frame, 0.53 GB) but training it costs about 7.5 h/epoch at batch 4, which
fills 15.9 GB of the 16 GB card, and the log above shows capacity is not the bottleneck.

### Deployed 2026-09-19 12:28

Live on :9053 with `DRONE_IMGSZ_MULTI=960:1280`, `DRONE_MERGE_IOU=0.75`, `DRONE_CONF=0.10` (defaults now in
serve.sh so a restart cannot drop them). Local scorer 0.647 against 0.599 single-scale.

Latency, same 40 consecutive recorded frames through the server, so the pipeline cost is held constant:

| | server-side p50 | p90 | max |
|---|---|---|---|
| single scale | 239 ms | 259 ms | 273 ms |
| 960+1280 | 270 ms | 300 ms | 312 ms |

The extra scale costs about 31 ms; the rest is the pipeline (SIFT homography, tracker, camera policy), which
dominates at roughly 240 ms on these frames. Frames arrive every 333 ms, so 960+1152+1280 (local 0.685) was
rejected: it measured 296 ms mean and 364 ms max per request, over the interval. Live check through the public
IP after deploying: p50 256 ms, max 301 ms.

Beware of measuring latency by posting one frame repeatedly - the tracker accumulates duplicates and inflates it.

### Validation 0.625 with 960+1280 (2026-09-19 12:42), now serving 960+1152

The attempt scored **0.625**, against 0.518 for the best single-scale run. It overlapped a medical attempt whose
speech model shares GPU 0: 47 of 278 requests ran over the 333 ms frame interval and **12 frames were never
delivered**. Submitting the two use cases at different times avoids that.

Scale pairs on the local scorer (conf 0.10, merge IoU 0.75): 960+1152 **0.673**, 960+1280 0.647, and
960+1152+1280 0.685 but too slow. Live latency after switching to 960+1152: p50 287 ms, p90 325 ms.

Where a request goes, measured stage by stage: decode_view 17 ms, detector 57 ms, world model and annotations
1 ms, response 0 ms, motion 38 ms with a full window, plus the wait on the previous frame's motion thread, which
holds the state lock. Nothing else is worth cutting; earlier readings of 237 ms for motion were CPU contention
from concurrent jobs, not the window fit.

motion.py now reads DRONE_MOTION_POINTS, DRONE_MOTION_WINDOW and DRONE_MOTION_ITERS (defaults unchanged at
1500/12/3000). Cheaper fits scored 0.686 (800/8/1500) and 0.687 (600/6/1000) on this flight at the same 38 ms,
so they are not faster; left at the defaults because the gain is small and the evaluation flight is unseen.

## Camera direction is measured, not assumed (2026-09-19)

The patrol band was hard-coded to the top edge because that is where new ground enters when the ground
drifts downwards. Nothing verified that against the flight in hand: a scene flown the other way would have
put the camera on the edge objects are *leaving* by, and each one would only be registered at the end of
its life - late, with few frames of evidence and low accumulated confidence.

Fitted the per-frame homography over the first 40 frames of every recorded sequence (many are byte-identical
replays, so ~4 distinct flights) and compared with the Helsinki prior, which was fitted on the *training*
scene:

| | ty (px/frame) | tx | sx |
|---|---|---|---|
| recordings (validation scene) | +52.31 … +52.63 | -12.11 … -12.89 | 1.00591 … 1.00680 |
| Helsinki prior | +53.12 | -12.96 | 1.00689 |

Two different scenes agree to within ~0.8 px per frame, so camera motion looks like a constant of the
simulator rather than a per-flight variable. The change is therefore insurance, not a fix: `incoming_edge`
(camera_policy.py) reads the displacement of the frame centre off the fitted homography and
`CameraPlanner.observe_motion`, called from example.py, fixes the band once, after 3 window fits and only
when the drift exceeds 5 px. On every flight seen so far it selects 'top' and the schedule is bit-identical
to before; 'bottom', 'left' and 'right' bands exist for the case that it is not.

Checked offline: real prior -> top, reversed -> bottom, sideways -> left/right, static -> no decision; no
switch before 3 fits; the band is held once fixed; every band's steady-state step is within the 1102 px
Level 1 limit ('right' has a single 1920 px entry hop that takes next_target's nearest-legal fallback for
one frame). Not testable on the local scorer, since replay feeds back recorded requests and cannot respond
to a different camera command - hence the conservative gating.

Smoke-tested live on :9053 over 14 consecutive recorded frames: sweep then top band, 2-3 annotations per
frame, 263-296 ms per request with the motion sweep competing for GPU 0.

## Homography window was oversized (2026-09-19)

The rolling window fit is the most expensive stage of a request and it holds the sequence lock, so it sets
the floor on latency. Swept it on the local scorer at the deployed detector settings (960:1152, merge 0.75,
conf 0.10):

| DRONE_MOTION_POINTS / WINDOW / ITERS | local mAP50 |
|---|---|
| 1500 / 12 / 3000 (deployed) | 0.673 |
| 800 / 8 / 1500 | 0.686 |
| **600 / 6 / 1000** | **0.687** |

Cheaper is not worse, it is slightly better - a shorter window tracks the flight's motion with less lag, and
the correspondences are so plentiful (median 1,906 SIFT inliers per pair) that 600 of them already pin the
fit. Deployed as the serve.sh defaults.

Timing from validation run on the previous settings, which is the honest baseline to beat: 247 frames
scored, median 82 ms, p90 203 ms, max 262 ms against the 333 ms interval, no frames lost. Note that posting
recorded frames back to back reads much higher (265-298 ms) because the motion thread for frame N still
holds the lock when N+1 arrives; the evaluator's 333 ms pacing is what the numbers above reflect.

### Validation 2026-09-19 15:23

0.6243 with the measured patrol band, against 0.6253 for the same configuration without it. The band
selector chose 'top' and never logged a switch, so the schedule was identical and the difference is run
noise - which is the intended result: the insurance is free.
