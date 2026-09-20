#!/usr/bin/env bash
# Usage: ./serve.sh start|stop|status|restart   (env: DRONE_MODEL, DRONE_DEVICE, DRONE_RECORD_DIR,
# DRONE_CONF, DRONE_IMGSZ_MULTI, DRONE_MERGE_IOU, DRONE_MOTION_*). The detector and motion settings are
# defaulted here rather than passed by hand so a restart, or the watchdog, cannot quietly drop them:
# measured on the local scorer, 960 alone is 0.599 at 84 ms per request, 960+1280 is 0.647, 960+1152+1280
# is 0.685 but measured 296 ms mean and 364 ms max per request against a 333 ms frame interval, and a
# frame we are too slow for is skipped, so two scales it is.
# The homography window was oversized: 600 points / 6 pairs / 1000 RANSAC iterations scores 0.687 against
# 0.673 for 1500/12/3000, and the window fit is the most expensive stage of the request and holds the
# sequence lock, so the cheaper settings buy accuracy and wall-clock at once.
set -euo pipefail
cd "$(dirname "$0")"
DATA=/mnt/data/nordicai/drone-flyby
PIDFILE=$DATA/server.pid
LOG=$DATA/runs/logs/server_$(date -u +%Y%m%dT%H%M%S).log
export DRONE_MODEL=${DRONE_MODEL:-$DATA/models/detector.pt}
export DRONE_DEVICE=${DRONE_DEVICE:-0}
export DRONE_RECORD_DIR=${DRONE_RECORD_DIR-$DATA/recordings}
# Dip to Level 2 for native pixels once per patrol position. On the reference scene, matched on
# skipped frames: 0.703 -> 0.762 mAP over 18 runs (p<0.0001), and 0.819 -> 0.842 offline.
export DRONE_PATROL=${DRONE_PATROL:-l2dip}
export DRONE_CONF=${DRONE_CONF:-0.10}
export DRONE_IMGSZ_MULTI=${DRONE_IMGSZ_MULTI:-960:1152}
export DRONE_MERGE_IOU=${DRONE_MERGE_IOU:-0.75}
export DRONE_MOTION_POINTS=${DRONE_MOTION_POINTS:-600}
export DRONE_MOTION_WINDOW=${DRONE_MOTION_WINDOW:-6}
export DRONE_MOTION_ITERS=${DRONE_MOTION_ITERS:-1000}

running() { [[ -f $PIDFILE ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-status}" in
  start)
    if running; then echo "already running (pid $(cat "$PIDFILE"))"; exit 0; fi
    source /mnt/data/virtual_environments/nordicai-drone-flyby/bin/activate
    nohup python api.py > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    for _ in $(seq 60); do curl -sf localhost:9053/ >/dev/null && break; sleep 1; done
    if running; then echo "started pid $(cat "$PIDFILE"), model $DRONE_MODEL, scales $DRONE_IMGSZ_MULTI, conf $DRONE_CONF, motion $DRONE_MOTION_POINTS/$DRONE_MOTION_WINDOW/$DRONE_MOTION_ITERS, log $LOG"; else echo "failed, see $LOG"; tail -20 "$LOG"; exit 1; fi
    ;;
  stop)
    if running; then kill "$(cat "$PIDFILE")"; sleep 2; fi
    rm -f "$PIDFILE"
    echo stopped
    ;;
  restart) "$0" stop; "$0" start ;;
  status) if running; then echo "running pid $(cat "$PIDFILE")"; else echo "not running"; fi; ss -ltnp | grep ':9053' || true ;;
esac
