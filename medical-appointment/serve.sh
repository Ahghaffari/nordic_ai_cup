#!/usr/bin/env bash
# Usage: ./serve.sh start|stop|status|restart
# env: MEDICAL_PORT (9054), MED_PIPELINE=0 serves the baseline, any other MED_* from settings.py
# Picks the model by what the GPUs can hold. The big Gemma-4-26B-A4B needs a card to itself (15.4 GB), so it goes
# on GPU 1 and speech recognition (4.6 GB) joins the drone server on GPU 0; the drone endpoint must keep its room,
# hence the check on both cards. Otherwise everything shares GPU 1 with the smaller Gemma-4-12B. MED_LLM overrides.
set -euo pipefail
cd "$(dirname "$0")"
DATA=/mnt/data/nordicai/medical-appointment
PIDFILE=$DATA/server.pid
LOG=$DATA/runs/logs/server_$(date -u +%Y%m%dT%H%M%S).log
PORT=${MEDICAL_PORT:-9054}
MODELS=/mnt/data/nordicai/medical-appointment/models/gguf
free_gpu() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null || echo 0; }
FREE_GPU0=$(free_gpu 0)
FREE_GPU1=$(free_gpu 1)
# The model is pinned, not chosen. Picking it from whatever memory happened to be free meant the
# deployed model changed whenever something else on the box started or stopped, which is how an
# unvalidated model ended up serving. If the card cannot hold the pinned one, refuse to start rather
# than quietly serve a different one than the one that was validated.
export MED_LLM=${MED_LLM:-$MODELS/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}
if [[ $MED_LLM == *26B* ]]; then
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
  export MED_DEVICE=${MED_DEVICE:-cuda:0}
  export MED_LLM_MAIN_GPU=${MED_LLM_MAIN_GPU:-1}
  NEED_GPU1=15500; NEED_GPU0=5500
else
  export CUDA_VISIBLE_DEVICES=${MED_GPU:-1}
  export MED_DEVICE=${MED_DEVICE:-cuda:0}
  NEED_GPU1=15500; NEED_GPU0=0
fi

# Conversations vary in length, so the caching allocator reserves a new block for every longer one and the
# high-water mark creeps towards the card. Expandable segments reuse one growing arena instead.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Decode settings, pinned for the same reason the model is: they are what was validated, and passing
# them on the command line meant any restart silently reverted to different ones. Validated at 0.7863
# against 0.7856 for threshold 0.3 / line_quote / 0.0 / -0.15 - the same system within a thousandth,
# measured twice offline and twice by the graders.
export MED_YES_THRESHOLD=${MED_YES_THRESHOLD:-0.10}
export MED_SPAN_MODE=${MED_SPAN_MODE:-quote}
export MED_PAD_START=${MED_PAD_START:-0.05}
export MED_PAD_END=${MED_PAD_END:--0.20}

running() { [[ -f $PIDFILE ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-status}" in
  start)
    if running; then echo "already running (pid $(cat "$PIDFILE"))"; exit 0; fi
    FREE_GPU0=$(free_gpu 0); FREE_GPU1=$(free_gpu 1)
    if (( FREE_GPU1 < NEED_GPU1 || FREE_GPU0 < NEED_GPU0 )); then
      echo "refusing to start: $(basename "$MED_LLM") needs ${NEED_GPU1} MiB on GPU1 and ${NEED_GPU0} MiB on GPU0," \
           "have ${FREE_GPU1} and ${FREE_GPU0}. Free the card or set MED_LLM deliberately."
      exit 1
    fi
    mkdir -p "$(dirname "$LOG")"
    source /mnt/data/virtual_environments/nordicai-medical-appointment/bin/activate
    nohup python api.py > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    for _ in $(seq 600); do
      curl -sf localhost:$PORT/ >/dev/null && break
      running || break
      sleep 1
    done
    if running; then echo "started pid $(cat "$PIDFILE") on GPU $CUDA_VISIBLE_DEVICES, model $(basename "${MED_LLM:-default}"), log $LOG"; else echo "failed, see $LOG"; tail -20 "$LOG"; exit 1; fi
    ;;
  stop)
    if running; then kill "$(cat "$PIDFILE")"; sleep 2; fi
    rm -f "$PIDFILE"
    echo stopped
    ;;
  restart) "$0" stop; "$0" start ;;
  status) if running; then echo "running pid $(cat "$PIDFILE")"; else echo "not running"; fi; ss -ltnp | grep ":$PORT" || true ;;
esac
