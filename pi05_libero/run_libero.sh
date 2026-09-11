#!/bin/bash
# usage: ./run_libero.sh <TASK_ID> <PORT> <GPU> <SEED> <TAG>
set -e
TASK=${1:?task id}; PORT=${2:-8040}; GPU=${3:-0}; SEED=${4:-1000}; TAG=${5:-run}
ACTOR=${PERTURB_ACTOR:?set PERTURB_ACTOR}
cd ${OPENPI_ROOT:?set OPENPI_ROOT}/examples/libero
env -u _PERT_TP -u _PERT_DONE -u _PERT_STEP \
  CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU \
  TASK_IDS=$TASK \
  CHUNK_MODE=${CHUNK_MODE:-cs_q} CS_DELTA=${CS_DELTA:-1.80} \
  CS_NCAND=${CS_NCAND:-16} CS_VERBOSE=${CS_VERBOSE:-1} \
  PERTURB_FWD=1 PERTURB_LOCK=1 PERTURB_RANDOM=1 PERTURB_AXIS=1 \
  PERTURB_MODE=${PERTURB_MODE:-teleport} PERTURB_DIST=${PERTURB_DIST:-0.05} \
  PERTURB_DUR=${PERTURB_DUR:-40} PERTURB_STEP_RANGE=${PERTURB_STEP_RANGE:-15,30} \
  PERTURB_ACTOR=$ACTOR PERTURB_SEED=$SEED \
  nohup python -u main_adaptive.py --args.host 0.0.0.0 --args.port $PORT \
    --args.task-suite-name libero_10 --args.num-trials-per-task 50 --args.seed 1000 \
    > ${RESULTS_DIR:-./results}/${TAG}_t${TASK}_s${SEED}.log 2>&1 &
echo "[$TAG] launched task=$TASK port=$PORT seed=$SEED"
